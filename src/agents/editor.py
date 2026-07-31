"""Editor Agent — quality review across multiple dimensions."""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft
from ..models.characters import CharacterRegistry
from ..models.memory import MemoryState
from ..models.outline import ChapterPlan
from ..models.review import Issue, ReviewReport
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================

class ReviewOutput(BaseModel):
    """LLM output for quality review."""
    overall_score: float = Field(default=7.0, ge=0.0, le=10.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ConsistencyOutput(BaseModel):
    """LLM output for continuity/consistency check."""
    issues: list[Issue] = Field(default_factory=list)


class AiFlavorOutput(BaseModel):
    """LLM output for AI-flavor detection."""
    score: float = Field(default=8.0, ge=0.0, le=10.0, description="AI味评分（越高越自然）")
    issues: list[Issue] = Field(default_factory=list)


# ============================================================
# Agent
# ============================================================

class EditorAgent(BaseAgent):
    """Performs multi-dimensional quality review of chapter drafts.

    Review dimensions:
    - consistency: Setting/world consistency
    - character: Character behavior consistency
    - pacing: Narrative pacing
    - hook: Hook and pleasure-point effectiveness
    - style: Writing style stability
    - ai_flavor: AI-generated text detection
    """

    agent_type = "editor"

    async def review_chapter(
        self,
        draft: ChapterDraft,
        chapter_plan: ChapterPlan,
        bible: NovelBible,
        characters: CharacterRegistry,
        memory: Optional[MemoryState] = None,
    ) -> ReviewReport:
        """Run all quality checks and produce a review report."""
        logger.info(f"Reviewing chapter {draft.chapter_number}...")

        # Run checks (could be parallel, but sequential is simpler for MVP)
        consistency_issues = await self.check_consistency(draft, bible, characters, memory)
        pacing_issues = await self.check_pacing(draft, chapter_plan)
        style_issues = await self.check_style(draft, bible)
        ai_flavor_result = await self.detect_ai_flavor(draft)

        # Combine into a comprehensive review
        system = self.build_system_prompt(
            role="主编审",
            expertise="对小说章节进行全面的质量审查。你能从多个维度给出精准的评分和"
                      "有建设性的修改建议。",
        )

        all_issues = consistency_issues + pacing_issues + style_issues + ai_flavor_result.issues

        issues_text = "\n".join(
            f"[{i.severity}][{i.category}] {i.description}" for i in all_issues
        )

        user = f"""请对第{draft.chapter_number}章进行综合评审：

【章节目标】{chapter_plan.goal}
【预期情绪曲线】{[(b.position, b.emotion) for b in chapter_plan.emotional_curve]}
【章末钩子设计】{chapter_plan.ending_hook}

【章节正文（开头500字）】
{draft.content[:500]}

【章节正文（结尾500字）】
{draft.content[-500:] if len(draft.content) > 500 else draft.content}

【已发现的问题】
{issues_text or '无特定问题'}

请给出：
1. 总体评分（0-10）
2. 各维度评分：
   - consistency: 设定一致性
   - character: 角色行为一致性
   - pacing: 节奏
   - hook: 爽点/钩子效果
   - style: 文风稳定性
   - ai_flavor: AI味程度（越高越自然）
3. 主要问题列表
4. 本章优点
5. 改进建议

注意：评分要客观，有问题就指出，不要过度赞美。"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ReviewOutput,
            temperature_override=0.3,
        )

        # Add AI flavor score from the specialized check
        if "ai_flavor" not in result.dimension_scores:
            result.dimension_scores["ai_flavor"] = ai_flavor_result.score
        result.issues.extend(ai_flavor_result.issues)

        # Determine pass/fail
        avg_score = sum(result.dimension_scores.values()) / max(len(result.dimension_scores), 1)
        has_critical = any(i.severity == "critical" for i in result.issues)
        passed = avg_score >= 6.5 and not has_critical

        report = ReviewReport(
            chapter_number=draft.chapter_number,
            overall_score=result.overall_score,
            dimension_scores=result.dimension_scores,
            issues=result.issues,
            strengths=result.strengths,
            suggestions=result.suggestions,
            passed=passed,
        )

        logger.info(f"Chapter {draft.chapter_number} review: score={report.overall_score}, "
                     f"passed={report.passed}, issues={len(report.issues)}")
        return report

    # ================================================================
    # Individual check methods
    # ================================================================

    async def check_consistency(
        self,
        draft: ChapterDraft,
        bible: NovelBible,
        characters: CharacterRegistry,
        memory: Optional[MemoryState] = None,
    ) -> list[Issue]:
        """Check world setting and character behavior consistency."""
        system = self.build_system_prompt(
            role="一致性检查员",
            expertise="你有一双挑剔的眼睛，能发现小说中任何与设定、前文或角色性格不一致的地方。",
        )

        # Build known facts summary
        known_facts = ""
        if memory and memory.long_term:
            facts_list = list(memory.long_term.facts.values())
            known_facts = "\n".join(
                f"- [{f.category}] {f.description}" for f in facts_list[:20]
            )

        user = f"""检查以下章节是否存在一致性问题：

【世界观规则】{bible.rules.model_dump_json() if bible.rules else '无特定规则'}
【角色性格摘要】{self._format_character_briefs(characters)}
【已知事实】{known_facts or '无（第一章）'}

【章节内容（摘要）】
{self._summarize_chapter(draft)}

请找出：
- 与世界观设定矛盾的地方
- 角色行为不符合其性格/动机的地方
- 与已知事实冲突的地方
- 时间线或因果关系问题"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ConsistencyOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def check_pacing(
        self,
        draft: ChapterDraft,
        chapter_plan: ChapterPlan,
    ) -> list[Issue]:
        """Check narrative pacing."""
        system = self.build_system_prompt(
            role="节奏分析员",
            expertise="分析小说章节的叙事节奏。能判断场景推进速度是否合理，"
                      "情绪起伏是否有层次，是否出现拖沓或过快的段落。",
        )

        user = f"""分析以下章节的叙事节奏：

【规划的场景数】{len(chapter_plan.scenes)}
【规划的情绪曲线】{[(b.position, b.emotion, b.intensity) for b in chapter_plan.emotional_curve]}
【目标字数】{chapter_plan.word_count_target}
【实际字数】{draft.word_count}

【章节开头】{draft.content[:300]}

【章节结尾】{draft.content[-300:]}

请检查：
- 节奏是否合理（是否有拖沓或过快的段落）
- 场景转换是否自然
- 情绪起伏是否符合规划
- 高潮/冲突场景是否给够篇幅"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ConsistencyOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def check_style(
        self,
        draft: ChapterDraft,
        bible: NovelBible,
    ) -> list[Issue]:
        """Check writing style stability against the style contract."""
        style = bible.style_contract
        system = self.build_system_prompt(
            role="文风检查员",
            expertise="精确判断小说文本是否遵守了文风契约。能发现句式重复、用词不当、"
                      "语调偏离等问题。",
        )

        user = f"""检查以下章节是否符合文风契约：

【文风契约】
- 语调：{style.tone}
- 句式风格：{style.sentence_style}
- 叙事距离：{style.narrative_distance}
- 禁用表达：{style.forbidden_phrases}
- 推荐技法：{style.preferred_techniques}

【章节样本（中间段落）】
{draft.content[len(draft.content)//2 - 300 : len(draft.content)//2 + 300]}

请找出：
- 句式重复（如连续使用相同句式结构）
- 用词不当或滥用
- 语调偏离
- 出现了禁用表达
- 对话/描写比例是否合理"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ConsistencyOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def detect_ai_flavor(self, draft: ChapterDraft) -> AiFlavorOutput:
        """Detect AI-generated text patterns."""
        system = self.build_system_prompt(
            role="AI文本检测专家",
            expertise="你能精确识别AI生成文本的典型特征：机械的过渡词、过于工整的结构、"
                      "缺乏真实情感波动、套路化的表达方式、过度使用某些连接词等。",
        )

        user = f"""请分析以下文本的"AI味"程度：

【文本】
{draft.content[:1000]}

AI味常见特征：
- 「综上所述」「值得注意的是」「在...的过程中」「不仅...而且...」等机械表达
- 段落结构过于工整（总是总分总、问题-分析-结论）
- 情感描写空洞（「他感到一阵XX」而不是通过行动展示）
- 对话过于功能化（角色说的话都是为推进剧情服务，缺乏个性）
- 过渡词滥用（「然而」「因此」「与此同时」过度使用）
- 描写过于平均（每个场景都分配了差不多的字数，缺乏重点）

请给出：
1. AI味评分（0=完全像AI写的, 10=完全自然的人类写作）
2. 具体的问题段落和修改建议"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=AiFlavorOutput,
            temperature_override=0.3,
        )
        return result

    # ================================================================
    # Helpers
    # ================================================================

    def _format_character_briefs(self, characters: CharacterRegistry) -> str:
        """Format brief character descriptions."""
        lines = []
        for cid, char in characters.characters.items():
            lines.append(f"- [{cid}] {char.name}: {char.personality[:80]} | 动机: {char.motivation[:60]}")
        return "\n".join(lines)

    def _summarize_chapter(self, draft: ChapterDraft) -> str:
        """Create a brief summary of the chapter content."""
        content = draft.content
        if len(content) <= 500:
            return content
        # Return first 250 + last 250 chars as summary context
        return content[:250] + "\n...\n" + content[-250:]
