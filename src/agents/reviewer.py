"""Adversarial Reviewer Agent — fresh-context independent chapter critique.

Inspired by ai-job-search-master's drafter-reviewer dual-agent pattern:
  WriterAgent drafts → ReviewerAgent (fresh context) adversarially critiques
  → WriterAgent/RefinerAgent revises

KEY DESIGN: The reviewer uses an INDEPENDENT ModelScheduler call with a
fresh context. It does NOT share the writer's reasoning process, preventing
"self-satisfied" blind spots where an agent misses its own errors.

The reviewer's job is to be a SKEPTIC — find problems, not be nice.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft
from ..models.characters import CharacterRegistry
from ..models.memory import MemoryState
from ..models.outline import ChapterPlan, MasterOutline
from ..models.project import ProjectConfig
from ..models.review import Issue, ReviewReport
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================


class AdversarialReviewOutput(BaseModel):
    """Structured output from the adversarial reviewer.

    More detailed than the standard ReviewOutput — designed to catch issues
    that a same-context reviewer would miss.
    """

    overall_score: float = Field(default=7.0, ge=0.0, le=10.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    # ── Adversarial-specific fields ──
    plot_holes: list[str] = Field(
        default_factory=list,
        description="逻辑漏洞或情节矛盾（可能被作者忽略的）",
    )
    character_inconsistencies: list[str] = Field(
        default_factory=list,
        description="角色行为与设定不符的地方",
    )
    missed_opportunities: list[str] = Field(
        default_factory=list,
        description="本章应该做但没有做的事（爽点、伏笔、信息增量）",
    )
    reader_drop_points: list[str] = Field(
        default_factory=list,
        description="读者可能弃书的段落及原因",
    )
    revision_required: bool = Field(
        default=False,
        description="是否需要强制修订",
    )


# ============================================================
# Agent
# ============================================================


class ReviewerAgent(BaseAgent):
    """Adversarial chapter reviewer with fresh, independent context.

    This agent is designed to be instantiated with a SEPARATE scheduler
    instance (or at minimum, a separate provider) so it doesn't share
    the writer's LLM context. This is the key architectural insight
    from ai-job-search-master: a single agent reviewing its own work
    is "self-satisfied" and misses blind spots.

    Usage:
        # In orchestrator:
        reviewer = ReviewerAgent(scheduler)  # same scheduler, but...
        report = await reviewer.adversarial_review(
            draft=draft,
            chapter_plan=chapter_plan,
            bible=bible,
            characters=characters,
            outline=outline,
            memory=memory,
            # NOTE: does NOT receive the writer's internal reasoning
        )
    """

    agent_type = "reviewer"

    async def adversarial_review(
        self,
        draft: ChapterDraft,
        chapter_plan: ChapterPlan,
        bible: NovelBible,
        characters: CharacterRegistry,
        outline: Optional[MasterOutline] = None,
        memory: Optional[MemoryState] = None,
        config: Optional[ProjectConfig] = None,
    ) -> AdversarialReviewOutput:
        """Perform an adversarial review of a chapter draft.

        Key difference from EditorAgent.review_chapter():
        - This is a SINGLE comprehensive pass (not multiple specialized checks)
        - The prompt is designed to make the LLM a SKEPTIC, not an editor
        - The reviewer does NOT see the writer's reasoning or the draft process
        - The reviewer independently verifies against the source materials

        Args:
            draft: The chapter draft to review.
            chapter_plan: The chapter plan the writer was supposed to follow.
            bible: World bible for consistency checking.
            characters: Character registry for behavior verification.
            outline: Master outline for plot coherence.
            memory: Memory state for continuity.
            config: Project config for style/tone verification.

        Returns:
            AdversarialReviewOutput with detailed critique.
        """
        logger.info(f"Adversarial review of chapter {draft.chapter_number}...")

        system = self._build_adversarial_system_prompt(bible, config)
        user = self._build_adversarial_user_prompt(
            draft=draft,
            chapter_plan=chapter_plan,
            bible=bible,
            characters=characters,
            outline=outline,
            memory=memory,
        )

        try:
            result = await self.generate_structured(
                system_prompt=system,
                user_prompt=user,
                response_model=AdversarialReviewOutput,
                temperature_override=0.4,  # slightly more creative than Editor's 0.3
                max_tokens_override=4096,
            )
            logger.info(
                f"Adversarial review complete: score={result.overall_score}, "
                f"issues={len(result.issues)}, plot_holes={len(result.plot_holes)}, "
                f"revision_required={result.revision_required}"
            )
            return result
        except Exception as e:
            logger.error(f"Adversarial review failed: {e}")
            # Return a minimal output — don't block the pipeline
            return AdversarialReviewOutput(
                overall_score=7.0,
                issues=[],
                suggestions=[f"Adversarial review failed: {e}. Using Editor review only."],
                revision_required=False,
            )

    def to_review_report(self, adversarial: AdversarialReviewOutput) -> ReviewReport:
        """Convert adversarial review output to a standard ReviewReport.

        This allows the adversarial review to be merged into the existing
        review pipeline without changing downstream consumers.
        """
        # Add plot holes and reader drop points as critical issues
        all_issues = list(adversarial.issues)

        for hole in adversarial.plot_holes:
            all_issues.append(
                Issue(
                    severity="critical",
                    category="plot_hole",
                    description=hole,
                    suggestion="需要修复此逻辑漏洞：确保因果链完整",
                )
            )

        for drop in adversarial.reader_drop_points:
            all_issues.append(
                Issue(
                    severity="major",
                    category="reader_engagement",
                    description=f"读者弃书风险：{drop}",
                    suggestion="增强此处的爽点、悬念或信息增量",
                )
            )

        for missed in adversarial.missed_opportunities:
            all_issues.append(
                Issue(
                    severity="major",
                    category="missed_opportunity",
                    description=missed,
                    suggestion="考虑在本章中加入此元素",
                )
            )

        for inc in adversarial.character_inconsistencies:
            all_issues.append(
                Issue(
                    severity="major",
                    category="character",
                    description=inc,
                    suggestion="检查角色设定，确保行为有合理的动机铺垫",
                )
            )

        # Calculate pass/fail
        avg_score = sum(adversarial.dimension_scores.values()) / max(len(adversarial.dimension_scores), 1)
        has_critical = any(i.severity == "critical" for i in all_issues) or bool(adversarial.plot_holes)

        return ReviewReport(
            chapter_number=0,  # Will be set by caller
            overall_score=adversarial.overall_score,
            dimension_scores=adversarial.dimension_scores,
            issues=all_issues,
            strengths=adversarial.strengths,
            suggestions=adversarial.suggestions,
            passed=avg_score >= 6.5 and not has_critical,
        )

    # ================================================================
    # Prompt builders
    # ================================================================

    def _build_adversarial_system_prompt(
        self,
        bible: NovelBible,
        config: Optional[ProjectConfig] = None,
    ) -> str:
        """Build the adversarial system prompt.

        The key phrase is "挑剔的审稿人" (picky reviewer) — we want the LLM
        to actively look for problems, not to be balanced or diplomatic.
        """
        style = bible.style_contract

        prompt = f"""你是一位极其挑剔的小说审稿人。

你的专长：你有一双能发现任何问题的眼睛。你的职责不是赞美，而是找出每一处
瑕疵、矛盾、逻辑漏洞、节奏问题和读者可能弃书的地方。你对AI生成的套路表达
有天然的敏感，你对角色行为的一致性有强迫症级别的执着。

你的审稿哲学：
1. 默认假设有问题，直到确认没有 — 而不是默认没问题
2. 从读者角度思考：读到这段会不会想弃书？
3. 从作者角度思考：这个设定后面会不会崩？
4. 从编辑角度思考：这段能不能删掉而不影响故事？
5. 你不需要「平衡」评价 — 有问题就直接指出

文风基准：
- 语调：{style.tone}
- 句式风格：{style.sentence_style}
- 禁用表达：{", ".join(style.forbidden_phrases) if style.forbidden_phrases else "无"}
- 推荐技法：{", ".join(style.preferred_techniques) if style.preferred_techniques else "无"}
"""

        if config:
            prompt += f"\n目标读者：{config.target_readers}"

        return prompt

    def _build_adversarial_user_prompt(
        self,
        draft: ChapterDraft,
        chapter_plan: ChapterPlan,
        bible: NovelBible,
        characters: CharacterRegistry,
        outline: Optional[MasterOutline] = None,
        memory: Optional[MemoryState] = None,
    ) -> str:
        """Build the adversarial user prompt with independent context.

        IMPORTANT: This prompt is designed to give the reviewer the RAW MATERIALS
        (bible, characters, outline, chapter plan) but NOT the writer's reasoning
        or the drafting process. The reviewer forms its own judgment independently.
        """
        parts = []

        # ── Chapter metadata ──
        parts.append(f"## 审稿任务：第{draft.chapter_number}章《{draft.title}》")
        parts.append("")
        parts.append("请以最挑剔的标准审阅本章。你的目标是找出所有问题，包括那些作者自己可能没意识到的盲点。")

        # ── What the chapter was SUPPOSED to do ──
        parts.append("")
        parts.append("## 本章规划（对照标准）")
        parts.append(f"- 目标：{chapter_plan.goal}")
        parts.append(f"- 核心冲突：{chapter_plan.conflict}")
        parts.append(f"- 信息增量：{chapter_plan.information_increment}")
        parts.append(f"- 章末钩子设计：{chapter_plan.ending_hook}")
        parts.append(f"- 规划场景数：{len(chapter_plan.scenes)}")
        parts.append(f"- 目标字数：{chapter_plan.word_count_target} | 实际字数：{draft.word_count}")
        parts.append(f"- 出场角色：{', '.join(chapter_plan.characters_involved)}")

        # ── Source materials (independent verification) ──
        parts.append("")
        parts.append("## 世界观规则（用于验证一致性）")
        if bible.rules:
            parts.append(f"- POV约束：{bible.rules.pov_constraint}")
            parts.append(f"- 时态：{bible.rules.tense}")
            parts.append(f"- 场景分隔符：{bible.rules.scene_break_style}")
        parts.append(f"- 世界类型：{bible.world.world_type}")
        if bible.world.magic_system:
            parts.append(f"- 力量体系：{bible.world.magic_system}")

        # ── Character references ──
        parts.append("")
        parts.append("## 角色设定（用于验证角色行为一致性）")
        involved_ids = set()
        for cid, char in characters.characters.items():
            if char.name in chapter_plan.characters_involved:
                involved_ids.add(cid)
                parts.append(
                    f"- [{cid}] {char.name} ({char.role}): "
                    f"性格={char.personality[:80]} | 动机={char.motivation[:80]} | 缺陷={char.flaw[:60]}"
                )
        if not involved_ids:
            # Include all characters if none matched
            for cid, char in list(characters.characters.items())[:5]:
                parts.append(
                    f"- [{cid}] {char.name} ({char.role}): 性格={char.personality[:80]} | 动机={char.motivation[:80]}"
                )

        # ── Outline context ──
        if outline:
            parts.append("")
            parts.append("## 故事大纲（用于验证情节一致性）")
            parts.append(f"- 主线：{outline.logline}")
            current_volume = None
            for vol in outline.volumes or []:
                if hasattr(vol, "start_chapter") and hasattr(vol, "end_chapter"):
                    if vol.start_chapter <= draft.chapter_number <= vol.end_chapter:
                        current_volume = vol
                        break
            if current_volume:
                parts.append(f"- 当前卷：{current_volume.title}")
                parts.append(f"- 卷范围：第{current_volume.start_chapter}-{current_volume.end_chapter}章")

        # ── Memory context ──
        if memory:
            st = memory.short_term
            if st:
                parts.append("")
                parts.append("## 前情提要")
                parts.append(f"- 上一章摘要：{st.current_chapter_summary}")
                if st.unresolved_hooks:
                    parts.append(f"- 未解决钩子：{st.unresolved_hooks}")

        # ── THE CHAPTER TEXT ──
        parts.append("")
        parts.append("## 本章正文（请逐段审阅）")
        parts.append("")
        # Include full text for thorough review
        max_len = 8000
        content = draft.content
        if len(content) > max_len:
            # Include first 1/3 and last 1/3, summarize middle
            third = max_len // 3
            parts.append(content[: third * 2])
            parts.append(f"\n... (中间{len(content) - third * 3}字省略) ...\n")
            parts.append(content[-third:])
        else:
            parts.append(content)

        # ── Review instructions ──
        parts.append("")
        parts.append("## 审阅指令")
        parts.append("""
请从以下维度逐一审查（不要跳过任何维度）：

### 1. 情节逻辑 (plot_holes)
- 有没有"因为剧情需要所以发生"的机械事件？
- 因果链是否完整？每个事件有合理的前因吗？
- 有没有角色做出不符合其智商的决策？
- 时间线是否合理？

### 2. 角色一致性 (character_inconsistencies)
- 角色行为是否与其性格、动机、当前状态一致？
- 角色对话是否有个性，还是所有角色说话都一样？
- 有没有OOC（角色崩坏）的瞬间？

### 3. 爽点与节奏 (reader_drop_points)
- 哪些段落会让读者觉得无聊想跳过？
- 爽点密度是否足够？开篇300字有没有钩子？
- 章末是否有让人必须点下一章的冲动？
- 情绪曲线是否有起伏，还是平的？

### 4. 错失机会 (missed_opportunities)
- 本章应该做但没有做的事是什么？
- 有没有可以加入的世界观展示？
- 有没有可以推进的伏笔？
- 有没有可以强化的爽点？

### 5. AI味检测
- 有没有「综上所述」「值得注意的是」等机械表达？
- 段落结构是否过于工整？
- 情感描写是否空洞（tell而非show）？
- 对话是否过于功能化？

### 6. 总体评价
- 本章最大的问题是什么？（必须指出至少一个）
- revision_required: 是否需要强制修订？
  - 如果有逻辑漏洞(plot_hole) → True
  - 如果有角色崩坏(OOC) → True
  - 如果评分 < 6.0 → True
  - 否则 → False
""")

        return "\n".join(parts)
