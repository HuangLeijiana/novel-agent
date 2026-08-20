"""Writer Agent — generates chapter drafts with full context awareness."""

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft, Fact, StateChange
from ..models.characters import CharacterRegistry
from ..models.memory import MemoryState
from ..models.outline import ChapterPlan, EmotionalBeat, MasterOutline, Scene
from ..models.project import ProjectConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================


class ChapterContentOutput(BaseModel):
    """LLM output for chapter content."""

    title: str = Field(default="", description="章节标题")
    content: str = Field(default="", description="章节正文（Markdown格式）")
    author_notes: str = Field(default="", description="作者附注")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Handle Qwen3 output quirks for chapter content."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Alias: description / notes → author_notes
        if "author_notes" not in d or not d.get("author_notes"):
            for alias in ("description", "notes", "备注", "附注"):
                if alias in d and alias != "content":
                    d["author_notes"] = d.pop(alias)
                    break
        # content is required by prompt; use empty string as safe fallback
        if "content" not in d or not d.get("content"):
            for alias in ("正文", "text", "body"):
                if alias in d:
                    d["content"] = d.pop(alias)
                    break
        return d


class FactExtractionOutput(BaseModel):
    """LLM output for fact extraction from chapter."""

    new_facts: list[Fact] = Field(default_factory=list, description="新增事实")
    state_changes: list[StateChange] = Field(default_factory=list, description="角色状态变化")
    contradictions: list[str] = Field(
        default_factory=list,
        description="与已有事实矛盾的新事实描述（如无矛盾则为空列表）",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Handle Qwen3 output quirks for fact extraction."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Coerce string items in state_changes list
        if "state_changes" in d and isinstance(d["state_changes"], list):
            d["state_changes"] = [{"reason": x} if isinstance(x, str) else x for x in d["state_changes"]]
        # Coerce string items in new_facts list (Fact already handles this,
        # but be extra safe)
        if "new_facts" in d and isinstance(d["new_facts"], list):
            d["new_facts"] = [{"description": x} if isinstance(x, str) else x for x in d["new_facts"]]
        return d


def _build_expansion_prompt(
    content: str,
    current_count: int,
    target_count: int,
    chapter_num: int,
    chapter_title: str,
) -> str:
    """Build a prompt asking the model to expand a too-short chapter."""
    return f"""你之前写的第{chapter_num}章「{chapter_title}」只有{current_count}字，但硬性要求是不少于{target_count}字。

请将以下章节内容扩充到至少{target_count}字。扩充方法：
- 每个场景增加环境描写细节（场景的氛围、光线、声音、气味）
- 对话之间增加角色的动作、表情、心理活动
- 关键情节增加铺垫和过渡，不要让剧情跳跃太快
- 角色的内心独白和感受要写充分
- 不要改变原有情节结构和结局，只是在现有内容基础上扩充细节

【当前章节内容】
{content}

【扩充要求】
请输出完整扩充后的章节（保持原有JSON格式，title和author_notes保持不变，只扩充content字段）。
扩充后content的字数必须不少于{target_count}字，当前只有{current_count}字，还需要增加至少{target_count - current_count}字。"""


# ============================================================
# Agent
# ============================================================


class WriterAgent(BaseAgent):
    """Writes chapter drafts with full awareness of the novel bible, outline,
    character states, and memory context."""

    agent_type = "writer"

    async def generate_chapter(
        self,
        chapter_plan: ChapterPlan,
        config: ProjectConfig,
        bible: NovelBible,
        characters: CharacterRegistry,
        outline: MasterOutline,
        memory: MemoryState | None = None,
        revision_feedback: str | None = None,
        rag_context: str = "",
    ) -> ChapterDraft:
        """Generate a complete chapter draft.

        Args:
            chapter_plan: Detailed chapter plan.
            config: Project configuration.
            bible: Novel bible for world/setting reference.
            characters: Character registry with current states.
            outline: Master outline for story context.
            memory: Memory state with recent summaries.
            revision_feedback: If set, this is a revision — include editor feedback.
            rag_context: Optional RAG-retrieved context string for semantic search.

        Returns:
            ChapterDraft with the written content and extracted metadata.
        """
        logger.info(f"Writing chapter {chapter_plan.chapter_number}...")

        # Build context block
        context = self._build_full_context(
            chapter_plan=chapter_plan,
            config=config,
            bible=bible,
            characters=characters,
            outline=outline,
            memory=memory,
            rag_context=rag_context,
        )

        chapter_num = chapter_plan.chapter_number
        chapter_title = chapter_plan.title

        # Template variables
        template_vars = {
            "language": config.language,
            "tone": bible.style_contract.tone,
            "sentence_style": bible.style_contract.sentence_style,
            "word_count_target": chapter_plan.word_count_target,
            "pov_constraint": bible.rules.pov_constraint,
            "tense": bible.rules.tense,
            "forbidden_phrases": ", ".join(bible.style_contract.forbidden_phrases) if bible.style_contract.forbidden_phrases else "无特定禁用",
            "scene_break_style": bible.rules.scene_break_style,
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "goal": chapter_plan.goal,
            "conflict": chapter_plan.conflict,
            "information_increment": chapter_plan.information_increment,
            "ending_hook": chapter_plan.ending_hook,
            "context": context,
            "scenes_formatted": self._format_scenes(chapter_plan.scenes),
            "emotional_curve_formatted": self._format_emotional_curve(chapter_plan.emotional_curve),
            "characters_involved": ", ".join(chapter_plan.characters_involved),
            "scene_count": len(chapter_plan.scenes),
            "per_scene_min": max(1000, chapter_plan.word_count_target // max(len(chapter_plan.scenes), 1)),
            "revision_feedback": revision_feedback or "",
        }

        # Inline defaults (kept in sync with templates)
        system_default = self.build_system_prompt(
            role="小说作家",
            expertise="创作引人入胜的小说章节。你的文字要有质感——精准的描写、自然的对话、"
            "恰到好处的节奏。你能自然地融入世界观，让角色行为有动机可循，"
            "让每个场景都有存在的理由。",
            constraints=f"""写作约束：
- 语言：{config.language}
- 文风语调：{bible.style_contract.tone}
- 句式风格：{bible.style_contract.sentence_style}
- 【最重要】全章总字数必须不少于{chapter_plan.word_count_target}字。这是强制要求——如果你写不够，系统会拒绝你的输出。请务必在完成每个场景后自查字数，不足立刻扩充。
- 叙事视角：{bible.rules.pov_constraint}
- 时态：{bible.rules.tense}
- 禁用表达：{", ".join(bible.style_contract.forbidden_phrases) if bible.style_contract.forbidden_phrases else "无特定禁用"}
- 场景分隔符：{bible.rules.scene_break_style}""",
        )

        user_default = f"""请根据以下章节规划，写作第{chapter_num}章。

【⚠️ 字数强制要求 — 请先读这条】
全章总字数不得少于{chapter_plan.word_count_target}字。共{len(chapter_plan.scenes)}个场景，每个场景至少写{max(1000, chapter_plan.word_count_target // len(chapter_plan.scenes))}字。写完后请认真数一遍字数，不足必须扩充到达标为止。字数不够的章节会被系统打回重写。

{context}

【章节规划】
- 标题：{chapter_title}
- 目标：{chapter_plan.goal}
- 核心冲突：{chapter_plan.conflict}
- 信息增量：{chapter_plan.information_increment}
- 章末钩子：{chapter_plan.ending_hook}

【场景列表】
{self._format_scenes(chapter_plan.scenes)}

【情绪曲线】
{self._format_emotional_curve(chapter_plan.emotional_curve)}

【出场角色】
{", ".join(chapter_plan.characters_involved)}

写作要求：
1. 严格遵循章节规划的场景顺序和情绪曲线
2. 每个场景都要充分展开：环境描写（至少100字）→ 角色行动与对话（至少200字）→ 心理活动（至少100字）→ 情节推进（至少100字），确保每个场景有实质内容
3. 对话要符合每个角色的性格和说话方式，对话量要充足，不要一句话带过
4. 描写要有画面感和细节，用具体的动作、表情、环境来传达情绪，不要直接说「他很愤怒」
5. 章末必须有一个强有力的钩子：{chapter_plan.ending_hook}
6. 用Markdown格式，场景切换用「***」分隔
7. 不要出现AI味的表达（如「综上所述」「值得注意的是」「在...的过程中」等）
8. 【再次强调】全章总字数不得少于{chapter_plan.word_count_target}字。你现在开始写，写完后请自查字数，如果不够{chapter_plan.word_count_target}字，请回到场景中继续扩充细节、对话、心理描写，直到达标。"""

        if revision_feedback:
            user_default += f"\n\n【修改意见】请按以下反馈修改：\n{revision_feedback}"

        system, user = self.render_prompts(
            "generate_chapter",
            system_default=system_default,
            user_default=user_default,
            **template_vars,
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ChapterContentOutput,
            temperature_override=0.9,
            max_tokens_override=16384,
        )

        # Count words (simple split for Chinese — count characters as rough estimate)
        word_count = len(result.content.replace(" ", "").replace("\n", ""))

        # ── Auto-expand if chapter is too short ──
        target = chapter_plan.word_count_target
        if word_count < target * 0.85:
            logger.info(f"Chapter {chapter_num} first pass only {word_count} chars (< {target}), expanding...")
            try:
                expansion_prompt = _build_expansion_prompt(
                    content=result.content,
                    current_count=word_count,
                    target_count=target,
                    chapter_num=chapter_num,
                    chapter_title=chapter_title,
                )
                expand_result = await self.generate_structured(
                    system_prompt=system,
                    user_prompt=expansion_prompt,
                    response_model=ChapterContentOutput,
                    temperature_override=0.8,
                    max_tokens_override=16384,
                )
                expand_count = len(expand_result.content.replace(" ", "").replace("\n", ""))
                if expand_count > word_count:
                    delta = expand_count - word_count
                    result = expand_result
                    word_count = expand_count
                    logger.info(f"Chapter {chapter_num} expanded: {word_count} chars (+{delta})")
                else:
                    logger.warning(
                        f"Expansion didn't increase length ({expand_count} <= {word_count}), keeping first pass"
                    )
            except Exception as e:
                logger.warning(f"Expansion failed ({e}), keeping first pass: {word_count} chars")

        draft = ChapterDraft(
            chapter_number=chapter_num,
            title=result.title or chapter_title,
            content=result.content,
            word_count=word_count,
            author_notes=result.author_notes,
        )

        logger.info(f"Chapter {chapter_num} drafted: {word_count} chars")
        return draft

    async def extract_facts_and_changes(
        self,
        draft: ChapterDraft,
        bible: NovelBible,
        characters: CharacterRegistry,
        memory: MemoryState | None = None,
    ) -> ChapterDraft:
        """Extract new facts and character state changes from the chapter draft.

        Also checks for contradictions with existing facts in long-term memory.
        """
        logger.info(f"Extracting facts from chapter {draft.chapter_number}...")

        system = self.build_system_prompt(
            role="事实提取器",
            expertise="从小说章节中精确提取新增的世界观事实、角色状态变化和关系变化。"
            "你需要仔细区分'早已存在的事实'和'本章新建立的事实'。"
            "同时你要发现新事实与已有事实之间的矛盾。",
        )

        # Build character ID mapping
        char_names = {c.name: cid for cid, c in characters.characters.items()}

        # ── Gather existing relevant facts for contradiction detection ──
        existing_facts_text = ""
        if memory and memory.long_term.facts:
            # Filter to facts relevant to characters in this chapter
            relevant_chars = set(char_names.keys())
            relevant_facts = []
            for fact in memory.long_term.facts.values():
                desc = fact.description
                if any(c in desc for c in relevant_chars):
                    relevant_facts.append(fact)
            if relevant_facts:
                existing_facts_text = "\n".join(
                    f"- [{f.category}] {f.description} (确定性: {f.certainty})"
                    for f in relevant_facts[:30]
                )

        chapter_text = draft.content[:8000] if len(draft.content) > 8000 else draft.content
        truncation_note = "...(内容过长已截断)" if len(draft.content) > 8000 else ""

        user = f"""请从以下章节中提取：

【章节正文】
{chapter_text}{truncation_note}

【已知角色】{char_names}

【已有事实（用于矛盾检测）】
{existing_facts_text or "无（第一章）"}

请提取：

### 1. 新增事实（new_facts）
本章新建立的事实，分为以下类别：
- character: 角色相关（新角色、新能力、背景揭示）
- world: 世界观相关（新地点、新规则、新历史）
- plot: 剧情相关（新事件、新信息）
- relationship: 关系变化
- item: 重要物品

每个事实标注：
- id: 唯一ID
- category: 类别
- description: 事实描述
- certainty: 确定性（1.0=确认事实, 0.5=角色主观认知, 0.0=传闻/可能为假）

### 2. 角色状态变化（state_changes）
本章中角色状态的变化：
- character_id: 角色ID
- attribute: 变化的属性（如 physical.location, emotional.mood, social.status）
- old_value: 变化前的值
- new_value: 变化后的值
- reason: 变化原因

### 3. 矛盾检测（contradictions）
检查本章新事实是否与已有事实矛盾。例如：
- 已有事实说"李明左手在战斗中废掉了" → 本章写"李明双手握剑" → 矛盾
- 已有事实说"北域终年冰封" → 本章写"北域春暖花开" → 矛盾（除非有合理解释）
如果发现矛盾，将矛盾描述写入 contradictions 列表。无矛盾则为空列表。"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=FactExtractionOutput,
            temperature_override=0.3,
            max_tokens_override=2048,
        )

        draft.new_facts = result.new_facts
        draft.character_state_changes = result.state_changes

        if result.contradictions:
            logger.warning(
                f"Chapter {draft.chapter_number}: {len(result.contradictions)} fact contradictions detected!"
            )
            for c in result.contradictions:
                logger.warning(f"  Contradiction: {c}")

        logger.info(
            f"Extracted {len(draft.new_facts)} facts, {len(draft.character_state_changes)} state changes, "
            f"{len(result.contradictions)} contradictions"
        )
        return draft

    # ================================================================
    # Helpers
    # ================================================================

    def _filter_relevant_facts(
        self,
        chapter_plan: ChapterPlan,
        facts: dict[str, Fact],
        max_facts: int = 20,
    ) -> list[Fact]:
        """Filter long-term facts to only those relevant to the current chapter.

        Relevance is determined by keyword overlap between fact text and the
        chapter's characters, settings, and goal.
        """
        if not facts:
            return []

        # Build keyword set from the chapter plan
        keywords: set[str] = set()
        for char_name in chapter_plan.characters_involved:
            keywords.add(char_name)
        for scene in chapter_plan.scenes:
            keywords.add(scene.setting)
            for c in scene.characters_present:
                keywords.add(c)

        # Also include chapter goal and conflict words
        goal_words = set(chapter_plan.goal.replace("，", ",").replace("、", ",").split(","))
        conflict_words = set(chapter_plan.conflict.replace("，", ",").replace("、", ",").split(","))
        keywords.update(w.strip() for w in goal_words | conflict_words if len(w.strip()) >= 2)

        # Score each fact by keyword overlap
        scored: list[tuple[int, Fact]] = []
        for fact in facts.values():
            desc = fact.description + fact.category
            score = sum(1 for kw in keywords if kw and kw in desc)
            if score > 0:
                scored.append((score, fact))

        # Return top-N by relevance score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_facts]]

    def _build_full_context(
        self,
        chapter_plan: ChapterPlan,
        config: ProjectConfig,
        bible: NovelBible,
        characters: CharacterRegistry,
        outline: MasterOutline,
        memory: MemoryState | None = None,
        rag_context: str = "",
    ) -> str:
        """Build the comprehensive context block for the writer."""
        ctx = self.build_context_block(
            project_config=config.model_dump(),
            bible=bible.model_dump(),
            characters={cid: c.model_dump() for cid, c in characters.characters.items()}
            if hasattr(characters, "characters")
            else {},
            outline=outline.model_dump(),
            memory=memory.model_dump() if memory else None,
        )

        # ── Inject relevant historical facts from long-term memory ──
        if memory and memory.long_term.facts:
            relevant = self._filter_relevant_facts(
                chapter_plan=chapter_plan,
                facts=memory.long_term.facts,
                max_facts=20,
            )
            if relevant:
                ctx += "\n=== 相关历史事实 ===\n"
                ctx += "（以下是从前文章节中提取的、与本章相关的事实，请确保不与之矛盾）\n"
                for fact in relevant:
                    ctx += f"- [{fact.category}] {fact.description} (确定性: {fact.certainty})\n"

        # ── Inject RAG-retrieved semantic context ──
        if rag_context:
            ctx += "\n=== 语义检索上下文 ===\n"
            ctx += rag_context
            ctx += "\n"

        # Add specific scene breakdown
        ctx += "\n=== 本章场景详解 ===\n"
        for i, scene in enumerate(chapter_plan.scenes, 1):
            ctx += f"场景{i}: {scene.setting}\n"
            ctx += f"  POV: {scene.pov}\n"
            ctx += f"  在场: {', '.join(scene.characters_present)}\n"
            ctx += f"  目标: {scene.goal}\n"
            ctx += f"  冲突: {scene.conflict}\n"
            ctx += f"  结果: {scene.outcome}\n\n"

        return ctx

    def _format_scenes(self, scenes: list[Scene]) -> str:
        """Format scene list for prompt."""
        if not scenes:
            return "（无场景规划）"
        lines = []
        for s in scenes:
            lines.append(f"场景{s.number}: [{s.setting}] POV={s.pov}")
            lines.append(f"  角色: {', '.join(s.characters_present)}")
            lines.append(f"  目标: {s.goal}")
            lines.append(f"  冲突: {s.conflict}")
            lines.append(f"  结果: {s.outcome}")
        return "\n".join(lines)

    def _format_emotional_curve(self, beats: list[EmotionalBeat]) -> str:
        """Format emotional curve for prompt."""
        if not beats:
            return "（无特定情绪规划）"
        lines = []
        for b in sorted(beats, key=lambda x: x.position):
            lines.append(f"  {b.position:.0%}: {b.emotion} (强度{b.intensity:.0%})")
        return "\n".join(lines)
