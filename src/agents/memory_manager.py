"""Memory Manager Agent — updates memory artifacts after each chapter."""

import logging

from pydantic import BaseModel, Field, field_validator

from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft, Fact, PolishedChapter
from ..models.characters import CharacterRegistry
from ..models.memory import (
    ForeshadowingEntry,
    LongTermMemory,
    MemoryState,
    ShortTermMemory,
    TimelineEvent,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Consolidation thresholds
STAGE_CHAPTERS = 10   # Every 10 chapters → stage summary
ARC_CHAPTERS = 50     # Every 50 chapters → arc summary
GLOBAL_CHAPTERS = 100  # Every 100 chapters → global summary


# ============================================================
# Structured output schemas
# ============================================================


class ChapterSummaryOutput(BaseModel):
    """LLM output for chapter summary and memory updates."""

    chapter_summary: str = Field(default="", description="本章摘要（200字以内）")
    timeline_events: list[TimelineEvent] = Field(default_factory=list, description="时间线事件")
    foreshadowing_updates: list[ForeshadowingEntry] = Field(
        default_factory=list,
        description="伏笔更新（新增/推进/回收）",
    )
    world_changes: list[str] = Field(default_factory=list, description="世界观变化")
    unresolved_issues: list[str] = Field(default_factory=list, description="未解决的问题/线索")

    @field_validator("world_changes", "unresolved_issues", mode="before")
    @classmethod
    def coerce_list_field(cls, v):
        """Coerce non-list values into a list for list-typed fields.

        Local LLMs sometimes return booleans (True) or strings ('True')
        instead of a list — wrap them into a single-element list or
        return an empty list for bare True.
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, bool):
            return [] if v is True else [v]
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                import json

                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            return [stripped]
        if isinstance(v, (int, float)):
            return [str(v)]
        return [str(v)]


# ============================================================
# Agent
# ============================================================


class MemoryManagerAgent(BaseAgent):
    """Manages short-term and long-term memory after each chapter is completed.

    Updates:
    - Chapter summaries (short-term + long-term)
    - Timeline events
    - Foreshadowing tracker (new / advance / payoff)
    - Character states
    - World changes
    - Known facts registry
    """

    agent_type = "memory_manager"

    async def update_memory(
        self,
        polished: PolishedChapter,
        draft: ChapterDraft,
        bible: NovelBible,
        characters: CharacterRegistry,
        existing_memory: MemoryState | None = None,
    ) -> MemoryState:
        """Update all memory artifacts after chapter completion.

        Args:
            polished: The polished/final chapter.
            draft: The original draft (has extracted facts and state changes).
            bible: Novel bible for context.
            characters: Character registry.
            existing_memory: Previous memory state to update.

        Returns:
            Updated MemoryState.
        """
        logger.info(f"Updating memory for chapter {polished.chapter_number}...")

        # Build context
        prev_summary = ""
        prev_foreshadowing = ""
        if existing_memory:
            prev_summary = existing_memory.short_term.current_chapter_summary
            active_fs = existing_memory.foreshadowing.active
            prev_foreshadowing = "\n".join(
                f"- [{e.id}] 第{e.planted_chapter}章埋下: {e.description} (预期第{e.expected_payoff_chapter}章回收)"
                for e in active_fs
            )

        system_default = self.build_system_prompt(
            role="记忆管理者",
            expertise="你是故事的忠实记录者。你精确地记录每章发生了什么、角色有什么变化、"
            "埋下了什么伏笔、推进了什么线索。你不会遗漏任何重要信息。",
        )
        user_default = f"""请分析以下章节，更新故事记忆：

【第{polished.chapter_number}章】{polished.title}

【章节摘要（供参考）】{draft.content[:600]}...

【本章新增事实】
{self._format_facts(draft.new_facts)}

【角色状态变化】
{self._format_state_changes(draft.character_state_changes)}

【上一章摘要】{prev_summary or "这是第一章"}

【活跃伏笔】
{prev_foreshadowing or "无"}

请生成：

1. **本章摘要**（200字以内，概括本章核心事件和推进）

2. **时间线事件**：
   - 提取本章中的重要时间线事件（1-3个）
   - 每个事件标注：在故事内的时间、描述、涉及角色、重要性

3. **伏笔更新**：
   - 新增伏笔（本章埋下的新伏笔）
   - 推进已有伏笔（如有）
   - 回收伏笔（如有）

4. **世界观变化**：本章是否改变了世界观状态

5. **未解决问题**：本章留下了什么未解决的问题或线索"""

        system, user = self.render_prompts(
            "update_memory",
            system_default=system_default,
            user_default=user_default,
            chapter_number=polished.chapter_number,
            chapter_title=polished.title,
            chapter_excerpt=draft.content[:600],
            facts_text=self._format_facts(draft.new_facts),
            state_changes_text=self._format_state_changes(draft.character_state_changes),
            prev_summary=prev_summary or "这是第一章",
            prev_foreshadowing=prev_foreshadowing or "无",
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ChapterSummaryOutput,
            temperature_override=0.3,
        )

        # Build updated memory
        memory = existing_memory or MemoryState()

        # Short-term memory
        memory.short_term = ShortTermMemory(
            current_chapter_summary=result.chapter_summary,
            previous_chapter_summary=(memory.short_term.current_chapter_summary if memory.short_term else ""),
            recent_character_states=memory.character_states if memory.character_states else {},
            active_foreshadowing=[e.id for e in result.foreshadowing_updates if e.status == "active"],
            unresolved_hooks=result.unresolved_issues,
        )

        # Long-term memory
        if memory.long_term is None:
            memory.long_term = LongTermMemory()

        memory.long_term.chapter_summaries[polished.chapter_number] = result.chapter_summary

        # Merge facts
        for fact in draft.new_facts:
            memory.long_term.facts[fact.id] = fact

        # Timeline
        memory.timeline.extend(result.timeline_events)

        # Foreshadowing
        for entry in result.foreshadowing_updates:
            existing = next(
                (e for e in memory.foreshadowing.entries if e.id == entry.id),
                None,
            )
            if existing:
                # Update existing entry
                existing.status = entry.status
                existing.payoff_chapter = entry.payoff_chapter
            else:
                memory.foreshadowing.entries.append(entry)

        # Character states
        for change in draft.character_state_changes:
            if change.character_id not in memory.character_states:
                memory.character_states[change.character_id] = {}
            memory.character_states[change.character_id][change.attribute] = change.new_value

        logger.info(
            f"Memory updated: {len(result.timeline_events)} timeline events, "
            f"{len(result.foreshadowing_updates)} foreshadowing updates"
        )
        return memory

    # ================================================================
    # Hierarchical consolidation
    # ================================================================

    async def consolidate_periodically(
        self,
        memory: MemoryState,
        chapter_num: int,
    ) -> MemoryState:
        """Generate hierarchical summaries at threshold boundaries.

        - Every 10 chapters → stage summary (~500 chars)
        - Every 50 chapters → arc summary (~800 chars)
        - Every 100 chapters → global summary (~1000 chars)

        Returns the updated memory (mutated in place).
        """
        if chapter_num % STAGE_CHAPTERS == 0 and chapter_num > 0:
            memory = await self._consolidate_stage(memory, chapter_num)

        if chapter_num % ARC_CHAPTERS == 0 and chapter_num > 0:
            memory = await self._consolidate_arc(memory, chapter_num)

        if chapter_num % GLOBAL_CHAPTERS == 0 and chapter_num > 0:
            memory = await self._consolidate_global(memory, chapter_num)

        return memory

    async def _consolidate_stage(
        self, memory: MemoryState, chapter_num: int
    ) -> MemoryState:
        """Compress the last 10 chapters into a stage summary."""
        stage_num = chapter_num // STAGE_CHAPTERS
        start_ch = chapter_num - STAGE_CHAPTERS + 1
        end_ch = chapter_num

        # Gather chapter summaries for this stage
        summaries = []
        for ch in range(start_ch, end_ch + 1):
            s = memory.long_term.chapter_summaries.get(ch, "")
            if s:
                summaries.append(f"第{ch}章: {s}")

        if not summaries:
            logger.warning(f"No summaries found for stage {stage_num} (chapters {start_ch}-{end_ch})")
            return memory

        # Gather timeline events in this range
        stage_events = [e for e in memory.timeline if start_ch <= e.chapter <= end_ch]
        events_text = "\n".join(
            f"- 第{e.chapter}章 {e.in_story_time}: {e.description}" for e in stage_events[-20:]
        )

        system = self.build_system_prompt(
            role="故事压缩器",
            expertise="你能将多章内容压缩为简洁的阶段摘要，保留核心情节推进、角色变化和重要伏笔，"
            "同时丢弃次要细节。",
        )

        user = f"""请将以下{STAGE_CHAPTERS}章内容压缩为一段阶段摘要（500字以内）。

【章节范围】第{start_ch}章 ~ 第{end_ch}章

【各章摘要】
{chr(10).join(summaries)}

【关键时间线事件】
{events_text or "无"}

【当前活跃伏笔】
{chr(10).join(f"- {e.description}" for e in memory.foreshadowing.active[:10]) or "无"}

请生成阶段摘要，包含：
1. 本阶段的核心情节推进（最重要的2-3条事件线）
2. 关键角色的状态变化
3. 重要的伏笔埋下或推进
4. 阶段结尾的故事状态（为下一阶段提供衔接上下文）

只输出摘要文本，不需要JSON格式。控制在500字以内。"""

        result = await self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature_override=0.3,
        )

        summary = result.content.strip()
        memory.long_term.stage_summaries[stage_num] = summary
        memory.stage_boundaries.append(chapter_num)
        logger.info(
            f"Stage {stage_num} summary generated ({len(summary)} chars): chapters {start_ch}-{end_ch}"
        )
        return memory

    async def _consolidate_arc(
        self, memory: MemoryState, chapter_num: int
    ) -> MemoryState:
        """Compress the last 50 chapters into an arc summary."""
        arc_num = chapter_num // ARC_CHAPTERS
        start_ch = chapter_num - ARC_CHAPTERS + 1
        end_ch = chapter_num

        # Use stage summaries as building blocks
        stage_texts = []
        for sn in sorted(memory.long_term.stage_summaries.keys()):
            s_start = (sn - 1) * STAGE_CHAPTERS + 1
            s_end = sn * STAGE_CHAPTERS
            if s_start >= start_ch and s_end <= end_ch:
                stage_texts.append(f"阶段{sn} (第{s_start}-{s_end}章): {memory.long_term.stage_summaries[sn]}")

        if not stage_texts:
            # Fallback: use raw chapter summaries
            summaries = []
            for ch in range(start_ch, end_ch + 1):
                s = memory.long_term.chapter_summaries.get(ch, "")
                if s:
                    summaries.append(f"第{ch}章: {s}")
            if not summaries:
                logger.warning(f"No data for arc {arc_num}")
                return memory
            stage_texts = [chr(10).join(summaries[:50])]

        # Key timeline events for this arc
        arc_events = [
            e for e in memory.timeline
            if start_ch <= e.chapter <= end_ch and e.importance == "major"
        ]
        events_text = "\n".join(
            f"- 第{e.chapter}章: {e.description}" for e in arc_events[-15:]
        )

        system = self.build_system_prompt(
            role="故事弧压缩器",
            expertise="你能将一个大故事弧（50章）的内容提炼为高级摘要，聚焦于弧级结构："
            "主要冲突的演变、角色弧的推进、主题的发展。",
        )

        user = f"""请将以下{ARC_CHAPTERS}章内容压缩为一段弧摘要（800字以内）。

【弧范围】第{start_ch}章 ~ 第{end_ch}章

【阶段摘要】
{chr(10).join(stage_texts)}

【重大事件】
{events_text or "无"}

【活跃伏笔】
{chr(10).join(f"- {e.description} (第{e.planted_chapter}章埋下)" for e in memory.foreshadowing.active[:15]) or "无"}

请生成弧摘要，包含：
1. 本弧的主要冲突及演变
2. 主角及其他关键角色的弧线推进
3. 已回收的重要伏笔和仍在活跃的伏笔
4. 世界观的重要揭示或变化
5. 弧结尾的故事状态（为下一弧提供衔接）

只输出摘要文本，控制在800字以内。"""

        result = await self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature_override=0.3,
        )

        summary = result.content.strip()
        memory.long_term.arc_summaries[arc_num] = summary
        memory.arc_boundaries.append(chapter_num)
        logger.info(f"Arc {arc_num} summary generated ({len(summary)} chars): chapters {start_ch}-{end_ch}")
        return memory

    async def _consolidate_global(
        self, memory: MemoryState, chapter_num: int
    ) -> MemoryState:
        """Update the global story summary from arc summaries."""
        arc_texts = []
        for an in sorted(memory.long_term.arc_summaries.keys()):
            arc_texts.append(f"弧{an}: {memory.long_term.arc_summaries[an]}")

        if not arc_texts:
            logger.warning("No arc summaries for global consolidation")
            return memory

        system = self.build_system_prompt(
            role="全书脉络压缩器",
            expertise="你能将整本书的多个大弧提炼为一个简明的全局摘要，"
            "让作者一眼看清故事的宏观结构。",
        )

        user = f"""请将以下全书内容压缩为全局摘要（1000字以内）。

【已完成的弧】
{chr(10).join(arc_texts)}

请生成全局摘要，包含：
1. 故事的总体脉络（从开始到现在，经历了哪些大阶段）
2. 主角的完整成长轨迹
3. 已解决和仍在进行的主要冲突
4. 当前故事所处的位置和未完成的主要线索

只输出摘要文本，控制在1000字以内。"""

        result = await self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature_override=0.3,
        )

        memory.long_term.global_summary = result.content.strip()
        logger.info(
            f"Global summary updated ({len(memory.long_term.global_summary)} chars) at chapter {chapter_num}"
        )
        return memory

    def _format_facts(self, facts: list[Fact]) -> str:
        if not facts:
            return "（无新增事实）"
        return "\n".join(f"- [{f.category}] {f.description} (确定性: {f.certainty})" for f in facts)

    def _format_state_changes(self, changes: list) -> str:
        if not changes:
            return "（无状态变化）"
        return "\n".join(f"- {c.character_id}: {c.attribute} ({c.old_value} → {c.new_value})" for c in changes)
