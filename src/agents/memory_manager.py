"""Memory Manager Agent — updates memory artifacts after each chapter."""

import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft, Fact, PolishedChapter
from ..models.characters import CharacterRegistry
from ..models.memory import (
    ForeshadowingEntry,
    ForeshadowingTracker,
    LongTermMemory,
    MemoryState,
    ShortTermMemory,
    TimelineEvent,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


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
        existing_memory: Optional[MemoryState] = None,
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

        system = self.build_system_prompt(
            role="记忆管理者",
            expertise="你是故事的忠实记录者。你精确地记录每章发生了什么、角色有什么变化、"
                      "埋下了什么伏笔、推进了什么线索。你不会遗漏任何重要信息。",
        )

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

        user = f"""请分析以下章节，更新故事记忆：

【第{polished.chapter_number}章】{polished.title}

【章节摘要（供参考）】{draft.content[:600]}...

【本章新增事实】
{self._format_facts(draft.new_facts)}

【角色状态变化】
{self._format_state_changes(draft.character_state_changes)}

【上一章摘要】{prev_summary or '这是第一章'}

【活跃伏笔】
{prev_foreshadowing or '无'}

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
            previous_chapter_summary=(
                memory.short_term.current_chapter_summary
                if memory.short_term else ""
            ),
            recent_character_states=memory.character_states if memory.character_states else {},
            active_foreshadowing=[
                e.id for e in result.foreshadowing_updates
                if e.status == "active"
            ],
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

    def _format_facts(self, facts: list[Fact]) -> str:
        if not facts:
            return "（无新增事实）"
        return "\n".join(f"- [{f.category}] {f.description} (确定性: {f.certainty})" for f in facts)

    def _format_state_changes(self, changes: list) -> str:
        if not changes:
            return "（无状态变化）"
        return "\n".join(
            f"- {c.character_id}: {c.attribute} ({c.old_value} → {c.new_value})"
            for c in changes
        )
