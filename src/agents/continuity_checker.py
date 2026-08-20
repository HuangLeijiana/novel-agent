"""Continuity Checker Agent — timeline, causality, and foreshadowing verification."""

import logging

from pydantic import BaseModel, Field

from ..models.chapter import ChapterDraft
from ..models.characters import CharacterRegistry
from ..models.memory import MemoryState
from ..models.review import Issue
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ContinuityOutput(BaseModel):
    """LLM output for continuity check."""

    issues: list[Issue] = Field(default_factory=list)


class ContinuityCheckerAgent(BaseAgent):
    """Verifies timeline consistency, causality chains, and character behavior continuity.

    This agent focuses on factual/logical consistency — it's not about style or pacing.
    It answers: "Does this make sense given everything we know?"
    """

    agent_type = "continuity_checker"

    async def check_timeline(
        self,
        draft: ChapterDraft,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check that timeline events are consistent and correctly ordered."""
        prev_events = ""
        if memory and memory.timeline:
            recent = memory.timeline[-30:]  # Last 30 events (was 10)
            prev_events = "\n".join(
                f"- [{e.id}] 第{e.chapter}章: {e.description} (故事时间: {e.in_story_time})" for e in recent
            )
        prev_events = prev_events or "无（第一章）"

        chapter_text = self._chapter_excerpt(draft.content)

        system_default = self.build_system_prompt(
            role="时间线核查员",
            expertise="你精确追踪故事中的时间流动。你能发现时间跳跃错误、事件顺序矛盾、以及任何时间相关的不一致。"
            "你会被给予完整的前序时间线和本章全文，请认真比对。",
        )
        user_default = f"""检查本章的时间线一致性：

【前序时间线事件（最近30条）】
{prev_events}

【本章全文】
{chapter_text}

请逐项检查：
- 事件顺序是否合理，与前序时间线是否一致
- 故事内时间引用（如"三天后""次日清晨"）是否与前文衔接
- 是否有时间跳跃但没有合理解释
- 是否有"同一天发生了不可能完成的事"之类的问题
- 请列出所有发现的不一致，即使看起来很微小"""

        system, user = self.render_prompts(
            "check_timeline",
            system_default=system_default,
            user_default=user_default,
            prev_events=prev_events,
            chapter_text=chapter_text,
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
            max_tokens_override=2048,
        )
        return result.issues

    async def check_character_consistency(
        self,
        draft: ChapterDraft,
        characters: CharacterRegistry,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check that characters behave consistently with their profiles and recent states."""
        char_briefs = []
        for cid, char in characters.characters.items():
            state = char.current_state
            if memory and cid in memory.character_states:
                state = memory.character_states[cid]
            char_briefs.append(
                f"- [{cid}] {char.name} ({char.role}): 性格={char.personality[:80]}, "
                f"动机={char.motivation[:80]}, 缺陷={char.flaw[:60]}, 当前状态={state}"
            )
        char_text = "\n".join(char_briefs)

        chapter_text = self._chapter_excerpt(draft.content)

        system_default = self.build_system_prompt(
            role="角色一致性检查员",
            expertise="你能发现角色行为与其设定性格、动机、当前状态之间的矛盾。"
            "你不会阻止角色成长或变化，但会标记没有合理解释的突然转变。"
            "你会被给予完整角色档案和本章全文，请仔细比对每一个出场角色的言行。",
        )
        user_default = f"""检查本章中角色行为是否一致：

【角色设定（完整档案）】
{char_text}

【本章全文】
{chapter_text}

请逐角色检查：
- 每个出场角色的言行是否符合其性格、动机和缺陷
- 角色的决策是否有合理的心理铺垫（不能"突然就做了"）
- 角色是否出现了无解释的OOC（Out of Character）
- 角色间互动是否与关系设定和历史一致
- 如果角色有成长或变化，变化过程是否可信
- 请列出所有发现的问题，包括轻微的不自然之处"""

        system, user = self.render_prompts(
            "check_character_consistency",
            system_default=system_default,
            user_default=user_default,
            char_text=char_text,
            chapter_text=chapter_text,
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
            max_tokens_override=2048,
        )
        return result.issues

    async def check_causality(
        self,
        draft: ChapterDraft,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check that events have proper cause-and-effect relationships."""
        chapter_text = self._chapter_excerpt(draft.content)

        # Include recent timeline for cross-chapter causality context
        prev_context = ""
        if memory and memory.timeline:
            recent = memory.timeline[-15:]
            prev_context = "【前序事件（用于判断因果链是否连贯）】\n" + "\n".join(
                f"- 第{e.chapter}章: {e.description}" for e in recent
            )

        system_default = self.build_system_prompt(
            role="因果关系检查员",
            expertise="你精准分析事件之间的因果链。你能发现'因为剧情需要所以发生'的机械事件，"
            "以及缺乏合理前因的'巧合'。你会被给予本章全文，请逐一检查每个情节点。",
        )
        user_default = f"""{prev_context}

【本章全文】
{chapter_text}

请逐事件检查：
- 每个事件是否有合理的前因（不能凭空发生）
- 是否有太多"巧合"推动剧情（超过2处即为异常）
- 角色的关键决策是否有足够的动机支撑
- 是否有"机械降神"（deus ex machina）式的解决方式
- 因果链是否有断裂或跳跃"""

        system, user = self.render_prompts(
            "check_causality",
            system_default=system_default,
            user_default=user_default,
            prev_context=prev_context,
            chapter_text=chapter_text,
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
            max_tokens_override=2048,
        )
        return result.issues

    async def check_foreshadowing(
        self,
        draft: ChapterDraft,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check foreshadowing usage — what's being planted, advanced, or paid off."""
        active_fs = ""
        if memory and memory.foreshadowing:
            all_entries = memory.foreshadowing.entries
            active_fs = "\n".join(
                f"- [{e.id}] 第{e.planted_chapter}章埋下: {e.description} "
                f"(状态: {e.status}, 预期第{e.expected_payoff_chapter}章回收)"
                for e in all_entries
            )
        active_fs = active_fs or "无"

        chapter_text = self._chapter_excerpt(draft.content)

        system_default = self.build_system_prompt(
            role="伏笔追踪员",
            expertise="你精确追踪故事中每一个伏笔的生命周期：埋下、暗示推进、回收。"
            "你能发现被遗忘的伏笔和回收不充分的伏笔。"
            "你会被给予所有活跃伏笔和本章全文，请仔细比对。",
        )
        user_default = f"""检查本章的伏笔处理：

【全部伏笔记录】
{active_fs}

【本章全文】
{chapter_text}

请逐项检查：
- 本章是否埋下了新伏笔（请指出具体内容和位置）
- 是否推进了已有伏笔（暗示加深读者期待）
- 是否有伏笔在本章回收（回收是否自然、充分）
- 是否有伏笔回收过于生硬（一笔带过、缺乏仪式感）
- 是否有伏笔被遗忘（状态为active但过去50章以上未提及）
- 预期回收章节已过但未回收的伏笔"""

        system, user = self.render_prompts(
            "check_foreshadowing",
            system_default=system_default,
            user_default=user_default,
            active_fs=active_fs,
            chapter_text=chapter_text,
        )

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
            max_tokens_override=2048,
        )
        return result.issues
