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
        system = self.build_system_prompt(
            role="时间线核查员",
            expertise="你精确追踪故事中的时间流动。你能发现时间跳跃错误、事件顺序矛盾、以及任何时间相关的不一致。",
        )

        prev_events = ""
        if memory and memory.timeline:
            recent = memory.timeline[-10:]  # Last 10 events
            prev_events = "\n".join(
                f"- [{e.id}] 第{e.chapter}章: {e.description} (故事时间: {e.in_story_time})" for e in recent
            )

        user = f"""检查本章的时间线一致性：

【前序时间线事件】
{prev_events or "无（第一章）"}

【本章内容（节选）】
{draft.content[:800]}

请检查：
- 事件顺序是否合理
- 故事内时间引用是否与前文一致
- 是否有时间跳跃但没有合理解释
- 是否有"同一天发生了不可能完成的事"之类的问题"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def check_character_consistency(
        self,
        draft: ChapterDraft,
        characters: CharacterRegistry,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check that characters behave consistently with their profiles and recent states."""
        system = self.build_system_prompt(
            role="角色一致性检查员",
            expertise="你能发现角色行为与其设定性格、动机、当前状态之间的矛盾。"
            "你不会阻止角色成长或变化，但会标记没有合理解释的突然转变。",
        )

        char_briefs = []
        for cid, char in characters.characters.items():
            state = char.current_state
            if memory and cid in memory.character_states:
                state = memory.character_states[cid]
            char_briefs.append(
                f"- [{cid}] {char.name}: 性格={char.personality[:60]}, 动机={char.motivation[:60]}, 当前状态={state}"
            )
        char_text = "\n".join(char_briefs)

        user = f"""检查本章中角色行为是否一致：

【角色设定】
{char_text}

【本章内容（节选）】
{draft.content[:800]}

请检查：
- 角色行为是否符合其性格和动机
- 角色决策是否有合理的铺垫
- 角色是否出现了无解释的突然转变（OOC）
- 角色间互动是否与关系设定一致"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def check_causality(
        self,
        draft: ChapterDraft,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check that events have proper cause-and-effect relationships."""
        system = self.build_system_prompt(
            role="因果关系检查员",
            expertise="你精准分析事件之间的因果链。你能发现'因为剧情需要所以发生'的机械事件，"
            "以及缺乏合理前因的'巧合'。",
        )

        user = f"""检查本章事件的因果关系：

【本章内容（节选）】
{draft.content[:800]}

请检查：
- 事件是否有合理的前因
- 是否有太多"巧合"推动剧情
- 角色的关键决策是否有足够的动机支撑
- 是否有"机械降神"式的解决方式"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
        )
        return result.issues

    async def check_foreshadowing(
        self,
        draft: ChapterDraft,
        memory: MemoryState | None = None,
    ) -> list[Issue]:
        """Check foreshadowing usage — what's being planted, advanced, or paid off."""
        system = self.build_system_prompt(
            role="伏笔追踪员",
            expertise="你精确追踪故事中每一个伏笔的生命周期：埋下、暗示推进、回收。"
            "你能发现被遗忘的伏笔和回收不充分的伏笔。",
        )

        active_fs = ""
        if memory and memory.foreshadowing:
            active_fs = "\n".join(
                f"- [{e.id}] 第{e.planted_chapter}章: {e.description} (状态: {e.status})"
                for e in memory.foreshadowing.entries
            )

        user = f"""检查本章的伏笔处理：

【活跃伏笔】
{active_fs or "无"}

【本章内容（节选）】
{draft.content[:800]}

请检查：
- 本章是否埋下了新伏笔
- 是否推进或回收了已有伏笔
- 是否有伏笔回收过于生硬
- 是否有伏笔被遗忘（离预期回收章节已过去很久仍未回收）"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ContinuityOutput,
            temperature_override=0.3,
        )
        return result.issues
