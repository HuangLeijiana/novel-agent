"""Character Manager Agent — creates and maintains character profiles."""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible, Theme
from ..models.characters import (
    ArcBeat,
    CharacterProfile,
    CharacterRegistry,
    CharacterState,
    Relationship,
)
from ..models.project import ProjectConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================

class CharacterProfilesOutput(BaseModel):
    """LLM output schema for character profiles."""
    characters: list[CharacterProfile] = Field(default_factory=list)


class RelationshipMapOutput(BaseModel):
    """LLM output schema for relationship mapping."""
    relationships: dict[str, list[Relationship]] = Field(
        default_factory=dict,
        description="Mapping: character_id -> list of relationships",
    )


class CharacterArcOutput(BaseModel):
    """LLM output schema for character arcs."""
    arcs: dict[str, list[ArcBeat]] = Field(
        default_factory=dict,
        description="Mapping: character_id -> arc beats",
    )


# ============================================================
# Agent
# ============================================================

class CharacterManagerAgent(BaseAgent):
    """Creates and manages character profiles, relationships, and arcs."""

    agent_type = "character_manager"

    async def create_all_characters(
        self,
        config: ProjectConfig,
        bible: NovelBible,
    ) -> CharacterRegistry:
        """Create all major characters for the novel.

        This is a multi-step process:
        1. Generate core character profiles
        2. Build relationships between characters
        3. Design character arcs
        """
        logger.info("Creating characters...")

        # Step 1: Generate profiles
        profiles = await self._generate_profiles(config, bible)

        # Build registry
        registry = CharacterRegistry()
        for profile in profiles:
            registry.characters[profile.id] = profile

        # Step 2: Build relationships
        registry = await self.build_relationships(registry, bible)

        # Step 3: Design arcs (for major characters)
        registry = await self.design_arcs(registry, bible)

        logger.info(f"Created {len(registry.characters)} characters")
        return registry

    async def _generate_profiles(
        self,
        config: ProjectConfig,
        bible: NovelBible,
    ) -> list[CharacterProfile]:
        """Generate core character profiles."""
        system = self.build_system_prompt(
            role="角色设计师",
            expertise="创造立体、有深度、令人难忘的小说角色。精通角色原型（英雄、导师、"
                      "伙伴、信使、捣蛋鬼、阴影等），能为每个角色赋予独特的声音、动机和缺陷。",
        )

        # Build context
        world = bible.world
        themes = bible.themes
        conflicts = bible.core_conflicts

        theme_names = ", ".join(t.name for t in themes)
        conflict_descs = "\n".join(f"- {c.description}" for c in conflicts)

        user = f"""请为以下小说创建角色阵容：

【故事灵感】{config.inspiration}
【题材】{', '.join(config.genre)}
【世界】{world.name}（{world.world_type}）
【主题】{theme_names}
【核心冲突】
{conflict_descs}

请创建以下角色（至少 4-6 个）：

1. **主角（protagonist）** — 必须创建
2. **反派/对手（antagonist）** — 必须创建
3. **重要配角（deuteragonist）** — 主角的盟友或伙伴
4. **导师角色（mentor）** — 可选的导师/引导者
5. **其他配角（supporting）** — 1-2个重要配角

每个角色需要：
- id: 唯一ID（如 char_mc, char_ant, char_ally 等）
- name: 名字
- role: protagonist/antagonist/deuteragonist/supporting/minor
- archetype: 原型（hero/mentor/herald/trickster/shadow 等）
- age: 年龄
- gender: 性别
- appearance: 外貌描述（具体、有辨识度）
- personality: 性格特征（包括优点和缺点，要有矛盾性）
- motivation: 核心动机（ta真正想要什么）
- flaw: 主要缺陷（阻碍ta成长的障碍）
- backstory: 背景故事（1-2段，解释ta为何成为现在的样子）
- abilities: 能力列表

角色设计要求：
- 主角要有让人共鸣的动机和明显的缺陷
- 反派要有合理的动机，不能是纯粹的邪恶
- 角色之间要有潜在的冲突和互补关系
- 每个角色要有独特的声音和行事风格
- 服务于主题和核心冲突"""

        if config.taboo_content:
            user += f"\n\n【避讳】避免：{', '.join(config.taboo_content)}"

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=CharacterProfilesOutput,
            temperature_override=0.7,
            max_tokens_override=8192,
        )
        return result.characters

    async def build_relationships(
        self,
        registry: CharacterRegistry,
        bible: NovelBible,
    ) -> CharacterRegistry:
        """Build relationship network between all characters."""
        system = self.build_system_prompt(
            role="角色关系设计师",
            expertise="设计复杂、真实、有张力的角色关系网络。理解权力动态、情感层次、"
                      "信任构建和关系演变。",
        )

        char_summaries = []
        for cid, char in registry.characters.items():
            char_summaries.append(
                f"[{cid}] {char.name} ({char.role}) - {char.personality[:100]}"
            )
        char_list = "\n".join(char_summaries)

        user = f"""基于以下角色列表，为每对角色设计关系：

【角色列表】
{char_list}

【主题】{', '.join(t.name for t in bible.themes)}

为每个角色与其他角色的关系定义：
- relationship_type: friend/rival/lover/enemy/family/mentor/subordinate/acquaintance
- trust: 信任度 0.0-1.0
- intimacy: 亲密度 0.0-1.0
- power_balance: 权力关系（0.0=服从对方, 0.5=平等, 1.0=支配对方）
- history: 共同的过往（1-2句话）
- notes: 补充说明

注意：
- 关系要服务于故事冲突和主题
- 要有张力和变化空间，不要都是和谐的
- 主角和反派的关系是重中之重"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=RelationshipMapOutput,
            temperature_override=0.6,
        )

        # Apply relationships to characters
        for char_id, relationships in result.relationships.items():
            if char_id in registry.characters:
                registry.characters[char_id].relationships = relationships

        return registry

    async def design_arcs(
        self,
        registry: CharacterRegistry,
        bible: NovelBible,
    ) -> CharacterRegistry:
        """Design character arcs for major characters."""
        system = self.build_system_prompt(
            role="角色弧线设计师",
            expertise="为角色设计有层次的成长弧线。精通各种弧线类型（正向成长、堕落、"
                      "救赎、幻灭等），能让角色变化既出人意料又合情合理。",
        )

        major_chars = {
            cid: char for cid, char in registry.characters.items()
            if char.role in ("protagonist", "antagonist", "deuteragonist")
        }

        char_descriptions = []
        for cid, char in major_chars.items():
            char_descriptions.append(
                f"[{cid}] {char.name} - 动机: {char.motivation} - 缺陷: {char.flaw}"
            )
        char_list = "\n".join(char_descriptions)

        user = f"""为主要角色设计角色弧线（变化轨迹）：

【角色】
{char_list}

为每个角色设计 3-5 个弧线节点（ArcBeat），每个节点包含：
- chapter: 大致发生在哪一章（估算即可）
- event: 发生了什么事
- change: 角色如何因此改变
- new_trait: 获得/显露了什么新特质（可选）
- lost_trait: 失去/压抑了什么特质（可选）

注意：
- 弧线要体现角色的成长或堕落
- 变化要循序渐进，有因果关联
- 关键节点应和故事转折点对齐"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=CharacterArcOutput,
            temperature_override=0.7,
        )

        for char_id, arc_beats in result.arcs.items():
            if char_id in registry.characters:
                registry.characters[char_id].arc = arc_beats

        return registry
