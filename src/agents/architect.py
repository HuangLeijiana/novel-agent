"""Architect Agent — responsible for world building and novel bible construction."""

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from ..llm.scheduler import ModelScheduler
from ..models.bible import (
    CoreConflict,
    Faction,
    NarrativeRules,
    NovelBible,
    StyleContract,
    Theme,
    WorldBuilding,
)
from ..models.project import ProjectConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================

def _normalize_str_list(value: Any) -> list[str]:
    """Normalise a field that should be list[str] but may arrive as a string."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
        return [stripped] if stripped else []
    return [str(value)]


def _normalize_optional_str(value: Any) -> Optional[str]:
    """Normalise a field that should be Optional[str] but may arrive as dict/list."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class WorldBuildingOutput(BaseModel):
    """LLM output schema for world building."""
    name: str = Field(..., description="世界名称")
    world_type: str = Field(..., description="世界类型")
    geography: str = Field(default="", description="地理环境")
    history: str = Field(default="", description="关键历史")
    culture: str = Field(default="", description="文化习俗")
    technology_level: str = Field(default="", description="科技/魔法水平")
    magic_system: Optional[str] = Field(default=None, description="魔法/力量体系")
    power_progression: Optional[str] = Field(default=None, description="力量进阶体系")
    special_rules: list[str] = Field(default_factory=list, description="特殊世界规则")

    @field_validator("special_rules", mode="before")
    @classmethod
    def coerce_special_rules(cls, v: Any) -> list[str]:
        return _normalize_str_list(v)

    @field_validator("magic_system", "power_progression", mode="before")
    @classmethod
    def coerce_optional_str(cls, v: Any) -> Optional[str]:
        return _normalize_optional_str(v)


class FactionsOutput(BaseModel):
    """LLM output schema for faction design."""
    factions: list[Faction] = Field(default_factory=list)


class StyleContractOutput(BaseModel):
    """LLM output schema for style contract."""
    tone: str = Field(default="", description="语调")
    pacing_preference: str = Field(default="medium", description="节奏偏好")
    sentence_style: str = Field(default="varied", description="句式风格")
    dialogue_ratio: float = Field(default=0.35, description="对话比例")
    description_ratio: float = Field(default=0.30, description="描写比例")
    narrative_distance: str = Field(default="close", description="叙事距离")
    forbidden_phrases: list[str] = Field(default_factory=list, description="禁用表达")
    preferred_techniques: list[str] = Field(default_factory=list, description="推荐技法")


class ThemesOutput(BaseModel):
    """LLM output schema for themes."""
    themes: list[Theme] = Field(default_factory=list)


class ConflictsOutput(BaseModel):
    """LLM output schema for core conflicts."""
    conflicts: list[CoreConflict] = Field(default_factory=list)


class PleasurePointOutput(BaseModel):
    """LLM output schema for pleasure point design."""
    pleasure_point_model: str = Field(default="", description="爽点模型描述")
    narrative_constraints: list[str] = Field(default_factory=list, description="叙事约束")


# ============================================================
# Agent
# ============================================================

class ArchitectAgent(BaseAgent):
    """Designs the world, factions, style contract, themes, and conflicts.

    This agent builds the foundational "Novel Bible" that all other agents
    reference to maintain consistency.
    """

    agent_type = "architect"

    # ================================================================
    # World Building
    # ================================================================

    async def build_world(self, config: ProjectConfig) -> WorldBuilding:
        """Design the story world based on project configuration."""
        system = self.build_system_prompt(
            role="世界观架构师",
            expertise="构建宏大、自洽、有深度的虚构世界。擅长设计地理、历史、文化、力量体系，"
                      "并能确保世界观服务于故事主题和爽点。",
        )

        user = f"""请为以下小说创意构建完整的世界观设定：

【灵感/创意】
{config.inspiration}

【题材】{', '.join(config.genre)}
【目标读者】{config.target_readers}
【篇幅】{config.target_length.value}（约{config.target_word_count}字）
【语调】{config.tone}

请设计：
1. 世界名称和类型
2. 地理环境（主要地点、气候）
3. 关键历史事件（塑造当今世界格局的事件）
4. 文化习俗（社会规范、禁忌、节日等）
5. 科技/魔法水平
6. 如果有魔法或力量体系，请详细设计（包括名称、规则、进阶路径）
7. 这个世界与真实世界不同的特殊规则

注意：
- 设定要服务于故事，不是为设定而设定
- 要为目标读者（{config.target_readers}）的阅读体验优化
- 留白：不要填满所有细节，给写作时留发挥空间"""

        if config.taboo_content:
            user += f"\n\n【避讳内容】请避免涉及：{', '.join(config.taboo_content)}"

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=WorldBuildingOutput,
            temperature_override=0.8,
        )

        world = WorldBuilding(
            name=result.name,
            world_type=result.world_type,
            geography=result.geography,
            history=result.history,
            culture=result.culture,
            technology_level=result.technology_level,
            magic_system=result.magic_system,
            power_progression=result.power_progression,
            special_rules=result.special_rules,
        )
        logger.info(f"World built: {world.name} ({world.world_type})")
        return world

    # ================================================================
    # Faction Design
    # ================================================================

    async def design_factions(self, config: ProjectConfig, world: WorldBuilding) -> list[Faction]:
        """Design factions and organizations in the world."""
        system = self.build_system_prompt(
            role="势力架构师",
            expertise="设计有深度、有冲突、有层次的组织势力。每个势力应有明确的目标、"
                      "内部结构、资源掌控和与其他势力的关系。",
        )

        world_summary = json.dumps(world.model_dump(), ensure_ascii=False, indent=2)
        user = f"""基于以下世界观，设计 3-6 个主要势力/组织：

【世界观】
{world_summary}

【题材】{', '.join(config.genre)}
【目标读者】{config.target_readers}

要求：
1. 每个势力必须有：名称、类型（宗门/王国/公会/家族等）、目标、层级结构、资源、意识形态
2. 势力之间应有明确的盟友/敌人/中立关系
3. 势力设计要服务于核心冲突
4. 至少有一个势力是主角所属或对立的

请为每个势力生成唯一ID（格式：faction_xxx）"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=FactionsOutput,
            temperature_override=0.7,
        )
        logger.info(f"Designed {len(result.factions)} factions")
        return result.factions

    # ================================================================
    # Style Contract
    # ================================================================

    async def create_style_contract(self, config: ProjectConfig) -> StyleContract:
        """Define the writing style contract for the novel."""
        system = self.build_system_prompt(
            role="文风设计师",
            expertise="为小说定义统一的文风契约。精通各类文学风格，能根据题材和目标读者"
                      "精准定位最适合的文风参数。",
        )

        user = f"""请为以下小说定义「文风契约」：

【灵感】{config.inspiration}
【题材】{', '.join(config.genre)}
【目标读者】{config.target_readers}
【语调】{config.tone}
【风格参考】{config.style_reference or '无特定参考'}

请确定：
1. 整体语调（dark/light/gritty/whimsical 等）
2. 叙事节奏偏好（fast/medium/slow）
3. 句式风格（simple/ornate/varied/minimalist）
4. 对话大致占比（0.0-1.0）
5. 描写大致占比（0.0-1.0）
6. 叙事距离（close沉浸式 / distant观察式 / omniscient全知）
7. 应避免的陈词滥调或特定表达
8. 推荐的写作技法

注意：这是{config.target_length.value}级别的小说，文风需要能支撑长篇写作。"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=StyleContractOutput,
            temperature_override=0.6,
        )
        style = StyleContract(
            tone=result.tone,
            pacing_preference=result.pacing_preference,
            sentence_style=result.sentence_style,
            dialogue_ratio=result.dialogue_ratio,
            description_ratio=result.description_ratio,
            narrative_distance=result.narrative_distance,
            forbidden_phrases=result.forbidden_phrases,
            preferred_techniques=result.preferred_techniques,
        )
        logger.info(f"Style contract created: tone={style.tone}, pacing={style.pacing_preference}")
        return style

    # ================================================================
    # Themes & Conflicts
    # ================================================================

    async def generate_themes(self, config: ProjectConfig, world: WorldBuilding) -> list[Theme]:
        """Generate thematic elements for the novel."""
        system = self.build_system_prompt(
            role="主题设计师",
            expertise="为小说提炼核心主题，并设计主题在人物、情节和世界观中的呈现方式。",
        )

        user = f"""基于以下信息，为小说提炼 3-5 个核心主题：

【灵感】{config.inspiration}
【题材】{', '.join(config.genre)}
【世界观】{world.name} - {world.world_type}

每个主题需要：
- 名称（如「救赎」「牺牲」「权力的代价」）
- 描述（这个主题在故事中的含义）
- 呈现方式（如何通过人物、情节、世界设定来表达）"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ThemesOutput,
            temperature_override=0.7,
        )
        logger.info(f"Generated {len(result.themes)} themes")
        return result.themes

    async def define_conflicts(
        self,
        config: ProjectConfig,
        world: WorldBuilding,
        factions: list[Faction],
    ) -> list[CoreConflict]:
        """Define core conflicts driving the narrative."""
        system = self.build_system_prompt(
            role="冲突架构师",
            expertise="设计推动故事发展的核心冲突。理解不同类型冲突（人与人、人与社会、"
                      "人与自我、人与自然）的叙事功能。",
        )

        factions_summary = json.dumps([f.model_dump() for f in factions], ensure_ascii=False, indent=2)
        user = f"""基于以下信息，设计 3-5 个核心冲突：

【灵感】{config.inspiration}
【题材】{', '.join(config.genre)}
【世界观】{world.name}
【势力】{factions_summary}

每个冲突需要：
- 冲突类型（person_vs_person / person_vs_society / person_vs_self / person_vs_nature / person_vs_technology）
- 描述
- 涉及方（角色或势力）
- 赌注（如果冲突不解决，会失去什么）

要求：
- 至少有一个核心冲突是 person_vs_self（内在冲突）
- 至少有一个冲突涉及势力对抗
- 冲突之间应有层次和关联"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ConflictsOutput,
            temperature_override=0.7,
        )
        logger.info(f"Defined {len(result.conflicts)} core conflicts")
        return result.conflicts

    # ================================================================
    # Pleasure Points & Constraints
    # ================================================================

    async def design_pleasure_points(
        self,
        config: ProjectConfig,
        world: WorldBuilding,
        themes: list[Theme],
    ) -> tuple[str, list[str]]:
        """Design the pleasure point model and narrative constraints."""
        system = self.build_system_prompt(
            role="爽点架构师",
            expertise="精通网文和小说的爽点设计。理解不同类型读者的阅读期待，"
                      "能设计出令人欲罢不能的阅读体验节奏。",
        )

        user = f"""为以下小说设计「爽点模型」和「叙事约束」：

【灵感】{config.inspiration}
【题材】{', '.join(config.genre)}
【目标读者】{config.target_readers}
【世界观类型】{world.world_type}"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=PleasurePointOutput,
            temperature_override=0.7,
        )
        return result.pleasure_point_model, result.narrative_constraints
