"""Novel Bible models — world building, factions, rules, style, themes."""

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _short_id() -> str:
    """Generate a short unique ID for model entities."""
    return uuid.uuid4().hex[:12]


class WorldBuilding(BaseModel):
    """Complete world-building description."""

    name: str = Field(..., description="World/setting name")
    world_type: str = Field(
        ...,
        description="World category, e.g. 'fantasy', 'sci-fi', 'historical', 'contemporary'",
    )
    geography: str = Field(default="", description="Major locations, climate, terrain")
    history: str = Field(default="", description="Key historical events shaping the current era")
    culture: str = Field(default="", description="Social norms, customs, festivals, taboos")
    technology_level: str = Field(default="", description="Technology/magic development level")

    # Magic / Power system (if applicable)
    magic_system: str | None = Field(
        default=None,
        description="Magic/power system name and description",
    )
    power_progression: str | None = Field(
        default=None,
        description="How powers/skills progress (cultivation levels, tech tiers, etc.)",
    )

    # Misc
    special_rules: list[str] = Field(
        default_factory=list,
        description="Special world rules that differ from reality",
    )
    factions: list[str] = Field(
        default_factory=list,
        description="List of faction IDs defined in the world",
    )


class Faction(BaseModel):
    """A faction or organization in the story world."""

    id: str = Field(default_factory=_short_id, description="Unique faction identifier")
    name: str = Field(..., description="Display name")
    faction_type: str = Field(
        default="",
        description="Type, e.g. 'sect', 'kingdom', 'guild', 'corporation', 'family'",
    )
    goal: str = Field(default="", description="Primary goal or motivation")
    hierarchy: str = Field(default="", description="Internal hierarchy and power structure")
    resources: str = Field(default="", description="Key resources, territories, or assets")
    ideology: str = Field(default="", description="Core beliefs or ideology")
    relationship_with_others: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping: faction_id -> 'ally' / 'enemy' / 'neutral' / 'complicated'",
    )
    members: list[str] = Field(
        default_factory=list,
        description="Character IDs that belong to this faction",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Model uses 'type' → map to 'faction_type'
        if "type" in d and "faction_type" not in d:
            d["faction_type"] = d.pop("type")
        if "name" not in d:
            d["name"] = d.get("title", "")
        # Chinese LLMs often return resources/goal/hierarchy/ideology as lists
        for field in ("resources", "goal", "hierarchy", "ideology"):
            val = d.get(field)
            if isinstance(val, list):
                d[field] = "；".join(str(v) for v in val)
        return d


class NarrativeRules(BaseModel):
    """Narrative constraints and conventions for the novel."""

    pov_constraint: str = Field(
        default="third_person_limited",
        description="POV style: 'single_first_person', 'third_person_limited', 'multi_pov', 'omniscient'",
    )
    tense: str = Field(default="past", description="Narrative tense: 'past', 'present'")
    allowed_narrative_devices: list[str] = Field(
        default_factory=list,
        description="Allowed devices, e.g. 'flashback', 'foreshadowing', 'epistolary', 'dream_sequence'",
    )
    banned_devices: list[str] = Field(
        default_factory=list,
        description="Banned narrative devices",
    )
    chapter_length_range: tuple[int, int] = Field(
        default=(2500, 5000),
        description="(min_words, max_words) per chapter",
    )
    scene_break_style: str = Field(
        default="***",
        description="Scene break marker, e.g. '***', '---', or blank line",
    )


class StyleContract(BaseModel):
    """Defines the writing style for the novel."""

    tone: str = Field(default="", description="Overall tone: 'dark', 'light', 'whimsical', 'gritty', etc.")
    pacing_preference: str = Field(default="medium", description="'fast', 'medium', 'slow'")
    sentence_style: str = Field(
        default="varied",
        description="'simple', 'ornate', 'varied', 'minimalist'",
    )
    dialogue_ratio: float = Field(
        default=0.35,
        ge=0.0,
        description="Target ratio of dialogue to total text (0.0–1.0)",
    )
    description_ratio: float = Field(
        default=0.30,
        ge=0.0,
        description="Target ratio of description to total text (0.0–1.0)",
    )

    @field_validator("dialogue_ratio", "description_ratio", mode="before")
    @classmethod
    def coerce_ratios(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            if v > 1.0:
                return v / 100.0 if v > 10 else v / 10.0
            return float(v)
        return 0.0

    narrative_distance: str = Field(
        default="close",
        description="'close' (immersive), 'distant' (observational), 'omniscient'",
    )
    forbidden_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases to avoid (e.g. clichés, overused expressions)",
    )
    preferred_techniques: list[str] = Field(
        default_factory=list,
        description="Preferred writing techniques",
    )


class Theme(BaseModel):
    """A thematic element of the novel."""

    id: str = Field(default_factory=_short_id, description="Unique theme identifier")
    name: str = Field(..., description="Theme name, e.g. 'redemption', 'sacrifice', 'power'")
    description: str = Field(default="", description="How this theme operates in the story")
    manifestation: str = Field(
        default="",
        description="How the theme manifests through characters, plot, and setting",
    )
    related_character_ids: list[str] = Field(
        default_factory=list,
        description="Characters embodying this theme",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "name" not in d:
            d["name"] = d.get("title", "")
        return d


class CoreConflict(BaseModel):
    """A core conflict driving the narrative."""

    id: str = Field(default_factory=_short_id, description="Unique conflict identifier")
    conflict_type: str = Field(
        default="person_vs_person",
        description="Type: 'person_vs_person', 'person_vs_society', 'person_vs_self', 'person_vs_nature', 'person_vs_technology'",
    )
    description: str = Field(default="", description="Description of the conflict")
    parties_involved: list[str] = Field(
        default_factory=list,
        description="Character or faction IDs involved",
    )
    stakes: str = Field(default="", description="What's at stake if this conflict is not resolved")
    resolution_chapter: int | None = Field(
        default=None,
        description="Chapter where this conflict resolves (null = not yet planned)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Model uses 'type' instead of 'conflict_type'
        if "type" in d and "conflict_type" not in d:
            d["conflict_type"] = d.pop("type")
        return d


class NovelBible(BaseModel):
    """The complete Novel Bible — the authoritative reference for the entire novel."""

    world: WorldBuilding = Field(default_factory=WorldBuilding)
    factions: list[Faction] = Field(default_factory=list)
    rules: NarrativeRules = Field(default_factory=NarrativeRules)
    style_contract: StyleContract = Field(default_factory=StyleContract)
    themes: list[Theme] = Field(default_factory=list)
    core_conflicts: list[CoreConflict] = Field(default_factory=list)
    pleasure_point_model: str = Field(
        default="",
        description="Description of 'pleasure points' (爽点) design — what makes the story satisfying to read",
    )
    narrative_constraints: list[str] = Field(
        default_factory=list,
        description="Hard constraints or rules the narrative must follow",
    )
