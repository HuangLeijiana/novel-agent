"""Character profile, relationship, and state models."""

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import CharacterRole


def _short_id() -> str:
    """Generate a short unique ID for model entities."""
    return uuid.uuid4().hex[:12]


class CharacterState(BaseModel):
    """Current state of a character at a specific point in the story."""

    physical: dict[str, Any] = Field(
        default_factory=dict,
        description="Physical state: location, health, injuries, appearance changes",
    )
    emotional: dict[str, Any] = Field(
        default_factory=dict,
        description="Emotional/mental state: mood, mental health, key emotions",
    )
    social: dict[str, Any] = Field(
        default_factory=dict,
        description="Social state: status, reputation, affiliations",
    )
    resources: dict[str, Any] = Field(
        default_factory=dict,
        description="Resources: items, money, knowledge, abilities gained/lost",
    )
    arc_progress: float = Field(
        default=0.0,
        ge=0.0,
        description="Progress through character arc (0.0 = start, 1.0 = complete)",
    )


class Relationship(BaseModel):
    """A directed relationship between two characters."""

    target_id: str = Field(default="", description="The other character's ID")
    relationship_type: str = Field(
        default="acquaintance",
        description="Type: 'friend', 'rival', 'lover', 'enemy', 'family', 'mentor', 'subordinate', etc.",
    )
    trust: float = Field(default=0.5, ge=0.0, description="Trust level (0 = none, 1 = absolute)")
    intimacy: float = Field(default=0.0, ge=0.0, description="Emotional intimacy (0 = distant, 1 = deepest)")
    power_balance: float = Field(
        default=0.5,
        ge=0.0,
        description="Power dynamic (0 = subject is submissive, 1 = subject is dominant relative to target)",
    )
    history: str = Field(default="", description="Shared history between the two characters")
    notes: str = Field(default="", description="Additional notes about this relationship")


class ArcBeat(BaseModel):
    """A single beat in a character's arc."""

    chapter: int = Field(default=0, description="Chapter where this beat occurs")
    event: str = Field(default="", description="What happens")
    change: str = Field(default="", description="How the character changes as a result")
    new_trait: Optional[str] = Field(default=None, description="New trait gained or revealed")
    lost_trait: Optional[str] = Field(default=None, description="Trait lost or suppressed")


class CharacterProfile(BaseModel):
    """Complete profile of a story character."""

    id: str = Field(default_factory=_short_id, description="Unique character identifier, e.g. 'char_mc'")
    name: str = Field(..., description="Character's name")
    role: str = Field(default="supporting", description="Narrative role: protagonist, antagonist, mentor, etc.")
    archetype: str = Field(
        default="",
        description="Character archetype: 'hero', 'mentor', 'herald', 'trickster', 'shadow', etc.",
    )

    # Demographics
    age: int = Field(default=0, description="Age in years")

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            try:
                return int(s)
            except ValueError:
                pass
            # Try to extract digits, e.g. "约20岁" → 20, "3000+岁" → 3000
            import re
            m = re.search(r"(\d+)", s)
            if m:
                return int(m.group(1))
        return 0  # "未知", "不详", etc.
    gender: str = Field(default="", description="Gender identity")

    # Traits
    appearance: str = Field(default="", description="Physical appearance description")
    personality: str = Field(default="", description="Personality traits and quirks")
    motivation: str = Field(default="", description="Core motivation driving the character")
    flaw: str = Field(default="", description="Primary character flaw or weakness")
    backstory: str = Field(default="", description="Character backstory")

    # Abilities
    abilities: list[str] = Field(
        default_factory=list,
        description="Skills, powers, or special abilities",
    )

    # Arc
    arc: list[ArcBeat] = Field(default_factory=list, description="Planned character arc beats")

    # Relationships
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="All relationships to other characters",
    )

    # Current dynamic state
    current_state: CharacterState = Field(
        default_factory=CharacterState,
        description="Current state (updated each chapter)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Handle unknown/missing fields and normalize role values."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Normalize role to a canonical short form
        if "role" in d:
            raw = str(d["role"]).strip().lower().replace(" ", "_")
            role_map = {
                "protagonist": "protagonist",
                "main": "protagonist",
                "lead": "protagonist",
                "hero": "protagonist",
                "antagonist": "antagonist",
                "villain": "antagonist",
                "enemy": "antagonist",
                "deuteragonist": "deuteragonist",
                "supporting": "supporting",
                "support": "supporting",
                "side": "supporting",
                "minor": "minor",
                "background": "minor",
                "mentor": "mentor",
                "love_interest": "love_interest",
                "rival": "rival",
                "foil": "foil",
            }
            d["role"] = role_map.get(raw, "supporting")
        return d

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Search/grouping tags")
    notes: str = Field(default="", description="Author's notes about this character")


class CharacterRegistry(BaseModel):
    """Complete registry of all characters in the novel."""

    characters: dict[str, CharacterProfile] = Field(
        default_factory=dict,
        description="Mapping: character_id -> CharacterProfile",
    )

    def get(self, character_id: str) -> Optional[CharacterProfile]:
        """Get a character by ID."""
        return self.characters.get(character_id)

    def get_protagonist(self) -> Optional[CharacterProfile]:
        """Get the protagonist."""
        for char in self.characters.values():
            if char.role == CharacterRole.PROTAGONIST:
                return char
        return None

    def get_antagonists(self) -> list[CharacterProfile]:
        """Get all antagonists."""
        return [c for c in self.characters.values() if c.role == CharacterRole.ANTAGONIST]
