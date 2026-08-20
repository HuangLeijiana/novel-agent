"""Chapter draft, fact extraction, and polished chapter models."""

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _short_id() -> str:
    """Generate a short unique ID for model entities."""
    return uuid.uuid4().hex[:12]


class Fact(BaseModel):
    """A single fact established in a chapter — tracked for consistency."""

    id: str = Field(default_factory=_short_id, description="Unique fact identifier")
    category: str = Field(
        default="other",
        description="Category: character, world, plot, relationship, item, rule, other",
    )
    description: str = Field(default="", description="The fact itself")
    source_chapter: int = Field(default=0, description="Chapter where this fact was established")
    certainty: float = Field(
        default=1.0,
        ge=0.0,
        description="Certainty: 1.0 = confirmed fact, 0.5 = character belief, 0.0 = rumor/misinformation",
    )
    character_id: str = Field(
        default="",
        description="Character this fact is about (if applicable)",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        """Wrap plain strings and normalize Chinese-key dicts.

        Local LLMs sometimes return a bare string instead of a structured
        Fact object — treat the string as the description.
        Qwen3 may also use Chinese keys like 描述/类别/确定性.
        """
        if isinstance(data, str):
            return {"description": data}
        if isinstance(data, dict):
            d = dict(data)
            # Normalize Chinese keys to English
            cn_map = {
                "描述": "description",
                "说明": "description",
                "类别": "category",
                "分类": "category",
                "类型": "category",
                "确定性": "certainty",
                "可信度": "certainty",
                "角色": "character_id",
                "角色ID": "character_id",
                "来源章节": "source_chapter",
                "章节": "source_chapter",
            }
            for cn_key, en_key in cn_map.items():
                if cn_key in d and en_key not in d:
                    d[en_key] = d.pop(cn_key)
            return d
        return data


class StateChange(BaseModel):
    """A change to a character's state caused by this chapter."""

    character_id: str = Field(default="", description="Character whose state changed")
    attribute: str = Field(default="", description="Attribute path, e.g. 'physical.location', 'emotional.mood'")
    old_value: Any = Field(default=None, description="Value before the change")
    new_value: Any = Field(default=None, description="Value after the change")
    reason: str = Field(default="", description="Why this change occurred")

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        """Wrap plain strings into a StateChange with reason."""
        if isinstance(data, str):
            return {"reason": data}
        return data


class ChapterDraft(BaseModel):
    """Raw chapter draft produced by the Writer agent."""

    chapter_number: int = Field(default=0, description="Chapter number")
    title: str = Field(default="", description="Chapter title")
    content: str = Field(default="", description="Chapter body text (markdown)")
    word_count: int = Field(default=0, description="Actual word count")
    author_notes: str = Field(default="", description="Writer's notes on this chapter")
    new_facts: list[Fact] = Field(
        default_factory=list,
        description="New facts established in this chapter",
    )
    character_state_changes: list[StateChange] = Field(
        default_factory=list,
        description="Character state changes caused by this chapter",
    )


class PolishedChapter(BaseModel):
    """Final polished chapter after refinement."""

    chapter_number: int = Field(default=0, description="Chapter number")
    title: str = Field(default="", description="Chapter title")
    content: str = Field(default="", description="Polished chapter body text (markdown)")
    word_count: int = Field(default=0, description="Actual word count after polishing")
    revision_notes: str = Field(
        default="",
        description="Summary of changes made during refinement",
    )
