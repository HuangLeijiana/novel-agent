"""Memory system models — short-term, long-term, timeline, foreshadowing."""

import uuid

from pydantic import BaseModel, Field, field_validator, model_validator

from .chapter import Fact


def _short_id() -> str:
    """Generate a short unique ID for model entities."""
    return uuid.uuid4().hex[:12]


class ShortTermMemory(BaseModel):
    """Short-term memory — recent context for the Writer agent."""

    current_chapter_summary: str = Field(
        default="",
        description="Summary of the most recently completed chapter",
    )
    previous_chapter_summary: str = Field(
        default="",
        description="Summary of the chapter before that (for context continuity)",
    )
    recent_character_states: dict[str, dict] = Field(
        default_factory=dict,
        description="Recent state snapshots: character_id -> state dict",
    )
    active_foreshadowing: list[str] = Field(
        default_factory=list,
        description="Foreshadowing entry IDs that are still unresolved",
    )
    unresolved_hooks: list[str] = Field(
        default_factory=list,
        description="Hooks from previous chapters that haven't been resolved",
    )


class TimelineEvent(BaseModel):
    """A single event on the story timeline."""

    id: str = Field(default_factory=_short_id, description="Unique event identifier")
    chapter: int = Field(default=0, description="Chapter where this event occurs")
    in_story_time: str = Field(
        default="",
        description="In-story time reference, e.g. 'Day 3, Morning', 'Year 3021, Third Moon'",
    )
    description: str = Field(default="", description="What happened")
    characters: list[str] = Field(
        default_factory=list,
        description="Character IDs involved in this event",
    )
    location: str = Field(default="", description="Where this event takes place")
    causality: list[str] = Field(
        default_factory=list,
        description="IDs of timeline events that directly caused this one",
    )
    importance: str = Field(
        default="minor",
        description="'major' (plot-critical), 'minor' (supporting), 'background'",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data):
        """Wrap plain strings into a TimelineEvent with description.

        Local LLMs sometimes return a bare string instead of a structured
        TimelineEvent object — treat the string as the description.
        """
        if isinstance(data, str):
            return {"description": data}
        return data

    @field_validator("importance", mode="before")
    @classmethod
    def coerce_importance(cls, v):
        if isinstance(v, (int, float)):
            return "major" if v >= 2 else "minor" if v == 1 else "background"
        return str(v) if v else "minor"


class ForeshadowingEntry(BaseModel):
    """A single foreshadowing instance being tracked."""

    id: str = Field(default_factory=_short_id, description="Unique entry identifier")
    planted_chapter: int = Field(default=0, description="Chapter where this was first hinted")
    description: str = Field(default="", description="What was hinted / foreshadowed")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        # Model outputs plain string → wrap as description
        if isinstance(data, str):
            return {"description": data}
        return data

    hint_strength: float = Field(
        default=0.3,
        ge=0.0,
        description="How obvious the hint is (0 = invisible, 1 = blatant)",
    )
    expected_payoff_chapter: int | None = Field(
        default=None,
        description="Chapter where payoff is planned (null = not yet planned)",
    )
    payoff_chapter: int | None = Field(
        default=None,
        description="Chapter where payoff actually occurred (null = not yet paid off)",
    )
    status: str = Field(
        default="active",
        description="Current lifecycle status: active, advanced, paid_off, abandoned",
    )


class ForeshadowingTracker(BaseModel):
    """Tracks all foreshadowing across the entire novel."""

    entries: list[ForeshadowingEntry] = Field(default_factory=list)

    @property
    def active(self) -> list[ForeshadowingEntry]:
        """Get all active (unresolved) foreshadowing."""
        return [e for e in self.entries if e.status == "active"]

    @property
    def paid_off(self) -> list[ForeshadowingEntry]:
        """Get all paid-off foreshadowing."""
        return [e for e in self.entries if e.status == "paid_off"]

    @property
    def overdue(self) -> list[ForeshadowingEntry]:
        """Get foreshadowing that should have paid off by now but hasn't."""
        return [e for e in self.entries if e.status == "active" and e.expected_payoff_chapter is not None]


class LongTermMemory(BaseModel):
    """Long-term memory — accumulates across the entire novel."""

    chapter_summaries: dict[int, str] = Field(
        default_factory=dict,
        description="Mapping: chapter_number -> chapter summary",
    )
    # ── Hierarchical summaries (periodically consolidated) ──
    stage_summaries: dict[int, str] = Field(
        default_factory=dict,
        description="Every 10 chapters: stage_num -> ~500 char summary of that stage",
    )
    arc_summaries: dict[int, str] = Field(
        default_factory=dict,
        description="Every 50 chapters: arc_num -> ~800 char summary of that arc",
    )
    global_summary: str = Field(
        default="",
        description="Summary of the entire story so far (~1000 chars, updated every 100 chapters)",
    )
    consolidated_summary: str = Field(
        default="",
        description="[deprecated] Use global_summary instead",
    )
    facts: dict[str, Fact] = Field(
        default_factory=dict,
        description="All established facts: fact_id -> Fact",
    )
    world_changes: list[str] = Field(
        default_factory=list,
        description="Major world changes that have occurred (in order)",
    )


class MemoryState(BaseModel):
    """Complete memory state — both short-term and long-term."""

    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
    timeline: list[TimelineEvent] = Field(
        default_factory=list,
        description="Complete story timeline (ordered by chapter)",
    )
    foreshadowing: ForeshadowingTracker = Field(default_factory=ForeshadowingTracker)
    character_states: dict[str, dict] = Field(
        default_factory=dict,
        description="Snapshot: character_id -> most recent state dict",
    )
    unresolved_issues: list[str] = Field(
        default_factory=list,
        description="Plot threads or questions that remain unresolved",
    )
    # ── Hierarchical summary boundaries ──
    stage_boundaries: list[int] = Field(
        default_factory=list,
        description="Chapter numbers where stage summaries were generated, e.g. [10, 20, 30, ...]",
    )
    arc_boundaries: list[int] = Field(
        default_factory=list,
        description="Chapter numbers where arc summaries were generated, e.g. [50, 100, 150, ...]",
    )
