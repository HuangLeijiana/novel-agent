"""Master outline, volumes, chapter planning models."""

import json
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator



def _short_id() -> str:
    """Generate a short unique ID for model entities."""
    return uuid.uuid4().hex[:12]


def _str_list(value: Any) -> list[str]:
    """Coerce a value to list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
        return [s] if s else []
    return [str(value)]


class PlotArc(BaseModel):
    """A single plot arc spanning multiple chapters."""

    id: str = Field(default_factory=_short_id, description="Unique arc identifier")
    arc_type: str = Field(default="main", description="Arc type: main, subplot, b_plot, etc.")
    name: str = Field(default="", description="Arc name")
    description: str = Field(default="", description="Arc summary")
    start_chapter: int = Field(default=0, description="Starting chapter number")
    end_chapter: int = Field(default=0, description="Ending chapter number")
    beats: list[str] = Field(
        default_factory=list,
        description="Key plot beats in this arc (ordered)",
    )
    characters_involved: list[str] = Field(
        default_factory=list,
        description="Character IDs in this arc",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalise field names from Chinese-native model output."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Model often uses 'arc' or 'subplot' instead of 'name'
        if "name" not in d:
            for alias in ("arc", "subplot", "title"):
                if alias in d:
                    d["name"] = d.pop(alias)
                    break
        # Model may use 'type' → map to 'arc_type' (do this BEFORE name fallback)
        if "type" in d and "arc_type" not in d:
            raw_type = d.pop("type")
            d["arc_type"] = str(raw_type).strip().lower().replace(" ", "_")
        # Normalize arc_type to canonical form
        if "arc_type" in d:
            arc_map = {"main": "main", "subplot": "subplot", "b_plot": "b_plot",
                       "b-plot": "b_plot", "bplot": "b_plot", "secondary": "b_plot",
                       "资源争夺": "subplot", "阴谋": "subplot",
                       "resource_struggle": "subplot", "devious_machination": "b_plot"}
            d["arc_type"] = arc_map.get(str(d["arc_type"]).lower(), d["arc_type"])
        # Still no name? Derive from arc_type or description
        if "name" not in d or not d.get("name"):
            desc = d.get("description", "")
            if isinstance(desc, str) and desc:
                # Truncate description for a short name (max 20 chars)
                d["name"] = desc[:20]
            elif "arc_type" in d:
                type_names = {"main": "主线", "subplot": "支线", "b_plot": "副线"}
                d["name"] = type_names.get(str(d["arc_type"]), "剧情线")
            else:
                d["name"] = "剧情线"
        # Coerce beats to list[str]
        if "beats" in d:
            d["beats"] = _str_list(d["beats"])
        return d


class Volume(BaseModel):
    """A volume (book/arc grouping) in the novel."""

    number: int = Field(default=0, description="Volume number (1-indexed)")
    title: str = Field(default="", description="Volume title")
    logline: str = Field(default="", description="One-sentence summary of the volume")
    start_chapter: int = Field(default=0, description="First chapter in this volume")
    end_chapter: int = Field(default=0, description="Last chapter in this volume")
    major_events: list[str] = Field(
        default_factory=list,
        description="Key events occurring in this volume",
    )
    emotional_arc: str = Field(
        default="",
        description="Emotional trajectory across this volume",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalise field names; fill missing number."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "number" not in d:
            d["number"] = 0
        # Alias: name / volume_name → title
        if "name" in d and "title" not in d:
            d["title"] = d.pop("name")
        # Alias: introduction / summary / description → logline
        if "logline" not in d:
            for alias in ("introduction", "summary", "description"):
                if alias in d:
                    d["logline"] = d.pop(alias)
                    break
        # Alias: key_events / events → major_events
        if "major_events" not in d:
            for alias in ("key_events", "events"):
                if alias in d:
                    d["major_events"] = d.pop(alias)
                    break
        if "major_events" in d:
            d["major_events"] = _str_list(d["major_events"])
        return d


class TurningPoint(BaseModel):
    """A major turning point in the story."""

    id: str = Field(default_factory=_short_id, description="Unique identifier")
    chapter: int = Field(default=0, description="Chapter where this turning point occurs")
    turning_type: str = Field(
        default="",
        description="Type: 'inciting_incident', 'first_plot_point', 'midpoint', "
        "'all_is_lost', 'dark_night_of_soul', 'climax', 'denouement'",
    )
    description: str = Field(default="", description="What happens at this turning point")
    impact: str = Field(default="", description="How this changes the story direction")
    characters_affected: list[str] = Field(
        default_factory=list,
        description="Character IDs affected by this turning point",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalise field names from model output."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "type" in d and "turning_type" not in d:
            d["turning_type"] = d.pop("type")
        if "chapter" not in d or d["chapter"] == 0:
            d["chapter"] = 0  # Will be auto-assigned by MasterOutline
        if "characters_affected" in d:
            d["characters_affected"] = _str_list(d["characters_affected"])
        return d


class Hook(BaseModel):
    """A narrative hook — typically at chapter end to pull readers forward."""

    hook_type: str = Field(default="cliffhanger", description="Type of hook: cliffhanger, mystery, emotional, revelation, etc.")
    description: str = Field(default="", description="The hook content")
    resolve_chapter: Optional[int] = Field(
        default=None,
        description="Chapter where this hook pays off (null = unresolved)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        # Plain string → wrap as description
        if isinstance(data, str):
            return {"description": data}
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "description" not in d:
            for alias in ("content", "hook", "text"):
                if alias in d:
                    d["description"] = d.pop(alias)
                    break
        if "type" in d and "hook_type" not in d:
            d["hook_type"] = d.pop("type")
        return d


class Scene(BaseModel):
    """A single scene within a chapter."""

    number: int = Field(default=0, description="Scene number within the chapter (1-indexed)")
    setting: str = Field(default="", description="Location and time of the scene")
    pov: str = Field(default="", description="Point-of-view character ID")
    characters_present: list[str] = Field(
        default_factory=list,
        description="Character IDs present in this scene",
    )
    goal: str = Field(default="", description="Scene goal — what the POV character wants")
    conflict: str = Field(default="", description="What prevents the goal from being achieved")
    outcome: str = Field(default="", description="Scene outcome — how it ends")
    word_count_estimate: int = Field(default=800, description="Estimated word count for this scene")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        # Plain string → try "地点：内容" format
        if isinstance(data, str):
            s = data.strip()
            parts = s.split("：", 1)
            if len(parts) == 2:
                return {"setting": parts[0], "goal": parts[1]}
            return {"setting": s}
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "number" not in d:
            d["number"] = 0
        if "characters_present" in d:
            d["characters_present"] = _str_list(d["characters_present"])
        if "pov_character" in d and "pov" not in d:
            d["pov"] = d.pop("pov_character")
        return d


class EmotionalBeat(BaseModel):
    """A point on the chapter's emotional curve."""

    position: float = Field(
        default=0.0,
        ge=0.0,
        description="Position within the chapter (0.0 = start, 1.0 = end)",
    )
    emotion: str = Field(default="neutral", description="Primary emotion at this point")
    intensity: float = Field(default=0.5, ge=0.0, description="Emotion intensity (0.0–1.0)")

    @field_validator("intensity", mode="before")
    @classmethod
    def coerce_intensity(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            if v > 1.0:
                return v / 10.0  # 0–10 scale → 0–1
            return float(v)
        if isinstance(v, str):
            try:
                val = float(v)
                if val > 1.0:
                    return val / 10.0
                return val
            except ValueError:
                pass
        return 0.5

    @field_validator("intensity", mode="after")
    @classmethod
    def clamp_intensity(cls, v: float) -> float:
        if v > 1.0:
            return v / 10.0
        return v


class ChapterPlan(BaseModel):
    """Detailed plan for a single chapter."""

    chapter_number: int = Field(default=0, description="Chapter number (1-indexed)")
    title: str = Field(default="", description="Chapter title")
    goal: str = Field(
        default="",
        description="What this chapter must accomplish for the story",
    )
    scenes: list[Scene] = Field(default_factory=list, description="Scenes in this chapter")
    pov_character: str = Field(default="", description="Primary POV character ID")
    conflict: str = Field(default="", description="Core conflict driving this chapter")
    hooks: list[Hook] = Field(default_factory=list, description="Hooks in this chapter")
    characters_involved: list[str] = Field(
        default_factory=list,
        description="All character IDs appearing in this chapter",
    )
    information_increment: str = Field(
        default="",
        description="New information the reader learns in this chapter",
    )
    reveals: list[str] = Field(
        default_factory=list,
        description="Specific reveals or twists in this chapter",
    )
    foreshadowing: list[str] = Field(
        default_factory=list,
        description="Foreshadowing planted in this chapter",
    )
    emotional_curve: list[EmotionalBeat] = Field(
        default_factory=list,
        description="Emotional arc across the chapter",
    )
    ending_hook: str = Field(default="", description="Chapter-ending hook description")
    word_count_target: int = Field(default=4000, description="Target word count")
    status: str = Field(default="planned", description="'planned', 'writing', 'reviewing', 'polishing', 'done'")


class MasterOutline(BaseModel):
    """Complete story outline with all arcs, volumes, and turning points."""

    title: str = Field(default="", description="Novel title")
    subtitle: Optional[str] = Field(default=None, description="Novel subtitle or tagline")
    logline: str = Field(default="", description="One-sentence story summary (logline)")
    main_plot: list[PlotArc] = Field(default_factory=list, description="Main plot arcs")
    subplots: list[PlotArc] = Field(default_factory=list, description="Subplot arcs")
    volumes: list[Volume] = Field(default_factory=list, description="Volume structure")
    major_turning_points: list[TurningPoint] = Field(
        default_factory=list,
        description="All major turning points",
    )
    chapter_count: int = Field(default=0, description="Planned total chapter count")

    @model_validator(mode="after")
    def auto_number(self):
        """Auto-number volumes and assign chapters to turning points."""
        total = self.chapter_count or 15

        # Auto-number volumes
        for i, v in enumerate(self.volumes):
            if v.number == 0:
                v.number = i + 1

        # Assign chapter numbers to turning points based on position
        tp_positions = {
            "inciting_incident": 1,
            "first_plot_point": max(2, int(total * 0.25)),
            "midpoint": max(3, int(total * 0.5)),
            "all_is_lost": max(4, int(total * 0.7)),
            "dark_night_of_soul": max(4, int(total * 0.75)),
            "climax": max(5, int(total * 0.9)),
            "denouement": total,
        }
        for tp in self.major_turning_points:
            if tp.chapter == 0 and tp.turning_type:
                tp.chapter = tp_positions.get(tp.turning_type, 0)

        return self
