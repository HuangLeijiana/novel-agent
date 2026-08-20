"""Shared types, enums, and base models used across the system."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Enums ---

class LengthCategory(str, Enum):
    """Story length classification."""
    SHORT_STORY = "short_story"     # < 10K words
    NOVELLA = "novella"             # 10K - 40K
    NOVEL = "novel"                 # 40K - 100K
    EPIC = "epic"                   # > 100K


class ProjectStatus(str, Enum):
    """Overall project lifecycle status."""
    INITIALIZED = "initialized"
    BIBLE_BUILDING = "bible_building"
    OUTLINING = "outlining"
    WRITING = "writing"
    REVIEWING = "reviewing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowPhase(str, Enum):
    """Current phase within the LangGraph workflow."""
    IDLE = "idle"
    # Upstream: commercial research pipeline
    PLATFORM_SCAN = "platform_scan"           # Steps 1A/1B: 飞卢+番茄扫榜
    TOPIC_SELECTION = "topic_selection"        # Steps 1C-4A: 交叉分析→对标→选题→书名
    MINI_ARC_OUTLINE = "mini_arc_outline"      # Step 5: 10章小事件闭环大纲
    # Core pipeline
    PROJECT_INIT = "project_init"
    BIBLE_CONSTRUCTION = "bible_construction"
    MASTER_OUTLINE = "master_outline"
    CHAPTER_PLANNING = "chapter_planning"
    CHAPTER_WRITING = "chapter_writing"
    QUALITY_REVIEW = "quality_review"
    POLISH_REVISION = "polish_revision"
    MEMORY_UPDATE = "memory_update"
    HUMAN_CONFIRMATION = "human_confirmation"


class HumanDecision(str, Enum):
    """Possible human decisions after chapter review."""
    ACCEPT = "accept"
    REVISE = "revise"
    REWRITE = "rewrite"
    ROLLBACK = "rollback"


class CharacterRole(str, Enum):
    """Narrative role of a character."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    DEUTERAGONIST = "deuteragonist"
    SUPPORTING = "supporting"
    MINOR = "minor"
    MENTOR = "mentor"
    LOVE_INTEREST = "love_interest"
    RIVAL = "rival"
    FOIL = "foil"
    OTHER = "other"


class PlotArcType(str, Enum):
    """Type of plot arc."""
    MAIN = "main"
    SUBPLOT = "subplot"
    B_PLOT = "b_plot"


class HookType(str, Enum):
    """Type of chapter-ending hook."""
    CLIFFHANGER = "cliffhanger"
    MYSTERY = "mystery"
    EMOTIONAL = "emotional"
    REVELATION = "revelation"


class IssueSeverity(str, Enum):
    """Severity of a review issue."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


class ForeshadowingStatus(str, Enum):
    """Lifecycle status of a foreshadowing entry."""
    ACTIVE = "active"
    ADVANCED = "advanced"
    PAID_OFF = "paid_off"
    ABANDONED = "abandoned"


# --- Shared Types ---

class Reference(BaseModel):
    """A reference to an artifact section for traceability."""
    artifact: str       # e.g. "character", "faction", "chapter_plan"
    id: str             # Artifact-specific identifier
    field: str = ""     # Optional field within the artifact


class AgentLogEntry(BaseModel):
    """Log entry for an agent action."""
    agent: str
    phase: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    summary: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: str = ""
    duration_ms: float = 0.0
