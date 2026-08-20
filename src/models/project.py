"""Project configuration and metadata models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .common import LengthCategory, ProjectStatus, WorkflowPhase


class ProjectConfig(BaseModel):
    """User-provided project configuration (Phase 1 input)."""

    # --- Core identity ---
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Novel title / project name, also used as the folder name",
    )
    inspiration: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="One-sentence inspiration or core idea for the novel",
    )
    genre: list[str] = Field(
        ...,
        min_length=1,
        description="Genre tags, e.g. ['玄幻', '修仙']",
    )
    target_readers: str = Field(
        ...,
        min_length=1,
        description="Target audience description, e.g. '18-35岁男性读者'",
    )
    target_length: LengthCategory = Field(
        default=LengthCategory.NOVEL,
        description="Rough length category",
    )
    target_word_count: int = Field(
        default=80000,
        ge=5000,
        le=500000,
        description="Target total word count",
    )

    # --- Style & tone ---
    tone: str = Field(
        default="",
        description="Desired tone, e.g. 'dark', 'light', 'whimsical', 'gritty'",
    )
    style_reference: Optional[str] = Field(
        default=None,
        description="Reference works or authors for style emulation",
    )
    language: str = Field(
        default="zh-CN",
        description="Primary writing language (BCP 47 tag)",
    )

    # --- Constraints ---
    taboo_content: list[str] = Field(
        default_factory=list,
        description="Content to avoid, e.g. ['色情', '过度暴力', '政治敏感']",
    )
    special_requirements: list[str] = Field(
        default_factory=list,
        description="Any special requirements or notes",
    )


class ProjectMeta(BaseModel):
    """Runtime project metadata tracked across the workflow."""

    project_id: str = Field(..., description="Unique project identifier")
    title: Optional[str] = Field(default=None, description="Novel title (can be set later)")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of project creation",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of last update",
    )
    status: ProjectStatus = Field(
        default=ProjectStatus.INITIALIZED,
        description="Current project status",
    )
    current_phase: WorkflowPhase = Field(
        default=WorkflowPhase.IDLE,
        description="Current workflow phase",
    )
    current_chapter: int = Field(default=0, description="Current chapter being worked on (0 = not started)")
    total_chapters: int = Field(default=0, description="Planned total chapter count")
    iteration_count: int = Field(default=0, description="Total review iterations so far")
    word_count_written: int = Field(default=0, description="Total words written so far")
