"""LangGraph root state — the single state object flowing through the workflow."""

from typing import Any

from pydantic import BaseModel, Field

from .bible import NovelBible
from .chapter import ChapterDraft, PolishedChapter
from .characters import CharacterRegistry
from .common import AgentLogEntry, HumanDecision, WorkflowPhase
from .feedback import FeedbackEntry
from .memory import MemoryState
from .outline import ChapterPlan, MasterOutline
from .project import ProjectConfig, ProjectMeta
from .review import ReviewReport
from .topic import TopicResearchState


class MainState(BaseModel):
    """
    Root state for the LangGraph StateGraph.

    This is the single state object that flows through every node in the workflow.
    Each node reads from and writes partial updates to this state.
    """

    # ===== Identity =====
    project_meta: ProjectMeta | None = Field(
        default=None,
        description="Project metadata (created in Phase 1)",
    )

    # ===== Core Artifacts (populated progressively) =====
    project_config: ProjectConfig | None = Field(
        default=None,
        description="User-provided project configuration (Phase 1 input)",
    )
    bible: NovelBible | None = Field(
        default=None,
        description="Novel Bible (populated in Phase 2)",
    )
    characters: CharacterRegistry | None = Field(
        default=None,
        description="Character registry (populated in Phase 2)",
    )
    outline: MasterOutline | None = Field(
        default=None,
        description="Master outline with volumes (populated in Phase 3)",
    )
    chapter_plan: ChapterPlan | None = Field(
        default=None,
        description="Current chapter plan (populated in Phase 4)",
    )
    chapter_draft: ChapterDraft | None = Field(
        default=None,
        description="Current chapter draft (populated in Phase 5)",
    )
    review_report: ReviewReport | None = Field(
        default=None,
        description="Latest review report (populated in Phase 6)",
    )
    chapter_inspection: Any | None = Field(
        default=None,
        description="Structural inspection result for the current chapter (populated in Phase 5)",
    )
    polished_chapter: PolishedChapter | None = Field(
        default=None,
        description="Polished chapter (populated in Phase 7)",
    )
    memory: MemoryState | None = Field(
        default=None,
        description="Memory state (updated in Phase 8)",
    )

    # ===== Upstream: Commercial Research Pipeline =====
    topic_research: TopicResearchState | None = Field(
        default=None,
        description="Topic research pipeline state (Steps 1-4: scan→benchmark→select→title)",
    )
    mini_arc_outline: dict | None = Field(
        default=None,
        description="10-chapter mini-arc outlines keyed by genre name",
    )

    # ===== Previous Chapter Artifacts (for rollback) =====
    previous_chapter_plan: ChapterPlan | None = Field(default=None)
    previous_chapter_draft: ChapterDraft | None = Field(default=None)
    previous_polished_chapter: PolishedChapter | None = Field(default=None)

    # ===== Control Flow =====
    current_phase: WorkflowPhase = Field(
        default=WorkflowPhase.IDLE,
        description="Current workflow phase",
    )
    current_chapter_number: int = Field(
        default=0,
        description="Chapter currently being worked on (0 = not started)",
    )
    total_chapters: int = Field(
        default=0,
        description="Total planned chapters from the master outline",
    )
    review_iteration: int = Field(
        default=0,
        description="Number of review iterations for the current chapter",
    )
    max_review_iterations: int = Field(
        default=3,
        description="Maximum review iterations before forcing acceptance",
    )

    # ===== Human-in-the-Loop =====
    human_feedback: str | None = Field(
        default=None,
        description="Textual feedback from the human author",
    )
    human_decision: HumanDecision | None = Field(
        default=None,
        description="Human decision: accept / revise / rewrite / rollback",
    )
    current_inspiration: str | None = Field(
        default=None,
        description="User-provided inspiration/ideas injected before the next phase",
    )
    rollback_target: str | None = Field(
        default=None,
        description="What to rollback to: 'chapter_plan', 'chapter_draft', 'bible'",
    )

    # ===== User Feedback (Data Flywheel) =====
    feedback_records: list[FeedbackEntry] = Field(
        default_factory=list,
        description="User feedback records (thumbs up/down + reason tags) for data flywheel",
    )

    # ===== Logging & Errors =====
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages accumulated during the workflow",
    )
    agent_log: list[AgentLogEntry] = Field(
        default_factory=list,
        description="Log of all agent actions for debugging and tracking",
    )

    # ===== Convenience Methods =====

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def log_agent(self, entry: AgentLogEntry) -> None:
        """Log an agent action."""
        self.agent_log.append(entry)

    def sync_character_states(self) -> None:
        """Push memory character states back into the character registry.

        Without this, WriterAgent sees the static current_state from character
        creation and never the per-chapter updates tracked in memory.

        Memory stores flat attribute paths like 'physical.location' or
        'emotional.mood', while CharacterState has nested dict fields
        (physical, emotional, social, resources). This method maps the dotted
        paths back into the correct nested dict.
        """
        if self.memory is None or self.characters is None:
            return
        if not self.memory.character_states:
            return
        for cid, state_updates in self.memory.character_states.items():
            if cid not in self.characters.characters:
                continue
            char = self.characters.characters[cid]
            cs = char.current_state
            if cs is None:
                from ..models.characters import CharacterState

                cs = CharacterState()
                char.current_state = cs

            for attr, value in state_updates.items():
                # Parse dotted path: "physical.location" → cs.physical["location"]
                parts = attr.split(".", 1)
                if len(parts) == 2:
                    category, key = parts
                    if hasattr(cs, category) and isinstance(getattr(cs, category), dict):
                        getattr(cs, category)[key] = value
                # Fallback: set as top-level attribute for flat paths
                elif hasattr(cs, attr):
                    setattr(cs, attr, value)

    def advance_chapter(self) -> None:
        """Move to the next chapter and reset per-chapter state."""
        self.current_chapter_number += 1
        self.previous_chapter_plan = self.chapter_plan
        self.previous_chapter_draft = self.chapter_draft
        self.previous_polished_chapter = self.polished_chapter
        self.chapter_plan = None
        self.chapter_draft = None
        self.review_report = None
        self.polished_chapter = None
        self.review_iteration = 0
        self.human_feedback = None
        self.human_decision = None

    def has_more_chapters(self) -> bool:
        """Check if there are more chapters to write."""
        if self.total_chapters == 0:
            return True  # Unknown total — keep going
        return self.current_chapter_number < self.total_chapters
