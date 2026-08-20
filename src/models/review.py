"""Review report and issue models."""

from pydantic import BaseModel, Field, model_validator


class Issue(BaseModel):
    """A single issue found during quality review."""

    severity: str = Field(default="minor", description="Issue severity: critical, major, minor, suggestion")
    category: str = Field(
        default="other",
        description="Category: continuity, character, timeline, causality, pacing, style, etc.",
    )
    location: str = Field(
        default="",
        description="Approximate location in text (e.g. 'paragraph 3', 'scene 2')",
    )
    description: str = Field(default="", description="What the issue is")
    suggestion: str = Field(default="", description="Suggested fix")

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data):
        """Wrap plain strings into an Issue with description.

        Local LLMs often return a bare string instead of a structured Issue
        object — treat the string as the description.
        """
        if isinstance(data, str):
            return {"description": data}
        return data


class ReviewReport(BaseModel):
    """Complete quality review report for a chapter."""

    chapter_number: int = Field(default=0, description="Chapter being reviewed")
    overall_score: float = Field(
        default=7.0,
        description="Overall quality score (0-10)",
    )

    # Dimension scores (0-10)
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual dimension scores",
    )

    # Issues found
    issues: list[Issue] = Field(default_factory=list, description="All issues found")

    # Positive feedback
    strengths: list[str] = Field(
        default_factory=list,
        description="What worked well in this chapter",
    )

    # Actionable suggestions
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete improvement suggestions",
    )

    # Pass/fail
    passed: bool = Field(
        default=False,
        description="Whether this chapter passes all quality thresholds",
    )

    @property
    def critical_issues(self) -> list[Issue]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def major_issues(self) -> list[Issue]:
        """Get all major issues."""
        return [i for i in self.issues if i.severity == "major"]

    @property
    def has_critical(self) -> bool:
        """Whether there are any critical issues."""
        return any(i.severity == "critical" for i in self.issues)
