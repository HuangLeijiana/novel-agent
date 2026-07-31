"""Tests for the feedback model and data flywheel layer.

Covers:
- FeedbackEntry creation and validation
- FeedbackSentiment enum
- FeedbackReasonTag enum
- FeedbackSummary aggregation
- Edge cases (empty, multiple tags)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.feedback import (
    FeedbackEntry,
    FeedbackSentiment,
    FeedbackReasonTag,
    FeedbackSummary,
    REASON_TAG_LABELS,
)


class TestFeedbackEntry:
    """Tests for FeedbackEntry model."""

    def test_positive_feedback(self):
        entry = FeedbackEntry(
            chapter_number=1,
            sentiment=FeedbackSentiment.THUMBS_UP,
            notes="这章写得很好！",
        )
        assert entry.is_positive
        assert entry.chapter_number == 1
        assert entry.reason_tags == []

    def test_negative_feedback_with_reasons(self):
        entry = FeedbackEntry(
            chapter_number=3,
            sentiment=FeedbackSentiment.THUMBS_DOWN,
            reason_tags=[
                FeedbackReasonTag.CHARACTER_BROKEN,
                FeedbackReasonTag.AI_FLAVOR_HEAVY,
            ],
            notes="主角性格前后不一致",
            decision="rewrite",
            revision_count=2,
            review_score=5.5,
        )
        assert not entry.is_positive
        assert len(entry.reason_tags) == 2
        assert entry.decision == "rewrite"
        assert entry.revision_count == 2
        assert 5.0 < entry.review_score < 6.0

    def test_reason_labels(self):
        entry = FeedbackEntry(
            chapter_number=1,
            sentiment=FeedbackSentiment.THUMBS_DOWN,
            reason_tags=[FeedbackReasonTag.PLOT_BORING],
        )
        assert "情节无聊" in entry.reason_labels

    def test_multiple_reason_labels(self):
        entry = FeedbackEntry(
            chapter_number=1,
            sentiment=FeedbackSentiment.THUMBS_DOWN,
            reason_tags=[
                FeedbackReasonTag.PLOT_BORING,
                FeedbackReasonTag.PACING_ISSUE,
                FeedbackReasonTag.AI_FLAVOR_HEAVY,
            ],
        )
        labels = entry.reason_labels
        assert "情节无聊" in labels
        assert "节奏问题" in labels
        assert "AI味太重" in labels

    def test_timestamp_auto_generated(self):
        entry = FeedbackEntry(chapter_number=1)
        assert entry.timestamp
        assert "T" in entry.timestamp  # ISO format

    def test_default_values(self):
        entry = FeedbackEntry(chapter_number=1)
        assert entry.sentiment == FeedbackSentiment.THUMBS_UP
        assert entry.reason_tags == []
        assert entry.notes == ""
        assert entry.revision_count == 0


class TestFeedbackSentiment:
    """Tests for FeedbackSentiment enum."""

    def test_values(self):
        assert FeedbackSentiment.THUMBS_UP.value == "thumbs_up"
        assert FeedbackSentiment.THUMBS_DOWN.value == "thumbs_down"


class TestFeedbackReasonTag:
    """Tests for FeedbackReasonTag enum."""

    def test_all_tags_have_labels(self):
        for tag in FeedbackReasonTag:
            assert tag in REASON_TAG_LABELS, f"Missing label for {tag}"
            assert REASON_TAG_LABELS[tag]  # Non-empty label

    def test_tag_values(self):
        assert FeedbackReasonTag.NOT_MEETING_EXPECTATIONS.value == "not_meeting_expectations"
        assert FeedbackReasonTag.CHARACTER_BROKEN.value == "character_broken"
        assert FeedbackReasonTag.PLOT_BORING.value == "plot_boring"
        assert FeedbackReasonTag.AI_FLAVOR_HEAVY.value == "ai_flavor_heavy"
        assert FeedbackReasonTag.OTHER.value == "other"


class TestFeedbackSummary:
    """Tests for FeedbackSummary aggregation."""

    def test_empty_summary(self):
        summary = FeedbackSummary(project_id="test")
        assert summary.total_feedback == 0
        assert summary.acceptance_rate == 0.0

    def test_all_positive(self):
        summary = FeedbackSummary(
            project_id="test",
            total_feedback=10,
            positive_count=10,
            negative_count=0,
            acceptance_rate=1.0,
        )
        assert summary.acceptance_rate == 1.0

    def test_mixed(self):
        summary = FeedbackSummary(
            project_id="test",
            total_feedback=10,
            positive_count=7,
            negative_count=3,
            acceptance_rate=0.7,
            reason_distribution={"ai_flavor_heavy": 2, "plot_boring": 1},
        )
        assert summary.acceptance_rate == 0.7
        assert summary.reason_distribution["ai_flavor_heavy"] == 2


class TestFeedbackSerialization:
    """Tests for model serialization."""

    def test_entry_to_dict(self):
        entry = FeedbackEntry(
            chapter_number=2,
            sentiment=FeedbackSentiment.THUMBS_DOWN,
            reason_tags=[FeedbackReasonTag.CHARACTER_BROKEN],
            notes="test note",
        )
        data = entry.model_dump()
        assert data["chapter_number"] == 2
        assert data["sentiment"] == "thumbs_down"
        assert data["reason_tags"] == ["character_broken"]
        assert data["notes"] == "test note"

    def test_entry_from_dict(self):
        data = {
            "chapter_number": 3,
            "sentiment": "thumbs_up",
            "reason_tags": [],
            "notes": "good",
        }
        entry = FeedbackEntry.model_validate(data)
        assert entry.chapter_number == 3
        assert entry.is_positive
