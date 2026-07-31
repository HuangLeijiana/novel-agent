"""LangGraph conditional edge routing functions."""

import logging
from typing import Literal

from ..models.state import MainState

logger = logging.getLogger(__name__)

# Route labels
ROUTE_ACCEPT = "accept"
ROUTE_REVISE = "revise"
ROUTE_REWRITE = "rewrite"
ROUTE_NEXT = "next_chapter"
ROUTE_DONE = "done"
ROUTE_ROLLBACK = "rollback"
ROUTE_RESTART_BIBLE = "restart_bible"


def review_decision(state: MainState) -> Literal["accept", "revise", "rewrite"]:
    """Determine post-review routing based on scores and iteration count.

    Decision logic:
    - If max iterations reached → accept (force through)
    - If critical issues or score < 4.0 → rewrite (back to chapter_writing)
    - If score < 6.5 or AI-flavor < 5.0 → revise (back to chapter_writing with feedback)
    - Otherwise → accept (proceed to polish)
    """
    report = state.review_report
    if report is None:
        logger.warning("No review report — accepting by default")
        return ROUTE_ACCEPT

    if state.review_iteration >= state.max_review_iterations:
        logger.info(f"Max review iterations ({state.max_review_iterations}) reached — forcing accept")
        return ROUTE_ACCEPT

    avg_score = report.overall_score
    ai_flavor = report.dimension_scores.get("ai_flavor", 10.0)

    if report.has_critical or avg_score < 4.0:
        logger.info(f"Critical issues or very low score ({avg_score:.1f}) — rewrite")
        return ROUTE_REWRITE

    if avg_score < 6.5 or ai_flavor < 5.0:
        logger.info(f"Below threshold (score={avg_score:.1f}, ai_flavor={ai_flavor:.1f}) — revise")
        return ROUTE_REVISE

    logger.info(f"Review passed (score={avg_score:.1f}) — accept")
    return ROUTE_ACCEPT


def next_chapter_or_done(state: MainState) -> Literal["next_chapter", "done"]:
    """Check if there are more chapters to write.

    Returns 'next_chapter' to loop back to chapter_planning,
    or 'done' to proceed to human_confirmation.
    """
    if state.has_more_chapters():
        logger.info(f"More chapters remain (at {state.current_chapter_number}/{state.total_chapters})")
        return ROUTE_NEXT
    logger.info("All chapters complete")
    return ROUTE_DONE


def human_confirmation_decision(
    state: MainState,
) -> Literal["accept", "revise", "rewrite", "rollback", "restart_bible"]:
    """Route based on human decision after chapter review.

    - accept → proceed to next chapter or end
    - revise → back to polish_revision with feedback
    - rewrite → back to chapter_writing with feedback
    - rollback → back to chapter_planning
    - restart_bible → back to bible_construction (deep rollback)
    """
    decision = state.human_decision
    if decision is None:
        logger.info("No human decision yet — defaulting to accept")
        return ROUTE_ACCEPT

    decision_value = decision.value if hasattr(decision, 'value') else str(decision)
    logger.info(f"Human decision: {decision_value}")

    if decision_value == "accept":
        return ROUTE_ACCEPT
    elif decision_value == "revise":
        return ROUTE_REVISE
    elif decision_value == "rewrite":
        return ROUTE_REWRITE
    elif decision_value == "rollback":
        if state.rollback_target == "bible":
            return ROUTE_RESTART_BIBLE
        return ROUTE_ROLLBACK
    else:
        return ROUTE_ACCEPT
