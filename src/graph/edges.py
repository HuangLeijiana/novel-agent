"""LangGraph conditional edge routing functions.

These functions are the *actual* routing used by the compiled workflow
(graph/workflow.py): the review accept/revise/rewrite decision and the
chapter-loop next/done decision.
"""

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


def review_decision(state: MainState) -> Literal["accept", "revise", "rewrite"]:
    """Determine post-review routing based on scores and iteration count.

    Decision logic:
    - If max iterations reached → accept (force through)
    - If critical issues or score < 4.0 → rewrite (back to chapter planning)
    - If score < 6.5 or AI-flavor < 5.0 → revise (back to chapter writing)
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

    Total chapters comes from the master outline when available, otherwise
    the state's total_chapters, with a floor of 3 (matching the original
    runtime so the loop always terminates).
    """
    total = max((state.outline.chapter_count if state.outline else 0), state.total_chapters, 3)
    if state.current_chapter_number < total:
        logger.info(f"More chapters remain (at {state.current_chapter_number}/{total})")
        return ROUTE_NEXT
    logger.info(f"All {total} chapters complete")
    return ROUTE_DONE
