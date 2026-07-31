"""LangGraph StateGraph construction — the backbone of the novel writing workflow.

Builds a state machine with 9 phase nodes and conditional routing for:
- Review decision (accept/revise/rewrite)
- Chapter iteration (next chapter / done)
- Human confirmation (accept/revise/rewrite/rollback)
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import MainState
from .nodes import (
    project_init_node,
    bible_construction_node,
    master_outline_node,
    chapter_planning_node,
    chapter_writing_node,
    quality_review_node,
    polish_revision_node,
    memory_update_node,
    human_confirmation_node,
)
from .edges import (
    review_decision,
    next_chapter_or_done,
    human_confirmation_decision,
    ROUTE_ACCEPT,
    ROUTE_REVISE,
    ROUTE_REWRITE,
    ROUTE_NEXT,
    ROUTE_DONE,
    ROUTE_ROLLBACK,
    ROUTE_RESTART_BIBLE,
)

logger = logging.getLogger(__name__)


def build_workflow() -> StateGraph:
    """Build the complete novel-writing StateGraph.

    Flow:
        START
          ↓
        project_init
          ↓
        bible_construction
          ↓
        master_outline
          ↓
        chapter_planning ←──────────────────────┐
          ↓                                      │
        chapter_writing ←────────┐               │
          ↓                      │               │
        quality_review           │               │
          ├─ accept → polish ────┤               │
          ├─ revise → writing ───┘               │
          └─ rewrite → planning ─────────────────┘
          ↓
        polish_revision
          ↓
        memory_update
          ├─ more → chapter_planning
          └─ done → human_confirmation
                      ├─ accept → END
                      ├─ revise → polish
                      ├─ rewrite → writing
                      └─ rollback → planning
    """
    builder = StateGraph(MainState)

    # ---- Register all nodes ----
    builder.add_node("project_init", project_init_node)
    builder.add_node("bible_construction", bible_construction_node)
    builder.add_node("master_outline", master_outline_node)
    builder.add_node("chapter_planning", chapter_planning_node)
    builder.add_node("chapter_writing", chapter_writing_node)
    builder.add_node("quality_review", quality_review_node)
    builder.add_node("polish_revision", polish_revision_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("human_confirmation", human_confirmation_node)

    # ---- Linear flow (start → planning) ----
    builder.add_edge(START, "project_init")
    builder.add_edge("project_init", "bible_construction")
    builder.add_edge("bible_construction", "master_outline")
    builder.add_edge("master_outline", "chapter_planning")
    builder.add_edge("chapter_planning", "chapter_writing")
    builder.add_edge("chapter_writing", "quality_review")

    # ---- Review routing ----
    builder.add_conditional_edges(
        "quality_review",
        review_decision,
        {
            ROUTE_ACCEPT: "polish_revision",
            ROUTE_REVISE: "chapter_writing",      # Rewrite with editor feedback
            ROUTE_REWRITE: "chapter_planning",    # Replan then rewrite
        },
    )

    # ---- Polish → Memory ----
    builder.add_edge("polish_revision", "memory_update")

    # ---- After memory: next chapter or done? ----
    builder.add_conditional_edges(
        "memory_update",
        next_chapter_or_done,
        {
            ROUTE_NEXT: "chapter_planning",
            ROUTE_DONE: "human_confirmation",
        },
    )

    # ---- Human confirmation routing ----
    builder.add_conditional_edges(
        "human_confirmation",
        human_confirmation_decision,
        {
            ROUTE_ACCEPT: END,
            ROUTE_REVISE: "polish_revision",          # Back to polish with feedback
            ROUTE_REWRITE: "chapter_writing",          # Back to writing with feedback
            ROUTE_ROLLBACK: "chapter_planning",        # Back to chapter planning
            ROUTE_RESTART_BIBLE: "bible_construction", # Deep rollback
        },
    )

    # ---- Compile with checkpointer ----
    checkpointer = MemorySaver()
    compiled = builder.compile(checkpointer=checkpointer)

    logger.info("Workflow compiled successfully")
    return compiled


# ============================================================
# Convenience: build with async SQLite checkpointer
# ============================================================

async def build_async_workflow(db_path: Optional[str] = None):
    """Build workflow with async SQLite checkpointer for persistence.

    Args:
        db_path: Path to SQLite database. If None, uses in-memory storage.

    Returns:
        Compiled StateGraph with AsyncSqliteSaver checkpointer.
    """
    try:
        from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
    except ImportError:
        logger.warning("aiosqlite not available, falling back to MemorySaver")
        return build_workflow()

    builder = StateGraph(MainState)

    # Same node registration as above
    builder.add_node("project_init", project_init_node)
    builder.add_node("bible_construction", bible_construction_node)
    builder.add_node("master_outline", master_outline_node)
    builder.add_node("chapter_planning", chapter_planning_node)
    builder.add_node("chapter_writing", chapter_writing_node)
    builder.add_node("quality_review", quality_review_node)
    builder.add_node("polish_revision", polish_revision_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("human_confirmation", human_confirmation_node)

    builder.add_edge(START, "project_init")
    builder.add_edge("project_init", "bible_construction")
    builder.add_edge("bible_construction", "master_outline")
    builder.add_edge("master_outline", "chapter_planning")
    builder.add_edge("chapter_planning", "chapter_writing")
    builder.add_edge("chapter_writing", "quality_review")

    builder.add_conditional_edges(
        "quality_review",
        review_decision,
        {ROUTE_ACCEPT: "polish_revision", ROUTE_REVISE: "chapter_writing", ROUTE_REWRITE: "chapter_planning"},
    )
    builder.add_edge("polish_revision", "memory_update")
    builder.add_conditional_edges(
        "memory_update",
        next_chapter_or_done,
        {ROUTE_NEXT: "chapter_planning", ROUTE_DONE: "human_confirmation"},
    )
    builder.add_conditional_edges(
        "human_confirmation",
        human_confirmation_decision,
        {
            ROUTE_ACCEPT: END,
            ROUTE_REVISE: "polish_revision",
            ROUTE_REWRITE: "chapter_writing",
            ROUTE_ROLLBACK: "chapter_planning",
            ROUTE_RESTART_BIBLE: "bible_construction",
        },
    )

    if db_path:
        checkpointer = await AsyncSqliteSaver.from_conn_string(db_path)
    else:
        checkpointer = MemorySaver()

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("Async workflow compiled successfully")
    return compiled
