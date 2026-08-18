"""LangGraph StateGraph construction — the actual workflow engine.

This graph IS the runtime: the FastAPI layer (routes.py) invokes it instead of
the former imperative ``_run_phased_workflow`` loop. It mirrors the original
flow exactly while making the conditional routing real:

    START → project_init
      → phase0_route (scan data?)
          scan:  platform_scan_work → confirm → topic_selection_work → confirm
                 → mini_arc_work → confirm
          skip:  (no scan data)
      → bible_route (bible exists?)
          work:  bible_work → confirm
          skip:  bible_skip (broadcast 'already exists')
      → characters_route (characters exist?)
          work:  characters_work → confirm
          skip:  characters_skip
      → outline_route (outline exists?)
          work:  outline_work → confirm
          skip:  outline_skip
      → chapter_route (more chapters?)
          chapter: chapter_plan → chapter_write → chapter_review
                   → review_route (accept → polish | revise → write | rewrite → plan)
                   → chapter_polish → chapter_memory → chapter_advance → chapter_route
          end:    END

Human confirmation between top-level phases uses LangGraph's ``interrupt()``:
the run pauses, the API layer broadcasts ``phase_blocked``, and the run is
resumed with ``Command(resume={"inspiration": ...})``. The checkpointer
(MemorySaver by default, AsyncSqliteSaver when a db path is given) persists
the paused state under the thread_id (= project_id).
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..models.state import MainState
from .edges import (
    ROUTE_ACCEPT,
    ROUTE_NEXT,
    ROUTE_REVISE,
    ROUTE_REWRITE,
    next_chapter_or_done,
    review_decision,
)
from .nodes import (
    bible_confirm_node,
    bible_skip_node,
    bible_work_node,
    chapter_advance_node,
    chapter_memory_node,
    chapter_plan_node,
    chapter_polish_node,
    chapter_review_node,
    chapter_write_node,
    characters_confirm_node,
    characters_skip_node,
    characters_work_node,
    mini_arc_confirm_node,
    mini_arc_work_node,
    outline_confirm_node,
    outline_skip_node,
    outline_work_node,
    platform_scan_confirm_node,
    platform_scan_work_node,
    project_init_node,
    topic_selection_confirm_node,
    topic_selection_work_node,
)

logger = logging.getLogger(__name__)


# ============================================================
# Routing functions
# ============================================================


def _has_scan_data(state: MainState) -> bool:
    """Whether scan data is pending (set by the /submit-scan endpoint)."""
    from ..api.phase_executor import _pending_scan_data

    return bool(_pending_scan_data.get("feilu") or _pending_scan_data.get("fanqie"))


def phase0_route(state: MainState) -> str:
    """Route into Phase 0 (commercial research) when scan data exists."""
    return "scan" if _has_scan_data(state) else "skip"


def bible_route(state: MainState) -> str:
    return "skip" if state.bible else "work"


def characters_route(state: MainState) -> str:
    return "skip" if state.characters else "work"


def outline_route(state: MainState) -> str:
    return "skip" if state.outline else "work"


def chapter_route(state: MainState) -> str:
    """Enter the chapter loop while chapters remain, otherwise END."""
    return "chapter" if next_chapter_or_done(state) == ROUTE_NEXT else "end"


# ============================================================
# Graph construction
# ============================================================


def _build_graph() -> StateGraph:
    """Register all nodes and edges; returns an uncompiled StateGraph."""
    builder = StateGraph(MainState)

    async def _passthrough(state: MainState, config: RunnableConfig | None = None) -> dict:
        """Router placeholder: the conditional edge attached to this node
        decides the actual next step."""
        return {}

    # ── Nodes ──────────────────────────────────────────────────────────
    builder.add_node("project_init", project_init_node)

    # Router nodes (conditional edges decide where to go next)
    for router in ("phase0_route", "bible_route", "characters_route", "outline_route", "chapter_route"):
        builder.add_node(router, _passthrough)

    # Phase 0 (commercial research)
    builder.add_node("platform_scan_work", platform_scan_work_node)
    builder.add_node("platform_scan_confirm", platform_scan_confirm_node)
    builder.add_node("topic_selection_work", topic_selection_work_node)
    builder.add_node("topic_selection_confirm", topic_selection_confirm_node)
    builder.add_node("mini_arc_work", mini_arc_work_node)
    builder.add_node("mini_arc_confirm", mini_arc_confirm_node)

    # Top-level creative phases
    builder.add_node("bible_work", bible_work_node)
    builder.add_node("bible_confirm", bible_confirm_node)
    builder.add_node("bible_skip", bible_skip_node)
    builder.add_node("characters_work", characters_work_node)
    builder.add_node("characters_confirm", characters_confirm_node)
    builder.add_node("characters_skip", characters_skip_node)
    builder.add_node("outline_work", outline_work_node)
    builder.add_node("outline_confirm", outline_confirm_node)
    builder.add_node("outline_skip", outline_skip_node)

    # Chapter loop
    builder.add_node("chapter_plan", chapter_plan_node)
    builder.add_node("chapter_write", chapter_write_node)
    builder.add_node("chapter_review", chapter_review_node)
    builder.add_node("chapter_polish", chapter_polish_node)
    builder.add_node("chapter_memory", chapter_memory_node)
    builder.add_node("chapter_advance", chapter_advance_node)

    # ── Edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "project_init")

    # Phase 0 gate
    builder.add_conditional_edges(
        "project_init",
        phase0_route,
        {"scan": "platform_scan_work", "skip": "bible_route"},
    )
    # Phase 0 chain (work → confirm → next)
    builder.add_edge("platform_scan_work", "platform_scan_confirm")
    builder.add_edge("platform_scan_confirm", "topic_selection_work")
    builder.add_edge("topic_selection_work", "topic_selection_confirm")
    builder.add_edge("topic_selection_confirm", "mini_arc_work")
    builder.add_edge("mini_arc_work", "mini_arc_confirm")
    builder.add_edge("mini_arc_confirm", "bible_route")

    # Bible
    builder.add_conditional_edges(
        "bible_route",
        bible_route,
        {"work": "bible_work", "skip": "bible_skip"},
    )
    builder.add_edge("bible_work", "bible_confirm")
    builder.add_edge("bible_confirm", "characters_route")
    builder.add_edge("bible_skip", "characters_route")

    # Characters
    builder.add_conditional_edges(
        "characters_route",
        characters_route,
        {"work": "characters_work", "skip": "characters_skip"},
    )
    builder.add_edge("characters_work", "characters_confirm")
    builder.add_edge("characters_confirm", "outline_route")
    builder.add_edge("characters_skip", "outline_route")

    # Outline
    builder.add_conditional_edges(
        "outline_route",
        outline_route,
        {"work": "outline_work", "skip": "outline_skip"},
    )
    builder.add_edge("outline_work", "outline_confirm")
    builder.add_edge("outline_confirm", "chapter_route")
    builder.add_edge("outline_skip", "chapter_route")

    # Chapter loop
    builder.add_conditional_edges(
        "chapter_route",
        chapter_route,
        {"chapter": "chapter_plan", "end": END},
    )
    builder.add_edge("chapter_plan", "chapter_write")
    builder.add_edge("chapter_write", "chapter_review")
    builder.add_conditional_edges(
        "chapter_review",
        review_decision,
        {
            ROUTE_ACCEPT: "chapter_polish",
            ROUTE_REVISE: "chapter_write",  # rewrite with editor feedback
            ROUTE_REWRITE: "chapter_plan",  # replan then rewrite
        },
    )
    builder.add_edge("chapter_polish", "chapter_memory")
    builder.add_edge("chapter_memory", "chapter_advance")
    builder.add_edge("chapter_advance", "chapter_route")

    return builder


# ============================================================
# Compiled graphs
# ============================================================


def build_workflow(checkpointer=None):
    """Build the compiled workflow with a checkpointer (MemorySaver by default)."""
    return _build_graph().compile(checkpointer=checkpointer or MemorySaver())


async def build_async_workflow(db_path: str | None = None):
    """Build the workflow with an async SQLite checkpointer for persistence.

    Falls back to MemorySaver when aiosqlite is unavailable.
    """
    try:
        from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
    except ImportError:
        logger.warning("aiosqlite not available, falling back to MemorySaver")
        return build_workflow()

    if db_path:
        checkpointer = await AsyncSqliteSaver.from_conn_string(db_path)
    else:
        checkpointer = MemorySaver()
    return _build_graph().compile(checkpointer=checkpointer)


# Singleton used by the API layer
_graph = None


def get_graph():
    """Get the shared compiled workflow (thread-safe enough for the MVP)."""
    global _graph
    if _graph is None:
        _graph = build_workflow()
    return _graph
