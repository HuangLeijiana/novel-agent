"""LangGraph node functions — each node corresponds to a workflow phase."""

import logging
from typing import Any

from ..agents.orchestrator import OrchestratorAgent
from ..llm.scheduler import ModelScheduler
from ..models.common import HumanDecision, WorkflowPhase
from ..models.state import MainState

logger = logging.getLogger(__name__)


async def _get_orchestrator(config: dict) -> OrchestratorAgent:
    """Get or create an OrchestratorAgent from runtime config."""
    scheduler = config.get("configurable", {}).get("scheduler")
    if scheduler is None:
        scheduler = ModelScheduler()
    if not hasattr(_get_orchestrator, "_instance"):
        _get_orchestrator._instance = OrchestratorAgent(scheduler)
    return _get_orchestrator._instance


# ================================================================
# Phase Nodes
# ================================================================


async def project_init_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 1: Initialize project structure and metadata."""
    logger.info("Node: project_init")
    state.current_phase = WorkflowPhase.PROJECT_INIT
    return {"current_phase": WorkflowPhase.PROJECT_INIT}


async def bible_construction_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 2: Build Novel Bible — world, factions, characters, style."""
    logger.info("Node: bible_construction")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.build_bible(state)
    return {
        "bible": updated_state.bible,
        "characters": updated_state.characters,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def master_outline_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 3: Create master outline, volumes, turning points."""
    logger.info("Node: master_outline")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.create_outline(state)
    return {
        "outline": updated_state.outline,
        "total_chapters": updated_state.total_chapters,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def chapter_planning_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 4: Plan the next chapter in detail."""
    logger.info(f"Node: chapter_planning (chapter {state.current_chapter_number + 1})")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.plan_chapter(state)
    return {
        "chapter_plan": updated_state.chapter_plan,
        "current_chapter_number": updated_state.current_chapter_number,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def chapter_writing_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 5: Write the chapter draft."""
    logger.info(f"Node: chapter_writing (chapter {state.current_chapter_number})")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.write_chapter(state)
    return {
        "chapter_draft": updated_state.chapter_draft,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
        "human_feedback": None,  # Clear after use
    }


async def quality_review_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 6: Run quality reviews — editor, continuity, reader simulation."""
    logger.info(
        f"Node: quality_review (chapter {state.current_chapter_number}, iteration {state.review_iteration + 1})"
    )
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.review_chapter(state)
    return {
        "review_report": updated_state.review_report,
        "review_iteration": updated_state.review_iteration,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def polish_revision_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 7: Polish and refine the chapter."""
    logger.info(f"Node: polish_revision (chapter {state.current_chapter_number})")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.polish_chapter(state)
    return {
        "polished_chapter": updated_state.polished_chapter,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def memory_update_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 8: Update memory — summaries, timeline, foreshadowing, character states."""
    logger.info(f"Node: memory_update (chapter {state.current_chapter_number})")
    orchestrator = await _get_orchestrator(config or {})
    updated_state = await orchestrator.update_memory(state)
    return {
        "memory": updated_state.memory,
        "agent_log": updated_state.agent_log,
        "current_phase": updated_state.current_phase,
    }


async def human_confirmation_node(state: MainState, config: dict = None) -> dict[str, Any]:
    """Phase 9: Pause for human confirmation.

    In a LangGraph context, this node would use `interrupt()` to pause
    the graph and wait for human input. For now, it sets the phase and
    returns — the API layer handles the pause/resume logic.
    """
    logger.info(f"Node: human_confirmation (chapter {state.current_chapter_number})")
    state.current_phase = WorkflowPhase.HUMAN_CONFIRMATION
    return {"current_phase": WorkflowPhase.HUMAN_CONFIRMATION}
