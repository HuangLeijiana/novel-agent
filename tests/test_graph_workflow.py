"""Graph-level tests for the LangGraph workflow.

Covers the previously-dead routing and the HITL plumbing:

1. interrupt()/Command(resume=...) — a run pauses at each top-level phase
   confirmation and resumes with the inspiration applied to state.
2. review_decision routing (accept / revise / rewrite / iteration cap).
3. next_chapter_or_done routing (including the floor-of-3 safety).

Run: pytest tests/test_graph_workflow.py -v   (no API key needed)
"""

import pytest

from src.graph.edges import next_chapter_or_done, review_decision
from src.graph.workflow import build_workflow
from src.models.review import Issue, ReviewReport
from src.models.state import MainState


def _result_phase(result: dict) -> str | None:
    """Extract the phase name from a run result's __interrupt__ list."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    payload = interrupts[0].value
    return payload.get("phase", "workflow") if isinstance(payload, dict) else "workflow"


class FakeOrchestrator:
    """Orchestrator stub: no LLM, records what it was asked to do."""

    def __init__(self):
        self.inspirations: list[str | None] = []

    async def build_bible_world(self, state: MainState) -> MainState:
        return state  # leave artifacts None so no file manager is needed

    async def create_characters(self, state: MainState) -> MainState:
        self.inspirations.append(state.current_inspiration)
        return state

    async def create_outline(self, state: MainState) -> MainState:
        return state


def _config(orch: FakeOrchestrator, thread_id: str = "test-thread") -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "project_id": "test-project",
            "fm": None,  # artifacts stay None, so no disk I/O is needed
            "scheduler": None,
            "orchestrator": orch,
        }
    }


@pytest.mark.asyncio
async def test_workflow_pauses_at_each_phase_and_resumes():
    """The run pauses after each top-level phase; resume continues the graph
    and applies the user's inspiration to state before the next phase."""
    from langgraph.types import Command

    orch = FakeOrchestrator()
    graph = build_workflow()
    config = _config(orch)

    # First run: project_init → bible_work → pause at bible confirm
    r1 = await graph.ainvoke(MainState(), config=config)
    assert _result_phase(r1) == "bible_construction"

    # Resume with inspiration → characters_work → pause at characters confirm
    r2 = await graph.ainvoke(Command(resume={"inspiration": "加一个隐藏主线"}), config=config)
    assert _result_phase(r2) == "character_creation"
    assert orch.inspirations == ["加一个隐藏主线"], "inspiration must reach state"

    # Resume again → outline_work → pause at outline confirm
    r3 = await graph.ainvoke(Command(resume={"inspiration": None}), config=config)
    assert _result_phase(r3) == "master_outline"


# ── Review routing (now live in the chapter loop) ───────────────────────────

def _state_with_report(report: ReviewReport, iterations: int = 1) -> MainState:
    return MainState(review_report=report, review_iteration=iterations)


def test_review_decision_accepts_high_score():
    report = ReviewReport(overall_score=8.0, dimension_scores={"ai_flavor": 7.0})
    assert review_decision(_state_with_report(report)) == "accept"


def test_review_decision_revises_low_score():
    report = ReviewReport(overall_score=5.0, dimension_scores={"ai_flavor": 6.0})
    assert review_decision(_state_with_report(report)) == "revise"


def test_review_decision_rewrites_ai_flavored_text():
    report = ReviewReport(overall_score=7.0, dimension_scores={"ai_flavor": 3.0})
    assert review_decision(_state_with_report(report)) == "revise"


def test_review_decision_rewrites_critical_issue():
    report = ReviewReport(
        overall_score=7.0,
        dimension_scores={},
        issues=[Issue(severity="critical", description="plot hole")],
    )
    assert review_decision(_state_with_report(report)) == "rewrite"


def test_review_decision_accepts_at_iteration_cap():
    report = ReviewReport(overall_score=3.0, dimension_scores={})
    state = MainState(review_report=report, review_iteration=3, max_review_iterations=3)
    assert review_decision(state) == "accept"


def test_review_decision_accepts_without_report():
    assert review_decision(MainState()) == "accept"


# ── Chapter loop routing ────────────────────────────────────────────────────

def test_next_chapter_when_remaining():
    state = MainState(current_chapter_number=2, total_chapters=5)
    assert next_chapter_or_done(state) == "next_chapter"


def test_done_when_all_chapters_written():
    state = MainState(current_chapter_number=5, total_chapters=5)
    assert next_chapter_or_done(state) == "done"


def test_loop_terminates_without_outline_floor_of_three():
    """Even with no outline and no total, the loop must terminate (floor 3)."""
    state = MainState(current_chapter_number=3)  # total_chapters=0, outline=None
    assert next_chapter_or_done(state) == "done"
