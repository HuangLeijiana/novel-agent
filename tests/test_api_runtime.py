"""Runtime-path smoke tests for the FastAPI layer.

Covers the previously-untested parts of ``src/api/routes.py`` and
``src/api/phase_executor.py``:

- Project CRUD roundtrip (file-manager + route layer, no LLM)
- LLM-backed endpoint (``/projects/suggest-titles``) driven by a fake
  scheduler — no network, no API keys
- ``/start`` workflow failure survival (a failing provider must not crash
  the app; the error is caught by the workflow's top-level handler)
- Error paths (``retry-phase`` on unknown project → 404)
- Phase-executor scan-data handoff, including the documented cross-project
  pollution hazard (``_pending_scan_data`` keyed by platform, not project)

Run: pytest tests/test_api_runtime.py -v   (no API key needed)
"""

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Isolate the workspace BEFORE any settings/app import.
os.environ["WORKSPACE_ROOT"] = str(REPO_ROOT / ".test_workspace")

sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402
import pytest  # noqa: E402

from src.api import dependencies, phase_executor  # noqa: E402
from src.api.server import create_app  # noqa: E402
from src.llm.client import LLMResponse  # noqa: E402
from src.llm.scheduler import ModelScheduler  # noqa: E402
from src.models.state import MainState  # noqa: E402


class FakeScheduler(ModelScheduler):
    """Deterministic ModelScheduler stub — no network, no API keys.

    ``generate`` returns canned text; ``generate_structured`` returns a valid
    instance of the requested response model when it looks like a titles
    response, otherwise a default-constructed instance.
    """

    def __init__(self, *, content: str = "书名：测试书名\n简介：这是一个测试简介。",
                 titles: list[str] | None = None, fail: bool = False):
        super().__init__(assignments=[])
        self.content = content
        self.titles = titles or ["书名一", "书名二", "书名三"]
        self.fail = fail
        self.calls: list[str] = []

    async def generate(self, agent_type, system_prompt, user_prompt, response_model=None,
                       temperature_override=None, max_tokens_override=None):
        self.calls.append(agent_type)
        if self.fail:
            raise RuntimeError("fake provider failure (intentional)")
        return LLMResponse(content=self.content, model="fake-model", provider="fake")

    async def generate_structured(self, agent_type, system_prompt, user_prompt, response_model,
                                  temperature_override=None, max_tokens_override=None):
        self.calls.append(agent_type)
        if self.fail:
            raise RuntimeError("fake provider failure (intentional)")
        if "titles" in response_model.model_fields:
            return response_model.model_validate({"titles": self.titles})
        # Generic fallback: construct with defaults, no validation.
        return response_model.model_construct()


@pytest.fixture(autouse=True)
def _isolate_scheduler():
    """Give every test a fresh fake scheduler, and reset phase-executor globals."""
    fake = FakeScheduler()
    dependencies._scheduler = fake
    phase_executor._pending_scan_data.clear()
    yield fake


@pytest.fixture
def client():
    app = create_app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


PROJECT_PAYLOAD = {
    "config": {
        "title": "测试仙侠之逆天改命",
        "inspiration": "山村少年偶得残破仙书，逆天改命。",
        "genre": ["玄幻", "修仙"],
        "target_readers": "18-35岁男性读者",
        "target_word_count": 80000,
    }
}


async def test_project_crud_roundtrip(client):
    # create
    r = await client.post("/api/projects", json=PROJECT_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    project_id = body["project_id"]
    assert body["status"] == "initialized"

    # list contains it
    r = await client.get("/api/projects")
    assert r.status_code == 200
    ids = [p["project_id"] for p in r.json()]
    assert project_id in ids

    # detail
    r = await client.get(f"/api/projects/{project_id}")
    assert r.status_code == 200
    assert r.json()["config"]["title"] == PROJECT_PAYLOAD["config"]["title"]

    # update title
    r = await client.put(f"/api/projects/{project_id}/title", json={"title": "改名后的书名"})
    assert r.status_code == 200
    r = await client.get(f"/api/projects/{project_id}")
    assert r.json()["meta"]["title"] == "改名后的书名"

    # delete
    r = await client.delete(f"/api/projects/{project_id}")
    assert r.status_code == 200
    r = await client.get(f"/api/projects/{project_id}")
    assert r.status_code == 404


async def test_suggest_titles_uses_fake_scheduler(client, _isolate_scheduler):
    fake: FakeScheduler = _isolate_scheduler
    r = await client.post(
        "/api/projects/suggest-titles",
        json={"inspiration": "一个会说话的猫", "genre": ["都市"]},
    )
    assert r.status_code == 200, r.text
    titles = r.json()["titles"]
    assert titles == fake.titles
    assert fake.calls, "scheduler should have been called by suggest-titles"


async def test_start_workflow_survives_fake_llm_failure(client, _isolate_scheduler):
    fake: FakeScheduler = _isolate_scheduler
    fake.fail = True

    r = await client.post("/api/projects", json=PROJECT_PAYLOAD)
    project_id = r.json()["project_id"]

    r = await client.post(f"/api/projects/{project_id}/start")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "started"

    # Give the background workflow task a moment to run (and fail gracefully).
    await asyncio.sleep(0.5)
    assert fake.calls, "workflow should have hit the (failing) scheduler"

    # The app must remain healthy: project still listed, no 500s.
    r = await client.get("/api/projects")
    assert r.status_code == 200
    assert project_id in [p["project_id"] for p in r.json()]

    # Cancel any lingering background task so the loop closes cleanly.
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    await asyncio.gather(*[t for t in asyncio.all_tasks()
                           if t is not asyncio.current_task()], return_exceptions=True)


async def test_retry_phase_unknown_project_returns_404(client):
    r = await client.post("/api/projects/nonexistent/retry-phase/bible_construction",
                          json={"inspiration": None})
    assert r.status_code == 404


# ── phase_executor: scan-data handoff ────────────────────────────────────────

class FakeOrchestrator:
    """Orchestrator stub that records what scan data it received."""

    def __init__(self):
        self.received: dict[str, str | None] = {}

    async def scan_platforms(self, state, feilu_content=None, fanqie_content=None):
        self.received["feilu"] = feilu_content
        self.received["fanqie"] = fanqie_content
        return state


async def test_platform_scan_consumes_pending_data():
    phase_executor._pending_scan_data["feilu"] = "<html>project-A-feilu</html>"
    orch = FakeOrchestrator()
    state = MainState()
    await phase_executor.execute_phase_platform_scan(state, orch, None)
    assert orch.received == {"feilu": "<html>project-A-feilu</html>", "fanqie": None}
    assert "feilu" not in phase_executor._pending_scan_data


async def test_platform_scan_without_data_initializes_topic_state():
    orch = FakeOrchestrator()
    state = MainState()
    await phase_executor.execute_phase_platform_scan(state, orch, None)
    assert orch.received == {}
    assert state.topic_research is not None


@pytest.mark.xfail(
    reason="Known issue: _pending_scan_data is keyed by platform, not project — "
           "two concurrent projects can steal each other's scan data. "
           "Fix: key by project_id and pass it through the phase executor.",
    strict=False,
)
async def test_scan_data_is_project_scoped():
    """Bug documentation: project B's scan data overwrites project A's before
    A's phase runs, so A receives B's data. This test xfails until the
    executor keys pending data by project_id."""
    phase_executor._pending_scan_data["feilu"] = "project-A-content"
    phase_executor._pending_scan_data["feilu"] = "project-B-content"  # B writes after A

    orch = FakeOrchestrator()
    await phase_executor.execute_phase_platform_scan(MainState(), orch, None)
    assert orch.received["feilu"] == "project-A-content"
