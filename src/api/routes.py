"""FastAPI REST routes for novel agent workflow control and artifact access."""

import asyncio
import logging
import re
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..models.common import HumanDecision
from ..models.project import ProjectConfig
from ..models.state import MainState
from ..graph.workflow import build_workflow
from .dependencies import get_file_manager, get_scheduler
from .websocket import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory project state registry (for MVP; replace with DB later)
_active_projects: dict[str, dict] = {}  # project_id -> {state, workflow, thread_config}


def _slugify_title(title: str, max_len: int = 30) -> str:
    """Convert a novel title to a safe folder-name segment.

    Strips characters illegal on Windows/macOS/Linux file systems,
    collapses whitespace, and truncates to *max_len* chars.
    """
    # Remove characters illegal in file names on common platforms
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    # Collapse runs of whitespace / dots
    sanitized = re.sub(r"\s+", "", sanitized)
    # Strip leading/trailing dots, spaces, hyphens
    sanitized = sanitized.strip(". -")
    if not sanitized:
        return "novel"
    # Truncate to max_len, avoiding trailing surrogate pairs
    return sanitized[:max_len]


# ============================================================
# Request/Response Schemas
# ============================================================


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""

    config: ProjectConfig


class ProjectResponse(BaseModel):
    """Project summary in list/detail views."""

    project_id: str
    title: Optional[str] = None
    status: str
    current_phase: str
    current_chapter: int
    total_chapters: int


class SuggestTitlesRequest(BaseModel):
    """Request to generate title suggestions."""

    inspiration: str = Field(..., min_length=1, max_length=2000)
    genre: list[str] = Field(default_factory=list)


class TitleSuggestionsResponse(BaseModel):
    """AI-generated title suggestions."""

    titles: list[str] = Field(..., min_length=1)


class HumanDecisionRequest(BaseModel):
    """Request to submit a human decision with structured feedback.

    The feedback fields implement the evaluation report's required
    explicit feedback entry: thumbs up/down + reason tags.
    """

    decision: str = Field(..., description="accept / revise / rewrite / rollback")
    feedback: Optional[str] = Field(default=None, description="Human feedback text (free-form)")
    rollback_target: Optional[str] = Field(default=None, description="For rollback: 'chapter_plan' or 'bible'")
    # Structured feedback (evaluation report pre-launch requirement #1)
    sentiment: Optional[str] = Field(
        default=None,
        description="Feedback sentiment: 'thumbs_up' or 'thumbs_down'",
    )
    reason_tags: Optional[list[str]] = Field(
        default=None,
        description="Reason tags for negative feedback: not_meeting_expectations, character_broken, plot_boring, ai_flavor_heavy, pacing_issue, dialogue_issue, worldbuilding_inconsistent, other",
    )


# ============================================================
# Title Suggestions
# ============================================================


@router.post("/projects/suggest-titles", response_model=TitleSuggestionsResponse)
async def suggest_titles(req: SuggestTitlesRequest):
    """Generate novel title suggestions based on inspiration and genre."""
    genre_text = "、".join(req.genre) if req.genre else "未指定"

    system_prompt = (
        "你是一位资深网络文学编辑，擅长为小说取名。"
        "根据用户提供的故事灵感和题材，生成3-5个备选书名。"
        "要求：\n"
        "- 书名简洁有力，2-8个字最佳\n"
        "- 符合题材风格（如玄幻、修仙、都市等）\n"
        "- 有吸引力，能让读者产生点击欲望\n"
        "- 优先使用中文书名\n"
        '- 必须返回以下JSON格式：{"titles": ["书名1", "书名2", "书名3"]}'
    )
    user_prompt = f"故事灵感：{req.inspiration}\n题材：{genre_text}\n\n请为这部小说生成3-5个备选书名。"

    class _TitleList(BaseModel):
        titles: list[str]

    try:
        scheduler = get_scheduler()
        result = await scheduler.generate_structured(
            agent_type="orchestrator",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_TitleList,
            temperature_override=0.9,
        )
        return TitleSuggestionsResponse(titles=result.titles[:5])
    except Exception as e:
        logger.error(f"Title suggestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成书名失败: {str(e)}")


# ============================================================
# Project CRUD
# ============================================================


@router.post("/projects", response_model=ProjectResponse)
async def create_project(req: CreateProjectRequest):
    """Create a new novel project and initialize its workspace."""
    slug = _slugify_title(req.config.title)
    short_uid = uuid.uuid4().hex[:6]
    project_id = f"{slug}_{short_uid}"

    fm = get_file_manager(project_id)

    if fm.exists():
        raise HTTPException(status_code=409, detail="Project ID collision")

    fm.initialize(req.config)

    # Also persist project metadata with the title
    from ..models.project import ProjectMeta

    meta = ProjectMeta(project_id=project_id, title=req.config.title)
    fm.save_project_meta(meta)

    return ProjectResponse(
        project_id=project_id,
        title=req.config.title,
        status="initialized",
        current_phase="idle",
        current_chapter=0,
        total_chapters=0,
    )


@router.get("/projects")
async def list_projects():
    """List all projects."""
    settings = __import__("src.config.settings", fromlist=["get_settings"]).get_settings()
    projects_dir = settings.workspace_path / "projects"
    if not projects_dir.exists():
        return []

    projects = []
    for proj_dir in sorted(projects_dir.iterdir()):
        if proj_dir.is_dir() and (proj_dir / "project.yaml").exists():
            fm = get_file_manager(proj_dir.name)
            meta = fm.load_project_meta()
            config = fm.load_project_config()
            projects.append(
                {
                    "project_id": proj_dir.name,
                    "title": meta.title if meta else config.inspiration[:50] if config else proj_dir.name,
                    "status": meta.status if meta else "initialized",
                    "current_phase": meta.current_phase if meta else "idle",
                    "current_chapter": meta.current_chapter if meta else 0,
                    "total_chapters": meta.total_chapters if meta else 0,
                }
            )
    return projects


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details."""
    fm = get_file_manager(project_id)
    meta = fm.load_project_meta()
    config = fm.load_project_config()
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "project_id": project_id,
        "meta": meta.model_dump() if meta else None,
        "config": config.model_dump(),
    }


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its artifacts."""
    import shutil

    fm = get_file_manager(project_id)
    if not fm.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(fm.root)
    _active_projects.pop(project_id, None)
    return {"status": "deleted"}


class UpdateTitleRequest(BaseModel):
    title: str


@router.put("/projects/{project_id}/title")
async def update_project_title(project_id: str, req: UpdateTitleRequest):
    """Update a project's title."""
    fm = get_file_manager(project_id)
    if not fm.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    config = fm.load_project_config()
    if config:
        config.title = req.title
        fm.save_project_config(config)
    from ..models.project import ProjectMeta

    fm.save_project_meta(ProjectMeta(project_id=project_id, title=req.title))
    return {"status": "ok", "title": req.title}


# ============================================================
# Workflow Control — Phased with Human-in-the-Loop
# ============================================================

# Phase sequence for the interactive workflow
PHASE_SEQUENCE = [
    "bible_construction",
    "character_creation",
    "master_outline",
]

CHAPTER_PHASES = [
    "chapter_planning",
    "chapter_writing",
    "quality_review",
    "polish_revision",
    "memory_update",
]


class NextPhaseRequest(BaseModel):
    """Request to confirm current phase and proceed."""

    phase: str = Field(..., description="Phase being confirmed")
    inspiration: Optional[str] = Field(default=None, description="User inspiration for next phase")
    edits: Optional[dict] = Field(default=None, description="User edits to current phase output")


class PhaseConfirmationRequest(BaseModel):
    """Request to confirm phase and provide input."""

    inspiration: Optional[str] = Field(default=None)


@router.post("/projects/{project_id}/start")
async def start_workflow(project_id: str):
    """Start the phased interactive workflow. Skips phases that already have content."""
    fm = get_file_manager(project_id)
    config = fm.load_project_config()
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")

    from ..models.project import ProjectMeta

    meta = fm.load_project_meta() or ProjectMeta(project_id=project_id)

    initial_state = MainState(project_meta=meta, project_config=config)

    # Load existing artifacts so we can skip completed phases
    existing_bible = fm.load_bible()
    if existing_bible:
        initial_state.bible = existing_bible
        logger.info(f"Loaded existing bible for {project_id}")

    existing_chars = fm.load_characters()
    if existing_chars:
        initial_state.characters = existing_chars
        logger.info(f"Loaded existing characters for {project_id}")

    existing_outline = fm.load_master_outline()
    if existing_outline:
        initial_state.outline = existing_outline
        initial_state.total_chapters = existing_outline.chapter_count or 15
        # Find highest FULLY COMPLETE chapter (phase 5 = memory updated)
        ch = 0
        for n in range(1, 100):
            phase = _get_chapter_phase(fm, n)
            if phase >= 5:
                ch = n
            elif phase > 0:
                # Chapter started but not complete — resume from this chapter
                ch = n - 1
                break
            else:
                break
        initial_state.current_chapter_number = ch
        logger.info(
            f"Loaded outline, {ch} chapters fully complete, next chapter has phase {_get_chapter_phase(fm, ch + 1)}"
        )

    _active_projects[project_id] = {
        "state": initial_state,
        "file_manager": fm,
        "phase_event": asyncio.Event(),
        "phase_input": {},
        "current_phase": None,
    }

    asyncio.create_task(_run_phased_workflow(project_id))
    return {"status": "started", "project_id": project_id}


class AiTitlesRequest(BaseModel):
    inspiration: str = Field(..., min_length=1, max_length=2000)


class GenerateSynopsisRequest(BaseModel):
    """Request to generate title + synopsis for a candidate topic."""

    genre_name: str = Field(..., min_length=1)
    inspiration: str = Field(default="", description="User inspiration or additional context")


@router.post("/projects/{project_id}/generate-synopsis")
async def generate_synopsis_for_candidate(project_id: str, req: GenerateSynopsisRequest):
    """Generate a book title + synopsis for a single candidate topic on demand.

    Uses plain-text generation (not structured output) for reliability with
    all API providers including ModelScope.
    """
    scheduler = get_scheduler()
    from ..agents.base import BaseAgent

    agent = BaseAgent.__new__(BaseAgent)
    agent.scheduler = scheduler
    agent.agent_type = "topic_scout"

    system_prompt = """你是一个专业的网文包装策划师，擅长为网文设计高点击率的书名和简介。

书名要求：
- 6-15个字
- 突出核心卖点/金手指
- 符合番茄免费文风格（直白、有冲突感、一眼看懂）
- 能吸引目标读者点击

简介要求：
- 2-4句话，80-150字
- 快速建立期待、明确爽点、留钩子
- 适合做短视频推广文案

请严格按以下格式输出：
书名：<书名>
简介：<简介>"""

    user_prompt = f"""请为以下题材生成一个吸睛的书名和简介。

题材：{req.genre_name}
灵感：{req.inspiration or "无"}

请直接输出书名和简介。"""

    try:
        result = await agent.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_override=0.8,
        )
        content = result.content if hasattr(result, "content") else str(result)

        # Parse title and synopsis from text
        import re

        title_match = re.search(r"书名[：:]\s*(.+?)(?:\n|$)", content)
        synopsis_match = re.search(r"简介[：:]\s*(.+?)(?:\n|$)", content, re.DOTALL)

        title = title_match.group(1).strip() if title_match else req.genre_name
        synopsis = synopsis_match.group(1).strip() if synopsis_match else ""

        # Fallback: if no label found, use first line as title, rest as synopsis
        if not title_match and not synopsis_match:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if len(lines) >= 2:
                title = lines[0]
                synopsis = " ".join(lines[1:])
            elif lines:
                title = lines[0]

        # Clean up common artifacts
        title = re.sub(r"^[《「]|[》」]$", "", title).strip()[:50]
        synopsis = synopsis.strip()[:300]

        return {
            "genre_name": req.genre_name,
            "final_title": title or req.genre_name,
            "final_synopsis": synopsis or req.inspiration or "",
            "title_candidates": [],
        }
    except Exception as e:
        logger.error(f"Synopsis generation failed: {e}")
        return {
            "genre_name": req.genre_name,
            "final_title": req.genre_name,
            "final_synopsis": req.inspiration or f"一部{req.genre_name}题材的精彩小说",
            "title_candidates": [],
        }


@router.post("/projects/{project_id}/ai-titles")
async def ai_generate_titles(project_id: str, req: AiTitlesRequest):
    """Generate book title suggestions based on user inspiration and scan data."""
    proj = _active_projects.get(project_id)

    # Try to get scan context from active state
    scan_context = ""
    if proj and proj.get("state"):
        state = proj["state"]
        if hasattr(state, "topic_research") and state.topic_research:
            tr = state.topic_research
            if tr.feilu_scan and not tr.feilu_scan.scan_failed:
                titles = [e.title for e in (tr.feilu_scan.entries or [])[:10] if e.title]
                if titles:
                    scan_context += "飞卢热书：" + "、".join(titles) + "\n"
            if tr.fanqie_scan and not tr.fanqie_scan.scan_failed:
                titles = [e.title for e in (tr.fanqie_scan.entries or [])[:10] if e.title]
                if titles:
                    scan_context += "番茄热书：" + "、".join(titles) + "\n"

    scheduler = get_scheduler()
    from ..agents.base import BaseAgent

    agent = BaseAgent.__new__(BaseAgent)
    agent.scheduler = scheduler

    system_prompt = f"""你是一个专业的网文书名创作专家。你需要基于用户的灵感，结合当前榜单趋势，生成吸睛的书名。

{scan_context}

书名要求：
- 6-15个字
- 突出核心卖点/金手指
- 符合当前网文流行风格
- 能吸引目标读者点击
"""

    user_prompt = f"""用户的灵感/想法：{req.inspiration}

请生成5个有吸引力的书名，直接列出，每行一个书名，不要编号。"""

    try:
        result = await agent.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_override=0.8,
        )
        # Parse titles from result (split by newlines, filter empty)
        titles = [line.strip() for line in result.content.split("\n") if line.strip()]
        titles = [t for t in titles if len(t) >= 4 and not t.startswith("#")][:5]
        return {"titles": titles}
    except Exception as e:
        logger.error(f"AI title generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# ============================================================
# Topic Research — Scan Data Submission (Phase 0)
# ============================================================


class SubmitScanRequest(BaseModel):
    """Submit platform HTML content for scanning."""

    feilu_html: Optional[str] = Field(default=None, description="飞卢榜单页面HTML/文本")
    fanqie_html: Optional[str] = Field(default=None, description="番茄榜单页面HTML/文本")


@router.post("/projects/{project_id}/submit-scan")
async def submit_scan_data(project_id: str, req: SubmitScanRequest):
    """Submit browser-scraped HTML content for platform scanning.

    This stores the data and starts the Phase 0 pipeline if a workflow
    is active, or just stores it for when the workflow starts.
    """
    from .phase_executor import _pending_scan_data

    proj = _active_projects.get(project_id)
    if not proj:
        # Create a lightweight entry — the workflow hasn't started yet
        _active_projects[project_id] = {
            "state": None,
            "file_manager": get_file_manager(project_id),
            "phase_event": asyncio.Event(),
            "phase_input": {},
            "current_phase": None,
        }

    _pending_scan_data["feilu"] = req.feilu_html
    _pending_scan_data["fanqie"] = req.fanqie_html

    return {
        "status": "scan_data_received",
        "project_id": project_id,
        "feilu_chars": len(req.feilu_html) if req.feilu_html else 0,
        "fanqie_chars": len(req.fanqie_html) if req.fanqie_html else 0,
    }


@router.post("/projects/{project_id}/auto-scan")
async def auto_scan(project_id: str):
    """Automatically scrape Feilu and Fanqie ranking pages using headless browser.

    Fetches both platform ranking pages, stores the HTML, and starts
    the Phase 0 pipeline. Falls back gracefully if scraping fails.
    """
    from ..utils.scraper import scrape_all
    from .phase_executor import _pending_scan_data

    # Ensure project entry exists
    proj = _active_projects.get(project_id)
    if not proj:
        _active_projects[project_id] = {
            "state": None,
            "file_manager": get_file_manager(project_id),
            "phase_event": asyncio.Event(),
            "phase_input": {},
            "current_phase": None,
        }

    # Scrape both platforms
    results = await scrape_all()
    feilu_html = results.get("feilu")
    fanqie_html = results.get("fanqie")

    _pending_scan_data["feilu"] = feilu_html
    _pending_scan_data["fanqie"] = fanqie_html

    feilu_ok = bool(feilu_html and len(feilu_html) > 500)
    fanqie_ok = bool(fanqie_html and len(fanqie_html) > 500)

    if feilu_ok and fanqie_ok:
        scan_status = "ok"
    elif feilu_ok or fanqie_ok:
        scan_status = "partial"
    else:
        scan_status = "fail"

    return {
        "status": scan_status,
        "project_id": project_id,
        "feilu": {
            "success": feilu_ok,
            "chars": len(feilu_html) if feilu_html else 0,
        },
        "fanqie": {
            "success": fanqie_ok,
            "chars": len(fanqie_html) if fanqie_html else 0,
        },
        "message": (
            "扫榜完成"
            if (feilu_ok and fanqie_ok)
            else "部分平台扫榜成功"
            if (feilu_ok or fanqie_ok)
            else "自动扫榜失败，请手动粘贴榜单内容"
        ),
    }


@router.post("/projects/{project_id}/confirm-phase")
async def confirm_phase(project_id: str, req: PhaseConfirmationRequest):
    """Confirm current phase and proceed to next."""
    proj = _active_projects.get(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="No active workflow")

    proj["phase_input"] = {"inspiration": req.inspiration}

    # Signal the waiting event to unblock
    if proj.get("phase_event"):
        proj["phase_event"].set()

    return {"status": "confirmed", "phase": proj.get("current_phase")}


@router.post("/projects/{project_id}/pause")
async def pause_workflow(project_id: str):
    """Pause a running workflow."""
    return {"status": "paused", "project_id": project_id}


@router.post("/projects/{project_id}/retry-phase/{phase}")
async def retry_phase(project_id: str, phase: str, req: PhaseConfirmationRequest = PhaseConfirmationRequest()):
    """Retry a failed phase with optional updated inspiration.

    Re-runs the phase executor for the given phase and broadcasts results.
    Only works when the workflow is blocked waiting for confirmation.
    """
    from .phase_executor import (
        PHASE_EXECUTORS,
        PHASE_LABELS,
        get_phase_data,
    )

    proj = _active_projects.get(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="No active workflow")

    state: MainState = proj.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="No state available")

    executor_fn = PHASE_EXECUTORS.get(phase)
    if not executor_fn:
        raise HTTPException(status_code=400, detail=f"Unknown phase: {phase}")

    # Apply inspiration if provided
    if req and req.inspiration:
        state.current_inspiration = req.inspiration

    fm = proj["file_manager"]
    scheduler = get_scheduler()
    from ..agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent(scheduler)

    label = PHASE_LABELS.get(phase, phase)
    await ws_manager.broadcast_phase_update(project_id, phase, 0.1, f"正在重新{label}...")

    try:
        await executor_fn(state, orchestrator, fm)
    except Exception as e:
        logger.error(f"Retry phase {phase} failed: {e}", exc_info=True)
        await ws_manager.broadcast_error(project_id, phase, str(e))
        raise HTTPException(status_code=500, detail=str(e))

    phase_data = get_phase_data(state, phase)
    await ws_manager.broadcast_phase_update(project_id, phase, 1.0, f"{label}完成")
    await ws_manager.broadcast_phase_complete(project_id, phase, phase_data)

    return {"status": "retried", "phase": phase}


async def _run_phased_workflow(project_id: str):
    """Execute workflow phases sequentially with human confirmation between each."""
    from .phase_executor import (
        PHASE_EXECUTORS,
        PHASE_TAB_MAP,
        PHASE_LABELS,
        get_phase_data,
        execute_phase_bible,
        execute_phase_characters,
        execute_phase_outline,
        execute_phase_platform_scan,
        execute_phase_topic_selection,
        execute_phase_mini_arc,
    )

    proj = _active_projects.get(project_id)
    if not proj:
        return

    state: MainState = proj["state"]
    fm = proj["file_manager"]
    scheduler = get_scheduler()
    from ..agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent(scheduler)

    try:
        # ================================================================
        # Phase 0a: Platform Scanning (optional — requires scan data)
        # ================================================================
        from .phase_executor import _pending_scan_data

        has_scan_data = bool(_pending_scan_data.get("feilu") or _pending_scan_data.get("fanqie"))
        if has_scan_data:
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "platform_scan",
                execute_phase_platform_scan,
            )
            if not await _wait_for_confirmation(project_id, "platform_scan"):
                return

            # Phase 0b: Topic Selection
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "topic_selection",
                execute_phase_topic_selection,
            )
            if not await _wait_for_confirmation(project_id, "topic_selection"):
                return
            state = _apply_inspiration(project_id, state)

            # Phase 0c: Mini-Arc Outline
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "mini_arc_outline",
                execute_phase_mini_arc,
            )
            if not await _wait_for_confirmation(project_id, "mini_arc_outline"):
                return
            state = _apply_inspiration(project_id, state)
        else:
            logger.info("No scan data — skipping Phase 0 (topic research)")

        # === Phase 1: Bible Construction (世界观) ===
        if not state.bible:
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "bible_construction",
                execute_phase_bible,
            )
            if not await _wait_for_confirmation(project_id, "bible_construction"):
                return
            state = _apply_inspiration(project_id, state)
        else:
            await ws_manager.broadcast_phase_update(project_id, "bible_construction", 1.0, "世界观已存在，跳过")
            await ws_manager.broadcast_phase_complete(
                project_id, "bible_construction", get_phase_data(state, "bible_construction")
            )

        # === Phase 2: Character Creation (角色) ===
        if not state.characters:
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "character_creation",
                execute_phase_characters,
            )
            if not await _wait_for_confirmation(project_id, "character_creation"):
                return
            state = _apply_inspiration(project_id, state)
        else:
            await ws_manager.broadcast_phase_update(project_id, "character_creation", 1.0, "角色已存在，跳过")
            await ws_manager.broadcast_phase_complete(
                project_id, "character_creation", get_phase_data(state, "character_creation")
            )

        # === Phase 3: Master Outline (大纲) ===
        if not state.outline:
            await _run_single_phase(
                project_id,
                state,
                orchestrator,
                fm,
                "master_outline",
                execute_phase_outline,
            )
            if not await _wait_for_confirmation(project_id, "master_outline"):
                return
            state = _apply_inspiration(project_id, state)
        else:
            await ws_manager.broadcast_phase_update(project_id, "master_outline", 1.0, "大纲已存在，跳过")
            await ws_manager.broadcast_phase_complete(
                project_id, "master_outline", get_phase_data(state, "master_outline")
            )

        # === Phase 4: Chapter Loop ===
        from .phase_executor import (
            execute_phase_chapter_planning,
            execute_phase_chapter_writing,
            execute_phase_review,
            execute_phase_polish,
            execute_phase_memory,
        )

        total = max(state.outline.chapter_count if state.outline else 0, state.total_chapters, 3)
        # Calculate per-chapter progress increment
        chapter_progress_increment = 85.0 / max(total, 1)  # 5%→90% range

        while state.current_chapter_number < total:
            ch = state.current_chapter_number + 1
            base_progress = 5 + (state.current_chapter_number * chapter_progress_increment)

            # Track per-chapter phase: 0=not started, 1=planned, 2=written, 3=reviewed, 4=polished, 5=done
            ch_phase = _get_chapter_phase(fm, ch)
            await ws_manager.broadcast_phase_update(
                project_id, "chapter_loop", base_progress / 100, f"第{ch}章（共{total}章）· 阶段{ch_phase}/5"
            )

            # Plan chapter (phase 0→1)
            if ch_phase < 1:
                await _run_single_phase(
                    project_id,
                    state,
                    orchestrator,
                    fm,
                    "chapter_planning",
                    execute_phase_chapter_planning,
                )
                if state.chapter_plan:
                    fm.save_chapter_plan(state.chapter_plan)
                _set_chapter_phase(fm, ch, 1)

            # Write chapter (phase 1→2)
            if _get_chapter_phase(fm, ch) < 2:
                proj = _active_projects.get(project_id)
                if proj:
                    proj["current_phase"] = "chapter_writing"
                await ws_manager.broadcast_phase_update(
                    project_id, "chapter_writing", (base_progress + 4) / 100, f"正在写作第{ch}章..."
                )
                state = await orchestrator.write_chapter(state)
                if state.chapter_draft:
                    fm.save_chapter_draft(state.chapter_draft)
                _set_chapter_phase(fm, ch, 2)

            # Review (phase 2→3)
            if _get_chapter_phase(fm, ch) < 3:
                state = await orchestrator.review_chapter(state)
                if state.review_report:
                    fm.save_review_report(state.review_report)
                    await ws_manager.broadcast(
                        project_id,
                        {
                            "type": "chapter_complete",
                            "chapter": ch,
                            "scores": state.review_report.dimension_scores,
                        },
                    )
                _set_chapter_phase(fm, ch, 3)

            # Polish (phase 3→4)
            if _get_chapter_phase(fm, ch) < 4:
                await ws_manager.broadcast_phase_update(
                    project_id, "polish_revision", (base_progress + 8) / 100, f"润色第{ch}章..."
                )
                state = await orchestrator.polish_chapter(state)
                if state.polished_chapter:
                    fm.save_chapter_markdown(state.polished_chapter)
                _set_chapter_phase(fm, ch, 4)

            # Memory (phase 4→5)
            if _get_chapter_phase(fm, ch) < 5:
                state = await orchestrator.update_memory(state)
                if state.memory:
                    fm.save_memory(state.memory)
                _set_chapter_phase(fm, ch, 5)

            state.current_chapter_number = ch
            await ws_manager.broadcast_phase_update(
                project_id, "chapter_loop", (base_progress + 16) / 100, f"第{ch}章完成 ✓"
            )

        await ws_manager.broadcast(
            project_id,
            {
                "type": "workflow_complete",
                "message": f"全部{total}章已完成！",
            },
        )

    except Exception as e:
        logger.error(f"Workflow error for {project_id}: {e}", exc_info=True)
        await ws_manager.broadcast_error(project_id, "workflow", str(e))


async def _run_single_phase(
    project_id: str,
    state: MainState,
    orchestrator,
    fm,
    phase: str,
    executor_fn,
) -> None:
    """Execute one phase and broadcast progress with heartbeat updates."""
    from .phase_executor import PHASE_LABELS, get_phase_data

    proj = _active_projects.get(project_id)
    if proj:
        proj["current_phase"] = phase

    label = PHASE_LABELS.get(phase, phase)

    # Heartbeat: send periodic "still working" updates so the user knows the process is alive
    heartbeat_stop = asyncio.Event()

    async def _heartbeat():
        dots = 0
        while not heartbeat_stop.is_set():
            dots = (dots + 1) % 4
            await ws_manager.broadcast_phase_update(
                project_id,
                phase,
                0.15 + (dots * 0.02),
                f"正在{label}{'.' * dots}",
            )
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=3)
            except asyncio.TimeoutError:
                pass

    heartbeat_task = asyncio.create_task(_heartbeat())
    await ws_manager.broadcast_phase_update(project_id, phase, 0.1, f"正在{label}...")

    try:
        await executor_fn(state, orchestrator, fm)
    except Exception as e:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        logger.error(f"Phase {phase} failed: {e}", exc_info=True)
        await ws_manager.broadcast_error(project_id, phase, str(e))
        raise

    heartbeat_stop.set()
    heartbeat_task.cancel()

    # Broadcast completion with data
    phase_data = get_phase_data(state, phase)
    await ws_manager.broadcast_phase_update(project_id, phase, 1.0, f"{label}完成")
    await ws_manager.broadcast_phase_complete(project_id, phase, phase_data)


async def _wait_for_confirmation(project_id: str, phase: str) -> bool:
    """Wait for human confirmation. Returns False on timeout."""
    from .phase_executor import PHASE_LABELS

    proj = _active_projects.get(project_id)
    if not proj:
        return False

    label = PHASE_LABELS.get(phase, phase)
    await ws_manager.broadcast_phase_blocked(project_id, phase)

    proj["phase_event"].clear()
    try:
        await asyncio.wait_for(proj["phase_event"].wait(), timeout=3600)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Phase confirmation timeout for {project_id}")
        return False


def _get_chapter_phase(fm, ch: int) -> int:
    """Get the completion phase of a chapter (0-5). Persisted to disk."""
    import os, json

    phase_file = os.path.join(fm.root, "output", "chapters", f"chapter_{ch:03d}_phase.json")
    try:
        if os.path.exists(phase_file):
            with open(phase_file) as f:
                return json.load(f).get("phase", 0)
    except Exception:
        pass
    # Fallback: detect from existing files
    if fm.load_chapter_markdown(ch):
        return 4  # Has polished content
    if fm.load_review_report(ch) if hasattr(fm, "load_review_report") else False:
        return 3
    if fm.load_chapter_draft(ch) if hasattr(fm, "load_chapter_draft") else False:
        return 2
    if fm.load_chapter_plan(ch):
        return 1
    return 0


def _set_chapter_phase(fm, ch: int, phase: int):
    """Save chapter completion phase to disk."""
    import os, json

    phase_dir = os.path.join(fm.root, "output", "chapters")
    os.makedirs(phase_dir, exist_ok=True)
    phase_file = os.path.join(phase_dir, f"chapter_{ch:03d}_phase.json")
    with open(phase_file, "w") as f:
        json.dump({"chapter": ch, "phase": phase}, f)


def _apply_inspiration(project_id: str, state: MainState) -> MainState:
    """Apply user inspiration to state for the next phase."""
    proj = _active_projects.get(project_id)
    if not proj:
        return state

    phase_input = proj.get("phase_input", {})
    inspiration = phase_input.get("inspiration", "")
    if inspiration:
        state.current_inspiration = inspiration
        # Clear for next round
        proj["phase_input"] = {}
    return state


# ============================================================
# Human Decision
# ============================================================


@router.post("/projects/{project_id}/human-decision")
async def submit_human_decision(project_id: str, req: HumanDecisionRequest):
    """Submit a human decision with structured feedback (accept/revise/rewrite/rollback).

    Records explicit feedback (thumbs up/down + reason tags) for the data flywheel,
    fulfilling the evaluation report's pre-launch requirement #1.
    """
    proj = _active_projects.get(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="No active workflow for this project")

    state: MainState = proj["state"]
    state.human_decision = HumanDecision(req.decision)
    state.human_feedback = req.feedback
    if req.rollback_target:
        state.rollback_target = req.rollback_target

    # Record structured feedback for data flywheel
    from ..models.feedback import FeedbackEntry, FeedbackSentiment, FeedbackReasonTag

    sentiment = FeedbackSentiment.THUMBS_UP
    if req.sentiment == "thumbs_down":
        sentiment = FeedbackSentiment.THUMBS_DOWN
    elif req.decision in ("revise", "rewrite", "rollback"):
        # Auto-infer thumbs_down from negative decisions if sentiment not explicit
        sentiment = FeedbackSentiment.THUMBS_DOWN

    reason_tags: list[FeedbackReasonTag] = []
    if req.reason_tags:
        for tag_str in req.reason_tags:
            try:
                reason_tags.append(FeedbackReasonTag(tag_str))
            except ValueError:
                logger.warning(f"Unknown reason tag: {tag_str}")

    entry = FeedbackEntry(
        chapter_number=state.current_chapter_number,
        phase=state.current_phase.value if state.current_phase else "",
        sentiment=sentiment,
        reason_tags=reason_tags,
        notes=req.feedback or "",
        decision=req.decision,
        revision_count=state.review_iteration,
        review_score=state.review_report.overall_score if state.review_report else 0.0,
    )
    state.feedback_records.append(entry)

    # Persist feedback to disk
    fm = proj.get("file_manager") or get_file_manager(project_id)
    _save_feedback(fm, state.feedback_records)

    return {
        "status": "decision_received",
        "decision": req.decision,
        "feedback_recorded": True,
        "sentiment": sentiment.value,
    }


# ============================================================
# Artifact Access
# ============================================================


@router.get("/projects/{project_id}/bible")
async def get_bible(project_id: str):
    """Get the complete novel bible."""
    fm = get_file_manager(project_id)
    bible = fm.load_bible()
    if not bible:
        raise HTTPException(status_code=404, detail="Bible not yet created")
    return bible.model_dump()


@router.get("/projects/{project_id}/characters")
async def get_characters(project_id: str):
    """Get all character profiles."""
    fm = get_file_manager(project_id)
    chars = fm.load_characters()
    if not chars:
        raise HTTPException(status_code=404, detail="Characters not yet created")
    return chars.model_dump()


@router.get("/projects/{project_id}/outline")
async def get_outline(project_id: str):
    """Get the master outline."""
    fm = get_file_manager(project_id)
    outline = fm.load_master_outline()
    if not outline:
        raise HTTPException(status_code=404, detail="Outline not yet created")
    return outline.model_dump()


@router.get("/projects/{project_id}/chapters")
async def list_chapters(project_id: str):
    """List all chapters with their status."""
    fm = get_file_manager(project_id)
    plans = fm.list_chapter_plans()
    chapters = []
    for num in plans:
        plan = fm.load_chapter_plan(num)
        chapters.append(
            {
                "chapter_number": num,
                "title": plan.title if plan else "",
                "status": plan.status if plan else "unknown",
            }
        )
    return chapters


@router.get("/projects/{project_id}/chapters/{chapter_number}")
async def get_chapter(project_id: str, chapter_number: int):
    """Get chapter details — plan, draft, and polished versions."""
    fm = get_file_manager(project_id)
    plan = fm.load_chapter_plan(chapter_number)
    draft = fm.load_chapter_draft(chapter_number)
    review = fm.load_review_report(chapter_number)
    polished_md = fm.load_chapter_markdown(chapter_number)

    return {
        "chapter_number": chapter_number,
        "plan": plan.model_dump() if plan else None,
        "draft": draft.model_dump() if draft else None,
        "review": review.model_dump() if review else None,
        "content": polished_md,
    }


@router.get("/projects/{project_id}/chapters/{chapter_number}/md")
async def download_chapter_md(project_id: str, chapter_number: int):
    """Download a chapter as Markdown."""
    from fastapi.responses import PlainTextResponse

    fm = get_file_manager(project_id)
    content = fm.load_chapter_markdown(chapter_number)
    if content is None:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return PlainTextResponse(content=content, media_type="text/markdown; charset=utf-8")


@router.get("/projects/{project_id}/memory")
async def get_memory(project_id: str):
    """Get the full memory state."""
    fm = get_file_manager(project_id)
    memory = fm.load_memory()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not yet created")
    return memory.model_dump()


@router.get("/projects/{project_id}/export/markdown")
async def export_book_md(project_id: str):
    """Export the complete book as a single Markdown file."""
    from fastapi.responses import FileResponse

    fm = get_file_manager(project_id)
    path = fm.export_book_markdown()
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@router.get("/projects/{project_id}/export/docx")
async def export_book_docx(project_id: str):
    """Export the complete book as a single DOCX file."""
    from fastapi.responses import FileResponse

    fm = get_file_manager(project_id)
    path = fm.export_book_docx()
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ============================================================
# Conversational Editor — file upload, analysis, polish, rewrite
# ============================================================


class EditorChatRequest(BaseModel):
    """Request for the conversational editor."""

    message: str = Field(..., description="User message or instruction")
    mode: str = Field(default="chat", description="Mode: chat, analyze, polish, rewrite")
    context: Optional[str] = Field(default=None, description="Current text content for context")


@router.post("/projects/{project_id}/editor/upload")
async def editor_upload(project_id: str, file: UploadFile = File(...)):
    """Upload a .txt or .md file for editing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Cannot decode file — use UTF-8 or GBK")

    # Store in active project
    proj = _active_projects.get(project_id, {})
    if "uploaded_files" not in proj:
        proj["uploaded_files"] = {}
    file_id = str(uuid.uuid4())[:8]
    proj["uploaded_files"][file_id] = {
        "filename": file.filename,
        "content": text,
        "size": len(text),
    }

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(text),
        "preview": text[:500] + ("..." if len(text) > 500 else ""),
    }


@router.post("/projects/{project_id}/editor/chat")
async def editor_chat(project_id: str, req: EditorChatRequest):
    """Conversational editor — stream AI response via SSE."""
    scheduler = get_scheduler()

    mode_prompts = {
        "analyze": "你是一位专业的小说编辑。请仔细分析以下文本，指出优点、问题和改进建议。",
        "polish": "你是一位文字润色专家。请在保持原意和风格的前提下，润色以下文本，使其更加流畅优美。直接输出润色后的文本。",
        "rewrite": "你是一位创意写手。请根据用户的要求，改写以下文本。",
        "chat": "你是一位专业的写作助手，帮助用户创作和修改小说。请用中文回答。",
    }

    system = mode_prompts.get(req.mode, mode_prompts["chat"])
    user = req.message
    if req.context:
        user += f"\n\n【参考文本】\n{req.context[:8000]}"  # Limit context

    async def generate():
        try:
            async for token in scheduler.generate_stream(
                agent_type="writer",
                system_prompt=system,
                user_prompt=user,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Editor chat error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/projects/{project_id}/editor/files")
async def editor_list_files(project_id: str):
    """List uploaded files for the editor."""
    proj = _active_projects.get(project_id, {})
    files = proj.get("uploaded_files", {})
    return {fid: {"filename": f["filename"], "size": f["size"]} for fid, f in files.items()}


@router.delete("/projects/{project_id}/chapters/{chapter_number}")
async def delete_chapter(project_id: str, chapter_number: int):
    """Delete a chapter and all its artifacts."""
    import os, glob

    fm = get_file_manager(project_id)
    deleted = []
    for pattern in [
        f"output/chapters/chapter_{chapter_number:03d}*",
        f"outline/chapters/chapter_{chapter_number:03d}*",
    ]:
        for f in glob.glob(os.path.join(fm.root, pattern)):
            os.remove(f)
            deleted.append(os.path.basename(f))
    return {"status": "deleted", "chapter": chapter_number, "files": deleted}


# ============================================================
# Inline Edit — save user edits to project artifacts
# ============================================================


class EditSectionRequest(BaseModel):
    """Request to edit a section of a project artifact."""

    section: str = Field(..., description="Section ID, e.g. 'bible-world', 'char-0', 'outline-vols'")
    value: str = Field(..., description="New text content for the section")


@router.put("/projects/{project_id}/edit-section")
async def edit_section(project_id: str, req: EditSectionRequest):
    """Save an edited section back to the project's YAML files."""
    fm = get_file_manager(project_id)
    section = req.section
    value = req.value

    try:
        if section.startswith("chapter-"):
            # Save chapter content edit
            ch_num = int(section.split("-")[1])
            from ..models.chapter import PolishedChapter

            existing = fm.load_chapter_markdown(ch_num)
            title = existing.title if existing else f"第{ch_num}章"
            updated = PolishedChapter(chapter_number=ch_num, title=title, content=value)
            fm.save_chapter_markdown(updated)
            return {"status": "saved", "section": section}

        if section.startswith("bible-"):
            bible = fm.load_bible()
            if bible:
                _apply_bible_edit(bible, section, value)
                fm.save_bible(bible)
            # Editing bible invalidates downstream: characters, outline, chapters
            _invalidate_downstream(fm, ["characters", "outline", "chapters"])

        elif section.startswith("char-"):
            chars = fm.load_characters()
            if chars:
                idx = int(section.split("-")[1]) if "-" in section else 0
                _apply_character_edit(chars, idx, value)
                fm.save_characters(chars)
            # Editing characters invalidates outline and chapters
            _invalidate_downstream(fm, ["outline", "chapters"])

        elif section.startswith("outline-"):
            outline = fm.load_master_outline()
            if outline:
                _apply_outline_edit(outline, section, value)
                fm.save_master_outline(outline)
            # Editing outline invalidates chapters
            _invalidate_downstream(fm, ["chapters"])

        # Also clear in-memory state so next start will regenerate
        proj = _active_projects.get(project_id)
        if proj and proj.get("state"):
            if section.startswith("bible-"):
                proj["state"].characters = None
                proj["state"].outline = None
            elif section.startswith("char-"):
                proj["state"].outline = None

        return {"status": "saved", "section": section, "downstream_invalidated": True}
    except Exception as e:
        logger.error(f"Edit failed for {section}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _invalidate_downstream(fm, artifacts: list):
    """Delete downstream artifacts so they get regenerated from updated content."""
    import os, glob

    for art in artifacts:
        try:
            if art == "characters":
                path = os.path.join(fm.root, "novel_bible", "characters.yaml")
                if os.path.exists(path):
                    os.remove(path)
            elif art == "outline":
                for f in glob.glob(os.path.join(fm.root, "outline", "*.yaml")):
                    os.remove(f)
            elif art == "chapters":
                for f in glob.glob(os.path.join(fm.root, "outline", "chapters", "*.yaml")):
                    os.remove(f)
                for f in glob.glob(os.path.join(fm.root, "output", "chapters", "*")):
                    os.remove(f)
        except Exception as e:
            logger.warning(f"Failed to invalidate {art}: {e}")


def _apply_bible_edit(bible, section: str, value: str):
    """Apply edit to the bible model."""
    import json

    sub = section.replace("bible-", "")
    if sub == "world" and bible.world:
        try:
            data = json.loads(value)
            for k, v in data.items():
                if hasattr(bible.world, k):
                    setattr(bible.world, k, v)
        except json.JSONDecodeError:
            pass
    elif sub == "factions":
        try:
            from ..models.bible import Faction

            data = json.loads(value)
            bible.factions = [Faction.model_validate(f) for f in data]
        except Exception:
            pass
    elif sub == "themes":
        try:
            from ..models.bible import Theme

            data = json.loads(value)
            bible.themes = [Theme.model_validate(t) for t in data]
        except Exception:
            pass
    elif sub == "conflicts":
        try:
            from ..models.bible import CoreConflict

            data = json.loads(value)
            bible.core_conflicts = [CoreConflict.model_validate(c) for c in data]
        except Exception:
            pass


def _apply_character_edit(chars, idx: int, value: str):
    """Apply edit to a character profile."""
    import json

    try:
        from ..models.characters import CharacterProfile

        data = json.loads(value)
        char_list = list(chars.characters.values())
        if idx < len(char_list):
            updated = CharacterProfile.model_validate(data)
            chars.characters[updated.id] = updated
    except Exception:
        pass


def _apply_outline_edit(outline, section: str, value: str):
    """Apply edit to the outline model."""
    import json

    sub = section.replace("outline-", "")
    try:
        if sub in ("info", "main", "subs", "vols", "tps"):
            data = json.loads(value)
            if sub == "main":
                from ..models.outline import PlotArc

                outline.main_plot = [PlotArc.model_validate(a) for a in data]
            elif sub == "subs":
                from ..models.outline import PlotArc

                outline.subplots = [PlotArc.model_validate(a) for a in data]
            elif sub == "vols":
                from ..models.outline import Volume

                outline.volumes = [Volume.model_validate(v) for v in data]
    except Exception:
        pass


# ============================================================
# Feedback Persistence & Analytics (Data Flywheel Layer)
# ============================================================


def _save_feedback(fm, feedback_records: list) -> None:
    """Persist feedback records to disk as JSON."""
    import os

    feedback_path = os.path.join(fm.root, "feedback.json")
    try:
        records_data = [r.model_dump() for r in feedback_records]
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")


def _load_feedback(fm) -> list:
    """Load feedback records from disk."""
    import os

    feedback_path = os.path.join(fm.root, "feedback.json")
    if not os.path.exists(feedback_path):
        return []
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from ..models.feedback import FeedbackEntry

        return [FeedbackEntry.model_validate(r) for r in data]
    except Exception as e:
        logger.error(f"Failed to load feedback: {e}")
        return []


@router.get("/projects/{project_id}/feedback")
async def get_feedback_history(project_id: str):
    """Get all feedback records for a project.

    Returns structured feedback data for Bad Case analysis and
    monitoring dashboards (evaluation report 4-week requirement #1 and #4).
    """
    fm = get_file_manager(project_id)
    if not fm.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    records = _load_feedback(fm)

    # Build summary
    from ..models.feedback import FeedbackSummary

    total = len(records)
    positive = sum(1 for r in records if r.get("sentiment") == "thumbs_up")
    negative = total - positive

    # Reason distribution
    reason_dist: dict[str, int] = {}
    for r in records:
        for tag in r.get("reason_tags", []):
            reason_dist[tag] = reason_dist.get(tag, 0) + 1

    # Score correlation
    accepted_scores = [r.get("review_score", 0) for r in records if r.get("sentiment") == "thumbs_up"]
    rejected_scores = [r.get("review_score", 0) for r in records if r.get("sentiment") == "thumbs_down"]

    # Recent acceptance rate (last 5)
    recent = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:5]
    recent_positive = sum(1 for r in recent if r.get("sentiment") == "thumbs_up")

    summary = FeedbackSummary(
        project_id=project_id,
        total_feedback=total,
        positive_count=positive,
        negative_count=negative,
        acceptance_rate=positive / max(total, 1),
        reason_distribution=reason_dist,
        avg_review_score_accepted=sum(accepted_scores) / max(len(accepted_scores), 1),
        avg_review_score_rejected=sum(rejected_scores) / max(len(rejected_scores), 1),
        recent_acceptance_rate=recent_positive / max(len(recent), 1),
        trend_direction=_compute_trend(records),
    )

    return {
        "records": records,
        "summary": summary.model_dump(),
    }


def _compute_trend(records: list) -> str:
    """Compute acceptance rate trend from feedback records."""
    if len(records) < 5:
        return "stable"
    # Split into first half and second half
    mid = len(records) // 2
    first_half = records[:mid]
    second_half = records[mid:]

    def _rate(recs):
        if not recs:
            return 0.0
        pos = sum(1 for r in recs if r.get("sentiment") == "thumbs_up")
        return pos / len(recs)

    rate1 = _rate(first_half)
    rate2 = _rate(second_half)
    diff = rate2 - rate1

    if diff > 0.05:
        return "improving"
    elif diff < -0.05:
        return "declining"
    return "stable"


@router.get("/projects/{project_id}/feedback/bad-cases")
async def get_bad_cases(project_id: str):
    """Get aggregated bad case report for weekly review.

    Automatically aggregates low-score chapters and high-revision
    chapters for team analysis (evaluation report 4-week requirement #1).
    """
    fm = get_file_manager(project_id)
    if not fm.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    records = _load_feedback(fm)
    from ..models.feedback import REASON_TAG_LABELS, FeedbackReasonTag

    bad_cases = []
    for r in records:
        if r.get("sentiment") != "thumbs_down":
            continue
        score = r.get("review_score", 0)
        if score < 6.5 or r.get("revision_count", 0) >= 2:
            reason_labels = [REASON_TAG_LABELS.get(FeedbackReasonTag(t), t) for t in r.get("reason_tags", [])]
            bad_cases.append(
                {
                    "chapter": r.get("chapter_number"),
                    "score": score,
                    "revisions": r.get("revision_count", 0),
                    "reasons": reason_labels,
                    "notes": r.get("notes", ""),
                    "timestamp": r.get("timestamp", ""),
                }
            )

    # Sort by score ascending (worst first)
    bad_cases.sort(key=lambda c: c["score"])

    # Top reasons
    reason_counts: dict[str, int] = {}
    for bc in bad_cases:
        for reason in bc["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "project_id": project_id,
        "bad_case_count": len(bad_cases),
        "top_reasons": sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "bad_cases": bad_cases,
    }
