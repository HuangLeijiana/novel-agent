"""Orchestrator Agent — coordinates all sub-agents across the writing workflow."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft, Fact, PolishedChapter, StateChange
from ..models.characters import CharacterRegistry
from ..models.common import AgentLogEntry
from ..models.memory import MemoryState
from ..models.outline import ChapterPlan, MasterOutline
from ..models.project import ProjectConfig
from ..models.review import ReviewReport
from ..models.state import MainState
from ..models.topic import CandidateTopic, ScanReport, TitleSynopsisReport, TopicResearchState
from ..storage.bible_markdown import BibleMarkdownStore
from .architect import ArchitectAgent
from .character_manager import CharacterManagerAgent
from .chapter_inspector import ChapterInspector, InspectionResult
from .continuity_checker import ContinuityCheckerAgent
from .editor import EditorAgent
from .memory_manager import MemoryManagerAgent
from .plot_planner import PlotPlannerAgent
from .reader_simulator import ReaderSimulatorAgent
from .refiner import RefinerAgent
from .reviewer import AdversarialReviewOutput, ReviewerAgent
from .topic_scout import TopicScoutAgent
from .writer import WriterAgent

logger = logging.getLogger(__name__)


def _build_topic_context(state: "MainState") -> tuple[str, list[str]]:
    """Build downstream context from upstream topic research.

    Extracts the selected topic's mini-arc outline, title/synopsis, golden finger,
    conflicts, and pacing — everything the architect, character designer, and
    plot planner need to build a coherent world.

    Returns (context_string, genre_tags).
    """
    research = state.topic_research
    parts: list[str] = []
    extra_genres: list[str] = []

    # ---- Mini-arc outline (pacing, conflicts, pleasure points) ----
    if state.mini_arc_outline:
        for genre_name, outline in state.mini_arc_outline.items():
            parts.append(f"【小事件闭环大纲 — {genre_name}】")
            parts.append(f"闭环总目标：{outline.arc_goal}")
            parts.append(f"总字数：{outline.total_words}")
            parts.append("10章节奏骨架：")
            for ch in (outline.chapters or [])[:5]:
                parts.append(
                    f"  第{ch.chapter_number}章 | 目标：{ch.goal} | 冲突：{ch.conflict} | 爽点：{ch.pleasure_point}"
                )
            if outline.chapters and len(outline.chapters) >= 10:
                last = outline.chapters[-1]
                parts.append(
                    f"  ... → 第{last.chapter_number}章(终章) | 目标：{last.goal} | 爽点：{last.pleasure_point}"
                )
            parts.append(f"下一事件钩子：{outline.next_arc_hook}")
            parts.append("")

    # ---- Selected topic details (title, synopsis, golden finger) ----
    if research and research.title_synopsis:
        ts = research.title_synopsis[0]
        parts.append(f"【选定题材】{ts.genre_name}")
        parts.append(f"书名：《{ts.final_title}》")
        parts.append(f"简介：{ts.final_synopsis}")
        extra_genres.append(ts.genre_name)

        # Matching candidate topic for detailed design specs
        if research.candidates:
            for c in research.candidates.topics:
                if c.genre_name == ts.genre_name:
                    parts.append(f"一句话设定：{c.one_line_setting}")
                    parts.append(f"核心金手指：{c.golden_finger}")
                    parts.append(f"第一章冲突：{c.chapter1_conflict}")
                    parts.append(f"小事件闭环方向：{c.first_event_direction}")
                    parts.append(f"第一波爽点：{c.first_pleasure_wave}")
                    parts.append(f"番茄适配度：{c.tomato_fit} | 视频表现力：{c.video_potential}")
                    if c.risk:
                        parts.append(f"风险点：{c.risk}")
                    break
        parts.append("")

    # ---- Benchmark insights ----
    if research and research.benchmarks and research.benchmarks.structural_summary:
        parts.append(f"【对标书结构参考】{research.benchmarks.structural_summary}")

    # ---- Cross-platform context ----
    if research and research.cross_platform and research.cross_platform.selected_directions:
        parts.append(f"【市场方向】{', '.join(research.cross_platform.selected_directions)}")

    return "\n".join(parts).strip(), extra_genres


def _extract_genres_from_scans(
    feilu: Optional[ScanReport] = None,
    fanqie: Optional[ScanReport] = None,
) -> list[str]:
    """Extract genre/direction names from scan reports when cross-platform
    analysis produces no directions (e.g., single-platform scan or thin data).

    Returns up to 6 genre names suitable as fallback topic directions.
    """
    genres: list[str] = []
    seen: set[str] = set()

    for report in (feilu, fanqie):
        if report is None or report.scan_failed:
            continue
        for entry in report.entries or []:
            g = (entry.genre or "").strip()
            # Normalize common genre patterns
            if not g:
                continue
            # Split combined genres like "都市/系统"
            for sub in g.replace("／", "/").split("/"):
                sub = sub.strip()
                if sub and sub not in seen:
                    seen.add(sub)
                    genres.append(sub)

    # If no genre tags found, try to infer from book titles
    if not genres:
        GENRE_KEYWORDS = {
            "战神": "都市战神",
            "修仙": "玄幻修仙",
            "系统": "系统流",
            "重生": "重生逆袭",
            "穿越": "穿越架空",
            "赘婿": "都市赘婿",
            "神医": "都市神医",
            "盗墓": "悬疑探险",
            "灵异": "灵异恐怖",
            "游戏": "游戏异界",
            "末日": "末日生存",
            "武侠": "武侠江湖",
        }
        for report in (feilu, fanqie):
            if report is None or report.scan_failed:
                continue
            for entry in report.entries or []:
                title = (entry.title or "").strip()
                for kw, genre in GENRE_KEYWORDS.items():
                    if kw in title and genre not in seen:
                        seen.add(genre)
                        genres.append(genre)

    return genres[:6]


class OrchestratorAgent:
    """Central coordinator that manages the entire novel-writing workflow.

    The Orchestrator doesn't call LLMs directly — it orchestrates sub-agents
    within each workflow phase, feeding outputs from one step as inputs to the next.
    """

    def __init__(self, scheduler: ModelScheduler):
        self.scheduler = scheduler

        # Initialize all sub-agents
        self.architect = ArchitectAgent(scheduler)
        self.character_manager = CharacterManagerAgent(scheduler)
        self.plot_planner = PlotPlannerAgent(scheduler)
        self.writer = WriterAgent(scheduler)
        self.editor = EditorAgent(scheduler)
        self.continuity_checker = ContinuityCheckerAgent(scheduler)
        self.reader_simulator = ReaderSimulatorAgent(scheduler)
        self.refiner = RefinerAgent(scheduler)
        self.memory_manager = MemoryManagerAgent(scheduler)
        self.topic_scout = TopicScoutAgent(scheduler)
        self.reviewer = ReviewerAgent(scheduler)
        self.chapter_inspector = ChapterInspector()

    # ================================================================
    # Phase 1: Project Initialization
    # ================================================================

    async def initialize_project(self, state: MainState) -> MainState:
        """Initialize project directory and metadata."""
        logger.info("Phase 1: Initializing project...")
        state.current_phase = "project_init"
        return state

    # ================================================================
    # Phase 0a: Platform Scanning (Steps 1A/1B)
    # ================================================================

    async def scan_platforms(
        self,
        state: MainState,
        feilu_content: Optional[str] = None,
        fanqie_content: Optional[str] = None,
    ) -> MainState:
        """Scan 飞卢 and 番茄 real rankings for trending topics.

        Pass page content from browser automation (Playwright MCP).
        If content is None, the scan fails gracefully with "本次扫榜失败".
        """
        logger.info("Phase 0a: Scanning platforms...")
        state.current_phase = "platform_scan"

        # Initialize topic research state if needed
        if state.topic_research is None:
            state.topic_research = TopicResearchState()

        # Run both scans (can be parallel if we implement async browser)
        feilu = await self.topic_scout.scan_feilu(page_content=feilu_content)
        state.topic_research.feilu_scan = feilu
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="scan_feilu",
                summary=f"飞卢扫榜: {'失败' if feilu.scan_failed else f'{len(feilu.entries)}本书'}",
            )
        )

        fanqie = await self.topic_scout.scan_fanqie(page_content=fanqie_content)
        state.topic_research.fanqie_scan = fanqie
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="scan_fanqie",
                summary=f"番茄扫榜: {'失败' if fanqie.scan_failed else f'{len(fanqie.entries)}本书'}",
            )
        )

        return state

    # ================================================================
    # Phase 0b: Topic Selection (Steps 1C-4A)
    # ================================================================

    async def select_topic(self, state: MainState) -> MainState:
        """Full topic selection pipeline: cross-platform → benchmarks → candidates → score → titles.

        Requires scan_platforms to have run first.
        """
        logger.info("Phase 0b: Selecting topic...")
        state.current_phase = "topic_selection"

        research = state.topic_research
        if research is None:
            raise ValueError("Topic research state must be initialized (run scan_platforms first)")

        feilu = research.feilu_scan or None
        fanqie = research.fanqie_scan or None

        # Step 1C: Cross-platform comparison
        cross = await self.topic_scout.cross_platform_analysis(
            feilu=feilu,
            fanqie=fanqie,
        )
        research.cross_platform = cross
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="cross_platform",
                summary=f"交叉分析: {len(cross.entries)}题材, 保留{len(cross.selected_directions)}方向",
            )
        )

        # Build fallback directions from scan data when cross-platform analysis
        # produces no directions (e.g., single-platform scan or thin data)
        effective_directions = list(cross.selected_directions) if cross.selected_directions else []
        if not effective_directions:
            effective_directions = _extract_genres_from_scans(feilu, fanqie)
            logger.info(f"Using {len(effective_directions)} fallback directions from scan data")

        if not effective_directions:
            logger.warning("No topic directions available from any source — generating generic topics")
            # Create minimal direction from user inspiration or generic prompt
            generic = state.current_inspiration or "都市生活"
            effective_directions = [generic, "玄幻修仙", "系统流", "重生逆袭"]

        # Step 2: Benchmark book skeleton analysis
        benchmarks = await self.topic_scout.analyze_benchmarks(
            directions=effective_directions,
            feilu_scan=feilu,
            fanqie_scan=fanqie,
        )
        research.benchmarks = benchmarks
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="benchmarks",
                summary=f"对标拆解: {len(benchmarks.entries)}本书",
            )
        )

        # Step 3A: Generate 12 candidate topics
        candidates = await self.topic_scout.generate_topics(
            cross_platform=cross,
            benchmarks=benchmarks,
        )
        research.candidates = candidates
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="generate_topics",
                summary=f"生成{len(candidates.topics)}个候选题材",
            )
        )

        if not candidates.topics:
            logger.warning("No candidate topics generated — building fallback topics")
            fallback_topics = []
            for direction in effective_directions[:8]:
                fallback_topics.append(
                    CandidateTopic(
                        genre_name=direction,
                        one_line_setting=f"一部{direction}题材的小说",
                        golden_finger="待定",
                        chapter1_conflict="待定",
                        first_event_direction="待定",
                        first_pleasure_wave="待定",
                    )
                )
            if fallback_topics:
                from ..models.topic import CandidateTopicsOutput

                candidates = CandidateTopicsOutput(topics=fallback_topics)
                research.candidates = candidates
                logger.info(f"Created {len(fallback_topics)} fallback topics from effective directions")
            else:
                # Ultimate fallback: generic topics
                generic = [
                    CandidateTopic(genre_name=g, one_line_setting=f"一部{g}题材的小说")
                    for g in ["都市生活", "玄幻修仙", "系统流", "重生逆袭"]
                ]
                candidates = CandidateTopicsOutput(topics=generic)
                research.candidates = candidates
                logger.info("Created 4 generic fallback topics")

        # Step 3B: Score and narrow to top 4
        scores = await self.topic_scout.score_topics(candidates)
        research.scores = scores
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="score_topics",
                summary=f"评分完成: 前4名={scores.top_4}",
            )
        )

        # Step 4A: Generate titles for top 4 (human picks final 2 later)
        top_genres = scores.top_4[:4] if len(scores.top_4) >= 4 else list(scores.top_4)

        # Fallback: when scoring produces no top_4 (e.g. LLM without JSON mode),
        # use candidate genre names directly so ALL candidates get titles
        if not top_genres and candidates.topics:
            top_genres = [t.genre_name for t in candidates.topics[:4] if t.genre_name]
            logger.info(f"Using candidate genres as fallback for title generation: {top_genres}")

        for genre_name in top_genres:
            # Find matching candidate topic
            matching = [t for t in candidates.topics if t.genre_name == genre_name]
            if not matching:
                continue
            topic = matching[0]
            title_report = await self.topic_scout.generate_titles(
                genre_name=genre_name,
                topic=topic,
            )
            research.title_synopsis.append(title_report)
            state.agent_log.append(
                AgentLogEntry(
                    agent="topic_scout",
                    phase="generate_titles",
                    summary=f"书名: {genre_name} → {title_report.final_title}",
                )
            )

        return state

    # ================================================================
    # Phase 0c: Mini-Arc Outline (Step 5)
    # ================================================================

    async def plan_mini_arc(self, state: MainState) -> MainState:
        """Generate 10-chapter mini-arc outline for the selected topic.

        Requires select_topic to have run first.
        Uses the FIRST topic from title_synopsis (the one the user selected).
        """
        logger.info("Phase 0c: Planning mini-arc outline...")
        state.current_phase = "mini_arc_outline"

        research = state.topic_research
        if research is None:
            raise ValueError("Topic research state must be initialized (run select_topic first)")

        if not research.title_synopsis:
            logger.warning("title_synopsis is empty — generating fallback from available data")
            fallback_reports = []

            # Source 1: candidate topics (preferred — real topic data)
            if research.candidates and research.candidates.topics:
                t = research.candidates.topics[0]
                fallback_reports.append(
                    TitleSynopsisReport(
                        genre_name=t.genre_name,
                        final_title=t.genre_name,
                        final_synopsis=t.one_line_setting or f"{t.genre_name}题材小说",
                    )
                )

            # Source 2: cross-platform directions
            if not fallback_reports and research.cross_platform and research.cross_platform.selected_directions:
                d = research.cross_platform.selected_directions[0]
                fallback_reports.append(
                    TitleSynopsisReport(
                        genre_name=d,
                        final_title=d,
                        final_synopsis=f"一部{d}题材的精彩小说",
                    )
                )

            # Source 3: user inspiration wrapped as a real topic
            if not fallback_reports and state.current_inspiration:
                fallback_reports.append(
                    TitleSynopsisReport(
                        genre_name="自定题材",
                        final_title=state.current_inspiration[:50],
                        final_synopsis=state.current_inspiration,
                    )
                )

            # Source 4: generic placeholder
            if not fallback_reports:
                logger.warning("No fallback data available — generating generic placeholder")
                fallback_reports = [
                    TitleSynopsisReport(
                        genre_name="都市生活",
                        final_title="未命名小说",
                        final_synopsis="一部精彩的网络小说",
                    )
                ]

            research.title_synopsis = fallback_reports

        outlines = {}
        # Only generate outline for the FIRST (user-selected) topic
        title_report = research.title_synopsis[0]
        genre = title_report.genre_name
        # Find matching candidate
        matching = [t for t in (research.candidates.topics if research.candidates else []) if t.genre_name == genre]
        if not matching:
            matching = [
                CandidateTopic(
                    genre_name=genre,
                    one_line_setting=title_report.final_synopsis,
                    golden_finger="",
                    chapter1_conflict="",
                    first_event_direction="",
                    first_pleasure_wave="",
                )
            ]

        outline = await self.topic_scout.plan_mini_arc(
            genre_name=genre,
            topic=matching[0],
            title_report=title_report,
        )
        outlines[genre] = outline
        chapter_count = len(outline.chapters) if outline.chapters else 0
        state.agent_log.append(
            AgentLogEntry(
                agent="topic_scout",
                phase="plan_mini_arc",
                summary=f"小事件大纲: {genre} → {chapter_count}章, 目标{outline.total_words}字",
            )
        )
        if chapter_count == 0:
            logger.warning(f"Mini-arc outline for '{genre}' has 0 chapters — all generation layers failed")

        state.mini_arc_outline = outlines

        # Post-check: warn if outline is empty
        total_chapters = sum(len(o.chapters) for o in outlines.values() if o.chapters)
        if total_chapters == 0:
            logger.warning("Mini-arc outline has 0 chapters — all generation layers failed")
            state.agent_log.append(
                AgentLogEntry(
                    agent="orchestrator",
                    phase="mini_arc_outline",
                    summary="⚠️ 大纲章节为空 — 所有生成层均失败，请检查LLM配置",
                )
            )

        return state

    # ================================================================
    # Phase 2: Bible Construction
    # ================================================================

    async def build_bible(self, state: MainState) -> MainState:
        """Build the complete Novel Bible (world + characters).

        Coordinates Architect and CharacterManager agents.
        Kept for backward compat — new phased flow uses build_bible_world + create_characters.
        """
        state = await self.build_bible_world(state)
        state = await self.create_characters(state)
        return state

    async def build_bible_world(self, state: MainState) -> MainState:
        """Phase 2a: Build world, factions, style, themes, conflicts, pleasure points.

        Produces state.bible but NOT state.characters.
        """
        logger.info("Phase 2a: Building world bible...")
        state.current_phase = "bible_construction"

        config = state.project_config
        if config is None:
            raise ValueError("Project config is required")

        # Inject human inspiration if provided
        if state.current_inspiration:
            config = config.model_copy()
            logger.info(f"Using human inspiration: {state.current_inspiration[:80]}...")

        # ── Inject upstream topic research into config ──
        # Build a context string from mini-arc + title/synopsis + candidate details.
        # The architect reads config.inspiration & config.genre to build the world,
        # so enriching them here makes the entire downstream pipeline coherent.
        topic_context, topic_genres = _build_topic_context(state)
        if topic_context:
            if config is state.project_config:
                config = config.model_copy()
            enriched = (
                f"{config.inspiration}\n\n"
                f"{'=' * 40}\n"
                f"以下为上游选题研究成果，请严格以此为基准构建世界观：\n"
                f"{'=' * 40}\n"
                f"{topic_context}"
            )
            config.inspiration = enriched[:2000]
            # Merge topic genres (preserve order, deduplicate)
            merged = list(dict.fromkeys(list(config.genre) + topic_genres))
            config.genre = merged[:5]
            logger.info(f"Topic context injected: genres={config.genre}")

        # Step 1: World building
        world = await self.architect.build_world(config)
        state.agent_log.append(
            AgentLogEntry(
                agent="architect",
                phase="build_world",
                summary=f"World '{world.name}' created",
            )
        )

        # Step 2: Faction design
        factions = await self.architect.design_factions(config, world)
        state.agent_log.append(
            AgentLogEntry(
                agent="architect",
                phase="design_factions",
                summary=f"{len(factions)} factions designed",
            )
        )

        # Step 3: Style contract
        style = await self.architect.create_style_contract(config)
        state.agent_log.append(
            AgentLogEntry(
                agent="architect",
                phase="create_style_contract",
                summary=f"Style contract: {style.tone}",
            )
        )

        # Step 4: Themes and conflicts
        themes = await self.architect.generate_themes(config, world)
        conflicts = await self.architect.define_conflicts(config, world, factions)
        state.agent_log.append(
            AgentLogEntry(
                agent="architect",
                phase="generate_themes",
                summary=f"{len(themes)} themes, {len(conflicts)} conflicts",
            )
        )

        # Step 5: Pleasure points
        pleasure_model, constraints = await self.architect.design_pleasure_points(
            config,
            world,
            themes,
        )

        # Assemble bible
        from ..models.bible import NarrativeRules

        bible = NovelBible(
            world=world,
            factions=factions,
            rules=NarrativeRules(),
            style_contract=style,
            themes=themes,
            core_conflicts=conflicts,
            pleasure_point_model=pleasure_model,
            narrative_constraints=constraints,
        )
        state.bible = bible
        # Persist enriched config so downstream phases (characters, outline)
        # get the topic research context too
        state.project_config = config
        return state

    def save_bible_as_markdown(self, state: MainState, project_dir: Path) -> None:
        """Export the bible as modular Markdown files (thin-pointer pattern).

        Creates bible/ directory with independent files for each concern:
        01-world-building.md, 02-factions.md, 03-writing-style.md, etc.

        Agents can then load only the files they need instead of the entire bible.
        """
        if state.bible is None:
            logger.warning("No bible to save")
            return
        store = BibleMarkdownStore(project_dir)
        store.save_bible(state.bible, state.characters)
        logger.info(f"Bible exported as modular Markdown to {store.bible_dir}")

    async def create_characters(self, state: MainState) -> MainState:
        """Phase 2b: Create character profiles based on the world bible."""
        logger.info("Phase 2b: Creating characters...")
        state.current_phase = "character_creation"

        characters = await self.character_manager.create_all_characters(
            state.project_config,
            state.bible,
        )
        state.characters = characters
        state.agent_log.append(
            AgentLogEntry(
                agent="character_manager",
                phase="create_all_characters",
                summary=f"{len(characters.characters)} characters created",
            )
        )
        logger.info(
            f"Bible construction complete: {len(state.bible.factions)} factions, "
            f"{len(characters.characters)} characters"
        )
        return state

    # ================================================================
    # Phase 3: Master Outline
    # ================================================================

    async def create_outline(self, state: MainState) -> MainState:
        """Create the master outline with volumes and turning points."""
        logger.info("Phase 3: Creating master outline...")
        state.current_phase = "master_outline"

        outline = await self.plot_planner.create_master_outline(
            config=state.project_config,
            bible=state.bible,
            characters=state.characters,
        )
        state.outline = outline
        state.total_chapters = outline.chapter_count

        state.agent_log.append(
            AgentLogEntry(
                agent="plot_planner",
                phase="create_master_outline",
                summary=f"Outline: {outline.title}, {outline.chapter_count} chapters",
            )
        )

        # Update project title if not set
        if state.project_meta and not state.project_meta.title and outline.title:
            state.project_meta.title = outline.title

        return state

    # ================================================================
    # Phase 4: Chapter Planning
    # ================================================================

    async def plan_chapter(self, state: MainState) -> MainState:
        """Plan the next chapter."""
        chapter_num = state.current_chapter_number + 1
        logger.info(f"Phase 4: Planning chapter {chapter_num}...")
        state.current_phase = "chapter_planning"

        plan = await self.plot_planner.plan_chapter(
            chapter_number=chapter_num,
            config=state.project_config,
            bible=state.bible,
            characters=state.characters,
            outline=state.outline,
            memory=state.memory,
        )
        state.chapter_plan = plan
        state.current_chapter_number = chapter_num

        state.agent_log.append(
            AgentLogEntry(
                agent="plot_planner",
                phase="plan_chapter",
                summary=f"Chapter {chapter_num}: {plan.title}, {len(plan.scenes)} scenes",
            )
        )

        return state

    # ================================================================
    # Phase 5: Chapter Writing
    # ================================================================

    async def write_chapter(self, state: MainState) -> MainState:
        """Write the chapter draft with structural inspection.

        After generation, runs a deterministic structural check (word count,
        dialogue ratio, paragraph density, hook presence, AI markers) before
        the expensive LLM review phase.
        """
        logger.info(f"Phase 5: Writing chapter {state.current_chapter_number}...")
        state.current_phase = "chapter_writing"

        revision_feedback = state.human_feedback if state.human_decision == "revise" else None

        draft = await self.writer.generate_chapter(
            chapter_plan=state.chapter_plan,
            config=state.project_config,
            bible=state.bible,
            characters=state.characters,
            outline=state.outline,
            memory=state.memory,
            revision_feedback=revision_feedback,
        )

        # Extract facts and state changes
        draft = await self.writer.extract_facts_and_changes(
            draft=draft,
            bible=state.bible,
            characters=state.characters,
        )

        # ── Structural inspection (deterministic, no LLM cost) ──
        inspection = self.chapter_inspector.inspect(
            draft=draft,
            chapter_plan=state.chapter_plan,
            bible=state.bible,
        )
        # Store inspection result for downstream use
        state.chapter_inspection = inspection

        # ── Content safety check (deterministic, no LLM cost) ──
        from ..utils.content_safety import ContentSafetyChecker

        safety_checker = ContentSafetyChecker()
        safety_result = safety_checker.check(
            draft.content,
            chapter_number=draft.chapter_number,
        )
        if not safety_result.passed:
            logger.error(
                f"Chapter {draft.chapter_number}: Content safety BLOCKED — {safety_result.block_count} blocking flag(s)"
            )
            state.errors.append(f"Chapter {draft.chapter_number} blocked by content safety: {safety_result.summary}")
        elif safety_result.warning_count > 0:
            logger.warning(f"Chapter {draft.chapter_number}: Content safety warnings — {safety_result.summary}")

        state.chapter_draft = draft
        state.agent_log.append(
            AgentLogEntry(
                agent="writer",
                phase="generate_chapter",
                summary=f"Chapter {draft.chapter_number}: {draft.word_count} chars, "
                f"{len(draft.new_facts)} facts | "
                f"Inspection: {'PASS' if inspection.passed else 'FAIL'} "
                f"({len(inspection.issues)} issues, {len(inspection.warnings)} warnings) | "
                f"Safety: {'PASS' if safety_result.passed else 'BLOCKED'} "
                f"({safety_result.warning_count} warnings, {safety_result.block_count} blocks)",
            )
        )

        if not inspection.passed:
            logger.warning(f"Chapter {draft.chapter_number} structural issues: {inspection.format_report(inspection)}")

        return state

    # ================================================================
    # Phase 6: Quality Review
    # ================================================================

    async def review_chapter(self, state: MainState) -> MainState:
        """Run comprehensive quality review with adversarial critique.

        Two-layer review:
        1. Standard review: Editor + ContinuityChecker + ReaderSimulator (parallel)
        2. Adversarial review: Independent ReviewerAgent with fresh perspective
        """
        logger.info(f"Phase 6: Reviewing chapter {state.current_chapter_number}...")
        state.current_phase = "quality_review"
        state.review_iteration += 1

        # ── Layer 1: Standard review (parallel checks) ──
        if state.review_iteration == 1:
            results = await asyncio.gather(
                self.editor.review_chapter(
                    draft=state.chapter_draft,
                    chapter_plan=state.chapter_plan,
                    bible=state.bible,
                    characters=state.characters,
                    memory=state.memory,
                ),
                self.continuity_checker.check_timeline(
                    draft=state.chapter_draft,
                    memory=state.memory,
                ),
                self.continuity_checker.check_character_consistency(
                    draft=state.chapter_draft,
                    characters=state.characters,
                    memory=state.memory,
                ),
                self.continuity_checker.check_causality(
                    draft=state.chapter_draft,
                    memory=state.memory,
                ),
                self.reader_simulator.simulate_reading(
                    draft=state.chapter_draft,
                    bible=state.bible,
                    target_reader=state.project_config.target_readers,
                ),
                # ── Layer 2: Adversarial review (independent fresh perspective) ──
                self.reviewer.adversarial_review(
                    draft=state.chapter_draft,
                    chapter_plan=state.chapter_plan,
                    bible=state.bible,
                    characters=state.characters,
                    outline=state.outline,
                    memory=state.memory,
                    config=state.project_config,
                ),
                return_exceptions=True,
            )

            report = results[0] if not isinstance(results[0], Exception) else None
            timeline_issues = results[1] if not isinstance(results[1], Exception) else []
            char_issues = results[2] if not isinstance(results[2], Exception) else []
            causality_issues = results[3] if not isinstance(results[3], Exception) else []
            reader_feedback = results[4] if not isinstance(results[4], Exception) else {}
            adversarial = results[5] if not isinstance(results[5], Exception) else None

            if report:
                # Add continuity checker issues
                report.issues.extend(timeline_issues)
                report.issues.extend(char_issues)
                report.issues.extend(causality_issues)

                # Add reader feedback as dimension
                if reader_feedback:
                    report.dimension_scores["reader_engagement"] = reader_feedback.get("engagement_score", 7.0)
                    report.dimension_scores["continuation_likelihood"] = reader_feedback.get(
                        "continuation_likelihood", 7.0
                    )

                # ── Merge adversarial review findings ──
                if adversarial and isinstance(adversarial, AdversarialReviewOutput):
                    adv_report = self.reviewer.to_review_report(adversarial)
                    # Merge issues (adversarial findings are often more critical)
                    report.issues.extend(adv_report.issues)
                    report.strengths.extend(adv_report.strengths)
                    report.suggestions.extend(adv_report.suggestions)
                    # Add adversarial-specific dimensions
                    report.dimension_scores["adversarial_plot"] = 10.0 - len(adversarial.plot_holes) * 2
                    report.dimension_scores["adversarial_character"] = (
                        10.0 - len(adversarial.character_inconsistencies) * 2
                    )
                    report.dimension_scores["adversarial_reader_risk"] = (
                        10.0 - len(adversarial.reader_drop_points) * 1.5
                    )
                    # Clamp scores
                    for key in list(report.dimension_scores.keys()):
                        report.dimension_scores[key] = max(0.0, min(10.0, report.dimension_scores[key]))

                    logger.info(
                        f"Adversarial review: score={adversarial.overall_score}, "
                        f"plot_holes={len(adversarial.plot_holes)}, "
                        f"char_issues={len(adversarial.character_inconsistencies)}, "
                        f"drop_points={len(adversarial.reader_drop_points)}, "
                        f"revision_required={adversarial.revision_required}"
                    )

                    # If adversarial reviewer demands revision, override pass/fail
                    if adversarial.revision_required:
                        report.passed = False
                        report.issues.append(
                            Issue(
                                severity="critical",
                                category="adversarial_review",
                                description="对抗性审稿要求强制修订",
                                suggestion="请根据对抗性审稿意见进行修订",
                            )
                        )

                # Recalculate pass/fail
                if not adversarial or not (
                    isinstance(adversarial, AdversarialReviewOutput) and adversarial.revision_required
                ):
                    avg = sum(report.dimension_scores.values()) / max(len(report.dimension_scores), 1)
                    report.passed = avg >= 6.5 and not report.has_critical
                    report.overall_score = avg

                state.review_report = report
        else:
            # Subsequent iterations: lighter review focused on remaining issues
            report = await self.editor.review_chapter(
                draft=state.chapter_draft,
                chapter_plan=state.chapter_plan,
                bible=state.bible,
                characters=state.characters,
                memory=state.memory,
            )
            state.review_report = report

        if state.review_report:
            state.agent_log.append(
                AgentLogEntry(
                    agent="editor",
                    phase="review_chapter",
                    summary=f"Score: {state.review_report.overall_score:.1f}, "
                    f"passed={state.review_report.passed}, "
                    f"issues={len(state.review_report.issues)}",
                )
            )
        else:
            state.agent_log.append(
                AgentLogEntry(
                    agent="editor",
                    phase="review_chapter",
                    summary="Review failed to produce a report",
                )
            )

        return state

    # ================================================================
    # Phase 7: Polish & Revision
    # ================================================================

    async def polish_chapter(self, state: MainState) -> MainState:
        """Polish and refine the chapter draft."""
        logger.info(f"Phase 7: Polishing chapter {state.current_chapter_number}...")
        state.current_phase = "polish_revision"

        polished = await self.refiner.polish_chapter(
            draft=state.chapter_draft,
            review=state.review_report if state.review_report else ReviewReport(),
            bible=state.bible,
            human_feedback=state.human_feedback,
        )

        state.polished_chapter = polished
        state.agent_log.append(
            AgentLogEntry(
                agent="refiner",
                phase="polish_chapter",
                summary=f"Polished: {polished.word_count} chars, notes: {polished.revision_notes[:100]}",
            )
        )

        return state

    # ================================================================
    # Phase 8: Memory Update
    # ================================================================

    async def update_memory(self, state: MainState) -> MainState:
        """Update all memory artifacts."""
        logger.info(f"Phase 8: Updating memory for chapter {state.current_chapter_number}...")
        state.current_phase = "memory_update"

        memory = await self.memory_manager.update_memory(
            polished=state.polished_chapter,
            draft=state.chapter_draft,
            bible=state.bible,
            characters=state.characters,
            existing_memory=state.memory,
        )

        state.memory = memory
        state.agent_log.append(
            AgentLogEntry(
                agent="memory_manager",
                phase="update_memory",
                summary=f"Memory updated: {len(memory.long_term.chapter_summaries)} chapter summaries, "
                f"{len(memory.timeline)} timeline events",
            )
        )

        return state
