"""End-to-end pipeline test: Phase 2→3→4→5 against ModelScope/Qwen3.

Runs the full novel generation pipeline from bible construction through
chapter writing, validating Pydantic output at each stage.

Usage:
    cd E:/projects/novel-agent
    uv run python -m pytest tests/test_e2e_modelscope.py -v -s
"""

import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.config.settings import get_settings
from src.llm.scheduler import ModelScheduler
from src.models.bible import NarrativeRules, NovelBible
from src.models.characters import CharacterRegistry
from src.models.chapter import ChapterDraft
from src.models.outline import ChapterPlan, MasterOutline
from src.models.project import ProjectConfig
from src.agents.architect import ArchitectAgent
from src.agents.character_manager import CharacterManagerAgent
from src.agents.plot_planner import PlotPlannerAgent
from src.agents.writer import WriterAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def scheduler():
    return ModelScheduler()


@pytest.fixture(scope="module")
def config():
    return ProjectConfig(
        title="逆命九霄",
        inspiration=(
            "一个天生废脉的少年秦默，被家族流放至偏远矿区。"
            "在一次矿难中，他意外发现了一块远古传承晶石，觉醒沉睡万年的血脉。"
            "从此，他踏上了一条逆天改命的修真之路。"
        ),
        genre=["玄幻", "修仙", "爽文"],
        target_readers="喜欢热血升级流和逆袭爽文的男性读者，18-35岁",
        tone="热血",
        target_word_count=90000,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Bible Construction
# ═══════════════════════════════════════════════════════════════════════

class TestPhase2Bible:
    """Phase 2: Build the complete Novel Bible."""

    @pytest.mark.asyncio
    async def test_build_world(self, scheduler, config):
        print("\n-- Phase 2a: World Building --")
        architect = ArchitectAgent(scheduler)
        world = await architect.build_world(config)
        assert world.name, "World must have a name"
        print(f"  [OK] World: {world.name} ({world.world_type})")
        print(f"    Geography: {world.geography[:80]}...")
        print(f"    Magic: {world.magic_system[:80]}...")

    @pytest.mark.asyncio
    async def test_build_bible(self, scheduler, config):
        """Full bible assembly — all Phase 2 sub-steps."""
        print("\n-- Phase 2: Full Bible Assembly --")
        architect = ArchitectAgent(scheduler)
        char_mgr = CharacterManagerAgent(scheduler)
        t0 = time.time()

        # Step 1: World
        world = await architect.build_world(config)
        print(f"  [OK] World: {world.name}")

        # Step 2: Factions
        factions = await architect.design_factions(config, world)
        print(f"  [OK] Factions: {len(factions)} [{', '.join(f.name for f in factions)}]")

        # Step 3: Style contract
        style = await architect.create_style_contract(config)
        print(f"  [OK] Style: {style.tone} / {style.sentence_style} / dialogue={style.dialogue_ratio}")

        # Step 4: Themes + Conflicts
        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        print(f"  [OK] Themes: {len(themes)} [{', '.join(t.name for t in themes)}]")
        print(f"  [OK] Conflicts: {len(conflicts)}")

        # Step 5: Pleasure points
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
        print(f"  [OK] Pleasure model: {pleasure_model[:80]}...")

        # Assemble bible
        bible = NovelBible(
            world=world, factions=factions, rules=NarrativeRules(),
            style_contract=style, themes=themes, core_conflicts=conflicts,
            pleasure_point_model=pleasure_model, narrative_constraints=constraints,
        )

        # Step 6: Characters
        registry = await char_mgr.create_all_characters(config, bible)
        print(f"  [OK] Characters: {len(registry.characters)}")
        for cid, c in registry.characters.items():
            print(f"    [{cid}] {c.name} ({c.role}) — {c.personality[:40]}...")

        elapsed = time.time() - t0
        print(f"\n  [TIME] Bible assembled in {elapsed:.0f}s: "
              f"{len(factions)} factions, {len(registry.characters)} characters, "
              f"{len(themes)} themes, {len(conflicts)} conflicts")

        assert len(factions) >= 2
        assert len(registry.characters) >= 2
        assert len(themes) >= 1
        assert len(conflicts) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Master Outline
# ═══════════════════════════════════════════════════════════════════════

class TestPhase3Outline:
    """Phase 3: Generate the master outline."""

    @pytest.fixture(scope="class")
    async def bible_and_characters(self, scheduler, config):
        """Build a complete bible for outline generation (shared across Phase 3 tests)."""
        architect = ArchitectAgent(scheduler)
        char_mgr = CharacterManagerAgent(scheduler)

        world = await architect.build_world(config)
        factions = await architect.design_factions(config, world)
        style = await architect.create_style_contract(config)
        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)

        bible = NovelBible(
            world=world, factions=factions, rules=NarrativeRules(),
            style_contract=style, themes=themes, core_conflicts=conflicts,
            pleasure_point_model=pleasure_model, narrative_constraints=constraints,
        )
        registry = await char_mgr.create_all_characters(config, bible)
        return bible, registry

    @pytest.mark.asyncio
    async def test_create_master_outline(self, scheduler, config, bible_and_characters):
        print("\n-- Phase 3: Master Outline --")
        bible, characters = bible_and_characters
        plot_planner = PlotPlannerAgent(scheduler)
        t0 = time.time()

        outline = await plot_planner.create_master_outline(
            config=config, bible=bible, characters=characters,
        )

        elapsed = time.time() - t0
        assert isinstance(outline, MasterOutline)
        assert outline.title, "Outline must have a title"
        assert outline.chapter_count >= 5, "Must plan at least 5 chapters"

        print(f"  [OK] Title: {outline.title}")
        print(f"  [OK] Logline: {outline.logline[:100]}...")
        print(f"  [OK] Volumes: {len(outline.volumes)}")
        for v in outline.volumes:
            print(f"    Vol {v.number}: {v.title} (ch {v.start_chapter}-{v.end_chapter})")
        print(f"  [OK] Main plots: {len(outline.main_plot)}")
        for p in outline.main_plot:
            print(f"    - {p.name}: {p.description[:60]}...")
        print(f"  [OK] Subplots: {len(outline.subplots)}")
        print(f"  [OK] Turning points: {len(outline.major_turning_points)}")
        for tp in outline.major_turning_points:
            print(f"    - [{tp.turning_type}] ch{tp.chapter}: {tp.description[:60]}...")
        print(f"  [OK] Total chapters: {outline.chapter_count}")
        print(f"  [TIME] Outline generated in {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Chapter Planning
# ═══════════════════════════════════════════════════════════════════════

class TestPhase4ChapterPlanning:
    """Phase 4: Plan individual chapters."""

    @pytest.fixture(scope="class")
    async def full_context(self, scheduler, config):
        """Build bible + outline for chapter planning (shared across Phase 4+5 tests)."""
        architect = ArchitectAgent(scheduler)
        char_mgr = CharacterManagerAgent(scheduler)
        plot_planner = PlotPlannerAgent(scheduler)

        # Bible
        world = await architect.build_world(config)
        factions = await architect.design_factions(config, world)
        style = await architect.create_style_contract(config)
        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
        bible = NovelBible(
            world=world, factions=factions, rules=NarrativeRules(),
            style_contract=style, themes=themes, core_conflicts=conflicts,
            pleasure_point_model=pleasure_model, narrative_constraints=constraints,
        )
        characters = await char_mgr.create_all_characters(config, bible)
        outline = await plot_planner.create_master_outline(config, bible, characters)
        return config, bible, characters, outline, plot_planner

    @pytest.mark.asyncio
    async def test_plan_chapter_1(self, scheduler, full_context):
        print("\n-- Phase 4: Chapter Planning (Chapter 1) --")
        config, bible, characters, outline, plot_planner = full_context
        t0 = time.time()

        plan = await plot_planner.plan_chapter(
            chapter_number=1,
            config=config, bible=bible, characters=characters, outline=outline,
        )

        elapsed = time.time() - t0
        assert isinstance(plan, ChapterPlan)
        assert plan.title, "Chapter must have a title"
        assert len(plan.scenes) >= 1, "Must have at least 1 scene"
        assert plan.goal, "Must have a goal"
        assert plan.conflict, "Must have a conflict"

        print(f"  [OK] Title: {plan.title}")
        print(f"  [OK] Goal: {plan.goal[:100]}...")
        print(f"  [OK] Scenes: {len(plan.scenes)}")
        for s in plan.scenes:
            print(f"    Scene {s.number}: [{s.setting}] {s.goal[:40]}...")
        print(f"  [OK] POV: {plan.pov_character}")
        print(f"  [OK] Conflict: {plan.conflict[:80]}...")
        print(f"  [OK] Characters: {plan.characters_involved}")
        print(f"  [OK] Emotional curve: {len(plan.emotional_curve)} beats")
        for beat in plan.emotional_curve:
            print(f"    pos={beat.position:.2f} {beat.emotion} (intensity={beat.intensity:.2f})")
        print(f"  [OK] Ending hook: {plan.ending_hook[:80]}...")
        print(f"  [OK] Word count target: {plan.word_count_target}")
        print(f"  [TIME] Chapter plan generated in {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Chapter Writing
# ═══════════════════════════════════════════════════════════════════════

class TestPhase5Writing:
    """Phase 5: Write chapter drafts."""

    @pytest.fixture(scope="class")
    async def writing_context(self, scheduler, config):
        """Build full context through Phase 4, then return all artifacts + plan."""
        architect = ArchitectAgent(scheduler)
        char_mgr = CharacterManagerAgent(scheduler)
        plot_planner = PlotPlannerAgent(scheduler)
        writer = WriterAgent(scheduler)

        # Bible
        world = await architect.build_world(config)
        factions = await architect.design_factions(config, world)
        style = await architect.create_style_contract(config)
        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
        bible = NovelBible(
            world=world, factions=factions, rules=NarrativeRules(),
            style_contract=style, themes=themes, core_conflicts=conflicts,
            pleasure_point_model=pleasure_model, narrative_constraints=constraints,
        )
        characters = await char_mgr.create_all_characters(config, bible)
        outline = await plot_planner.create_master_outline(config, bible, characters)
        plan = await plot_planner.plan_chapter(1, config, bible, characters, outline)
        return config, bible, characters, outline, plan, writer

    @pytest.mark.asyncio
    async def test_write_chapter_1(self, scheduler, writing_context):
        print("\n-- Phase 5: Chapter Writing (Chapter 1) --")
        config, bible, characters, outline, plan, writer = writing_context
        t0 = time.time()

        draft = await writer.generate_chapter(
            chapter_plan=plan, config=config, bible=bible,
            characters=characters, outline=outline,
        )

        elapsed = time.time() - t0
        assert isinstance(draft, ChapterDraft)
        assert draft.content, "Chapter must have content"
        assert draft.word_count > 0, "Word count must be positive"

        print(f"  [OK] Title: {draft.title}")
        print(f"  [OK] Word count: {draft.word_count}")
        print(f"  [OK] Content: {len(draft.content)} chars")
        print(f"    Preview: {draft.content[:200]}...")
        if draft.author_notes:
            print(f"  [OK] Author notes: {draft.author_notes[:100]}...")
        print(f"  [TIME] Chapter written in {elapsed:.0f}s")

    @pytest.mark.asyncio
    async def test_extract_facts_chapter_1(self, scheduler, writing_context):
        """Write chapter + extract facts: the full Phase 5 pipeline."""
        print("\n-- Phase 5b: Fact Extraction --")
        config, bible, characters, outline, plan, writer = writing_context
        t0 = time.time()

        # Write
        draft = await writer.generate_chapter(
            chapter_plan=plan, config=config, bible=bible,
            characters=characters, outline=outline,
        )
        print(f"  [OK] Written: {draft.word_count} chars")

        # Extract
        draft = await writer.extract_facts_and_changes(draft, bible, characters)

        elapsed = time.time() - t0
        print(f"  [OK] Facts extracted: {len(draft.new_facts)}")
        for f in draft.new_facts:
            print(f"    [{f.category}] {f.description[:60]}... (c={f.certainty})")
        print(f"  [OK] State changes: {len(draft.character_state_changes)}")
        for sc in draft.character_state_changes:
            print(f"    {sc.character_id}: {sc.attribute} | {str(sc.old_value)[:20]} → {str(sc.new_value)[:20]}")
        print(f"  [TIME] Write+Extract completed in {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline: All phases in sequence
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_full_pipeline(scheduler, config):
    """Run the complete pipeline Phases 2→3→4→5 in one test.

    This is the definitive end-to-end test. All phases call ModelScope API.
    """
    print("\n" + "=" * 70)
    print("=  END-TO-END PIPELINE: Phase 2 -> 3 -> 4 -> 5")
    print("=" * 70)

    architect = ArchitectAgent(scheduler)
    char_mgr = CharacterManagerAgent(scheduler)
    plot_planner = PlotPlannerAgent(scheduler)
    writer = WriterAgent(scheduler)

    errors = []
    pipeline_start = time.time()

    # === Phase 2: Bible Construction ===
    print("\n+-- Phase 2: Bible Construction " + "-" * 44)
    t0 = time.time()
    try:
        world = await architect.build_world(config)
        print(f"| World: {world.name} ({world.world_type})")

        factions = await architect.design_factions(config, world)
        print(f"| Factions: {len(factions)}")

        style = await architect.create_style_contract(config)
        print(f"| Style: {style.tone}")

        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        print(f"| Themes: {len(themes)}, Conflicts: {len(conflicts)}")

        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
        print(f"| Pleasure model: [OK]")

        bible = NovelBible(
            world=world, factions=factions, rules=NarrativeRules(),
            style_contract=style, themes=themes, core_conflicts=conflicts,
            pleasure_point_model=pleasure_model, narrative_constraints=constraints,
        )

        characters = await char_mgr.create_all_characters(config, bible)
        print(f"| Characters: {len(characters.characters)}")
        for cid, c in characters.characters.items():
            print(f"     [{cid}] {c.name} ({c.role})")
        print(f"   [TIME] Phase 2: {time.time() - t0:.0f}s")
    except Exception as e:
        errors.append(f"Phase 2: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Phase 2 failed: {e}")
        return

    # === Phase 3: Master Outline ===
    print("\n+-- Phase 3: Master Outline " + "-" * 45)
    t0 = time.time()
    try:
        outline = await plot_planner.create_master_outline(config, bible, characters)
        assert isinstance(outline, MasterOutline)
        print(f"| Title: {outline.title}")
        print(f"| Logline: {outline.logline[:100]}...")
        print(f"| Volumes: {len(outline.volumes)}")
        print(f"| Main plots: {len(outline.main_plot)}")
        print(f"| Subplots: {len(outline.subplots)}")
        print(f"| Turning points: {len(outline.major_turning_points)}")
        print(f"| Total chapters: {outline.chapter_count}")
        print(f"   [TIME] Phase 3: {time.time() - t0:.0f}s")
    except Exception as e:
        errors.append(f"Phase 3: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Phase 3 failed: {e}")
        return

    # === Phase 4: Chapter Planning ===
    print("\n+-- Phase 4: Chapter Planning " + "-" * 44)
    t0 = time.time()
    try:
        plan = await plot_planner.plan_chapter(1, config, bible, characters, outline)
        assert isinstance(plan, ChapterPlan)
        print(f"| Title: {plan.title}")
        print(f"| Goal: {plan.goal[:80]}...")
        print(f"| Scenes: {len(plan.scenes)}")
        for s in plan.scenes:
            print(f"|  Scene {s.number}: [{s.setting}] {s.goal[:30]}...")
        print(f"| POV: {plan.pov_character}")
        print(f"| Conflict: {plan.conflict[:60]}...")
        print(f"| Characters: {plan.characters_involved}")
        print(f"| Emotional curve: {len(plan.emotional_curve)} beats")
        print(f"| Hooks: {len(plan.hooks)}")
        print(f"| Word count target: {plan.word_count_target}")
        print(f"   [TIME] Phase 4: {time.time() - t0:.0f}s")
    except Exception as e:
        errors.append(f"Phase 4: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Phase 4 failed: {e}")
        return

    # === Phase 5: Chapter Writing ===
    print("\n+-- Phase 5: Chapter Writing " + "-" * 46)
    t0 = time.time()
    try:
        draft = await writer.generate_chapter(
            chapter_plan=plan, config=config, bible=bible,
            characters=characters, outline=outline,
        )
        assert isinstance(draft, ChapterDraft)
        assert draft.content, "Chapter must have content"
        print(f"| Title: {draft.title}")
        print(f"| Word count: {draft.word_count}")
        print(f"| Content: {len(draft.content)} chars")
        print(f"|  Preview: {draft.content[:150]}...")
        if draft.author_notes:
            print(f"| Author notes: {draft.author_notes[:80]}...")

        # Extract facts
        draft = await writer.extract_facts_and_changes(draft, bible, characters)
        print(f"| Facts: {len(draft.new_facts)}")
        for f in draft.new_facts[:5]:
            print(f"|  [{f.category}] {f.description[:50]}...")
        if len(draft.new_facts) > 5:
            print(f"|  ... and {len(draft.new_facts) - 5} more")
        print(f"| State changes: {len(draft.character_state_changes)}")
        for sc in draft.character_state_changes:
            print(f"     {sc.character_id}: {sc.attribute}")
        print(f"   [TIME] Phase 5: {time.time() - t0:.0f}s")
    except Exception as e:
        errors.append(f"Phase 5: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Phase 5 failed: {e}")
        return

    # === Summary ===
    total = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print(f"=  PIPELINE COMPLETE in {total:.0f}s ({total/60:.1f}m)")
    print(f"=  Phases: 2[OK] 3[OK] 4[OK] 5[OK]")
    print(f"=  Output: Chapter 1 -- '{draft.title}' ({draft.word_count} chars)")
    print(f"=  Facts: {len(draft.new_facts)} | State changes: {len(draft.character_state_changes)}")
    if errors:
        print(f"=  ERRORS: {errors}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        s = ModelScheduler()
        cfg = ProjectConfig(
            title="逆命九霄",
            inspiration="一个天生废脉的少年意外觉醒远古血脉，踏上逆天改命的修真之路。",
            genre=["玄幻", "修仙", "爽文"],
            target_readers="喜欢热血升级流的男性读者",
            tone="热血",
            target_word_count=90000,
        )
        await test_full_pipeline(s, cfg)

    asyncio.run(main())
