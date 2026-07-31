"""Integration test: Phase 2 Bible construction against ModelScope/Qwen3.

Tests ArchitectAgent (world, factions, style, themes, conflicts, pleasure points)
and CharacterManagerAgent (character profiles, relationships, arcs).

Usage:
    cd E:/projects/novel-agent
    PYTHONPATH="E:/projects/novel-agent" uv run python -m pytest tests/test_bible_modelscope.py -v -s
"""

import logging
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.llm.scheduler import ModelScheduler
from src.models.bible import (
    CoreConflict, Faction, NarrativeRules, NovelBible,
    StyleContract, Theme, WorldBuilding,
)
from src.models.characters import CharacterProfile, CharacterRegistry
from src.models.project import ProjectConfig
from src.agents.architect import ArchitectAgent
from src.agents.character_manager import CharacterManagerAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def scheduler():
    return ModelScheduler()


@pytest.fixture
def architect(scheduler):
    return ArchitectAgent(scheduler)


@pytest.fixture
def character_manager(scheduler):
    return CharacterManagerAgent(scheduler)


@pytest.fixture
def config():
    return ProjectConfig(
        title="测试小说",
        inspiration="一个废柴少年意外觉醒了远古血脉，在弱肉强食的修真世界一步步崛起的故事。",
        genre=["玄幻", "修仙", "爽文"],
        target_readers="喜欢热血升级流的男性读者",
        tone="热血",
        target_word_count=90000,
    )


# ── Individual Agent Tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_world(architect, config):
    """Test world building."""
    print("\n--- build_world ---")
    try:
        world = await architect.build_world(config)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"build_world: {type(e).__name__}: {e}")

    assert isinstance(world, WorldBuilding)
    assert world.name, "World must have a name"
    assert world.world_type, "World must have a type"
    print(f"  [OK] {world.name} ({world.world_type})")


@pytest.mark.asyncio
async def test_design_factions(architect, config):
    """Test faction design."""
    print("\n--- design_factions ---")
    # Need a world first
    world = await architect.build_world(config)
    try:
        factions = await architect.design_factions(config, world)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"design_factions: {type(e).__name__}: {e}")

    assert len(factions) >= 2, "Need at least 2 factions"
    for f in factions:
        assert f.name, "Faction must have a name"
        assert f.id, "Faction must have an id"
        print(f"  [OK] [{f.id}] {f.name} ({f.faction_type})")


@pytest.mark.asyncio
async def test_create_style_contract(architect, config):
    """Test style contract."""
    print("\n--- create_style_contract ---")
    try:
        style = await architect.create_style_contract(config)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"create_style_contract: {type(e).__name__}: {e}")

    assert isinstance(style, StyleContract)
    assert style.tone, "Must have a tone"
    print(f"  [OK] tone={style.tone}, pacing={style.pacing_preference}")


@pytest.mark.asyncio
async def test_generate_themes(architect, config):
    """Test theme generation."""
    print("\n--- generate_themes ---")
    world = await architect.build_world(config)
    try:
        themes = await architect.generate_themes(config, world)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"generate_themes: {type(e).__name__}: {e}")

    assert len(themes) >= 1, "Need at least 1 theme"
    for t in themes:
        assert t.name, "Theme must have a name"
        print(f"  [OK] {t.name}")


@pytest.mark.asyncio
async def test_define_conflicts(architect, config):
    """Test conflict definition."""
    print("\n--- define_conflicts ---")
    world = await architect.build_world(config)
    factions = await architect.design_factions(config, world)
    try:
        conflicts = await architect.define_conflicts(config, world, factions)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"define_conflicts: {type(e).__name__}: {e}")

    assert len(conflicts) >= 1, "Need at least 1 conflict"
    for c in conflicts:
        assert c.description, "Conflict must have description"
        print(f"  [OK] [{c.conflict_type}] {c.description[:60]}...")


@pytest.mark.asyncio
async def test_design_pleasure_points(architect, config):
    """Test pleasure point design."""
    print("\n--- design_pleasure_points ---")
    world = await architect.build_world(config)
    themes = await architect.generate_themes(config, world)
    try:
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"design_pleasure_points: {type(e).__name__}: {e}")

    assert pleasure_model, "Must have pleasure point model"
    print(f"  [OK] pleasure_model={pleasure_model[:80]}...")


# ── Character Manager Tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_all_characters(character_manager, config, architect):
    """Test full character creation pipeline."""
    print("\n--- create_all_characters ---")
    # Build a minimal bible first
    world = await architect.build_world(config)
    factions = await architect.design_factions(config, world)
    style = await architect.create_style_contract(config)
    themes = await architect.generate_themes(config, world)
    conflicts = await architect.define_conflicts(config, world, factions)
    pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)

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

    try:
        registry = await character_manager.create_all_characters(config, bible)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"create_all_characters: {type(e).__name__}: {e}")

    assert isinstance(registry, CharacterRegistry)
    assert len(registry.characters) >= 2, "Need at least 2 characters"
    for cid, char in registry.characters.items():
        assert char.name, "Character must have a name"
        assert char.role, "Character must have a role"
        print(f"  [OK] [{cid}] {char.name} ({char.role})")


# ── Full Bible Assembly Test ────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_bible_assembly(architect, character_manager, config):
    """End-to-end bible construction (what orchestrator.build_bible does)."""
    print("\n" + "=" * 60)
    print("FULL BIBLE ASSEMBLY TEST")
    print("=" * 60)

    errors = []

    # Step 1: World
    try:
        world = await architect.build_world(config)
        print(f"[OK] World: {world.name}")
    except Exception as e:
        errors.append(f"World: {e}")
        traceback.print_exc()
        pytest.fail(f"World: {type(e).__name__}: {e}")
        return

    # Step 2: Factions
    try:
        factions = await architect.design_factions(config, world)
        print(f"[OK] Factions: {len(factions)}")
    except Exception as e:
        errors.append(f"Factions: {e}")
        traceback.print_exc()
        pytest.fail(f"Factions: {type(e).__name__}: {e}")
        return

    # Step 3: Style
    try:
        style = await architect.create_style_contract(config)
        print(f"[OK] Style: {style.tone}")
    except Exception as e:
        errors.append(f"Style: {e}")
        traceback.print_exc()
        pytest.fail(f"Style: {type(e).__name__}: {e}")
        return

    # Step 4: Themes + Conflicts
    try:
        themes = await architect.generate_themes(config, world)
        conflicts = await architect.define_conflicts(config, world, factions)
        print(f"[OK] Themes: {len(themes)}, Conflicts: {len(conflicts)}")
    except Exception as e:
        errors.append(f"Themes/Conflicts: {e}")
        traceback.print_exc()
        pytest.fail(f"Themes/Conflicts: {type(e).__name__}: {e}")
        return

    # Step 5: Pleasure points
    try:
        pleasure_model, constraints = await architect.design_pleasure_points(config, world, themes)
        print(f"[OK] Pleasure points: {pleasure_model[:60]}...")
    except Exception as e:
        errors.append(f"Pleasure: {e}")
        traceback.print_exc()
        pytest.fail(f"Pleasure: {type(e).__name__}: {e}")
        return

    # Assemble bible
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

    # Step 6: Characters
    try:
        registry = await character_manager.create_all_characters(config, bible)
        print(f"[OK] Characters: {len(registry.characters)}")
        for cid, char in registry.characters.items():
            print(f"     [{cid}] {char.name} ({char.role}) — {char.personality[:40]}...")
    except Exception as e:
        errors.append(f"Characters: {e}")
        traceback.print_exc()
        pytest.fail(f"Characters: {type(e).__name__}: {e}")
        return

    print(f"\n[OK] Bible assembled: {len(bible.factions)} factions, "
          f"{len(registry.characters)} characters, "
          f"{len(bible.themes)} themes, {len(bible.core_conflicts)} conflicts")

    if errors:
        pytest.fail(f"Errors during assembly: {errors}")
