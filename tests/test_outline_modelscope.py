"""Integration test: Outline stage Pydantic validation with ModelScope/Qwen3.

Runs the PlotPlannerAgent.create_master_outline() against ModelScope
and reports any Pydantic validation failures with full tracebacks.

Usage:
    cd E:/projects/novel-agent
    uv run python -m pytest tests/test_outline_modelscope.py -v -s
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.agents.plot_planner import PlotPlannerAgent
from src.llm.scheduler import ModelScheduler
from src.models.bible import CoreConflict, Faction, NarrativeRules, NovelBible, StyleContract, Theme, WorldBuilding
from src.models.characters import CharacterProfile, CharacterRegistry
from src.models.outline import MasterOutline
from src.models.project import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def scheduler():
    """Create ModelScheduler from .env settings."""
    return ModelScheduler()


@pytest.fixture
def plot_planner(scheduler):
    """Create PlotPlannerAgent."""
    return PlotPlannerAgent(scheduler)


@pytest.fixture
def project_config():
    """Minimal project config for testing."""
    return ProjectConfig(
        title="测试小说",
        inspiration="一个废柴少年意外觉醒了远古血脉，在弱肉强食的修真世界一步步崛起的故事。",
        genre=["玄幻", "修仙", "爽文"],
        target_readers="喜欢热血升级流的男性读者",
        tone="热血",
        target_word_count=90000,
    )


@pytest.fixture
def bible():
    """Minimal novel bible for testing."""
    world = WorldBuilding(
        name="苍玄大陆",
        world_type="fantasy",
        geography="九大州，中央为圣地，四方为蛮荒之地",
        history="万年前仙魔大战后天地灵气稀薄，修真文明衰退",
        culture="以实力为尊，宗门为基本单位",
        technology_level="修真文明，凡人处于封建时代",
        magic_system="修真体系：炼气→筑基→金丹→元婴→化神→合体→大乘→渡劫",
        power_progression="每个大境界分九层小境界",
    )
    factions = [
        Faction(
            name="青云宗",
            faction_type="sect",
            goal="培养修真人才，维护正道",
            resources="灵石矿脉、藏经阁、炼丹房",
        ),
        Faction(
            name="魔渊教",
            faction_type="cult",
            goal="颠覆正道统治，夺取天地灵脉",
            resources="魔气源泉、暗杀组织",
        ),
    ]
    style = StyleContract(
        tone="热血",
        pacing_preference="fast",
        sentence_style="varied",
        dialogue_ratio=0.35,
        description_ratio=0.25,
    )
    themes = [
        Theme(name="逆袭成长", description="主角从废柴一步步走向巅峰"),
        Theme(name="友情与背叛", description="修真路上的盟友与敌人"),
    ]
    conflicts = [
        CoreConflict(
            conflict_type="person_vs_society",
            description="主角被家族抛弃后在宗门中证明自己",
            stakes="生存与尊严",
        ),
        CoreConflict(
            conflict_type="person_vs_person",
            description="主角与宿敌的终极对决",
            stakes="大道之争，不死不休",
        ),
    ]
    return NovelBible(
        world=world,
        factions=factions,
        rules=NarrativeRules(),
        style_contract=style,
        themes=themes,
        core_conflicts=conflicts,
        pleasure_point_model="每10章一个小高潮：突破境界、打脸反派、获得机缘",
    )


@pytest.fixture
def characters():
    """Minimal character registry."""
    return CharacterRegistry(
        characters={
            "char_001": CharacterProfile(
                id="char_001",
                name="秦默",
                role="protagonist",
                personality="坚韧不拔，心思缜密，但有时过于冒险",
                motivation="打破命运枷锁，成为最强修真者",
                flaw="过于固执，不轻易相信他人",
                backstory="秦家嫡系却天生废脉，被家族流放至偏远矿区，偶得远古血脉传承",
            ),
            "char_002": CharacterProfile(
                id="char_002",
                name="苏婉清",
                role="deuteragonist",
                personality="外冷内热，聪慧过人",
                motivation="寻找失散多年的师父",
                flaw="不善于表达感情",
                backstory="青云宗第一天才，身世成谜",
            ),
        }
    )


# ── Test ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_master_outline(plot_planner, project_config, bible, characters):
    """Execute the full master outline creation against ModelScope.

    This is the main test — it will surface any Pydantic validation
    errors that Qwen3's output causes.
    """
    print("\n" + "=" * 70)
    print("Testing create_master_outline() against ModelScope...")
    print("=" * 70)

    try:
        outline = await plot_planner.create_master_outline(
            config=project_config,
            bible=bible,
            characters=characters,
        )
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"FAILED: {type(e).__name__}: {e}")
        print("!" * 70)
        traceback.print_exc()
        pytest.fail(f"create_master_outline raised {type(e).__name__}: {e}")
        return

    # ── Validate the result ──
    assert isinstance(outline, MasterOutline), f"Expected MasterOutline, got {type(outline).__name__}"
    assert outline.title, "Title must not be empty"
    print(f"\n[OK] Title: {outline.title}")
    print(f"[OK] Logline: {outline.logline[:100]}...")
    print(f"[OK] Volumes: {len(outline.volumes)}")
    for v in outline.volumes:
        print(f"  - Vol {v.number}: {v.title} (ch {v.start_chapter}-{v.end_chapter})")
    print(f"[OK] Main plots: {len(outline.main_plot)}")
    for p in outline.main_plot:
        print(f"  - {p.name}: {p.description[:80]}...")
    print(f"[OK] Subplots: {len(outline.subplots)}")
    for p in outline.subplots:
        print(f"  - {p.name}: {p.description[:80]}...")
    print(f"[OK] Turning points: {len(outline.major_turning_points)}")
    for tp in outline.major_turning_points:
        print(f"  - [{tp.turning_type}] ch{tp.chapter}: {tp.description[:80]}...")
    print(f"[OK] Chapter count: {outline.chapter_count}")

    # ── Structural assertions ──
    assert len(outline.main_plot) >= 1, "Must have at least 1 main plot arc"
    assert len(outline.volumes) >= 1, "Must have at least 1 volume"
    assert len(outline.major_turning_points) >= 2, "Must have at least 2 turning points"
    assert outline.chapter_count >= 5, "Must plan at least 5 chapters"

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


# ── Pydantic model unit tests (no LLM calls) ───────────────────────


class TestMasterOutlineValidation:
    """Test Pydantic validation with various malformed inputs Qwen3 might produce."""

    def test_string_turning_points(self):
        """Turning points as plain strings should be coerced."""
        from src.agents.plot_planner import MasterOutlineOutput

        data = {
            "title": "测试小说",
            "logline": "一个测试故事",
            "main_plot": [
                {"name": "主线", "description": "主角的成长"},
            ],
            "subplots": [],
            "volumes": [
                {"title": "第一卷", "start_chapter": 1, "end_chapter": 10},
            ],
            "major_turning_points": [
                "主角意外获得传承",  # plain string!
                "主角击败第一个敌人",
            ],
            "chapter_count": 15,
        }
        result = MasterOutlineOutput.model_validate(data)
        assert len(result.major_turning_points) == 2
        assert result.major_turning_points[0].description == "主角意外获得传承"
        assert result.major_turning_points[1].description == "主角击败第一个敌人"

    def test_string_plot_arcs(self):
        """Plot arcs as plain strings should be coerced."""
        from src.agents.plot_planner import MasterOutlineOutput

        data = {
            "title": "测试小说",
            "logline": "一个测试故事",
            "main_plot": [
                "主角从废柴到强者",  # plain string!
            ],
            "subplots": [
                "爱情线",  # plain string!
            ],
            "volumes": [
                {"title": "第一卷", "start_chapter": 1, "end_chapter": 10},
            ],
            "major_turning_points": [],
            "chapter_count": 15,
        }
        result = MasterOutlineOutput.model_validate(data)
        assert len(result.main_plot) == 1
        assert result.main_plot[0].description == "主角从废柴到强者"
        assert len(result.subplots) == 1
        assert result.subplots[0].description == "爱情线"

    def test_volume_string_chapter_numbers(self):
        """Volume start/end chapter as strings should be coerced by Pydantic."""
        data = {
            "title": "测试小说",
            "logline": "测试",
            "main_plot": [],
            "subplots": [],
            "volumes": [
                {"title": "第一卷", "start_chapter": "1", "end_chapter": "10"},
            ],
            "major_turning_points": [],
            "chapter_count": 15,
        }
        from src.agents.plot_planner import MasterOutlineOutput

        result = MasterOutlineOutput.model_validate(data)
        assert result.volumes[0].start_chapter == 1
        assert result.volumes[0].end_chapter == 10

    def test_turning_point_chinese_type_key(self):
        """TurningPoint with Chinese key '类型' should be mapped to turning_type."""
        from src.models.outline import TurningPoint

        # Qwen might output "type" in English, which TurningPoint already handles
        data = {"type": "inciting_incident", "chapter": 1, "description": "主角觉醒"}
        result = TurningPoint.model_validate(data)
        assert result.turning_type == "inciting_incident"

    def test_plot_arc_chinese_keys(self):
        """PlotArc with Chinese-style keys."""
        from src.models.outline import PlotArc

        data = {"arc": "主角线", "description": "主角成长", "start_chapter": 1, "end_chapter": 30}
        result = PlotArc.model_validate(data)
        assert result.name == "主角线"

    def test_emotional_beat_intensity_clamp(self):
        """EmotionalBeat intensity 0-10 scale should be clamped to 0-1."""
        from src.models.outline import EmotionalBeat

        data = {"position": 0.5, "emotion": "激动", "intensity": 8}
        result = EmotionalBeat.model_validate(data)
        assert result.intensity == 0.8

    def test_emotional_beat_intensity_string(self):
        """EmotionalBeat intensity as string should be parsed."""
        from src.models.outline import EmotionalBeat

        data = {"position": 0.5, "emotion": "激动", "intensity": "0.7"}
        result = EmotionalBeat.model_validate(data)
        assert result.intensity == 0.7

    def test_hook_plain_string(self):
        """Hook as plain string should be coerced to description."""
        from src.models.outline import Hook

        result = Hook.model_validate("主角发现了一座隐藏的洞府")
        assert result.description == "主角发现了一座隐藏的洞府"
        assert result.hook_type == "cliffhanger"

    def test_scene_plain_string(self):
        """Scene as plain string should be coerced."""
        from src.models.outline import Scene

        result = Scene.model_validate("青云宗广场：主角参加入门考核")
        assert result.setting == "青云宗广场"
        assert "入门考核" in result.goal

    def test_empty_lists_accepted(self):
        """All empty lists should be accepted without issues."""
        from src.agents.plot_planner import MasterOutlineOutput

        data = {
            "title": "空测试",
            "logline": "",
            "main_plot": [],
            "subplots": [],
            "volumes": [],
            "major_turning_points": [],
            "chapter_count": 0,
        }
        result = MasterOutlineOutput.model_validate(data)
        assert result.title == "空测试"
        assert result.chapter_count == 0

    def test_volume_with_introduction_alias(self):
        """Volume with 'introduction' key should map to logline."""
        from src.models.outline import Volume

        data = {
            "title": "第一卷",
            "introduction": "从家族弃子到宗门弟子的成长之路",
            "start_chapter": 1,
            "end_chapter": 10,
        }
        result = Volume.model_validate(data)
        assert result.logline == "从家族弃子到宗门弟子的成长之路"

    def test_volume_with_key_events_alias(self):
        """Volume with 'key_events' key should map to major_events."""
        from src.models.outline import Volume

        data = {
            "title": "第一卷",
            "key_events": ["血脉觉醒", "宗门考核", "初战告捷"],
            "start_chapter": 1,
            "end_chapter": 10,
        }
        result = Volume.model_validate(data)
        assert len(result.major_events) == 3
        assert result.major_events[0] == "血脉觉醒"

    def test_plot_arc_auto_name_from_description(self):
        """PlotArc with no name should derive from description."""
        from src.models.outline import PlotArc

        data = {
            "type": "subplot",
            "description": "苏婉清寻找失散师父的调查线",
        }
        result = PlotArc.model_validate(data)
        assert result.name == "苏婉清寻找失散师父的调查线"
        assert result.arc_type == "subplot"

    def test_plot_arc_auto_name_from_arc_type(self):
        """PlotArc with no name and no description should derive from arc_type."""
        from src.models.outline import PlotArc

        data = {
            "type": "b_plot",
        }
        result = PlotArc.model_validate(data)
        assert result.name == "副线"
        assert result.arc_type == "b_plot"

    def test_volume_strings_in_output(self):
        """Volumes as plain strings should be coerced to Volume dicts."""
        from src.agents.plot_planner import MasterOutlineOutput

        data = {
            "title": "测试小说",
            "logline": "测试故事",
            "main_plot": [],
            "subplots": [],
            "volumes": [
                "第一卷：废土觉醒 — 从生存本能到守护信念",
                "第二卷：血契狂潮 — 危机与盟友羁绊",
            ],
            "major_turning_points": [],
            "chapter_count": 20,
        }
        result = MasterOutlineOutput.model_validate(data)
        assert len(result.volumes) == 2
        assert result.volumes[0].title == "废土觉醒 — 从生存本能到守护信念"
        assert result.volumes[0].number == 1
        assert result.volumes[0].start_chapter == 1
        assert result.volumes[0].end_chapter == 10
        assert result.volumes[1].title == "血契狂潮 — 危机与盟友羁绊"
        assert result.volumes[1].number == 2
        assert result.volumes[1].start_chapter == 11
        assert result.volumes[1].end_chapter == 20

    def test_volume_string_no_chapter_prefix(self):
        """Volume string without chapter prefix should use full string as title."""
        from src.agents.plot_planner import MasterOutlineOutput

        data = {
            "title": "测试",
            "logline": "测试",
            "main_plot": [],
            "subplots": [],
            "volumes": ["觉醒篇", "成长篇", "巅峰篇"],
            "major_turning_points": [],
            "chapter_count": 30,
        }
        result = MasterOutlineOutput.model_validate(data)
        assert len(result.volumes) == 3
        assert result.volumes[0].title == "觉醒篇"
        assert result.volumes[0].number == 1
        assert result.volumes[1].title == "成长篇"
        assert result.volumes[1].number == 2


# ── Test ChapterPlanOutput validation ───────────────────────────────


class TestChapterPlanValidation:
    """Test ChapterPlanOutput validation with malformed inputs."""

    def test_emotional_curve_intensity_clamp(self):
        """ChapterPlanOutput should clamp intensities from 0-10 scale."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "建立世界观",
            "scenes": [],
            "pov_character": "char_001",
            "conflict": "测试冲突",
            "emotional_curve": [
                {"position": 0.0, "emotion": "平静", "intensity": 3},
                {"position": 0.5, "emotion": "紧张", "intensity": 8},
                {"position": 1.0, "emotion": "激动", "intensity": 9},
            ],
            "ending_hook": "悬念结尾",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert result.emotional_curve[0].intensity == 0.3
        assert result.emotional_curve[1].intensity == 0.8
        assert result.emotional_curve[2].intensity == 0.9

    def test_scene_list_with_strings(self):
        """Scenes as mixed strings and dicts."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "测试",
            "scenes": [
                "修炼室：主角突破境界",
                {"setting": "宗门大殿", "goal": "接受任务", "conflict": "有人质疑主角"},
            ],
            "pov_character": "char_001",
            "conflict": "测试",
            "emotional_curve": [],
            "ending_hook": "",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert len(result.scenes) == 2
        assert result.scenes[0].setting == "修炼室"
        assert result.scenes[1].setting == "宗门大殿"

    def test_emotional_curve_string_items(self):
        """Emotional curve items as strings should be parsed to dicts."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "测试",
            "scenes": [],
            "pov_character": "char_001",
            "conflict": "测试",
            "emotional_curve": [
                "压抑，强度0.9",
                "愤怒，强度0.7",
                "紧张，强度0.95",
                "惊喜，强度0.85",
                "笃定，强度0.8",
            ],
            "ending_hook": "",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert len(result.emotional_curve) == 5
        assert result.emotional_curve[0].emotion == "压抑"
        assert result.emotional_curve[0].intensity == 0.9  # Already in 0-1 scale
        assert result.emotional_curve[2].emotion == "紧张"
        assert result.emotional_curve[2].intensity == 0.95

    def test_emotional_curve_string_wrong(self):
        """Emotional curve string items with 0-10 scale should be clamped."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "测试",
            "scenes": [],
            "pov_character": "char_001",
            "conflict": "测试",
            "emotional_curve": [
                "平静，强度3",
                "紧张，强度8",
                "激动，强度9",
            ],
            "ending_hook": "",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert result.emotional_curve[0].intensity == 0.3
        assert result.emotional_curve[1].intensity == 0.8
        assert result.emotional_curve[2].intensity == 0.9

    def test_emotional_curve_dict_with_parenthesized_intensity(self):
        """Dict items where emotion contains '(强度X.X)' should extract intensity."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "测试",
            "scenes": [],
            "pov_character": "char_001",
            "conflict": "测试",
            "emotional_curve": [
                {"position": 0.0, "emotion": "压抑（强度0.9）"},
                {"position": 0.3, "emotion": "担忧（强度0.7）"},
                {"position": 0.5, "emotion": "紧张（强度0.85）"},
            ],
            "ending_hook": "",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert len(result.emotional_curve) == 3
        assert result.emotional_curve[0].emotion == "压抑"
        assert result.emotional_curve[0].intensity == 0.9

    def test_emotional_curve_string_parenthesized(self):
        """String items with parenthesized intensity should be parsed."""
        from src.agents.plot_planner import ChapterPlanOutput

        data = {
            "chapter_number": 1,
            "title": "第一章",
            "goal": "测试",
            "scenes": [],
            "pov_character": "char_001",
            "conflict": "测试",
            "emotional_curve": [
                "压抑（强度0.9）",
                "愤怒（强度0.7）",
            ],
            "ending_hook": "",
            "word_count_target": 3000,
        }
        result = ChapterPlanOutput.model_validate(data)
        assert result.emotional_curve[0].emotion == "压抑"
        assert result.emotional_curve[0].intensity == 0.9


# ── Chapter Planning Integration Test ────────────────────────────────


@pytest.mark.asyncio
async def test_plan_chapter(plot_planner, project_config, bible, characters):
    """Test chapter-level planning against ModelScope.

    This requires a MasterOutline first, then plans chapter 1.
    """
    print("\n" + "=" * 70)
    print("Step 1: Creating master outline...")
    print("=" * 70)

    try:
        outline = await plot_planner.create_master_outline(
            config=project_config,
            bible=bible,
            characters=characters,
        )
        print(f"[OK] Outline: {outline.title}, {outline.chapter_count} chapters")
    except Exception as e:
        print(f"[FAIL] Master outline failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Master outline failed: {e}")
        return

    print("\n" + "=" * 70)
    print("Step 2: Planning chapter 1...")
    print("=" * 70)

    try:
        plan = await plot_planner.plan_chapter(
            chapter_number=1,
            config=project_config,
            bible=bible,
            characters=characters,
            outline=outline,
        )
    except Exception as e:
        print(f"[FAIL] Chapter planning failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"plan_chapter raised {type(e).__name__}: {e}")
        return

    # Validate
    print(f"[OK] Chapter title: {plan.title}")
    print(f"[OK] Goal: {plan.goal[:100]}...")
    print(f"[OK] Scenes: {len(plan.scenes)}")
    for i, scene in enumerate(plan.scenes):
        print(f"  Scene {i + 1}: {scene.setting} | goal={scene.goal[:40]}...")
    print(f"[OK] POV: {plan.pov_character}")
    print(f"[OK] Conflict: {plan.conflict[:80]}...")
    print(f"[OK] Hooks: {len(plan.hooks)}")
    for h in plan.hooks:
        print(f"  - [{h.hook_type}] {h.description[:60]}...")
    print(f"[OK] Characters involved: {plan.characters_involved}")
    print(f"[OK] Info increment: {plan.information_increment[:80]}...")
    print(f"[OK] Reveals: {plan.reveals}")
    print(f"[OK] Foreshadowing: {plan.foreshadowing}")
    print(f"[OK] Emotional curve: {len(plan.emotional_curve)} beats")
    for beat in plan.emotional_curve:
        print(f"  - pos={beat.position:.2f} {beat.emotion} intensity={beat.intensity:.2f}")
    print(f"[OK] Ending hook: {plan.ending_hook[:80]}...")
    print(f"[OK] Word count target: {plan.word_count_target}")

    assert plan.title, "Chapter must have a title"
    assert len(plan.scenes) >= 1, "Must have at least 1 scene"
    assert plan.goal, "Must have a goal"
    assert plan.conflict, "Must have a conflict"

    print("\n" + "=" * 70)
    print("CHAPTER PLANNING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    # Allow running directly for debugging
    async def main():
        logging.basicConfig(level=logging.DEBUG)
        s = ModelScheduler()
        pp = PlotPlannerAgent(s)

        pc = ProjectConfig(
            title="测试小说",
            inspiration="一个废柴少年意外觉醒了远古血脉，在弱肉强食的修真世界一步步崛起的故事。",
            genre=["玄幻", "修仙", "爽文"],
            target_readers="喜欢热血升级流的男性读者",
            tone="热血",
            target_word_count=90000,
        )

        world = WorldBuilding(
            name="苍玄大陆",
            world_type="fantasy",
            geography="九大州",
            history="万年前仙魔大战",
            culture="以实力为尊",
            magic_system="修真体系",
        )
        bible = NovelBible(
            world=world,
            factions=[
                Faction(name="青云宗", faction_type="sect", goal="维护正道"),
                Faction(name="魔渊教", faction_type="cult", goal="颠覆正道"),
            ],
            rules=NarrativeRules(),
            style_contract=StyleContract(tone="热血"),
            themes=[Theme(name="逆袭成长"), Theme(name="友情与背叛")],
            core_conflicts=[
                CoreConflict(conflict_type="person_vs_society", description="主角被家族抛弃"),
                CoreConflict(conflict_type="person_vs_person", description="主角与宿敌对决"),
            ],
        )
        chars = CharacterRegistry(
            characters={
                "char_001": CharacterProfile(
                    id="char_001",
                    name="秦默",
                    role="protagonist",
                    personality="坚韧不拔",
                    motivation="成为最强",
                    flaw="过于固执",
                    backstory="被家族流放",
                ),
            }
        )

        print("Calling create_master_outline...")
        try:
            outline = await pp.create_master_outline(pc, bible, chars)
            print(f"SUCCESS: {outline.title}")
            print(outline.model_dump_json(indent=2))
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    asyncio.run(main())
