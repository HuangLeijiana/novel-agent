"""Integration test: Phase 5 Chapter Writing with ModelScope/Qwen3.

Runs WriterAgent.generate_chapter() and extract_facts_and_changes()
against ModelScope and reports any Pydantic validation failures.

Usage:
    cd E:/projects/novel-agent
    uv run python -m pytest tests/test_writer_modelscope.py -v -s
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.config.settings import get_settings
from src.llm.scheduler import ModelScheduler
from src.models.bible import (
    CoreConflict,
    Faction,
    NarrativeRules,
    NovelBible,
    StyleContract,
    Theme,
    WorldBuilding,
)
from src.models.chapter import ChapterDraft
from src.models.characters import CharacterProfile, CharacterRegistry
from src.models.memory import MemoryState
from src.models.outline import (
    ChapterPlan,
    EmotionalBeat,
    Hook,
    MasterOutline,
    PlotArc,
    Scene,
    TurningPoint,
    Volume,
)
from src.models.project import ProjectConfig
from src.agents.writer import WriterAgent, ChapterContentOutput, FactExtractionOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def scheduler():
    """Create ModelScheduler from .env settings."""
    return ModelScheduler()


@pytest.fixture
def writer(scheduler):
    """Create WriterAgent."""
    return WriterAgent(scheduler)


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
        Faction(name="青云宗", faction_type="sect", goal="培养修真人才，维护正道", resources="灵石矿脉、藏经阁"),
        Faction(name="魔渊教", faction_type="cult", goal="颠覆正道统治，夺取天地灵脉", resources="魔气源泉、暗杀组织"),
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
            conflict_type="person_vs_society", description="主角被家族抛弃后在宗门中证明自己", stakes="生存与尊严"
        ),
        CoreConflict(conflict_type="person_vs_person", description="主角与宿敌的终极对决", stakes="大道之争，不死不休"),
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
            "char_003": CharacterProfile(
                id="char_003",
                name="赵铁山",
                role="antagonist",
                personality="骄横跋扈，心胸狭隘",
                motivation="维护自己在宗门中的地位和权力",
                flaw="嫉妒心极强，容不得别人超越自己",
                backstory="青云宗内门弟子，家世显赫，自视甚高",
            ),
        }
    )


@pytest.fixture
def outline():
    """Minimal master outline for testing."""
    return MasterOutline(
        title="逆命九霄",
        logline="废柴少年秦默觉醒远古血脉，在弱肉强食的修真世界逆天改命，一步步踏上巅峰。",
        main_plot=[
            PlotArc(
                name="主线", description="秦默从废柴到强者的成长之路", arc_type="main", start_chapter=1, end_chapter=30
            ),
        ],
        subplots=[
            PlotArc(
                name="苏婉清寻师线",
                description="苏婉清寻找失散师父的调查线",
                arc_type="subplot",
                start_chapter=3,
                end_chapter=20,
            ),
        ],
        volumes=[
            Volume(
                title="第一卷：废柴觉醒",
                logline="从家族弃子到宗门弟子的蜕变",
                number=1,
                start_chapter=1,
                end_chapter=10,
            ),
        ],
        major_turning_points=[
            TurningPoint(turning_type="inciting_incident", chapter=1, description="秦默在矿区意外获得远古血脉传承"),
            TurningPoint(turning_type="midpoint", chapter=15, description="宗门大比，秦默一战成名"),
        ],
        chapter_count=30,
    )


@pytest.fixture
def chapter_plan():
    """Minimal but realistic chapter plan for Chapter 1."""
    return ChapterPlan(
        chapter_number=1,
        title="废脉少年",
        goal="建立主角的困境和世界背景，让读者同情主角并期待转机",
        scenes=[
            Scene(
                number=1,
                setting="秦家矿区 — 正午",
                pov="char_001",
                characters_present=["char_001"],
                goal="展示秦默在家族中的卑微地位",
                conflict="其他矿工的嘲笑和监工的压迫",
                outcome="秦默默默忍受，但眼中闪过一丝不甘",
            ),
            Scene(
                number=2,
                setting="矿区深处 — 傍晚",
                pov="char_001",
                characters_present=["char_001"],
                goal="发现异常矿脉",
                conflict="矿洞塌方，秦默被困",
                outcome="在绝境中发现一块散发着远古气息的奇异晶石",
            ),
            Scene(
                number=3,
                setting="矿区深处 — 夜晚",
                pov="char_001",
                characters_present=["char_001"],
                goal="激活远古血脉",
                conflict="晶石中的力量涌入体内，剧烈的痛苦几乎撕裂他的身体",
                outcome="秦默成功融合远古血脉，感受到前所未有的力量",
            ),
        ],
        pov_character="char_001",
        conflict="秦默在家族中的卑微地位与他内心不甘的冲突；远古血脉觉醒带来的身体痛苦",
        hooks=[
            Hook(hook_type="cliffhanger", description="秦默体内的血脉觉醒后，一道古老的声音在他脑海中响起……"),
        ],
        characters_involved=["char_001"],
        information_increment="建立世界观（修真体系、秦家地位），引入主角困境，揭示远古血脉的存在",
        reveals=["秦默体内隐藏着远古血脉"],
        foreshadowing=["远古血脉的来源暗示", "青云宗与秦家的关系"],
        emotional_curve=[
            EmotionalBeat(position=0.0, emotion="压抑", intensity=0.7),
            EmotionalBeat(position=0.3, emotion="挣扎", intensity=0.6),
            EmotionalBeat(position=0.6, emotion="绝望", intensity=0.8),
            EmotionalBeat(position=0.8, emotion="痛苦", intensity=0.9),
            EmotionalBeat(position=1.0, emotion="震撼", intensity=0.95),
        ],
        ending_hook="秦默体内的血脉觉醒后，一道古老而威严的声音在他脑海中响起：'万年了……终于等到你。'",
        word_count_target=4000,
    )


@pytest.fixture
def memory():
    """Empty memory state for first chapter."""
    return MemoryState()


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Unit Tests: ChapterContentOutput
# ═══════════════════════════════════════════════════════════════════════


class TestChapterContentOutput:
    """Test ChapterContentOutput validation with malformed Qwen3 inputs."""

    def test_valid_output(self):
        """Normal valid output should pass."""
        data = {
            "title": "废脉少年",
            "content": "## 第一节\n\n苍玄大陆，秦家矿区……\n\n（正文内容）",
            "author_notes": "第一章聚焦于建立主角的困境和世界观",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.title == "废脉少年"
        assert "苍玄大陆" in result.content
        assert result.author_notes

    def test_missing_author_notes(self):
        """Missing author_notes should default to empty string."""
        data = {
            "title": "第一章",
            "content": "正文内容",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.author_notes == ""

    def test_notes_alias(self):
        """'notes' key should map to author_notes."""
        data = {
            "title": "第一章",
            "content": "正文",
            "notes": "这是作者的备注",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.author_notes == "这是作者的备注"

    def test_description_alias(self):
        """'description' key should map to author_notes."""
        data = {
            "title": "第一章",
            "content": "正文",
            "description": "章节描述",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.author_notes == "章节描述"

    def test_chinese_notes_alias(self):
        """Chinese '备注' key should map to author_notes."""
        data = {
            "title": "第一章",
            "content": "正文",
            "备注": "中文备注",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.author_notes == "中文备注"

    def test_chinese_attached_alias(self):
        """Chinese '附注' key should map to author_notes."""
        data = {
            "title": "第一章",
            "content": "正文",
            "附注": "作者附注内容",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.author_notes == "作者附注内容"

    def test_chinese_body_alias(self):
        """Chinese '正文' key should map to content."""
        data = {
            "title": "第一章",
            "正文": "这是正文内容",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.content == "这是正文内容"

    def test_text_alias(self):
        """'text' key should map to content."""
        data = {
            "title": "第一章",
            "text": "文本内容",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.content == "文本内容"

    def test_empty_content_fallback(self):
        """Empty content with body alias should still work."""
        data = {
            "title": "第一章",
            "content": "",
            "body": "真实内容在这里",
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.content == "真实内容在这里"

    def test_all_empty(self):
        """All empty fields should still validate."""
        data = {}
        result = ChapterContentOutput.model_validate(data)
        assert result.title == ""
        assert result.content == ""
        assert result.author_notes == ""

    def test_extra_fields_ignored(self):
        """Extra unknown fields should be ignored by Pydantic."""
        data = {
            "title": "第一章",
            "content": "正文",
            "extra_field": "should be ignored",
            "another_extra": 123,
        }
        result = ChapterContentOutput.model_validate(data)
        assert result.title == "第一章"


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Unit Tests: FactExtractionOutput
# ═══════════════════════════════════════════════════════════════════════


class TestFactExtractionOutput:
    """Test FactExtractionOutput validation with malformed Qwen3 inputs."""

    def test_valid_empty_output(self):
        """Valid but empty output."""
        data = {"new_facts": [], "state_changes": []}
        result = FactExtractionOutput.model_validate(data)
        assert result.new_facts == []
        assert result.state_changes == []

    def test_valid_with_facts(self):
        """Valid output with facts."""
        data = {
            "new_facts": [
                {
                    "id": "fact_001",
                    "category": "world",
                    "description": "苍玄大陆以修真为尊，凡人处于社会底层",
                    "certainty": 1.0,
                    "source_chapter": 1,
                },
                {
                    "id": "fact_002",
                    "category": "character",
                    "description": "秦默天生废脉，被家族流放至矿区",
                    "certainty": 1.0,
                    "source_chapter": 1,
                },
            ],
            "state_changes": [
                {
                    "character_id": "char_001",
                    "attribute": "physical.power",
                    "old_value": "凡人（废脉）",
                    "new_value": "炼气一层（远古血脉觉醒）",
                    "reason": "融合远古血脉传承晶石",
                },
            ],
        }
        result = FactExtractionOutput.model_validate(data)
        assert len(result.new_facts) == 2
        assert result.new_facts[0].category == "world"
        assert len(result.state_changes) == 1

    def test_state_changes_as_strings(self):
        """State changes as plain strings should be coerced to dicts."""
        data = {
            "new_facts": [],
            "state_changes": [
                "秦默觉醒了远古血脉",
                "秦默的修为从凡人提升到炼气一层",
            ],
        }
        result = FactExtractionOutput.model_validate(data)
        assert len(result.state_changes) == 2
        assert result.state_changes[0].reason == "秦默觉醒了远古血脉"
        assert result.state_changes[1].reason == "秦默的修为从凡人提升到炼气一层"

    def test_new_facts_as_strings(self):
        """New facts as plain strings should be coerced to dicts (via Fact validator)."""
        data = {
            "new_facts": [
                "苍玄大陆以修真为尊",
                "秦默天生废脉被流放",
            ],
            "state_changes": [],
        }
        result = FactExtractionOutput.model_validate(data)
        assert len(result.new_facts) == 2
        assert result.new_facts[0].description == "苍玄大陆以修真为尊"
        assert result.new_facts[1].description == "秦默天生废脉被流放"

    def test_facts_with_chinese_keys(self):
        """Facts with Chinese keys should be normalized."""
        data = {
            "new_facts": [
                {"描述": "测试事实", "类别": "world", "确定性": 0.8},
            ],
            "state_changes": [],
        }
        result = FactExtractionOutput.model_validate(data)
        assert len(result.new_facts) == 1
        assert result.new_facts[0].description == "测试事实"
        assert result.new_facts[0].category == "world"
        assert result.new_facts[0].certainty == 0.8

    def test_mixed_valid_and_string_items(self):
        """Mixed valid dicts and string items should both work."""
        data = {
            "new_facts": [
                {"category": "plot", "description": "测试剧情"},
                "另一个事实",
            ],
            "state_changes": [
                {"character_id": "char_001", "reason": "正常变更"},
                "角色的状态发生了变化",
            ],
        }
        result = FactExtractionOutput.model_validate(data)
        assert len(result.new_facts) == 2
        assert result.new_facts[0].category == "plot"
        assert result.new_facts[1].description == "另一个事实"
        assert len(result.state_changes) == 2
        assert result.state_changes[0].character_id == "char_001"
        assert result.state_changes[1].reason == "角色的状态发生了变化"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_generate_chapter(writer, project_config, bible, characters, outline, chapter_plan, memory):
    """Execute chapter generation against ModelScope.

    This is the main test — it surfaces Pydantic validation errors from Qwen3 output.
    """
    print("\n" + "=" * 70)
    print("Phase 5: Testing generate_chapter() against ModelScope...")
    print("=" * 70)

    try:
        draft = await writer.generate_chapter(
            chapter_plan=chapter_plan,
            config=project_config,
            bible=bible,
            characters=characters,
            outline=outline,
            memory=memory,
        )
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"FAILED: {type(e).__name__}: {e}")
        print("!" * 70)
        traceback.print_exc()
        pytest.fail(f"generate_chapter raised {type(e).__name__}: {e}")
        return

    # ── Validate the result ──
    assert isinstance(draft, ChapterDraft), f"Expected ChapterDraft, got {type(draft).__name__}"
    assert draft.chapter_number == 1, f"Expected chapter 1, got {draft.chapter_number}"
    assert draft.title, "Title must not be empty"
    assert draft.content, "Content must not be empty"
    assert draft.word_count > 0, "Word count must be positive"

    print(f"\n[OK] Title: {draft.title}")
    print(f"[OK] Word count: {draft.word_count}")
    print(f"[OK] Content preview ({len(draft.content)} chars):")
    print(f"    {draft.content[:300]}...")
    if draft.author_notes:
        print(f"[OK] Author notes: {draft.author_notes[:100]}...")

    # ── Content quality checks ──
    content_len = len(draft.content)
    print(f"\n[INFO] Content length: {content_len} chars")
    if content_len < 500:
        print("[WARN] Content is very short — Qwen3 may have truncated output")
    if content_len > 200:
        print(f"[OK] Content has reasonable length")

    print("\n" + "=" * 70)
    print("CHAPTER GENERATION TEST PASSED")
    print("=" * 70)


@pytest.mark.asyncio
async def test_extract_facts(writer, project_config, bible, characters, outline, chapter_plan, memory):
    """Test fact extraction from a generated chapter.

    First generates a chapter, then extracts facts from it.
    """
    print("\n" + "=" * 70)
    print("Step 1: Generating chapter for fact extraction test...")
    print("=" * 70)

    try:
        draft = await writer.generate_chapter(
            chapter_plan=chapter_plan,
            config=project_config,
            bible=bible,
            characters=characters,
            outline=outline,
            memory=memory,
        )
        print(f"[OK] Generated: {draft.title} ({draft.word_count} chars)")
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"generate_chapter: {type(e).__name__}: {e}")
        return

    print("\n" + "=" * 70)
    print("Step 2: Extracting facts and state changes...")
    print("=" * 70)

    try:
        draft = await writer.extract_facts_and_changes(
            draft=draft,
            bible=bible,
            characters=characters,
        )
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"FAILED: {type(e).__name__}: {e}")
        print("!" * 70)
        traceback.print_exc()
        pytest.fail(f"extract_facts_and_changes raised {type(e).__name__}: {e}")
        return

    print(f"[OK] New facts extracted: {len(draft.new_facts)}")
    for fact in draft.new_facts:
        print(f"  - [{fact.category}] {fact.description[:80]}... (certainty={fact.certainty})")

    print(f"[OK] State changes: {len(draft.character_state_changes)}")
    for sc in draft.character_state_changes:
        print(f"  - {sc.character_id}: {sc.attribute} {sc.old_value} → {sc.new_value}")
        print(f"    原因: {sc.reason[:80]}")

    # Validation
    assert isinstance(draft.new_facts, list), "new_facts must be a list"
    assert isinstance(draft.character_state_changes, list), "state_changes must be a list"

    print("\n" + "=" * 70)
    print("FACT EXTRACTION TEST PASSED")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Standalone runner (for debugging without pytest)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def main():
        logging.basicConfig(level=logging.DEBUG)
        s = ModelScheduler()
        w = WriterAgent(s)

        # Build fixtures inline
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
        b = NovelBible(
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
                "char_002": CharacterProfile(
                    id="char_002",
                    name="苏婉清",
                    role="deuteragonist",
                    personality="外冷内热",
                    motivation="寻找师父",
                    flaw="不善表达",
                    backstory="身世成谜",
                ),
            }
        )
        outline = MasterOutline(
            title="逆命九霄",
            logline="废柴少年逆天改命",
            main_plot=[PlotArc(name="主线", description="主角成长", arc_type="main")],
            subplots=[],
            volumes=[Volume(title="第一卷", number=1, start_chapter=1, end_chapter=10)],
            major_turning_points=[
                TurningPoint(turning_type="inciting_incident", chapter=1, description="血脉觉醒"),
            ],
            chapter_count=30,
        )
        plan = ChapterPlan(
            chapter_number=1,
            title="废脉少年",
            goal="建立主角困境和世界观",
            scenes=[
                Scene(
                    number=1,
                    setting="秦家矿区",
                    pov="char_001",
                    characters_present=["char_001"],
                    goal="展示主角卑微地位",
                    conflict="矿工嘲笑和监工压迫",
                    outcome="秦默眼中闪过一丝不甘",
                ),
                Scene(
                    number=2,
                    setting="矿区深处",
                    pov="char_001",
                    characters_present=["char_001"],
                    goal="发现异常矿脉",
                    conflict="矿洞塌方",
                    outcome="发现远古晶石",
                ),
            ],
            pov_character="char_001",
            conflict="主角内心的不甘与现实的压迫",
            hooks=[Hook(hook_type="cliffhanger", description="古老的声音在脑海中响起")],
            characters_involved=["char_001"],
            information_increment="建立世界观，引入主角困境",
            reveals=["远古血脉的存在"],
            foreshadowing=["血脉来源的暗示"],
            emotional_curve=[
                EmotionalBeat(position=0.0, emotion="压抑", intensity=0.7),
                EmotionalBeat(position=0.5, emotion="绝望", intensity=0.8),
                EmotionalBeat(position=1.0, emotion="震撼", intensity=0.95),
            ],
            ending_hook="古老的声音在脑海中响起：'万年了……终于等到你。'",
            word_count_target=4000,
        )
        mem = MemoryState()

        print("Testing generate_chapter...")
        try:
            draft = await w.generate_chapter(
                chapter_plan=plan,
                config=pc,
                bible=b,
                characters=chars,
                outline=outline,
                memory=mem,
            )
            print(f"SUCCESS: {draft.title} ({draft.word_count} chars)")
            print(f"Content preview:\n{draft.content[:500]}...")

            # Also test fact extraction
            print("\nTesting extract_facts_and_changes...")
            draft = await w.extract_facts_and_changes(draft, b, chars)
            print(f"SUCCESS: {len(draft.new_facts)} facts, {len(draft.character_state_changes)} changes")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    asyncio.run(main())
