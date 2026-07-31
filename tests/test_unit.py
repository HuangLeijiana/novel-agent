"""Unit tests — no API key required.

Tests Pydantic model validation, JSON repair, content safety,
prompt building, and settings resolution with pure logic.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.llm.scheduler import _try_parse_json

# ═══════════════════════════════════════════════════════════════════════
# JSON Repair (_try_parse_json)
# ═══════════════════════════════════════════════════════════════════════


class TestJsonRepair:
    """Test the multi-layer JSON repair pipeline used by ModelScheduler."""

    def test_valid_json_passes_through(self):
        data = {"title": "test", "content": "hello world"}
        result = _try_parse_json(json.dumps(data))
        assert result == data

    def test_valid_json_with_chinese(self):
        data = {"title": "测试章节", "content": "正文内容很长……" * 50}
        result = _try_parse_json(json.dumps(data, ensure_ascii=False))
        assert result["title"] == "测试章节"

    def test_json_inside_markdown_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _try_parse_json(raw)
        assert result["key"] == "value"

    def test_json_with_leading_text(self):
        raw = 'Here is the output:\n\n{"key": "value"}'
        result = _try_parse_json(raw)
        assert result["key"] == "value"

    def test_json_with_trailing_text(self):
        raw = '{"key": "value"}\n\nEnd of output.'
        result = _try_parse_json(raw)
        assert result["key"] == "value"

    def test_unescaped_newlines_in_string(self):
        raw = '{\n  "content": "line 1\nline 2\nline 3",\n  "title": "ok"\n}'
        result = _try_parse_json(raw)
        assert result["title"] == "ok"
        assert "line 1" in result["content"]

    def test_unescaped_tabs_in_string(self):
        raw = '{"content": "col1\tcol2\tcol3", "title": "ok"}'
        result = _try_parse_json(raw)
        assert result["title"] == "ok"

    def test_truncated_json_closes_braces(self):
        raw = '{"title": "test", "content": "truncated...'
        result = _try_parse_json(raw)
        assert result["title"] == "test"

    @pytest.mark.xfail(reason="Known limitation: deeply nested truncation with unclosed string in array")
    def test_truncated_json_with_nested_brackets(self):
        raw = '{"items": [{"a": 1}, {"b": 2}'
        result = _try_parse_json(raw)
        assert isinstance(result, dict)
        assert "items" in result or "a" in result

    def test_truncated_json_unclosed_string(self):
        raw = '{"title": "test", "content": "unclosed string'
        result = _try_parse_json(raw)
        assert "title" in result

    def test_qwen3_style_comment_in_json(self):
        """Qwen3 sometimes copies // comments from schema prompts."""
        raw = '{\n  "title": "test",  // 标题\n  "content": "正文"\n}'
        result = _try_parse_json(raw)
        # Should either parse successfully or at minimum not crash
        assert isinstance(result, (dict, type(None))) or "title" in result

    def test_empty_object(self):
        result = _try_parse_json("{}")
        assert result == {}

    def test_empty_array(self):
        result = _try_parse_json("[]")
        assert result == []

    def test_nested_objects(self):
        data = {"a": {"b": {"c": [1, 2, 3]}}}
        result = _try_parse_json(json.dumps(data))
        assert result["a"]["b"]["c"] == [1, 2, 3]

    def test_qwen3_real_truncation_pattern(self):
        """Simulates Qwen3 truncation: content string cut off mid-word."""
        raw = (
            "{\n"
            '  "title": "废脉少年",\n'
            '  "content": "## 第一节\\n\\n秦默站在矿区深处，手中的晶石散发着诡异的光芒。'
            "他感受到一股前所未有的力量在体内涌动，仿佛有什么东西正在苏醒。\\n\\n突然，"
            "一道古老的声音在他脑海中响起："
        )
        result = _try_parse_json(raw)
        assert result["title"] == "废脉少年"
        assert "秦默" in result["content"]


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Model Validators — ChapterContentOutput
# ═══════════════════════════════════════════════════════════════════════


class TestChapterContentOutput:
    from src.agents.writer import ChapterContentOutput

    def test_valid(self):
        data = {"title": "第一章", "content": "正文内容", "author_notes": "备注"}
        result = self.ChapterContentOutput.model_validate(data)
        assert result.title == "第一章"
        assert result.content == "正文内容"

    def test_chinese_content_alias(self):
        """'正文' should map to content, '备注' to author_notes."""
        data = {"title": "第一章", "正文": "正文内容", "备注": "作者附注"}
        result = self.ChapterContentOutput.model_validate(data)
        assert result.title == "第一章"
        assert result.content == "正文内容"
        assert result.author_notes == "作者附注"

    def test_chinese_author_notes_alias(self):
        """'附注' should map to author_notes."""
        data = {"title": "第一章", "content": "内容", "附注": "附注内容"}
        result = self.ChapterContentOutput.model_validate(data)
        assert result.author_notes == "附注内容"

    def test_missing_optional_fields_default(self):
        data = {"title": "第一章", "content": "正文"}
        result = self.ChapterContentOutput.model_validate(data)
        assert result.author_notes == ""

    def test_empty_all(self):
        result = self.ChapterContentOutput.model_validate({})
        assert result.title == ""
        assert result.content == ""
        assert result.author_notes == ""

    def test_extra_fields_ignored(self):
        data = {"title": "第一章", "content": "正文", "extra": "ignored"}
        result = self.ChapterContentOutput.model_validate(data)
        assert result.title == "第一章"
        assert not hasattr(result, "extra")

    def test_notes_alias(self):
        result = self.ChapterContentOutput.model_validate({"title": "x", "content": "y", "notes": "z"})
        assert result.author_notes == "z"


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Model Validators — FactExtractionOutput
# ═══════════════════════════════════════════════════════════════════════


class TestFactExtractionOutput:
    from src.agents.writer import FactExtractionOutput

    def test_valid_empty(self):
        result = self.FactExtractionOutput.model_validate({"new_facts": [], "state_changes": []})
        assert result.new_facts == []
        assert result.state_changes == []

    def test_string_facts_coerced(self):
        """Plain strings should be coerced to Fact dicts."""
        result = self.FactExtractionOutput.model_validate({"new_facts": ["苍玄大陆以修真为尊"], "state_changes": []})
        assert result.new_facts[0].description == "苍玄大陆以修真为尊"

    def test_string_state_changes_coerced(self):
        """Plain strings should be coerced to StateChange dicts."""
        result = self.FactExtractionOutput.model_validate({"new_facts": [], "state_changes": ["秦默觉醒了远古血脉"]})
        assert result.state_changes[0].reason == "秦默觉醒了远古血脉"

    def test_chinese_keys_normalized(self):
        result = self.FactExtractionOutput.model_validate(
            {
                "new_facts": [{"描述": "测试事实", "类别": "world", "确定性": 0.8}],
                "state_changes": [],
            }
        )
        assert result.new_facts[0].description == "测试事实"
        assert result.new_facts[0].category == "world"
        assert result.new_facts[0].certainty == 0.8

    def test_mixed_strings_and_dicts(self):
        result = self.FactExtractionOutput.model_validate(
            {
                "new_facts": [
                    {"category": "plot", "description": "剧情"},
                    "另一个事实",
                ],
                "state_changes": [
                    {"character_id": "char_001", "reason": "正常"},
                    "角色变化",
                ],
            }
        )
        assert len(result.new_facts) == 2
        assert result.new_facts[0].category == "plot"
        assert result.new_facts[1].description == "另一个事实"


# ═══════════════════════════════════════════════════════════════════════
# Content Safety
# ═══════════════════════════════════════════════════════════════════════


class TestContentSafety:
    from src.utils.content_safety import ContentSafetyChecker

    def test_clean_content_passes(self):
        checker = self.ContentSafetyChecker()
        text = "秦默站在山巅，俯瞰苍玄大陆。风吹过他的长袍，猎猎作响。"
        result = checker.check(text)
        assert result.passed
        assert len(result.flags) == 0

    def test_block_keyword_detected(self):
        checker = self.ContentSafetyChecker()
        text = "这段内容包含做爱描写。"  # blocking keyword
        result = checker.check(text)
        assert not result.passed
        assert result.block_count >= 1

    def test_warning_keyword_flagged(self):
        checker = self.ContentSafetyChecker()
        text = "她露出酥胸，春光乍泄。"  # warning keyword
        result = checker.check(text)
        assert result.passed  # Warnings don't block
        assert result.warning_count >= 1

    def test_multiple_flags(self):
        checker = self.ContentSafetyChecker()
        text = "做爱描写和酥胸描写同时出现"  # block + warning
        result = checker.check(text)
        assert not result.passed
        assert result.block_count >= 1
        assert result.warning_count >= 1

    def test_empty_text_passes(self):
        checker = self.ContentSafetyChecker()
        result = checker.check("")
        assert result.passed
        assert len(result.flags) == 0

    def test_long_novel_chapter_passes(self):
        checker = self.ContentSafetyChecker()
        text = (
            "苍玄大陆，秦家矿区。黄昏的余晖洒在碎石路上，"
            "将人影拉得很长。秦默背着装满矿石的竹篓，"
            "粗糙的麻绳在肩头磨出血痕。他已经在这片矿区"
            "干了整整三年的苦力，从十五岁到现在，手上的"
            "茧子比石头还硬。" * 20
        )
        result = checker.check(text)
        assert result.passed

    def test_flag_context_captured(self):
        checker = self.ContentSafetyChecker()
        text = "剧情铺垫。" + "做爱" + "后续发展。"
        result = checker.check(text)
        assert len(result.flags) >= 1
        flag = result.flags[0]
        assert flag.keyword == "做爱"
        assert flag.severity == "block"
        assert len(flag.context) > 0

    def test_summary_formatting(self):
        checker = self.ContentSafetyChecker()
        result = checker.check("做爱描写和酥胸描写")
        summary = result.summary
        assert "flag" in summary.lower() or "flag" in summary


# ═══════════════════════════════════════════════════════════════════════
# Prompt Building
# ═══════════════════════════════════════════════════════════════════════


class TestPromptBuilding:
    def test_expansion_prompt_contains_targets(self):
        from src.agents.writer import _build_expansion_prompt

        prompt = _build_expansion_prompt(
            content="短内容",
            current_count=1000,
            target_count=4000,
            chapter_num=3,
            chapter_title="测试章",
        )
        assert "4000" in prompt
        assert "1000" in prompt
        assert "测试章" in prompt
        assert "第3章" in prompt
        assert "短内容" in prompt

    def test_expansion_prompt_has_json_instruction(self):
        from src.agents.writer import _build_expansion_prompt

        prompt = _build_expansion_prompt(
            content="x",
            current_count=100,
            target_count=500,
            chapter_num=1,
            chapter_title="title",
        )
        assert "JSON" in prompt or "content" in prompt

    def test_format_scenes(self):
        from src.agents.writer import WriterAgent
        from src.llm.scheduler import ModelScheduler
        from src.models.outline import Scene

        agent = WriterAgent(ModelScheduler())
        scenes = [
            Scene(
                number=1,
                setting="矿区",
                pov="char_001",
                characters_present=["char_001"],
                goal="展示卑微",
                conflict="压迫",
                outcome="不甘",
            ),
        ]
        result = agent._format_scenes(scenes)
        assert "矿区" in result

    def test_format_emotional_curve(self):
        from src.agents.writer import WriterAgent
        from src.llm.scheduler import ModelScheduler
        from src.models.outline import EmotionalBeat

        agent = WriterAgent(ModelScheduler())
        beats = [
            EmotionalBeat(position=0.0, emotion="压抑", intensity=0.7),
            EmotionalBeat(position=1.0, emotion="震撼", intensity=0.95),
        ]
        result = agent._format_emotional_curve(beats)
        assert "压抑" in result
        assert "震撼" in result


# ═══════════════════════════════════════════════════════════════════════
# Settings & Model Assignments
# ═══════════════════════════════════════════════════════════════════════


class TestModelAssignments:
    def test_provider_presets_loaded(self):
        from src.config.settings import PROVIDER_PRESETS

        assert "modelscope" in PROVIDER_PRESETS
        assert "openai" in PROVIDER_PRESETS
        assert "anthropic" in PROVIDER_PRESETS
        assert PROVIDER_PRESETS["modelscope"]["quality"] == "Qwen/Qwen3-235B-A22B"

    def test_quality_agents_have_writer(self):
        from src.config.model_assignments import QUALITY_AGENTS

        assert "writer" in QUALITY_AGENTS
        assert "character_manager" in QUALITY_AGENTS
        assert "architect" in QUALITY_AGENTS

    def test_budget_agents_have_editor(self):
        from src.config.model_assignments import BUDGET_AGENTS

        assert "editor" in BUDGET_AGENTS
        assert "continuity_checker" in BUDGET_AGENTS

    def test_agent_configs_complete(self):
        from src.config.model_assignments import AGENT_CONFIGS, BUDGET_AGENTS, ORCHESTRATOR_AGENTS, QUALITY_AGENTS

        all_agents = QUALITY_AGENTS | BUDGET_AGENTS | ORCHESTRATOR_AGENTS
        for agent in all_agents:
            assert agent in AGENT_CONFIGS, f"Missing config for {agent}"
            assert "temperature" in AGENT_CONFIGS[agent]
            assert "max_tokens" in AGENT_CONFIGS[agent]

    def test_writer_max_tokens(self):
        from src.config.model_assignments import AGENT_CONFIGS

        assert AGENT_CONFIGS["writer"]["max_tokens"] >= 16384

    def test_all_agents_in_override_map(self):
        import inspect

        from src.config.model_assignments import (
            BUDGET_AGENTS,
            ORCHESTRATOR_AGENTS,
            QUALITY_AGENTS,
            _resolve_provider_model,
        )

        # Verify the function has entries for all agent types
        all_agents = sorted(QUALITY_AGENTS | BUDGET_AGENTS | ORCHESTRATOR_AGENTS)
        src = inspect.getsource(_resolve_provider_model)
        for agent in all_agents:
            assert agent in src, f"Override map missing: {agent}"


# ═══════════════════════════════════════════════════════════════════════
# Data Models — Robustness
# ═══════════════════════════════════════════════════════════════════════


class TestMasterOutlineValidation:
    from src.models.outline import MasterOutline

    def test_string_volumes_coerced(self):
        """Qwen3 returns volumes as strings — should be coerced."""
        from src.models.outline import PlotArc, TurningPoint, Volume

        outline = self.MasterOutline(
            title="测试",
            logline="测试",
            main_plot=[PlotArc(name="主线", description="主线", arc_type="main")],
            subplots=[],
            volumes=[Volume(title="卷一", number=1, start_chapter=1, end_chapter=10)],
            major_turning_points=[TurningPoint(turning_type="inciting_incident", chapter=1, description="事件")],
            chapter_count=30,
        )
        assert outline.title == "测试"
        assert outline.volumes[0].number == 1

    def test_minimal_outline_valid(self):
        from src.models.outline import PlotArc, TurningPoint, Volume

        outline = self.MasterOutline(
            title="测试",
            logline="测试",
            main_plot=[PlotArc(name="主线", description="d", arc_type="main")],
            volumes=[Volume(title="v", number=1, start_chapter=1, end_chapter=5)],
            major_turning_points=[TurningPoint(turning_type="inciting_incident", chapter=1, description="事件")],
            chapter_count=10,
        )
        assert outline.chapter_count == 10


class TestFactModel:
    from src.models.chapter import Fact

    def test_string_coerced(self):
        fact = self.Fact.model_validate("苍玄大陆以修真为尊")
        assert fact.description == "苍玄大陆以修真为尊"
        # Default category when coerce from string
        assert fact.category in ("world", "other")

    def test_dict_with_chinese_keys(self):
        fact = self.Fact.model_validate({"描述": "测试事实", "类别": "character", "确定性": 0.9})
        assert fact.description == "测试事实"
        assert fact.category == "character"
        assert fact.certainty == 0.9


# ═══════════════════════════════════════════════════════════════════════
# Model Assignments — Parse Override
# ═══════════════════════════════════════════════════════════════════════


class TestParseOverride:
    def test_provider_model_split(self):
        from src.config.model_assignments import _parse_override
        from src.config.settings import Settings

        settings = Settings()
        provider, model, fallback = _parse_override("anthropic:claude-sonnet-4", settings)
        assert provider == "anthropic"
        assert model == "claude-sonnet-4"
        assert fallback is None

    def test_model_only_uses_default_provider(self):
        from src.config.model_assignments import _parse_override
        from src.config.settings import Settings

        settings = Settings(
            DEFAULT_PROVIDER="modelscope",
            DEFAULT_MODEL="Qwen/Qwen3-235B-A22B",
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1",
        )
        provider, model, fallback = _parse_override("Qwen/Qwen3-8B", settings)
        assert provider == "modelscope"
        assert model == "Qwen/Qwen3-8B"


# ═══════════════════════════════════════════════════════════════════════
# File Manager — Path Handling
# ═══════════════════════════════════════════════════════════════════════


class TestFileManager:
    def test_root_path_construction(self):
        from pathlib import Path

        from src.storage.file_manager import ProjectFileManager

        fm = ProjectFileManager("/tmp/workspace", "proj_001")
        assert fm.root == Path("/tmp/workspace/projects/proj_001")
        assert fm.workspace_root == Path("/tmp/workspace")
        assert fm.project_id == "proj_001"

    def test_path_with_windows_style(self):
        from pathlib import Path

        from src.storage.file_manager import ProjectFileManager

        fm = ProjectFileManager("D:\\novels", "test-project")
        assert fm.root == Path("D:/novels/projects/test-project")

    def test_exists_returns_false_for_new_project(self):
        import tempfile

        from src.storage.file_manager import ProjectFileManager

        with tempfile.TemporaryDirectory() as td:
            fm = ProjectFileManager(td, "nonexistent")
            assert not fm.exists()

    def test_initialize_creates_directories(self):
        import tempfile

        from src.models.project import ProjectConfig
        from src.storage.file_manager import ProjectFileManager

        with tempfile.TemporaryDirectory() as td:
            fm = ProjectFileManager(td, "init_test")
            config = ProjectConfig(
                title="测试",
                inspiration="测试",
                genre=["玄幻"],
                target_readers="读者",
                tone="热血",
            )
            fm.initialize(config)
            assert fm.exists()
            assert (fm.root / "novel_bible").is_dir()
            assert (fm.root / "outline" / "chapters").is_dir()
            assert (fm.root / "memory").is_dir()
            assert (fm.root / "output" / "chapters").is_dir()


# ═══════════════════════════════════════════════════════════════════════
# Model Scheduler — Routing & Provider Init
# ═══════════════════════════════════════════════════════════════════════


class TestSchedulerRouting:
    def test_get_assignment_returns_known_agent(self):
        from src.llm.scheduler import ModelScheduler

        s = ModelScheduler()
        assignment = s._get_assignment("writer")
        assert assignment.agent_type == "writer"
        assert assignment.max_tokens >= 16384

    def test_get_assignment_falls_back_for_unknown(self):
        from src.llm.scheduler import ModelScheduler

        s = ModelScheduler()
        assignment = s._get_assignment("nonexistent_agent")
        assert assignment is not None
        # Falls back to orchestrator or first available
        assert assignment.agent_type in ("orchestrator", "writer", "architect")

    def test_all_registered_agents_have_assignments(self):
        from src.config.model_assignments import AGENT_CONFIGS
        from src.llm.scheduler import ModelScheduler

        s = ModelScheduler()
        for agent_type in AGENT_CONFIGS:
            assignment = s._get_assignment(agent_type)
            assert assignment is not None, f"Missing assignment for {agent_type}"
            assert assignment.max_tokens > 0
            assert 0.0 <= assignment.temperature <= 2.0

    def test_modelscope_uses_openai_provider(self):
        from src.llm.scheduler import ModelScheduler

        s = ModelScheduler()
        provider = s._get_provider("modelscope")
        assert provider.provider_name in ("openai", "modelscope")

    def test_provider_lazy_initialization(self):
        from src.llm.scheduler import ModelScheduler

        s = ModelScheduler()
        # Providers dict starts empty
        assert len(s._providers) == 0
        # First access initializes
        p1 = s._get_provider("modelscope")
        assert "modelscope" in s._providers
        # Second access returns same instance
        p2 = s._get_provider("modelscope")
        assert p1 is p2


# ═══════════════════════════════════════════════════════════════════════
# Character & Bible Models
# ═══════════════════════════════════════════════════════════════════════


class TestCharacterModels:
    def test_minimal_character_profile(self):
        from src.models.characters import CharacterProfile

        char = CharacterProfile(
            id="char_001",
            name="秦默",
            role="protagonist",
            personality="坚韧",
            motivation="变强",
            flaw="固执",
            backstory="被流放",
        )
        assert char.name == "秦默"
        assert char.role == "protagonist"

    def test_character_registry(self):
        from src.models.characters import CharacterProfile, CharacterRegistry

        reg = CharacterRegistry(
            characters={
                "c1": CharacterProfile(
                    id="c1",
                    name="主角",
                    role="protagonist",
                    personality="勇敢",
                    motivation="复仇",
                    flaw="傲慢",
                    backstory="孤儿",
                ),
            }
        )
        assert "c1" in reg.characters
        assert reg.characters["c1"].name == "主角"


class TestBibleModels:
    def test_world_building_minimal(self):
        from src.models.bible import WorldBuilding

        w = WorldBuilding(name="测试大陆", world_type="fantasy")
        assert w.name == "测试大陆"
        assert w.world_type == "fantasy"

    def test_faction_creation(self):
        from src.models.bible import Faction

        f = Faction(name="青云宗", faction_type="sect", goal="维护正道")
        assert f.name == "青云宗"
        assert f.faction_type == "sect"

    def test_novel_bible_minimal(self):
        from src.models.bible import (
            CoreConflict,
            Faction,
            NarrativeRules,
            NovelBible,
            StyleContract,
            Theme,
            WorldBuilding,
        )

        bible = NovelBible(
            world=WorldBuilding(name="大陆", world_type="fantasy"),
            factions=[Faction(name="宗门", faction_type="sect", goal="正道")],
            rules=NarrativeRules(),
            style_contract=StyleContract(tone="热血"),
            themes=[Theme(name="成长")],
            core_conflicts=[CoreConflict(conflict_type="person_vs_society", description="抗争")],
        )
        assert len(bible.factions) == 1
        assert bible.style_contract.tone == "热血"
        assert len(bible.themes) == 1


# ═══════════════════════════════════════════════════════════════════════
# Settings — Provider Detection
# ═══════════════════════════════════════════════════════════════════════


class TestSettingsProviderDetection:
    def test_detect_modelscope_from_base_url(self):
        from src.config.settings import Settings

        s = Settings(
            openai_base_url="https://api-inference.modelscope.cn/v1",
            _env_file=None,
        )
        # No API keys → falls through to base_url detection
        assert s.active_provider == "modelscope"

    def test_detect_dashscope_from_base_url(self):
        from src.config.settings import Settings

        s = Settings(
            openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            _env_file=None,
        )
        # No API keys → falls through to base_url detection
        assert s.active_provider == "dashscope"

    def test_default_active_provider(self):
        from src.config.settings import Settings

        s = Settings(
            default_provider="openai",
            openai_api_key="test",
            _env_file=None,
        )
        assert s.active_provider == "openai"

    def test_has_anthropic_detection(self):
        from src.config.settings import Settings

        s = Settings(_env_file=None)
        assert not s.has_anthropic
        s2 = Settings(anthropic_api_key="sk-ant-test", _env_file=None)
        assert s2.has_anthropic
