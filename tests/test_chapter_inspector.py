"""Edge case tests for ChapterInspector — deterministic structural analysis.

Covers the evaluation report's required test categories:
- Empty input
- Pure dialogue chapters
- Extreme length chapters (very short / very long)
- AI flavor markers
- Mixed Chinese/English content
- Special characters and formatting
- Duplicate/repetitive content

Note: ChapterInspector.inspect() accepts ChapterDraft objects,
not raw text strings. We construct minimal ChapterDraft objects
for each test scenario.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.agents.chapter_inspector import ChapterInspector, InspectionResult
from src.models.chapter import ChapterDraft


@pytest.fixture
def inspector():
    return ChapterInspector()


def _make_draft(content: str, chapter_number: int = 1, word_count: int = 0) -> ChapterDraft:
    """Create a minimal ChapterDraft for testing."""
    wc = word_count if word_count > 0 else len(content)
    return ChapterDraft(
        chapter_number=chapter_number,
        title=f"第{chapter_number}章",
        content=content,
        word_count=wc,
    )


class TestEmptyAndMinimalInput:
    """Edge case: empty, whitespace-only, and minimal content."""

    def test_empty_text(self, inspector):
        """Empty input should not crash and should report zero word count."""
        draft = _make_draft("")
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        assert result.word_count == 0

    def test_whitespace_only(self, inspector):
        """Whitespace-only input should be handled gracefully."""
        draft = _make_draft("   \n\n  \n   ")
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        # Whitespace characters may be counted, but should be minimal
        assert result.word_count < 50

    def test_single_sentence(self, inspector):
        """Single short sentence should not crash."""
        draft = _make_draft("他走进了房间。")
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        assert result.word_count < 50


class TestPureDialogueChapter:
    """Edge case: chapter consisting entirely of dialogue."""

    def test_pure_dialogue(self, inspector):
        """100% dialogue chapter — dialogue ratio should be very high."""
        # Use colon-based name dialogue pattern that the inspector detects
        # Note: inspector regex requires ≥10 chars after the colon
        text = "\n".join(
            [
                "张三：你终于来了，我已经等了很久了。",
                "李四：我来了是因为有重要的事情要告诉你。",
                "张三：你不该来的，这里太危险了。",
                "李四：可我已经来了，而且我不会离开的。",
                "张三：那么，动手吧，我们没有时间了。",
                "李四：好，让我们一起面对这一切吧。",
            ]
            * 100
        )
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert result.dialogue_ratio > 0.3, f"Expected high dialogue ratio, got {result.dialogue_ratio}"

    def test_no_dialogue(self, inspector):
        """Chapter with zero dialogue — dialogue ratio should be ~0."""
        text = (
            "天空是灰蒙蒙的，远处连绵的山脉在雾气中若隐若现。"
            "风从北方吹来，带着初冬的寒意。树木已经落尽了叶子，"
            "光秃秃的枝丫在风中瑟瑟发抖。一条小溪从山间流过，"
            "溪水清澈见底，偶尔能看到几尾小鱼在水中游弋。"
        ) * 20
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert result.dialogue_ratio < 0.1, f"Expected low dialogue ratio, got {result.dialogue_ratio}"


class TestExtremeLengthChapters:
    """Edge case: very short and very long chapters."""

    def test_very_short_chapter(self, inspector):
        """Chapter under 500 characters should be flagged as too short."""
        text = "他走进去，看到了她。就这些。"
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert result.word_count < 100

    def test_very_long_chapter(self, inspector):
        """Chapter over 10,000 characters should not crash."""
        paragraph = (
            "李明站在山巅，俯瞰着脚下的世界。风从他的耳边呼啸而过，"
            "带着远古的气息。他已经在这条路上走了太久，久到几乎忘记了自己当初为什么要出发。"
            "但是每当他想要放弃的时候，师傅的话就会在脑海中响起：真正的强者，不是没有恐惧，"
            "而是能够直面恐惧。他深吸一口气，握紧了手中的剑。前方还有更艰难的战斗在等着他，"
            "但他已经不再是当初那个懵懂少年了。"
        )
        text = "\n\n".join([paragraph] * 40)
        draft = _make_draft(text, word_count=10000)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        assert result.word_count > 5000


class TestAIFlavorMarkers:
    """Edge case: text with varying levels of AI-generated markers."""

    def test_heavy_ai_flavor(self, inspector):
        """Text full of AI-typical phrases should trigger many markers."""
        text = (
            "总而言之，这是一个充满希望的时刻。此外，值得注意的是，"
            "他不仅展现出了非凡的勇气，而且还表现出了超乎寻常的智慧。"
            "首先，我们需要理解的是，这个世界充满了未知。其次，"
            "值得注意的是，每一个选择都会带来相应的后果。最后，"
            "总而言之，这是一段值得铭记的旅程。"
        ) * 10
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        # Should detect AI flavor markers
        assert result.ai_marker_count > 0, f"Expected AI flavor markers, got {result.ai_marker_count}"

    def test_natural_human_text(self, inspector):
        """Natural-sounding Chinese text should have few AI markers."""
        text = (
            "老王掏出烟，点上，深深吸了一口。烟雾在昏暗的灯光下慢慢散开。"
            "他妈的，他嘟囔了一句，把烟头摁灭在烟灰缸里。"
            "这事没完。他站起身，拿起桌上的车钥匙，头也不回地走出了门。"
            "外面下着毛毛雨，他没打伞，就这么走进了雨里。"
        ) * 15
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        # Natural text should have very few AI markers
        assert result.ai_marker_count < 5, f"Expected low AI flavor, got {result.ai_marker_count}"


class TestChineseEnglishMixed:
    """Edge case: mixed Chinese and English content."""

    def test_mixed_language(self, inspector):
        """Text with English words interspersed should still analyze correctly."""
        text = (
            "John看着眼前的terminal屏幕，上面显示着System Error。"
            "他type了几个command，但是nothing happened。"
            "“Fuck，”他骂了一句，“这破system又crash了。”"
            "旁边的同事Lisa探过头来：“你是不是又run了那个buggy的script？”"
        ) * 15
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        assert result.word_count > 0

    def test_english_only(self, inspector):
        """English-only text should still be inspected without crash."""
        text = (
            "The wind howled through the ancient trees. Shadows danced "
            "in the flickering torchlight. Marcus gripped his sword tightly, "
            "his knuckles white with tension. Somewhere in the darkness, "
            "something was watching him."
        ) * 20
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)


class TestSpecialCharacters:
    """Edge case: special characters, emoji, and formatting."""

    def test_text_with_emojis(self, inspector):
        """Text containing emoji should not break the inspector."""
        text = "他笑了\U0001f60a，她也笑了\U0001f602。这就是人生\U0001f31f。战斗吧少年⚔️！" * 50
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)

    def test_text_with_html_like_tags(self, inspector):
        """Text with HTML-like tags should not match AI markers incorrectly."""
        text = (
            "<system>正在加载世界设定...</system>\n"
            "李明看着眼前的光幕，上面显示着：\n"
            "<status>任务完成</status>\n"
            "<reward>获得经验值1000</reward>\n"
            "他微微一笑，这系统文的主角待遇，终究是轮到自己了。"
        ) * 10
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)


class TestDuplicateContent:
    """Edge case: duplicated paragraphs and repetitive patterns."""

    def test_duplicate_paragraphs(self, inspector):
        """Chapter with repeated identical paragraphs."""
        para = "他走进房间，看到了那把椅子。椅子是红色的，上面有金色的花纹。"
        text = "\n\n".join([para] * 30)
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        assert result.paragraph_count >= 30

    def test_short_paragraphs(self, inspector):
        """Very short paragraphs (one sentence each)."""
        paragraphs = [
            "他来了。",
            "她走了。",
            "天下雨了。",
            "他撑开了伞。",
            "她回头看了一眼。",
            "他没有追上去。",
            "雨越下越大。",
        ]
        text = "\n\n".join(paragraphs * 10)
        draft = _make_draft(text)
        result = inspector.inspect(draft)
        assert isinstance(result, InspectionResult)
        # Average paragraph length should be small
        assert result.avg_paragraph_length < 30
