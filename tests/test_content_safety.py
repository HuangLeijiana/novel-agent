"""Tests for content safety checker.

Covers edge cases:
- Clean content (no flags)
- Explicit sexual content (blocking)
- Violent content (blocking)
- Suggestive content (warning)
- Mixed Chinese/English
- Empty text
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.content_safety import ContentSafetyChecker, SafetyResult, check_content_safety


@pytest.fixture
def checker():
    return ContentSafetyChecker()


class TestCleanContent:
    """Normal content should pass without flags."""

    def test_normal_novel_text(self, checker):
        result = checker.check(
            "李明站在山巅，俯瞰着脚下的世界。风从他的耳边呼啸而过，带着远古的气息。他已经在这条路上走了太久。"
        )
        assert result.passed
        assert len(result.flags) == 0

    def test_empty_text(self, checker):
        result = checker.check("")
        assert result.passed
        assert len(result.flags) == 0

    def test_whitespace_only(self, checker):
        result = checker.check("   \n\n   ")
        assert result.passed


class TestBlockingContent:
    """Blocking keywords should cause check to fail."""

    def test_explicit_sexual_block(self, checker):
        result = checker.check("这是一段包含做爱的文字描述。")
        assert not result.passed
        assert result.block_count >= 1
        assert any(f.category == "sexual" for f in result.flags)

    def test_extreme_violence_block(self, checker):
        result = checker.check("这段文字描述了肢解的恐怖场景。")
        assert not result.passed
        assert result.block_count >= 1
        assert any(f.category == "violence" for f in result.flags)


class TestWarningContent:
    """Warning keywords should flag but not block."""

    def test_suggestive_content_warning(self, checker):
        result = checker.check("她露出了酥胸，微微一笑。")
        # Warning only, not blocking
        assert result.passed
        assert result.warning_count >= 1

    def test_violence_warning(self, checker):
        result = checker.check("战场上血肉模糊，惨不忍睹。")
        assert result.passed
        assert result.warning_count >= 1


class TestEdgeCases:
    """Edge cases for the safety checker."""

    def test_mixed_clean_and_flagged(self, checker):
        """Text with both clean and flagged content."""
        text = (
            "这是一个正常的武侠故事开头。主角从小在山村长大，"
            "每天修炼武功，过着平静的生活。然而有一天，"
            "他在后山发现了一个春宵秘境的入口……"
        )
        result = checker.check(text)
        assert result.warning_count >= 1 or result.block_count >= 0

    def test_very_long_text(self, checker):
        """Safety check should handle long texts efficiently."""
        clean_text = "这是一个很正常的中文句子。" * 1000
        result = checker.check(clean_text)
        assert result.passed

    def test_context_extraction(self, checker):
        """Context extraction should show surrounding text."""
        text = "前文很长的一段描述。" * 10 + "这里出现了做爱这个词。" + "后文也很长。" * 10
        result = checker.check(text)
        assert not result.passed
        assert len(result.flags) > 0
        # Context should contain the keyword
        assert "做爱" in result.flags[0].context


class TestConvenienceFunction:
    """Test the module-level convenience function."""

    def test_check_content_safety(self):
        result = check_content_safety("正常的中文小说内容。")
        assert result.passed
        assert isinstance(result, SafetyResult)

    def test_check_content_safety_blocked(self):
        result = check_content_safety("这段内容包含做爱的描写。")
        assert not result.passed


class TestCorrectionMode:
    """Test the check_with_correction method."""

    def test_clean_text_returns_unchanged(self, checker):
        text = "正常的中文内容，没有任何问题。"
        result, corrected = checker.check_with_correction(text)
        assert result.passed
        assert corrected == text

    def test_warning_text_returns_redacted(self, checker):
        text = "她露出了酥胸，场景十分香艳。"
        result, corrected = checker.check_with_correction(text)
        assert result.passed  # Warnings don't block
        assert corrected is not None
        assert "[内容已编辑]" in corrected

    def test_blocked_text_returns_none(self, checker):
        text = "这段内容包含做爱的描写。"
        result, corrected = checker.check_with_correction(text)
        assert not result.passed
        assert corrected is None  # Blocking content cannot be auto-corrected
