"""Content safety detection for Writer output.

Implements the evaluation report's 4-week requirement #3:
Basic content safety keyword/pattern filtering aligned with target
platform (起点/番茄/飞卢) content review standards.

Checks:
- Explicit sexual content / erotica
- Excessive violence / gore
- Political sensitivity keywords
- Platform-specific banned terms

Design: Deterministic (zero LLM cost), runs before chapter is
presented to the user or saved to disk. Returns a SafetyResult
with pass/fail and flagged segments for human review.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SafetyFlag:
    """A single content safety flag with location info."""

    category: str  # 'sexual', 'violence', 'political', 'platform'
    severity: str  # 'warning', 'block'
    keyword: str  # The triggering term
    context: str  # Surrounding text (50 chars around the match)
    position: int  # Character position in text

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "keyword": self.keyword,
            "context": self.context,
            "position": self.position,
        }


@dataclass
class SafetyResult:
    """Result of content safety check."""

    passed: bool = True
    flags: list[SafetyFlag] = field(default_factory=list)
    warning_count: int = 0
    block_count: int = 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "warning_count": self.warning_count,
            "block_count": self.block_count,
            "flags": [f.to_dict() for f in self.flags],
        }

    @property
    def summary(self) -> str:
        if not self.flags:
            return "Content safety check passed — no issues found"
        parts = [f"Content safety: {len(self.flags)} flag(s)"]
        if self.block_count:
            parts.append(f"{self.block_count} blocking")
        if self.warning_count:
            parts.append(f"{self.warning_count} warning(s)")
        return ", ".join(parts)


class ContentSafetyChecker:
    """Checks chapter content against platform content safety standards.

    Uses deterministic keyword and pattern matching — zero LLM cost.
    Designed to be extensible: add keywords to the category lists below.

    Usage:
        checker = ContentSafetyChecker()
        result = checker.check(chapter_text)
        if not result.passed:
            print(result.summary)
            for flag in result.flags:
                print(f"  [{flag.severity}] {flag.category}: {flag.keyword}")
    """

    # ── Configuration ─────────────────────────────────────────────

    # Context window size (chars before/after match)
    CONTEXT_WINDOW = 50

    # Blocking keywords — chapter is rejected if any of these match
    BLOCK_KEYWORDS: dict[str, list[str]] = {
        "sexual": [
            # Explicit sexual acts (Chinese)
            "做爱",
            "性交",
            "口交",
            "肛交",
            "强奸",
            "轮奸",
            "乱伦",
            "淫乱",
            "性奴",
            "性虐待",
            # Explicit sexual acts (English/romanized)
        ],
        "violence": [
            # Extreme gore / torture
            "肢解",
            "剥皮",
            "活体解剖",
            "凌迟",
        ],
        "political": [
            # Politically sensitive — platform ban risk
            "法轮功",
            "六四",
            "天安门事件",
        ],
    }

    # Warning keywords — flagged for review but don't block
    WARNING_KEYWORDS: dict[str, list[str]] = {
        "sexual": [
            # Suggestive but not explicit
            "酥胸",
            "玉体",
            "胴体",
            "香艳",
            "春宵",
            "翻云覆雨",
            "巫山云雨",
            "鱼水之欢",
            # Platform-specific borderline terms
            "双修",
            "采补",
            "炉鼎",
        ],
        "violence": [
            # Graphic violence descriptors
            "血肉模糊",
            "脑浆迸裂",
            "开膛破肚",
            "尸横遍野",
            "血流成河",
        ],
        "platform": [
            # Terms that trigger platform auto-review
            "色情",
            "黄色小说",
            "成人内容",
        ],
    }

    # ── Public API ────────────────────────────────────────────────

    def check(self, text: str, chapter_number: int = 0) -> SafetyResult:
        """Run content safety check on a chapter.

        Args:
            text: The chapter content to check.
            chapter_number: Chapter number for logging context.

        Returns:
            SafetyResult with pass/fail and any flagged segments.
        """
        result = SafetyResult()

        # Check blocking keywords first
        for category, keywords in self.BLOCK_KEYWORDS.items():
            for keyword in keywords:
                matches = self._find_matches(text, keyword)
                for pos in matches:
                    flag = SafetyFlag(
                        category=category,
                        severity="block",
                        keyword=keyword,
                        context=self._extract_context(text, pos, len(keyword)),
                        position=pos,
                    )
                    result.flags.append(flag)
                    result.block_count += 1

        # Check warning keywords
        for category, keywords in self.WARNING_KEYWORDS.items():
            for keyword in keywords:
                matches = self._find_matches(text, keyword)
                for pos in matches:
                    flag = SafetyFlag(
                        category=category,
                        severity="warning",
                        keyword=keyword,
                        context=self._extract_context(text, pos, len(keyword)),
                        position=pos,
                    )
                    result.flags.append(flag)
                    result.warning_count += 1

        # Chapter passes if no block-level flags
        result.passed = result.block_count == 0

        if result.flags:
            logger.warning(f"Chapter {chapter_number}: {result.summary}")

        return result

    def check_with_correction(self, text: str, chapter_number: int = 0) -> tuple[SafetyResult, str | None]:
        """Check content and optionally suggest corrected text.

        For warning-level flags, replaces flagged terms with [内容已编辑]
        markers. For block-level flags, returns None for corrected text
        (chapter must be rewritten).

        Returns:
            Tuple of (SafetyResult, corrected_text_or_None).
        """
        result = self.check(text, chapter_number)

        if not result.passed:
            # Blocking issues — cannot auto-correct
            return result, None

        if result.warning_count == 0:
            return result, text

        # Auto-redact warning-level flagged terms
        corrected = text
        for flag in sorted(result.flags, key=lambda f: f.position, reverse=True):
            if flag.severity == "warning":
                start = flag.position
                end = start + len(flag.keyword)
                corrected = corrected[:start] + "[内容已编辑]" + corrected[end:]

        return result, corrected

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _find_matches(text: str, keyword: str) -> list[int]:
        """Find all positions of a keyword in text (case-insensitive)."""
        positions = []
        start = 0
        text_lower = text.lower()
        kw_lower = keyword.lower()
        while True:
            pos = text_lower.find(kw_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    @classmethod
    def _extract_context(cls, text: str, pos: int, kw_len: int) -> str:
        """Extract surrounding text around a match for context."""
        start = max(0, pos - cls.CONTEXT_WINDOW)
        end = min(len(text), pos + kw_len + cls.CONTEXT_WINDOW)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"


# Convenience function
def check_content_safety(text: str, chapter_number: int = 0) -> SafetyResult:
    """Quick content safety check. Returns SafetyResult."""
    checker = ContentSafetyChecker()
    return checker.check(text, chapter_number)
