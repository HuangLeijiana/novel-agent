"""Chapter Inspector — render-and-inspect verification for chapter drafts.

Inspired by ai-job-search-master's compile-and-inspect pattern:
  LaTeX → compile PDF → visual inspection → ATS text verification → iterate

Applied to novel writing:
  Draft → structural analysis → quality metrics → auto-flag → optional auto-revise

Checks performed:
  - Word count target (±15% tolerance)
  - Dialogue ratio (web novel target: 30-50%)
  - Paragraph density (no walls of text)
  - Chapter-end hook presence
  - Scene count vs plan
  - AI-flavor marker detection
  - Pacing variety (emotional curve spread)

This is a DETERMINISTIC checker — no LLM calls. It runs fast and catches
mechanical issues before the expensive LLM review phase.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft
from ..models.outline import ChapterPlan

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================

@dataclass
class InspectionResult:
    """Result of chapter structural inspection."""
    passed: bool = True
    word_count: int = 0
    target_word_count: int = 3000
    word_count_ok: bool = True
    word_count_deviation: float = 0.0  # e.g. 0.15 = 15% over

    dialogue_ratio: float = 0.0
    dialogue_ok: bool = True

    paragraph_count: int = 0
    avg_paragraph_length: float = 0.0
    max_paragraph_length: int = 0
    wall_of_text_count: int = 0  # paragraphs > 500 chars

    has_ending_hook: bool = True
    ending_hook_quality: str = ""  # "strong", "weak", "missing"

    scene_count_actual: int = 0
    scene_count_planned: int = 0

    emotional_spread: float = 0.0  # 0=flat, 1=good variety

    ai_markers: list[str] = field(default_factory=list)
    ai_marker_count: int = 0

    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================
# AI-flavor marker patterns
# ============================================================

AI_FLAVOR_PATTERNS: list[tuple[str, str]] = [
    (r'综上所述', '机械过渡词"综上所述"'),
    (r'值得注意的是', '机械表达"值得注意的是"'),
    (r'在.+的过程中', 'AI常用句式"在...的过程中"'),
    (r'不仅.+而且', 'AI常用关联词"不仅...而且"'),
    (r'然而\s*，', 'AI常用转折"然而"'),
    (r'因此\s*，', 'AI常用因果"因此"'),
    (r'与此同时\s*，', 'AI常用过渡"与此同时"'),
    (r'总而言之', '机械总结"总而言之"'),
    (r'从某种(?:意义|程度)上来说', 'AI常用模糊表达'),
    (r'不可否认的是', 'AI常用强调'),
    (r'可以(?:说|理解为|看作是)', 'AI常用解释句式'),
    (r'他感到一阵', 'AI常用情感描写"感到一阵"'),
    (r'他的(?:心中|内心|心里)', 'AI常用内心独白引入'),
    (r'显得(?:格外|异常|十分)', 'AI常用程度修饰'),
    (r'仿佛', 'AI常用比喻引入"仿佛"（检查是否滥用）'),
]


# ============================================================
# Inspector
# ============================================================

class ChapterInspector:
    """Deterministic structural inspector for chapter drafts.

    Usage:
        inspector = ChapterInspector()
        result = inspector.inspect(draft, chapter_plan, bible)
        if not result.passed:
            print(f"Found {len(result.issues)} issues to fix")
    """

    # ── Configuration ──────────────────────────────────────────

    WORD_COUNT_TOLERANCE: float = 0.15       # ±15%
    MIN_DIALOGUE_RATIO: float = 0.20         # 20% minimum
    MAX_DIALOGUE_RATIO: float = 0.60         # 60% maximum (web novel target: 30-50%)
    IDEAL_DIALOGUE_MIN: float = 0.30
    IDEAL_DIALOGUE_MAX: float = 0.50
    MAX_PARAGRAPH_LENGTH: int = 500          # Characters before "wall of text"
    MIN_ENDING_HOOK_LENGTH: int = 50         # Last paragraph should be > 50 chars
    AI_MARKER_WARN_THRESHOLD: int = 2        # Flag if ≥ this many AI markers found
    AI_MARKER_FAIL_THRESHOLD: int = 5

    def inspect(
        self,
        draft: ChapterDraft,
        chapter_plan: Optional[ChapterPlan] = None,
        bible: Optional[NovelBible] = None,
    ) -> InspectionResult:
        """Run all structural checks on a chapter draft.

        Args:
            draft: The chapter draft to inspect.
            chapter_plan: Optional chapter plan for comparison.
            bible: Optional novel bible for style contract reference.

        Returns:
            InspectionResult with all check outcomes.
        """
        result = InspectionResult()
        content = draft.content

        # ── 1. Word count ──
        result.word_count = draft.word_count or len(content.replace(" ", "").replace("\n", ""))
        if chapter_plan:
            result.target_word_count = chapter_plan.word_count_target
        result.word_count_deviation = (
            (result.word_count - result.target_word_count) / result.target_word_count
            if result.target_word_count > 0 else 0
        )
        result.word_count_ok = abs(result.word_count_deviation) <= self.WORD_COUNT_TOLERANCE
        if not result.word_count_ok:
            direction = "超" if result.word_count_deviation > 0 else "不足"
            result.issues.append(
                f"字数{direction}标：目标{result.target_word_count}字，实际{result.word_count}字 "
                f"({result.word_count_deviation:+.0%})"
            )

        # ── 2. Dialogue ratio ──
        result.dialogue_ratio = self._estimate_dialogue_ratio(content)
        if result.dialogue_ratio < self.MIN_DIALOGUE_RATIO:
            result.dialogue_ok = False
            result.warnings.append(
                f"对话比例偏低（{result.dialogue_ratio:.0%}），建议≥{self.MIN_DIALOGUE_RATIO:.0%}"
            )
        elif result.dialogue_ratio > self.MAX_DIALOGUE_RATIO:
            result.dialogue_ok = False
            result.warnings.append(
                f"对话比例偏高（{result.dialogue_ratio:.0%}），建议≤{self.MAX_DIALOGUE_RATIO:.0%}"
            )
        elif result.dialogue_ratio < self.IDEAL_DIALOGUE_MIN:
            result.warnings.append(
                f"对话比例略低（{result.dialogue_ratio:.0%}），网文推荐{self.IDEAL_DIALOGUE_MIN:.0%}-{self.IDEAL_DIALOGUE_MAX:.0%}"
            )

        # ── 3. Paragraph density ──
        paragraphs = self._split_paragraphs(content)
        result.paragraph_count = len(paragraphs)
        if paragraphs:
            lengths = [len(p) for p in paragraphs]
            result.avg_paragraph_length = sum(lengths) / len(lengths)
            result.max_paragraph_length = max(lengths)
            result.wall_of_text_count = sum(1 for l in lengths if l > self.MAX_PARAGRAPH_LENGTH)
            if result.wall_of_text_count > 0:
                result.warnings.append(
                    f"发现{result.wall_of_text_count}个超长段落（>{self.MAX_PARAGRAPH_LENGTH}字），"
                    f"建议拆分以提升移动端阅读体验"
                )
            if result.avg_paragraph_length > 300:
                result.warnings.append(
                    f"平均段落长度{result.avg_paragraph_length:.0f}字偏长，"
                    f"网文建议控制在200-300字/段"
                )

        # ── 4. Chapter-end hook ──
        result.has_ending_hook, result.ending_hook_quality = self._check_ending_hook(content)
        if not result.has_ending_hook:
            result.issues.append("章末缺少钩子：最后一段未能制造悬念或期待")
        elif result.ending_hook_quality == "weak":
            result.warnings.append("章末钩子偏弱，建议增强悬念感或信息增量")

        # ── 5. Scene count vs plan ──
        if chapter_plan:
            result.scene_count_planned = len(chapter_plan.scenes)
            result.scene_count_actual = self._count_scenes(content)
            if result.scene_count_actual < result.scene_count_planned * 0.5:
                result.warnings.append(
                    f"场景数不足：规划{result.scene_count_planned}个，实际约{result.scene_count_actual}个"
                )

        # ── 6. AI-flavor markers ──
        result.ai_markers = self._detect_ai_markers(content)
        result.ai_marker_count = len(result.ai_markers)
        if result.ai_marker_count >= self.AI_MARKER_FAIL_THRESHOLD:
            result.issues.append(
                f"AI味严重：检测到{result.ai_marker_count}处AI常用表达"
            )
        elif result.ai_marker_count >= self.AI_MARKER_WARN_THRESHOLD:
            result.warnings.append(
                f"检测到{result.ai_marker_count}处AI常用表达，建议润色"
            )

        # ── Determine overall pass/fail ──
        result.passed = len(result.issues) == 0

        if not result.passed:
            logger.info(
                f"Chapter {draft.chapter_number} inspection: FAILED — "
                f"{len(result.issues)} issues, {len(result.warnings)} warnings"
            )
        else:
            logger.info(
                f"Chapter {draft.chapter_number} inspection: PASSED "
                f"({len(result.warnings)} minor warnings)"
            )

        return result

    # ================================================================
    # Analysis helpers
    # ================================================================

    @staticmethod
    def _estimate_dialogue_ratio(content: str) -> float:
        """Estimate dialogue ratio by counting quoted text.

        Handles Chinese quotes 「」"" and Western quotes "".
        """
        if not content:
            return 0.0

        # Count characters inside Chinese dialogue markers
        dialogue_chars = 0
        # Chinese dialogue: 「...」 or "..."
        for pattern in [r'「[^」]*」', r'『[^』]*』', r'"[^"]*"', r'"[^"]*"']:
            for match in re.finditer(pattern, content):
                dialogue_chars += len(match.group())

        # Also count lines that start with character name patterns (play-style dialogue)
        # 角色名：... or 角色名:...
        name_dialogue = re.findall(r'(?:^|\n)\S{1,4}[：:][^\n]{10,}', content)
        for line in name_dialogue:
            dialogue_chars += len(line)

        total_chars = len(content.replace("\n", "").replace(" ", ""))
        if total_chars == 0:
            return 0.0

        ratio = dialogue_chars / total_chars
        return min(ratio, 1.0)  # Cap at 1.0

    @staticmethod
    def _split_paragraphs(content: str) -> list[str]:
        """Split content into paragraphs (non-empty lines)."""
        return [p.strip() for p in content.split('\n') if p.strip() and p.strip() != '***']

    @staticmethod
    def _check_ending_hook(content: str) -> tuple[bool, str]:
        """Check if the chapter has a strong ending hook.

        Returns (has_hook, quality).
        """
        if not content:
            return False, "missing"

        # Get last meaningful paragraph(s)
        paragraphs = [p.strip() for p in content.split('\n') if p.strip() and p.strip() != '***']
        if not paragraphs:
            return False, "missing"

        # Analyze last 2 paragraphs
        last = paragraphs[-1]
        last_two = ' '.join(paragraphs[-2:]) if len(paragraphs) >= 2 else last

        # Strong hook indicators
        strong_patterns = [
            r'[？?]',           # Question ending
            r'[！!]',           # Exclamation
            r'突然|忽然|就在这时|正在这时|猛地',
            r'却|竟然|居然|没想到|不料',
            r'下一[步个秒刻章]|明天|即将|将要|马上',
            r'暗[中处]|阴影|背后|秘密|真相',
            r'危险|危机|杀[意气机]|死亡',
            r'冷笑|诡异|神秘|未知|奇怪',
            r'难道|莫非|难道说',
            r'……$|\.{3}$',     # Ellipsis ending
            r'嘴角|眼神|目光',  # Micro-expression hook
        ]

        # Weak ending indicators
        weak_patterns = [
            r'^(?:就这样|于是|然后|接着|之后)',
            r'^(?:他|她|它)(?:想|觉得|知道|明白)',
        ]

        strong_count = sum(1 for p in strong_patterns if re.search(p, last_two))
        weak_count = sum(1 for p in weak_patterns if re.search(p, last))

        if len(last) < ChapterInspector.MIN_ENDING_HOOK_LENGTH:
            return False, "missing"

        if strong_count >= 3:
            return True, "strong"
        elif strong_count >= 1 and weak_count == 0:
            return True, "weak"
        elif weak_count > 0:
            return False, "missing"
        else:
            return True, "weak"

    @staticmethod
    def _count_scenes(content: str) -> int:
        """Count scene breaks (*** or --- markers)."""
        scene_breaks = re.findall(r'^(?:\*{3,}|-{3,})$', content, re.MULTILINE)
        return len(scene_breaks) + 1  # N breaks = N+1 scenes

    @staticmethod
    def _detect_ai_markers(content: str) -> list[str]:
        """Detect AI-generated text markers in the content.

        Returns list of marker descriptions found.
        """
        found = []
        for pattern, description in AI_FLAVOR_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                found.append(f"{description}（{len(matches)}次）")
        return found

    def format_report(self, result: InspectionResult) -> str:
        """Format inspection result as a human-readable report."""
        lines = [
            "=" * 50,
            "章节结构检查报告",
            "=" * 50,
            "",
            f"字数：{result.word_count}/{result.target_word_count} "
            f"({'[OK] 达标' if result.word_count_ok else '[FAIL] 偏差' + f'{result.word_count_deviation:+.0%}'}) ",
            f"对话比例：{result.dialogue_ratio:.0%} "
            f"({'[OK]' if result.dialogue_ok else '[WARN]'})",
            f"段落数：{result.paragraph_count} | 平均{result.avg_paragraph_length:.0f}字/段",
            f"超长段落：{result.wall_of_text_count}个",
            f"章末钩子：{'[OK] 强' if result.ending_hook_quality == 'strong' else '[WARN] 弱' if result.ending_hook_quality == 'weak' else '[FAIL] 缺失'}",

            f"场景数：约{result.scene_count_actual}个{' (规划' + str(result.scene_count_planned) + '个)' if result.scene_count_planned else ''}",
            f"AI味标记：{result.ai_marker_count}处",
        ]

        if result.ai_markers:
            lines.append("")
            lines.append("AI味详情：")
            for marker in result.ai_markers:
                lines.append(f"  - {marker}")

        if result.issues:
            lines.append("")
            lines.append("[FAIL] 必须修复：")
            for issue in result.issues:
                lines.append(f"  - {issue}")

        if result.warnings:
            lines.append("")
            lines.append("[WARN] 建议优化：")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        lines.append("")
        lines.append(f"总结：{'[OK] 通过' if result.passed else '[FAIL] 未通过，建议修订后重新检查'}")
        lines.append("=" * 50)

        return "\n".join(lines)
