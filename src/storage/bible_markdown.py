"""Bible Markdown Store — thin-pointer modular bible files.

Inspired by ai-job-search-master's modular profile system:
  01-candidate-profile.md  ← hard facts only
  02-behavioral-profile.md ← soft skills only
  03-writing-style.md      ← tone & voice rules
  ...

Applied to novel writing:
  bible/
  ├── 01-world-building.md    ← World setting (geography, history, rules)
  ├── 02-factions.md          ← Factions & organizations
  ├── 03-writing-style.md     ← Style contract (tone, sentence style, forbidden phrases)
  ├── 04-themes.md            ← Themes & core conflicts
  ├── 05-pleasure-points.md   ← Pleasure point model & narrative constraints
  └── 06-characters.md        ← Character registry (thin reference)

Key principle: NO INFORMATION DUPLICATION across files. Each fact lives in
exactly one file. Agents load only the files they need for their task.
"""

import logging
from datetime import datetime
from pathlib import Path

from ..models.bible import (
    CoreConflict,
    Faction,
    NovelBible,
    StyleContract,
    Theme,
    WorldBuilding,
)
from ..models.characters import CharacterRegistry

logger = logging.getLogger(__name__)

# File name constants
FILE_WORLD = "01-world-building.md"
FILE_FACTIONS = "02-factions.md"
FILE_STYLE = "03-writing-style.md"
FILE_THEMES = "04-themes.md"
FILE_PLEASURE = "05-pleasure-points.md"
FILE_CHARACTERS = "06-characters.md"

ALL_BIBLE_FILES = [
    FILE_WORLD,
    FILE_FACTIONS,
    FILE_STYLE,
    FILE_THEMES,
    FILE_PLEASURE,
    FILE_CHARACTERS,
]


class BibleMarkdownStore:
    """Reads and writes bible components as modular Markdown files.

    Each file is self-contained and can be loaded independently by agents.
    This enables the thin-pointer pattern: agents load only the specific
    files they need, reducing token waste and preventing information drift.

    Usage:
        store = BibleMarkdownStore(project_dir)
        store.save_bible(bible, characters)
        # ... later ...
        world = store.load_world()
        style = store.load_style()
    """

    def __init__(self, project_dir: Path):
        self._bible_dir = project_dir / "bible"
        self._bible_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────

    def save_bible(
        self,
        bible: NovelBible,
        characters: CharacterRegistry | None = None,
    ) -> None:
        """Save all bible components as modular Markdown files."""
        self._save_world(bible.world)
        self._save_factions(bible.factions)
        self._save_style(bible.style_contract)
        self._save_themes(bible.themes, bible.core_conflicts)
        self._save_pleasure_points(bible.pleasure_point_model, bible.narrative_constraints)
        if characters:
            self._save_characters(characters)
        logger.info(f"Bible saved to {self._bible_dir} ({len(ALL_BIBLE_FILES)} files)")

    def _save_world(self, world: WorldBuilding) -> None:
        """Save world building as Markdown."""
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: world-building",
            "---",
            "",
            f"# 世界观：{world.name}",
            "",
            f"**世界类型**：{world.world_type}",
            "",
            "## 地理环境",
            "",
            world.geography or "（待补充）",
            "",
            "## 关键历史",
            "",
            world.history or "（待补充）",
            "",
            "## 文化习俗",
            "",
            world.culture or "（待补充）",
            "",
            f"**科技/魔法水平**：{world.technology_level or '待定'}",
            "",
        ]
        if world.magic_system:
            lines.extend(
                [
                    "## 魔法/力量体系",
                    "",
                    world.magic_system,
                    "",
                ]
            )
        if world.power_progression:
            lines.extend(
                [
                    "## 力量进阶体系",
                    "",
                    world.power_progression,
                    "",
                ]
            )
        if world.special_rules:
            lines.extend(
                [
                    "## 特殊世界规则",
                    "",
                ]
            )
            for rule in world.special_rules:
                lines.append(f"- {rule}")
            lines.append("")

        self._write(FILE_WORLD, "\n".join(lines))

    def _save_factions(self, factions: list[Faction]) -> None:
        """Save factions as Markdown."""
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: factions",
            "---",
            "",
            f"# 势力与组织（共{len(factions)}个）",
            "",
        ]

        for i, f in enumerate(factions, 1):
            lines.extend(
                [
                    f"## {i}. {f.name}",
                    "",
                    f"- **ID**：{f.id}",
                    f"- **类型**：{f.type}",
                    f"- **目标**：{f.goal}",
                    f"- **层级结构**：{f.hierarchy or '待定'}",
                    f"- **资源**：{f.resources or '待定'}",
                    f"- **意识形态**：{f.ideology or '待定'}",
                    "",
                ]
            )
            if f.description:
                lines.append(f.description)
                lines.append("")

        self._write(FILE_FACTIONS, "\n".join(lines))

    def _save_style(self, style: StyleContract) -> None:
        """Save style contract as Markdown."""
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: writing-style",
            "---",
            "",
            "# 文风契约",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 语调 | {style.tone} |",
            f"| 节奏偏好 | {style.pacing_preference} |",
            f"| 句式风格 | {style.sentence_style} |",
            f"| 对话占比 | {style.dialogue_ratio:.0%} |",
            f"| 描写占比 | {style.description_ratio:.0%} |",
            f"| 叙事距离 | {style.narrative_distance} |",
            "",
        ]

        if style.forbidden_phrases:
            lines.append("## 禁用表达")
            lines.append("")
            for phrase in style.forbidden_phrases:
                lines.append(f"- {phrase}")
            lines.append("")

        if style.preferred_techniques:
            lines.append("## 推荐技法")
            lines.append("")
            for tech in style.preferred_techniques:
                lines.append(f"- {tech}")
            lines.append("")

        self._write(FILE_STYLE, "\n".join(lines))

    def _save_themes(self, themes: list[Theme], conflicts: list[CoreConflict]) -> None:
        """Save themes and conflicts as Markdown."""
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: themes-and-conflicts",
            "---",
            "",
            "# 主题与核心冲突",
            "",
            f"## 主题（共{len(themes)}个）",
            "",
        ]

        for t in themes:
            lines.append(f"### {t.name}")
            lines.append(f"{t.description}")
            if t.expression:
                lines.append(f"\n呈现方式：{t.expression}")
            lines.append("")

        lines.append(f"## 核心冲突（共{len(conflicts)}个）")
        lines.append("")

        for c in conflicts:
            lines.append(f"### {c.type}")
            lines.append(f"{c.description}")
            lines.append(f"- 涉及方：{c.parties_involved}")
            lines.append(f"- 赌注：{c.stakes}")
            lines.append("")

        self._write(FILE_THEMES, "\n".join(lines))

    def _save_pleasure_points(
        self,
        pleasure_model: str,
        constraints: list[str],
    ) -> None:
        """Save pleasure point model as Markdown."""
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: pleasure-points",
            "---",
            "",
            "# 爽点模型与叙事约束",
            "",
            "## 爽点模型",
            "",
            pleasure_model or "（待补充）",
            "",
        ]

        if constraints:
            lines.append("## 叙事约束")
            lines.append("")
            for c in constraints:
                lines.append(f"- {c}")
            lines.append("")

        self._write(FILE_PLEASURE, "\n".join(lines))

    def _save_characters(self, characters: CharacterRegistry) -> None:
        """Save character registry as thin-reference Markdown.

        Note: This is a THIN REFERENCE. Full character details live in
        the character YAML files. This file provides quick lookup for
        agents that need character names + roles without the full profile.
        """
        lines = [
            "---",
            'framework_version: "1.0.0"',
            f'generated_at: "{datetime.now().isoformat()}"',
            "category: characters",
            "---",
            "",
            f"# 角色列表（共{len(characters.characters)}人）",
            "",
            "> 注意：本文件为薄引用。完整角色卡请参见 `characters/` 目录。",
            "",
            "| ID | 姓名 | 角色 | 性格摘要 | 动机 |",
            "|----|------|------|----------|------|",
        ]

        for cid, char in characters.characters.items():
            personality_brief = (char.personality or "")[:40]
            motivation_brief = (char.motivation or "")[:40]
            lines.append(f"| {cid} | {char.name} | {char.role} | {personality_brief} | {motivation_brief} |")
        lines.append("")

        self._write(FILE_CHARACTERS, "\n".join(lines))

    # ── Load (individual files) ───────────────────────────────────

    def load_world(self) -> str | None:
        """Load world-building Markdown for agent context."""
        return self._read(FILE_WORLD)

    def load_factions(self) -> str | None:
        """Load factions Markdown for agent context."""
        return self._read(FILE_FACTIONS)

    def load_style(self) -> str | None:
        """Load style contract Markdown for agent context."""
        return self._read(FILE_STYLE)

    def load_themes(self) -> str | None:
        """Load themes & conflicts Markdown for agent context."""
        return self._read(FILE_THEMES)

    def load_pleasure_points(self) -> str | None:
        """Load pleasure point model Markdown for agent context."""
        return self._read(FILE_PLEASURE)

    def load_characters(self) -> str | None:
        """Load character reference Markdown for agent context."""
        return self._read(FILE_CHARACTERS)

    def load_all(self) -> dict[str, str]:
        """Load all bible files as a dict of filename → content."""
        result = {}
        for filename in ALL_BIBLE_FILES:
            content = self._read(filename)
            if content:
                result[filename] = content
        return result

    # ── Helpers ────────────────────────────────────────────────────

    def _write(self, filename: str, content: str) -> None:
        """Write a bible file."""
        path = self._bible_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.debug(f"Wrote {path}")

    def _read(self, filename: str) -> str | None:
        """Read a bible file, returning None if it doesn't exist."""
        path = self._bible_dir / filename
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @property
    def bible_dir(self) -> Path:
        return self._bible_dir

    def exists(self) -> bool:
        """Check if bible files have been saved."""
        return (self._bible_dir / FILE_WORLD).exists()
