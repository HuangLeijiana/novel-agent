"""Project file manager — handles all artifact persistence to disk."""

import json
from pathlib import Path
from typing import Optional

from ..models.project import ProjectConfig, ProjectMeta
from ..models.bible import NovelBible, WorldBuilding, Faction, NarrativeRules, StyleContract, Theme, CoreConflict
from ..models.characters import CharacterRegistry
from ..models.outline import MasterOutline, ChapterPlan, Volume, TurningPoint
from ..models.chapter import ChapterDraft, PolishedChapter
from ..models.review import ReviewReport
from ..models.memory import MemoryState
from .serializers import YamlSerializer, MarkdownSerializer, DocxSerializer


class ProjectFileManager:
    """Manages reading and writing all project artifacts to/from disk.

    Directory structure:
        {workspace}/projects/{project_id}/
            project.yaml
            novel_bible/
                world.yaml
                factions.yaml
                characters.yaml
                rules.yaml
                style_contract.yaml
                themes.yaml
                conflicts.yaml
            outline/
                master_outline.yaml
                volumes.yaml
                turning_points.yaml
                chapters/
                    chapter_001.yaml
                    ...
            memory/
                short_term.yaml
                long_term.json
                timeline.yaml
                foreshadowing.yaml
                character_states.yaml
            output/
                chapters/
                    chapter_001.md
                    chapter_001.docx
                    ...
    """

    def __init__(self, workspace_root: str | Path, project_id: str):
        self.workspace_root = Path(workspace_root)
        self.project_id = project_id
        self.root = self.workspace_root / "projects" / project_id

    # ================================================================
    # Directory initialization
    # ================================================================

    def initialize(self, config: ProjectConfig) -> None:
        """Create the full directory structure and save initial project.yaml."""
        dirs = [
            self.root / "novel_bible",
            self.root / "outline" / "chapters",
            self.root / "memory",
            self.root / "output" / "chapters",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.save_project_config(config)

    def exists(self) -> bool:
        """Check if this project exists on disk."""
        return (self.root / "project.yaml").exists()

    # ================================================================
    # Project Config & Meta
    # ================================================================

    def save_project_config(self, config: ProjectConfig) -> None:
        """Save project configuration."""
        YamlSerializer.to_yaml(config, self.root / "project.yaml")

    def load_project_config(self) -> Optional[ProjectConfig]:
        """Load project configuration."""
        path = self.root / "project.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, ProjectConfig)

    def save_project_meta(self, meta: ProjectMeta) -> None:
        """Save project metadata."""
        YamlSerializer.to_yaml(meta, self.root / "project_meta.yaml")

    def load_project_meta(self) -> Optional[ProjectMeta]:
        """Load project metadata."""
        path = self.root / "project_meta.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, ProjectMeta)

    # ================================================================
    # Novel Bible
    # ================================================================

    def save_bible(self, bible: NovelBible) -> None:
        """Save all sections of the Novel Bible."""
        bible_dir = self.root / "novel_bible"
        YamlSerializer.to_yaml(bible.world, bible_dir / "world.yaml")
        YamlSerializer.to_yaml(bible.factions, bible_dir / "factions.yaml")
        YamlSerializer.to_yaml(bible.rules, bible_dir / "rules.yaml")
        YamlSerializer.to_yaml(bible.style_contract, bible_dir / "style_contract.yaml")
        YamlSerializer.to_yaml(bible.themes, bible_dir / "themes.yaml")
        YamlSerializer.to_yaml(bible.core_conflicts, bible_dir / "conflicts.yaml")
        # Also save the complete bible for easy loading
        YamlSerializer.to_yaml(bible, bible_dir / "bible_full.yaml")

    def load_bible(self) -> Optional[NovelBible]:
        """Load the complete Novel Bible from the full file."""
        path = self.root / "novel_bible" / "bible_full.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, NovelBible)

    def save_world(self, world: WorldBuilding) -> None:
        YamlSerializer.to_yaml(world, self.root / "novel_bible" / "world.yaml")

    def save_factions(self, factions: list[Faction]) -> None:
        YamlSerializer.to_yaml(factions, self.root / "novel_bible" / "factions.yaml")

    def save_rules(self, rules: NarrativeRules) -> None:
        YamlSerializer.to_yaml(rules, self.root / "novel_bible" / "rules.yaml")

    def save_style_contract(self, style: StyleContract) -> None:
        YamlSerializer.to_yaml(style, self.root / "novel_bible" / "style_contract.yaml")

    def save_themes(self, themes: list[Theme]) -> None:
        YamlSerializer.to_yaml(themes, self.root / "novel_bible" / "themes.yaml")

    def save_conflicts(self, conflicts: list[CoreConflict]) -> None:
        YamlSerializer.to_yaml(conflicts, self.root / "novel_bible" / "conflicts.yaml")

    # ================================================================
    # Characters
    # ================================================================

    def save_characters(self, characters: CharacterRegistry) -> None:
        """Save character registry."""
        YamlSerializer.to_yaml(characters, self.root / "novel_bible" / "characters.yaml")

    def load_characters(self) -> Optional[CharacterRegistry]:
        """Load character registry."""
        path = self.root / "novel_bible" / "characters.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, CharacterRegistry)

    # ================================================================
    # Outline
    # ================================================================

    def save_master_outline(self, outline: MasterOutline) -> None:
        """Save master outline."""
        YamlSerializer.to_yaml(outline, self.root / "outline" / "master_outline.yaml")

    def load_master_outline(self) -> Optional[MasterOutline]:
        """Load master outline."""
        path = self.root / "outline" / "master_outline.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, MasterOutline)

    def save_volumes(self, volumes: list[Volume]) -> None:
        """Save volume structure."""
        YamlSerializer.to_yaml(volumes, self.root / "outline" / "volumes.yaml")

    def save_turning_points(self, turning_points: list[TurningPoint]) -> None:
        """Save major turning points."""
        YamlSerializer.to_yaml(turning_points, self.root / "outline" / "turning_points.yaml")

    def save_chapter_plan(self, plan: ChapterPlan) -> None:
        """Save a single chapter plan."""
        filename = f"chapter_{plan.chapter_number:03d}.yaml"
        YamlSerializer.to_yaml(plan, self.root / "outline" / "chapters" / filename)

    def load_chapter_plan(self, chapter_number: int) -> Optional[ChapterPlan]:
        """Load a single chapter plan."""
        filename = f"chapter_{chapter_number:03d}.yaml"
        path = self.root / "outline" / "chapters" / filename
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, ChapterPlan)

    def list_chapter_plans(self) -> list[int]:
        """List all chapter numbers that have plans."""
        chapters_dir = self.root / "outline" / "chapters"
        if not chapters_dir.exists():
            return []
        nums = []
        for f in chapters_dir.glob("chapter_*.yaml"):
            try:
                num = int(f.stem.replace("chapter_", ""))
                nums.append(num)
            except ValueError:
                pass
        return sorted(nums)

    # ================================================================
    # Chapter Output
    # ================================================================

    def save_chapter_markdown(self, chapter: PolishedChapter) -> Path:
        """Save polished chapter as Markdown."""
        filename = f"chapter_{chapter.chapter_number:03d}.md"
        path = self.root / "output" / "chapters" / filename
        MarkdownSerializer.save_chapter_markdown(
            path,
            chapter.chapter_number,
            chapter.title,
            chapter.content,
        )
        return path

    def save_chapter_docx(self, chapter: PolishedChapter) -> Path:
        """Save polished chapter as DOCX."""
        filename = f"chapter_{chapter.chapter_number:03d}.docx"
        path = self.root / "output" / "chapters" / filename
        DocxSerializer.chapter_to_docx(
            path,
            chapter.chapter_number,
            chapter.title,
            chapter.content,
        )
        return path

    def load_chapter_markdown(self, chapter_number: int) -> Optional[str]:
        """Load a chapter's Markdown content."""
        filename = f"chapter_{chapter_number:03d}.md"
        path = self.root / "output" / "chapters" / filename
        if not path.exists():
            return None
        _, _, content = MarkdownSerializer.load_chapter_markdown(path)
        return content

    def save_chapter_draft(self, draft: ChapterDraft) -> None:
        """Save raw chapter draft for potential recovery."""
        filename = f"chapter_{draft.chapter_number:03d}_draft.yaml"
        YamlSerializer.to_yaml(draft, self.root / "output" / "chapters" / filename)

    def load_chapter_draft(self, chapter_number: int) -> Optional[ChapterDraft]:
        """Load raw chapter draft."""
        filename = f"chapter_{chapter_number:03d}_draft.yaml"
        path = self.root / "output" / "chapters" / filename
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, ChapterDraft)

    # ================================================================
    # Review Report
    # ================================================================

    def save_review_report(self, report: ReviewReport) -> None:
        """Save review report."""
        filename = f"review_chapter_{report.chapter_number:03d}.yaml"
        YamlSerializer.to_yaml(report, self.root / "output" / "chapters" / filename)

    def load_review_report(self, chapter_number: int) -> Optional[ReviewReport]:
        """Load review report."""
        filename = f"review_chapter_{chapter_number:03d}.yaml"
        path = self.root / "output" / "chapters" / filename
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path, ReviewReport)

    # ================================================================
    # Memory
    # ================================================================

    def save_memory(self, memory: MemoryState) -> None:
        """Save all memory artifacts."""
        mem_dir = self.root / "memory"
        YamlSerializer.to_yaml(memory.short_term, mem_dir / "short_term.yaml")
        YamlSerializer.to_yaml(memory.long_term, mem_dir / "long_term.yaml")
        YamlSerializer.to_yaml(memory.timeline, mem_dir / "timeline.yaml")
        YamlSerializer.to_yaml(memory.foreshadowing, mem_dir / "foreshadowing.yaml")
        YamlSerializer.to_yaml(memory.character_states, mem_dir / "character_states.yaml")
        # Save long-term events as JSONL for streaming/append
        self._save_long_term_events_jsonl(memory, mem_dir / "long_term_events.jsonl")

    def load_memory(self) -> Optional[MemoryState]:
        """Load complete memory state."""
        mem_dir = self.root / "memory"
        if not (mem_dir / "short_term.yaml").exists():
            return None
        short_term = YamlSerializer.from_yaml(mem_dir / "short_term.yaml")
        long_term = YamlSerializer.from_yaml(mem_dir / "long_term.yaml")
        timeline = YamlSerializer.from_yaml(mem_dir / "timeline.yaml") or []
        foreshadowing = YamlSerializer.from_yaml(mem_dir / "foreshadowing.yaml")
        character_states = YamlSerializer.from_yaml(mem_dir / "character_states.yaml") or {}
        return MemoryState(
            short_term=short_term,
            long_term=long_term,
            timeline=timeline,
            foreshadowing=foreshadowing,
            character_states=character_states,
        )

    # ================================================================
    # Topic Research (Phase 0)
    # ================================================================

    def save_mini_arc_outlines(self, outlines: dict) -> None:
        """Save mini-arc outlines (Phase 0c output)."""
        topic_dir = self.root / "topic_research"
        YamlSerializer.to_yaml(outlines, topic_dir / "mini_arc_outlines.yaml")

    def load_mini_arc_outlines(self) -> Optional[dict]:
        """Load saved mini-arc outlines."""
        path = self.root / "topic_research" / "mini_arc_outlines.yaml"
        if not path.exists():
            return None
        return YamlSerializer.from_yaml(path)

    def _save_long_term_events_jsonl(self, memory: MemoryState, path: Path) -> None:
        """Save timeline events as JSONL for append-friendly format."""
        from ..models.memory import TimelineEvent

        events = memory.timeline
        with open(path, "w", encoding="utf-8") as f:
            for event in events:
                if isinstance(event, TimelineEvent):
                    f.write(event.model_dump_json() + "\n")
                elif isinstance(event, dict):
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ================================================================
    # Full Export
    # ================================================================

    def export_book_markdown(self) -> Path:
        """Export all chapters as a single Markdown file."""
        chapters_dir = self.root / "output" / "chapters"
        outline = self.load_master_outline()
        title = outline.title if outline else self.project_id

        parts = [f"# {title}\n\n"]
        for md_file in sorted(chapters_dir.glob("chapter_*.md")):
            content = md_file.read_text(encoding="utf-8")
            parts.append(content)
            parts.append("\n\n")

        output_path = self.root / "output" / f"{title}_full.md"
        output_path.write_text("\n".join(parts), encoding="utf-8")
        return output_path

    def export_book_docx(self) -> Path:
        """Export all chapters as a single DOCX file."""
        chapters_dir = self.root / "output" / "chapters"
        outline = self.load_master_outline()
        title = outline.title if outline else self.project_id

        chapters = []
        for md_file in sorted(chapters_dir.glob("chapter_*.md")):
            num, ch_title, content = MarkdownSerializer.load_chapter_markdown(md_file)
            # Extract chapter number from filename
            try:
                num = int(md_file.stem.replace("chapter_", ""))
            except ValueError:
                pass
            chapters.append((num, ch_title, content))

        output_path = self.root / "output" / f"{title}_full.docx"
        DocxSerializer.book_to_docx(output_path, title, chapters)
        return output_path
