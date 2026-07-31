"""Prompt template version management.

Implements the evaluation report's pre-launch requirement #2:
- Tracks every prompt template change with diffs and reasons
- Supports viewing version history
- Enables rollback to previous versions
- Uses content hashing for integrity verification

Design:
- Each template has a .version.json sibling file tracking its change history
- Content is hashed (SHA-256) for integrity verification
- Changes are recorded with timestamp, reason, and diff
- Version history is stored in a versions/ subdirectory with timestamped snapshots
"""

import hashlib
import json
import logging
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptVersionManager:
    """Manages version tracking for Jinja2 prompt templates.

    Each template file gets a corresponding .version.json file and
    historical snapshots in a versions/ subdirectory.

    Usage:
        mgr = PromptVersionManager()
        mgr.record_change("writer/generate_chapter_system.j2", "Added POV constraints")
        history = mgr.get_history("writer/generate_chapter_system.j2")
        mgr.rollback("writer/generate_chapter_system.j2", version=2)
    """

    def __init__(self, template_dir: Optional[Path] = None):
        from .prompt_loader import _TEMPLATE_DIR
        self._template_dir = template_dir or _TEMPLATE_DIR
        self._versions_dir = self._template_dir / "_versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────

    def record_change(self, template_path: str, reason: str, author: str = "") -> dict:
        """Record a change to a template file.

        Computes the current hash, diffs against the last recorded version,
        saves a snapshot, and appends to the version log.

        Args:
            template_path: Relative path within templates/ (e.g. 'writer/generate_chapter_system.j2').
            reason: Why this change was made.
            author: Who made the change (default: 'system').

        Returns:
            Version record dict with version number, hash, and diff.
        """
        full_path = self._template_dir / template_path
        if not full_path.exists():
            raise FileNotFoundError(f"Template not found: {full_path}")

        current_hash = self._hash_file(full_path)
        current_content = full_path.read_text(encoding="utf-8")

        # Load existing history
        history = self._load_history(template_path)

        # Check if actually changed
        if history and history[-1]["hash"] == current_hash:
            logger.debug(f"No changes detected for {template_path}")
            return history[-1]

        # Compute diff against previous version
        prev_content = ""
        if history:
            prev_snapshot = self._get_snapshot_path(template_path, history[-1]["version"])
            if prev_snapshot.exists():
                prev_content = prev_snapshot.read_text(encoding="utf-8")

        diff_lines = list(unified_diff(
            prev_content.splitlines(keepends=True),
            current_content.splitlines(keepends=True),
            fromfile=f"{template_path} (v{len(history)})",
            tofile=f"{template_path} (v{len(history) + 1})",
        ))

        # Create new version record
        new_version = len(history) + 1
        record = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "hash": current_hash,
            "reason": reason,
            "author": author or "system",
            "diff_summary": self._summarize_diff(diff_lines),
            "diff_lines": diff_lines,
            "file_size": len(current_content),
            "line_count": current_content.count("\n") + 1,
        }

        # Save snapshot
        snapshot_path = self._get_snapshot_path(template_path, new_version)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(current_content, encoding="utf-8")

        # Append to history
        history.append(record)
        self._save_history(template_path, history)

        logger.info(
            f"Recorded v{new_version} of {template_path}: {reason} "
            f"(hash={current_hash[:8]}..., {record['diff_summary']})"
        )
        return record

    def get_history(self, template_path: str) -> list[dict]:
        """Get the full version history for a template.

        Returns list of version records, oldest first.
        """
        return self._load_history(template_path)

    def get_version(self, template_path: str, version: int) -> Optional[str]:
        """Retrieve the content of a specific version."""
        snapshot_path = self._get_snapshot_path(template_path, version)
        if snapshot_path.exists():
            return snapshot_path.read_text(encoding="utf-8")
        return None

    def rollback(self, template_path: str, version: int, reason: str = "") -> dict:
        """Rollback a template to a previous version.

        Restores the template file from the snapshot, then records
        the rollback as a new version entry.

        Args:
            template_path: Relative path within templates/.
            version: Target version number to restore.
            reason: Why the rollback was performed.

        Returns:
            The new version record created by the rollback.
        """
        content = self.get_version(template_path, version)
        if content is None:
            raise ValueError(f"Version {version} not found for {template_path}")

        full_path = self._template_dir / template_path
        full_path.write_text(content, encoding="utf-8")

        rollback_reason = reason or f"Rollback to v{version}"
        return self.record_change(template_path, rollback_reason)

    def scan_all(self) -> dict[str, list[dict]]:
        """Scan all templates and return version histories.

        Returns dict mapping template path to history list.
        Useful for building a dashboard of all template changes.
        """
        results: dict[str, list[dict]] = {}
        for tmpl_path in self._find_all_templates():
            history = self._load_history(tmpl_path)
            if history:
                results[tmpl_path] = history
        return results

    def get_changed_since(self, since_timestamp: str) -> list[dict]:
        """Get all template changes since a given ISO timestamp.

        Useful for identifying which prompts changed when debugging
        quality regressions (evaluation report recommendation).
        """
        changes = []
        for tmpl_path, history in self.scan_all().items():
            for record in history:
                if record["timestamp"] >= since_timestamp:
                    changes.append({
                        "template": tmpl_path,
                        **record,
                    })
        changes.sort(key=lambda r: r["timestamp"])
        return changes

    def verify_integrity(self) -> dict[str, bool]:
        """Verify that all templates match their latest recorded hash.

        Returns dict mapping template path to validity.
        """
        results = {}
        for tmpl_path in self._find_all_templates():
            history = self._load_history(tmpl_path)
            if not history:
                results[tmpl_path] = True  # No history = no verification needed
                continue
            full_path = self._template_dir / tmpl_path
            current_hash = self._hash_file(full_path)
            results[tmpl_path] = current_hash == history[-1]["hash"]
        return results

    # ── internal helpers ──────────────────────────────────────────

    def _hash_file(self, path: Path) -> str:
        """Compute SHA-256 hash of a file's content."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _get_version_path(self, template_path: str) -> Path:
        """Get the path to the .version.json file for a template."""
        # Store in _versions/ mirroring the template directory structure
        rel = Path(template_path)
        version_file = self._versions_dir / rel.parent / f"{rel.name}.version.json"
        return version_file

    def _get_snapshot_path(self, template_path: str, version: int) -> Path:
        """Get the path to a version snapshot."""
        rel = Path(template_path)
        return self._versions_dir / rel.parent / "snapshots" / f"{rel.name}.v{version:04d}"

    def _load_history(self, template_path: str) -> list[dict]:
        """Load version history from disk."""
        vp = self._get_version_path(template_path)
        if not vp.exists():
            return []
        try:
            return json.loads(vp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Corrupted version history for {template_path}: {e}")
            return []

    def _save_history(self, template_path: str, history: list[dict]) -> None:
        """Save version history to disk."""
        vp = self._get_version_path(template_path)
        vp.parent.mkdir(parents=True, exist_ok=True)
        # Strip full diffs from stored history to save space;
        # keep only the summary and first 20 diff lines
        compact = []
        for record in history:
            r = dict(record)
            r["diff_lines"] = r.get("diff_lines", [])[:20]
            compact.append(r)
        vp.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_all_templates(self) -> list[str]:
        """Find all .j2 template files relative to template_dir."""
        templates = []
        for f in self._template_dir.rglob("*.j2"):
            if "_versions" in f.parts:
                continue
            templates.append(str(f.relative_to(self._template_dir)).replace("\\", "/"))
        return sorted(templates)

    @staticmethod
    def _summarize_diff(diff_lines: list[str]) -> str:
        """Create a human-readable summary of a diff."""
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        if added == 0 and removed == 0:
            return "no changes"
        parts = []
        if added:
            parts.append(f"+{added}")
        if removed:
            parts.append(f"-{removed}")
        return ", ".join(parts)


# Module-level singleton
_version_manager: Optional[PromptVersionManager] = None


def get_version_manager() -> PromptVersionManager:
    """Get the global PromptVersionManager singleton."""
    global _version_manager
    if _version_manager is None:
        _version_manager = PromptVersionManager()
    return _version_manager
