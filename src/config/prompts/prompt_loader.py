"""Jinja2 prompt template loader for all novel-writing agents.

Replaces inline Python string prompts with external .j2 templates that can be
edited without code changes or server restarts (when paired with hot-reload).

Design inspired by ai-job-search-master's modular profile files —
each agent loads only the templates it needs, no information duplication.

Usage:
    from ..config.prompts.prompt_loader import PromptLoader

    loader = PromptLoader()
    system, user = loader.render("writer", "generate_chapter", **variables)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, StrictUndefined

logger = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class PromptLoader:
    """Loads and renders Jinja2 prompt templates for agents.

    Templates are organized by agent type:
        templates/
        ├── writer/
        │   ├── generate_chapter_system.j2
        │   ├── generate_chapter_user.j2
        │   ├── extract_facts_system.j2
        │   └── extract_facts_user.j2
        ├── editor/
        │   ├── review_chapter_system.j2
        │   └── ...
        └── ...

    Each agent gets its own subdirectory. Each LLM call gets a pair of
    {method}_system.j2 and {method}_user.j2 templates.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        self._template_dir = template_dir or _TEMPLATE_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, bool] = {}  # template existence cache

    # ── public API ────────────────────────────────────────────────

    def render(
        self,
        agent: str,
        method: str,
        prompt_type: str = "system",
        **variables: Any,
    ) -> str:
        """Render a template for an agent method.

        Args:
            agent: Agent type name (e.g. 'writer', 'editor').
            method: Method name (e.g. 'generate_chapter', 'review_chapter').
            prompt_type: 'system' or 'user'.
            **variables: Variables to pass to the template.

        Returns:
            Rendered prompt string.

        Raises:
            TemplateNotFound: If the template file doesn't exist.
        """
        template_name = f"{agent}/{method}_{prompt_type}.j2"
        try:
            template = self._env.get_template(template_name)
            return template.render(**variables)
        except TemplateNotFound:
            logger.debug(f"Template not found: {template_name}")
            raise

    def render_or_none(
        self,
        agent: str,
        method: str,
        prompt_type: str = "system",
        **variables: Any,
    ) -> Optional[str]:
        """Render a template, returning None if it doesn't exist.

        Use this for graceful fallback to inline prompts.
        """
        try:
            return self.render(agent, method, prompt_type, **variables)
        except TemplateNotFound:
            return None

    def has_template(self, agent: str, method: str) -> bool:
        """Check if templates exist for an agent method (both system + user)."""
        key = f"{agent}/{method}"
        if key not in self._cache:
            sys_exists = (self._template_dir / agent / f"{method}_system.j2").exists()
            usr_exists = (self._template_dir / agent / f"{method}_user.j2").exists()
            self._cache[key] = sys_exists and usr_exists
        return self._cache[key]

    def list_agents(self) -> list[str]:
        """List agent directories that have templates."""
        if not self._template_dir.exists():
            return []
        return [
            d.name
            for d in self._template_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

    def list_methods(self, agent: str) -> list[str]:
        """List methods that have both system + user templates for an agent."""
        agent_dir = self._template_dir / agent
        if not agent_dir.exists():
            return []
        methods: set[str] = set()
        for f in agent_dir.glob("*_system.j2"):
            stem = f.stem  # e.g. "generate_chapter_system"
            method = stem.rsplit("_system", 1)[0]
            user_file = agent_dir / f"{method}_user.j2"
            if user_file.exists():
                methods.add(method)
        return sorted(methods)

    @property
    def template_dir(self) -> Path:
        return self._template_dir


# Module-level singleton for convenience
_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the global PromptLoader singleton."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader
