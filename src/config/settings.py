"""Application settings loaded from environment variables.

Provider-agnostic design for open source:
- Set DEFAULT_PROVIDER + DEFAULT_MODEL in .env, and all agents use that.
- Or set per-agent overrides for fine-grained control (e.g. WRITER_MODEL=openai:gpt-4o).
- Supported providers: anthropic, openai, openrouter, ollama, modelscope, dashscope.

Provider:Model format for overrides: "provider:model_name"
Examples:
  DEFAULT_PROVIDER=openai
  DEFAULT_MODEL=gpt-4o
  WRITER_MODEL=anthropic:claude-sonnet-4-20250514
  EDITOR_MODEL=openrouter:openai/gpt-4o-mini
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Known provider model lists (for documentation + UI hints)
# ---------------------------------------------------------------------------

PROVIDER_PRESETS = {
    "anthropic": {
        "quality": "claude-sonnet-4-20250514",
        "budget": "claude-haiku-4-5-20251001",
        "premium": "claude-opus-5",
    },
    "openai": {
        "quality": "gpt-4o",
        "budget": "gpt-4o-mini",
        "premium": "gpt-4.1",
    },
    "openrouter": {
        "quality": "anthropic/claude-sonnet-4",
        "budget": "openai/gpt-4o-mini",
        "premium": "anthropic/claude-opus-4",
    },
    "modelscope": {
        "quality": "Qwen/Qwen3-235B-A22B",
        "budget": "Qwen/Qwen3-8B",
    },
    "dashscope": {
        "quality": "qwen-max",
        "budget": "qwen-turbo",
    },
    "ollama": {
        "quality": "qwen3:14b",
        "budget": "qwen3:8b",
    },
}


class Settings(BaseSettings):
    """Global application settings with environment variable binding.

    Every setting can be configured via .env file or environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API Keys (all optional — only configure what you use) ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = ""   # Override for OpenAI-compatible APIs (ModelScope, DashScope, etc.)
    ollama_base_url: str = "http://localhost:11434"

    # --- Default Model Configuration ---
    # The provider:model that ALL agents use unless overridden per-agent.
    # Leave empty to use the built-in tiered defaults (quality/budget per agent type).
    default_provider: str = ""
    default_model: str = ""

    # Budget provider:model for analytical/checking agents.
    # If set, checking agents (editor, continuity_checker, reader_simulator, memory_manager)
    # use this instead of the quality model, saving cost.
    budget_provider: str = ""
    budget_model: str = ""

    # --- Per-Agent Overrides (format: "provider:model" or just "model") ---
    # When only a model name is given, the default_provider is used.
    # Example: WRITER_MODEL=openai:gpt-4o  or  WRITER_MODEL=gpt-4o
    architect_model: str = ""
    character_manager_model: str = ""
    plot_planner_model: str = ""
    writer_model: str = ""
    editor_model: str = ""
    continuity_checker_model: str = ""
    reader_simulator_model: str = ""
    refiner_model: str = ""
    memory_manager_model: str = ""
    orchestrator_model: str = ""
    topic_scout_model: str = ""
    reviewer_model: str = ""

    # --- Workspace ---
    workspace_root: str = "./workspace"

    # --- Application ---
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # --- Chapter Defaults ---
    default_chapter_word_count: int = 3000
    max_review_iterations: int = 3
    min_review_score_accept: float = 6.5

    # --- Retrieval ---
    max_context_chapters: int = 5
    max_context_facts: int = 20

    # ========================================================================
    # Convenience properties
    # ========================================================================

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_root).resolve()

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_any_provider(self) -> bool:
        """Whether at least one API key is configured."""
        return self.has_anthropic or self.has_openai or self.has_openrouter

    @property
    def active_provider(self) -> str:
        """Detect which provider the user intends to use."""
        if self.default_provider:
            return self.default_provider
        if self.has_anthropic:
            return "anthropic"
        if self.has_openrouter:
            return "openrouter"
        if self.has_openai:
            return "openai"
        # Fallback: check if openai_base_url suggests ModelScope or DashScope
        if self.openai_base_url:
            if "modelscope" in self.openai_base_url:
                return "modelscope"
            if "dashscope" in self.openai_base_url or "aliyuncs" in self.openai_base_url:
                return "dashscope"
        return "openai"  # Default: OpenAI-compatible


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
