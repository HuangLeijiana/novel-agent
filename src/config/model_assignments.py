"""Model assignment configuration — provider-agnostic, tier-based.

Design for open source:
- Creative/quality agents (writer, architect, plot_planner, character_manager,
  reviewer) default to the "quality" tier model.
- Analytical/budget agents (editor, continuity_checker, reader_simulator,
  memory_manager, refiner) default to the "budget" tier model if configured.
- Every agent can be individually overridden via .env (e.g. WRITER_MODEL=...).
- Supported providers: anthropic, openai, openrouter, modelscope, dashscope, ollama.

How to configure:
1. Simplest: set DEFAULT_PROVIDER + DEFAULT_MODEL in .env → all agents use it.
2. Tiered: also set BUDGET_PROVIDER + BUDGET_MODEL → checking agents use cheaper model.
3. Per-agent: set WRITER_MODEL=openai:gpt-4o → that agent uses the specific model.

Format for all *_MODEL settings: "provider:model_name"
If provider is omitted (just "model_name"), the default_provider is used.
"""

from typing import Optional

from pydantic import BaseModel, Field

from .settings import PROVIDER_PRESETS, Settings, get_settings


class ModelAssignment(BaseModel):
    """Which LLM model an agent type should use."""

    agent_type: str = Field(..., description="Matches BaseAgent.agent_type")
    provider: str = Field(
        default="openai",
        description="Provider: 'anthropic', 'openai', 'openrouter', 'ollama', 'modelscope', 'dashscope'",
    )
    primary_model: str = Field(..., description="Model ID, e.g. 'gpt-4o', 'claude-sonnet-4-20250514'")
    fallback_model: Optional[str] = Field(
        default=None,
        description="Fallback model if primary fails",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)


# ================================================================
# Agent categories — which tier each agent type belongs to
# ================================================================

# Creative / high-quality agents: worth spending on a good model
QUALITY_AGENTS = {
    "architect", "character_manager", "plot_planner",
    "writer", "reviewer", "topic_scout",
}

# Analytical / checking agents: can use a cheaper model
BUDGET_AGENTS = {
    "editor", "continuity_checker", "reader_simulator",
    "memory_manager", "refiner",
}

# Orchestrator is lightweight
ORCHESTRATOR_AGENTS = {"orchestrator"}

# ================================================================
# Agent configs: temperature + max_tokens per agent type
# (provider-agnostic — these are LLM-agnostic parameters)
# ================================================================

AGENT_CONFIGS: dict[str, dict] = {
    # Creative agents — higher temperature for variety
    "architect":      {"temperature": 0.8, "max_tokens": 8192,  "tier": "quality"},
    "character_manager": {"temperature": 0.7, "max_tokens": 8192,  "tier": "quality"},
    "plot_planner":   {"temperature": 0.7, "max_tokens": 8192,  "tier": "quality"},
    "writer":         {"temperature": 0.9, "max_tokens": 16384, "tier": "quality"},
    "reviewer":       {"temperature": 0.4, "max_tokens": 4096,  "tier": "quality"},
    "topic_scout":    {"temperature": 0.5, "max_tokens": 8192,  "tier": "quality"},
    # Analytical agents — lower temperature for consistency
    "editor":              {"temperature": 0.3, "max_tokens": 4096,  "tier": "budget"},
    "continuity_checker":  {"temperature": 0.3, "max_tokens": 4096,  "tier": "budget"},
    "reader_simulator":    {"temperature": 0.5, "max_tokens": 2048,  "tier": "budget"},
    "memory_manager":      {"temperature": 0.3, "max_tokens": 4096,  "tier": "budget"},
    "refiner":             {"temperature": 0.4, "max_tokens": 8192,  "tier": "budget"},
    # Orchestrator
    "orchestrator":   {"temperature": 0.3, "max_tokens": 2048,  "tier": "budget"},
}


def _resolve_provider_model(
    agent_type: str,
    settings: Settings,
) -> tuple[str, str, Optional[str]]:
    """Resolve (provider, primary_model, fallback_model) for an agent.

    Resolution order:
    1. Per-agent override (e.g. WRITER_MODEL=openai:gpt-4o)
    2. Tier-based defaults (quality → default_model, budget → budget_model)
    3. Auto-detect from configured API keys
    4. Hard fallback: openai + gpt-4o (user must configure an API key)
    """
    config = AGENT_CONFIGS.get(agent_type, {})
    tier = config.get("tier", "quality")

    # ---- Step 1: per-agent override ----
    override_map = {
        "architect": settings.architect_model,
        "character_manager": settings.character_manager_model,
        "plot_planner": settings.plot_planner_model,
        "writer": settings.writer_model,
        "editor": settings.editor_model,
        "continuity_checker": settings.continuity_checker_model,
        "reader_simulator": settings.reader_simulator_model,
        "refiner": settings.refiner_model,
        "memory_manager": settings.memory_manager_model,
        "orchestrator": settings.orchestrator_model,
        "topic_scout": settings.topic_scout_model,
        "reviewer": settings.reviewer_model,
    }

    override = override_map.get(agent_type, "")
    if override:
        return _parse_override(override, settings)

    # ---- Step 2: tier-based defaults ----
    provider = settings.default_provider
    model = settings.default_model

    if tier == "budget" and settings.budget_model:
        # Budget tier: use budget model if configured
        if settings.budget_provider:
            provider = settings.budget_provider
        model = settings.budget_model

    if provider and model:
        return provider, model, None

    # ---- Step 3: auto-detect from API keys ----
    if settings.has_openrouter:
        preset = PROVIDER_PRESETS["openrouter"]
        return "openrouter", preset[tier], None
    if settings.has_anthropic:
        preset = PROVIDER_PRESETS["anthropic"]
        return "anthropic", preset[tier], None
    if settings.has_openai:
        preset = PROVIDER_PRESETS["openai"]
        return "openai", preset[tier], None

    # ModelScope / DashScope: detected via openai_base_url
    if settings.openai_base_url:
        if "modelscope" in settings.openai_base_url:
            preset = PROVIDER_PRESETS["modelscope"]
            return "modelscope", preset[tier], None
        if "dashscope" in settings.openai_base_url or "aliyuncs" in settings.openai_base_url:
            preset = PROVIDER_PRESETS["dashscope"]
            return "dashscope", preset[tier], None

    # ---- Step 4: hard fallback (user must configure at least one API key) ----
    # Default to OpenAI; the system will error clearly if no key is set.
    preset = PROVIDER_PRESETS["openai"]
    return "openai", preset[tier], None


def _parse_override(override: str, settings: Settings) -> tuple[str, str, Optional[str]]:
    """Parse a 'provider:model' or 'model' override string."""
    if ":" in override:
        provider, model = override.split(":", 1)
        return provider.strip(), model.strip(), None
    else:
        # Model only — use default provider (or auto-detect)
        provider = settings.default_provider or settings.active_provider
        return provider, override.strip(), None


def get_default_assignments() -> list[ModelAssignment]:
    """Build the list of ModelAssignments from current settings.

    This is called once at startup. The result drives all agent LLM calls.
    """
    settings = get_settings()
    assignments: list[ModelAssignment] = []

    for agent_type, config in AGENT_CONFIGS.items():
        provider, model, fallback = _resolve_provider_model(agent_type, settings)
        assignments.append(ModelAssignment(
            agent_type=agent_type,
            provider=provider,
            primary_model=model,
            fallback_model=fallback,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        ))

    return assignments
