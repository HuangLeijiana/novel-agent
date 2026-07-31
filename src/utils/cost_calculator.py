"""Cost budget calculator — provider-agnostic, multi-model.

Estimates token + dollar costs for novel-writing tasks across providers:
Anthropic, OpenAI, OpenRouter, ModelScope, DashScope, and Ollama (free).

Usage:
    from src.utils.cost_calculator import CostCalculator
    calc = CostCalculator()
    # Auto-detect provider from your .env settings:
    est = calc.estimate_novel(total_chapters=30)
    # Or specify explicitly:
    est = calc.estimate_novel(total_chapters=30, provider="openai", quality_model="gpt-4o")
    print(est.format())
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import PROVIDER_PRESETS

logger = logging.getLogger(__name__)

# ========================================================================
# Provider pricing (USD per 1M tokens, as of 2026-07)
# ========================================================================

# Format: {model_id: (input_price_per_1M, output_price_per_1M)}
# Prices are estimates — always check the provider's official pricing page.
PROVIDER_PRICING = {
    "anthropic": {
        "claude-opus-5":             (15.00, 75.00),
        "claude-sonnet-4-20250514":  (3.00,  15.00),
        "claude-haiku-4-5-20251001": (1.00,   5.00),
    },
    "openai": {
        "gpt-4.1":    (2.00,  8.00),
        "gpt-4o":     (2.50, 10.00),
        "gpt-4o-mini":(0.15,  0.60),
        "gpt-4.1-nano":(0.10, 0.40),
    },
    "openrouter": {
        # OpenRouter prices fluctuate; these are mid-2026 estimates.
        # Use openrouter.ai/models for current pricing.
        "anthropic/claude-sonnet-4":    (3.00, 15.00),
        "anthropic/claude-haiku-4.5":   (1.00,  5.00),
        "anthropic/claude-opus-4":      (15.00, 75.00),
        "openai/gpt-4o":                (2.50, 10.00),
        "openai/gpt-4o-mini":           (0.15,  0.60),
        "google/gemini-2.5-pro":        (1.25, 10.00),
        "google/gemini-2.5-flash":      (0.15,  0.60),
    },
    "modelscope": {
        # ModelScope pricing varies by model. Most are free during beta.
        # Set MODELscope_API_KEY for higher rate limits.
        "Qwen/Qwen3-235B-A22B": (0.0, 0.0),
        "Qwen/Qwen3-8B":        (0.0, 0.0),
    },
    "dashscope": {
        # Alibaba DashScope (Qwen official API)
        "qwen-max":    (2.00, 6.00),
        "qwen-plus":   (1.00, 2.00),
        "qwen-turbo":  (0.15, 0.60),
    },
    "ollama": {
        # Local — electricity only
        "*": (0.0, 0.0),
    },
}

# Fallback pricing when a specific model isn't found
FALLBACK_PRICE = (1.0, 5.0)  # conservative estimate

# ========================================================================
# Per-agent token estimates (provider-agnostic)
# ========================================================================

SYSTEM_PROMPT_TOKENS = {
    "architect": 800,  "character_manager": 600, "plot_planner": 700,
    "writer": 1200,    "editor": 900,            "continuity_checker": 500,
    "reader_simulator": 400, "refiner": 600,     "reviewer": 800,
    "memory_manager": 500, "topic_scout": 700,   "orchestrator": 400,
}

BASE_USER_PROMPT_TOKENS = {
    "architect": 2000,  "character_manager": 1500, "plot_planner": 2000,
    "writer": 3000,     "editor": 3500,            "continuity_checker": 2500,
    "reader_simulator": 3000, "refiner": 3000,     "reviewer": 3500,
    "memory_manager": 2000, "topic_scout": 1000,   "orchestrator": 800,
}

OUTPUT_TOKENS = {
    "architect": 1500,  "character_manager": 2000, "plot_planner": 2500,
    "writer": 4000,     "editor": 1500,            "continuity_checker": 800,
    "reader_simulator": 600,  "refiner": 3000,     "reviewer": 1200,
    "memory_manager": 1000, "topic_scout": 2000,   "orchestrator": 500,
}

# Which agents are called in each phase
WRITING_AGENTS = ["writer"]
REVIEW_AGENTS = ["editor", "continuity_checker", "reader_simulator"]
ADVERSARIAL_AGENTS = ["reviewer"]
POLISH_AGENTS = ["refiner"]
MEMORY_AGENTS = ["memory_manager"]

# Context growth per chapter (accumulated bible/character/outline/memory)
CONTEXT_GROWTH_PER_CHAPTER = 500


@dataclass
class ChapterCost:
    """Cost breakdown for a single chapter."""
    chapter_number: int
    writing_cost: float = 0.0
    review_cost: float = 0.0
    adversarial_review_cost: float = 0.0
    polish_cost: float = 0.0
    memory_cost: float = 0.0
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class NovelCostEstimate:
    """Complete cost estimate for a novel."""
    total_chapters: int
    provider: str = ""
    quality_model: str = ""
    budget_model: str = ""
    per_chapter: list[ChapterCost] = field(default_factory=list)
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    initial_write_total: float = 0.0
    with_one_revision_total: float = 0.0
    with_two_revisions_total: float = 0.0

    def format(self) -> str:
        lines = [
            "=" * 60,
            "  Novel Writing Cost Estimate",
            "=" * 60,
            f"  Chapters:        {self.total_chapters}",
            f"  Provider:        {self.provider}",
            f"  Quality model:   {self.quality_model}",
            f"  Budget model:    {self.budget_model}",
            "",
            f"  [Scenarios]",
            f"  All pass 1st attempt:    ${self.initial_write_total:.2f}",
            f"  30% need 1 revision:     ${self.with_one_revision_total:.2f}",
            f"  15% need 2 revisions:    ${self.with_two_revisions_total:.2f}",
            "",
            f"  [Tokens]",
            f"  Total input:   {self.total_input_tokens:,}",
            f"  Total output:  {self.total_output_tokens:,}",
            f"  Avg/chapter:   {(self.total_input_tokens + self.total_output_tokens) // max(self.total_chapters, 1):,}",
            "=" * 60,
        ]
        return "\n".join(lines)


class CostCalculator:
    """Provider-agnostic cost estimator.

    Auto-detects provider and models from your Settings (.env), or
    accepts explicit overrides.

    Usage:
        calc = CostCalculator()              # auto-detect from .env
        calc = CostCalculator(provider="openai", quality_model="gpt-4o",
                              budget_model="gpt-4o-mini")
        est = calc.estimate_novel(30)
    """

    def __init__(
        self,
        provider: str = "",
        quality_model: str = "",
        budget_model: str = "",
        words_per_chapter: int = 3000,
    ):
        self.words_per_chapter = words_per_chapter

        # Auto-detect from settings if not provided
        if not provider:
            from ..config.settings import get_settings
            s = get_settings()
            self.provider = s.active_provider
            # Best-effort model detection
            if s.default_model:
                self.quality_model = s.default_model
                self.budget_model = s.budget_model or s.default_model
            else:
                preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["openai"])
                self.quality_model = s.default_model or preset["quality"]
                self.budget_model = s.budget_model or preset["budget"]
        else:
            self.provider = provider
            self.quality_model = quality_model or PROVIDER_PRESETS.get(provider, {}).get("quality", "gpt-4o")
            self.budget_model = budget_model or PROVIDER_PRESETS.get(provider, {}).get("budget", "gpt-4o-mini")

    # ── Public API ────────────────────────────────────────────────

    def estimate_novel(
        self,
        total_chapters: int = 30,
        words_per_chapter: Optional[int] = None,
    ) -> NovelCostEstimate:
        """Estimate total cost for writing a complete novel."""
        if words_per_chapter:
            self.words_per_chapter = words_per_chapter

        estimate = NovelCostEstimate(
            total_chapters=total_chapters,
            provider=self.provider,
            quality_model=self.quality_model,
            budget_model=self.budget_model,
        )

        for ch in range(1, total_chapters + 1):
            ch_cost = self._estimate_chapter(ch)
            estimate.per_chapter.append(ch_cost)
            estimate.total_cost += ch_cost.total_cost
            estimate.total_input_tokens += ch_cost.total_input_tokens
            estimate.total_output_tokens += ch_cost.total_output_tokens

        # Scenario analysis
        estimate.initial_write_total = estimate.total_cost
        revision_rate = 0.30
        revision_cost = sum(
            c.review_cost + c.polish_cost + c.adversarial_review_cost
            for c in estimate.per_chapter
        ) * revision_rate
        estimate.with_one_revision_total = estimate.total_cost + revision_cost
        estimate.with_two_revisions_total = estimate.total_cost + revision_cost * 2

        return estimate

    def estimate_single_chapter(self, chapter_number: int = 1) -> ChapterCost:
        """Estimate cost for a single chapter."""
        return self._estimate_chapter(chapter_number)

    def compare_providers(self, total_chapters: int = 30) -> dict[str, NovelCostEstimate]:
        """Estimate costs across all supported providers for comparison."""
        results: dict[str, NovelCostEstimate] = {}
        for provider_name, presets in PROVIDER_PRESETS.items():
            if provider_name == "ollama":
                continue  # Free, skip comparison
            calc = CostCalculator(
                provider=provider_name,
                quality_model=presets["quality"],
                budget_model=presets["budget"],
            )
            results[provider_name] = calc.estimate_novel(total_chapters)
        return results

    # ── Internal ──────────────────────────────────────────────────

    def _get_price(self, model_id: str) -> tuple[float, float]:
        """Get (input_price, output_price) per 1M tokens for a model."""
        provider_prices = PROVIDER_PRICING.get(self.provider, {})
        if model_id in provider_prices:
            return provider_prices[model_id]
        # Try wildcard match
        if "*" in provider_prices:
            return provider_prices["*"]
        return FALLBACK_PRICE

    def _estimate_chapter(self, chapter_number: int) -> ChapterCost:
        """Estimate cost for one chapter."""
        ch = ChapterCost(chapter_number=chapter_number)
        context_extra = (chapter_number - 1) * CONTEXT_GROWTH_PER_CHAPTER

        def _agent_cost(agent_type: str) -> tuple[float, int, int]:
            # Quality agents use quality_model, budget agents use budget_model
            from ..config.model_assignments import BUDGET_AGENTS
            model = self.budget_model if agent_type in BUDGET_AGENTS else self.quality_model
            input_tokens = (
                SYSTEM_PROMPT_TOKENS.get(agent_type, 600)
                + BASE_USER_PROMPT_TOKENS.get(agent_type, 2000)
                + context_extra
            )
            output_tokens = OUTPUT_TOKENS.get(agent_type, 1500)
            input_price, output_price = self._get_price(model)
            cost = (
                input_price * input_tokens / 1_000_000
                + output_price * output_tokens / 1_000_000
            )
            return cost, input_tokens, output_tokens

        for agent_list, cost_attr in [
            (WRITING_AGENTS, "writing_cost"),
            (REVIEW_AGENTS, "review_cost"),
            (ADVERSARIAL_AGENTS, "adversarial_review_cost"),
            (POLISH_AGENTS, "polish_cost"),
            (MEMORY_AGENTS, "memory_cost"),
        ]:
            for agent in agent_list:
                r_cost, r_in, r_out = _agent_cost(agent)
                setattr(ch, cost_attr, getattr(ch, cost_attr) + r_cost)
                ch.total_cost += r_cost
                ch.total_input_tokens += r_in
                ch.total_output_tokens += r_out

        return ch


# ── Convenience ───────────────────────────────────────────────────

def estimate_novel_cost(
    total_chapters: int = 30,
    provider: str = "",
    quality_model: str = "",
    budget_model: str = "",
) -> NovelCostEstimate:
    """Quick cost estimate. Auto-detects provider from .env if not specified."""
    calc = CostCalculator(
        provider=provider,
        quality_model=quality_model,
        budget_model=budget_model,
    )
    return calc.estimate_novel(total_chapters)


def print_cost_comparison(total_chapters: int = 30):
    """Print a comparison of all supported providers."""
    calc = CostCalculator()
    estimates = calc.compare_providers(total_chapters)
    print(f"\n{'='*70}")
    print(f"  Cost Comparison: {total_chapters} chapters ({calc.words_per_chapter} words/ch)")
    print(f"{'='*70}")
    print(f"  {'Provider':<20} {'Quality Model':<32} {'Base':>10} {'+Rev':>10}")
    print(f"  {'-'*65}")
    for name, est in estimates.items():
        print(f"  {name:<20} {est.quality_model:<32} ${est.initial_write_total:>8.2f} ${est.with_one_revision_total:>8.2f}")
    print(f"  {'ollama (local)':<20} {'any (free)':<32} {'$0.00':>10} {'$0.00':>10}")
    print(f"{'='*70}\n")
