"""Model scheduler — routes agent requests to the correct LLM model."""

import json
import logging
import re
from typing import Any, AsyncGenerator, Optional

from pydantic import BaseModel, ValidationError

from ..config.model_assignments import ModelAssignment
from ..config.settings import get_settings
from .client import BaseLLMProvider, LLMResponse
from .providers import (
    AnthropicProvider,
    OpenAIProvider,
    _build_field_map,
    _coerce_field_types,
    _extract_json,
    _translate_keys,
)

logger = logging.getLogger(__name__)


def _try_parse_json(raw: str) -> Any:
    """Attempt to parse JSON, fixing common LLM output issues.

    Returns the parsed object, or raises ValueError with details.
    """
    text = _extract_json(raw)

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        pass

    # Attempt 2: escape literal newlines/tabs within JSON strings
    result_chars = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            result_chars.append(ch)
            escaped = False
            continue
        if ch == '\\':
            result_chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            result_chars.append(ch)
            continue
        if in_string:
            if ch == '\n':
                result_chars.append('\\n')
            elif ch == '\r':
                result_chars.append('\\r')
            elif ch == '\t':
                result_chars.append('\\t')
            else:
                result_chars.append(ch)
        else:
            result_chars.append(ch)
    fixed = ''.join(result_chars)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 2b: fix missing commas too
    fixed2 = re.sub(r'"\s*\n\s*"', '",\n"', fixed)
    fixed2 = re.sub(r'"\s+"', '", "', fixed2)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # Attempt 3: try to extract just the first complete JSON object
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        continue
        elif ch == '[':
            stack.append('[')
        elif ch == ']':
            if stack:
                stack.pop()

    # Attempt 4: truncated JSON — try closing unclosed strings/objects/arrays
    # Track brace depth, bracket depth, and string state
    in_str = False
    escaped = False
    brace_count = 0
    bracket_count = 0
    for ch in text:
        if escaped:
            escaped = False; continue
        if ch == '\\':
            escaped = True; continue
        if ch == '"' and not escaped:
            in_str = not in_str
        if not in_str:
            if ch == '{': brace_count += 1
            elif ch == '}': brace_count -= 1
            elif ch == '[': bracket_count += 1
            elif ch == ']': bracket_count -= 1

    fixed = text.rstrip().rstrip(',')  # strip trailing commas too
    # If still inside a string, close it
    if in_str:
        fixed += '"'
    # Close structures inside-out: alternate braces/brackets,
    # starting with whichever type is innermost (has more opens)
    while brace_count > 0 or bracket_count > 0:
        if brace_count >= bracket_count and brace_count > 0:
            fixed += '}'
            brace_count -= 1
        elif bracket_count > 0:
            fixed += ']'
            bracket_count -= 1

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 5: try removing the last incomplete item (common with list truncation)
    # Find the last complete item before the truncation point
    if '[' in text:
        # Try closing at the last comma before an unclosed string
        last_comma = text.rfind(',')
        if last_comma > 0:
            truncated = text[:last_comma].rstrip()
            # Re-check brace/bracket balance on truncated text
            bc = truncated.count('{') - truncated.count('}')
            brc = truncated.count('[') - truncated.count(']')
            in_s = False
            esc = False
            for ch in truncated:
                if esc: esc = False; continue
                if ch == '\\': esc = True; continue
                if ch == '"': in_s = not in_s
            if in_s:
                truncated += '"'
            # Inside-out closing
            while bc > 0 or brc > 0:
                if bc >= brc and bc > 0:
                    truncated += '}'
                    bc -= 1
                elif brc > 0:
                    truncated += ']'
                    brc -= 1
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Cannot parse LLM JSON output (first 200 chars): {text[:200]}")

logger = logging.getLogger(__name__)


class ModelScheduler:
    """Routes agent generation requests to the configured LLM provider and model.

    Features:
    - Maps agent_type -> provider + model via ModelAssignment config
    - Primary/fallback model with automatic retry on failure
    - Per-agent token and cost tracking
    - Lazy provider initialization
    """

    def __init__(self, assignments: Optional[list[ModelAssignment]] = None):
        """
        Args:
            assignments: List of ModelAssignment configurations.
                If None, loads defaults from model_assignments.py.
        """
        if assignments is None:
            from ..config.model_assignments import get_default_assignments

            assignments = get_default_assignments()

        self._assignments: dict[str, ModelAssignment] = {
            a.agent_type: a for a in assignments
        }
        self._providers: dict[str, BaseLLMProvider] = {}
        self._usage_stats: dict[str, dict] = {
            agent_type: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "errors": 0}
            for agent_type in self._assignments
        }

    # ================================================================
    # Public API
    # ================================================================

    async def generate(
        self,
        agent_type: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[type[BaseModel]] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response using the model assigned to this agent type.

        Args:
            agent_type: Which agent is making the request (e.g. 'writer', 'editor').
            system_prompt: System-level instruction.
            user_prompt: Content to process.
            response_model: Optional Pydantic model for structured JSON output.
            temperature_override: Override the configured temperature.
            max_tokens_override: Override configured max tokens.

        Returns:
            LLMResponse with content and usage metadata.
        """
        assignment = self._get_assignment(agent_type)
        provider = self._get_provider(assignment.provider)

        temperature = temperature_override if temperature_override is not None else assignment.temperature
        max_tokens = max_tokens_override if max_tokens_override is not None else assignment.max_tokens

        try:
            response = await provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=assignment.primary_model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_model,
            )
            self._track_success(agent_type, response)
            return response

        except Exception as e:
            logger.warning(f"Primary model {assignment.primary_model} failed for {agent_type}: {e}")

            if assignment.fallback_model:
                logger.info(f"Retrying with fallback model {assignment.fallback_model}")
                try:
                    response = await provider.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=assignment.fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_model,
                    )
                    self._track_success(agent_type, response)
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed for {agent_type}: {fallback_error}")
                    self._track_error(agent_type)
                    raise

            self._track_error(agent_type)
            raise

    async def generate_structured(
        self,
        agent_type: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> BaseModel:
        """Generate and parse a structured response.

        Shortcut that combines generate() with Pydantic model parsing.
        Includes fallback Chinese→English key translation for local models.
        """
        response = await self.generate(
            agent_type=agent_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        )

        # Layer 1: direct validation
        try:
            return response_model.model_validate_json(response.content)
        except (ValidationError, ValueError) as e:
            logger.debug(f"Direct validation failed, trying Chinese-key fallback: {e}")

        # Layer 2: Chinese→English key translation + type coercion
        try:
            data = _try_parse_json(response.content)

            # Guard: structured output must be a JSON object, not a primitive
            if not isinstance(data, dict):
                logger.warning(
                    f"LLM returned non-object JSON ({type(data).__name__}): "
                    f"{str(data)[:200]}"
                )
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")

            cn_to_en = _build_field_map(response_model)

            # Unwrap nested wrapper objects like {"世界观设定": {...}}
            if isinstance(data, dict) and len(data) == 1:
                sole_key = next(iter(data))
                sole_val = data[sole_key]
                if sole_key not in cn_to_en and isinstance(sole_val, dict):
                    logger.debug(f"Unwrapping nested key '{sole_key}'")
                    data = sole_val

            translated = _translate_keys(data, cn_to_en)
            logger.debug(f"Translated keys: {list(translated.keys())}")
            coerced = _coerce_field_types(translated, response_model)
            return response_model.model_validate(coerced)
        except (ValidationError, ValueError):
            raise  # Fallback also failed, let the original error propagate
        except Exception as fallback_err:
            logger.error(f"Fallback processing error: {fallback_err}", exc_info=True)
            raise  # Re-raise original validation error

    async def generate_stream(
        self,
        agent_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a text generation, yielding tokens as they arrive.

        Structured output (JSON mode) is NOT supported in streaming mode.
        """
        assignment = self._get_assignment(agent_type)
        provider = self._get_provider(assignment.provider)

        temperature = temperature_override if temperature_override is not None else assignment.temperature
        max_tokens = max_tokens_override if max_tokens_override is not None else assignment.max_tokens

        try:
            async for token in provider.generate_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=assignment.primary_model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield token
        except Exception as e:
            logger.warning(f"Primary model stream failed for {agent_type}: {e}")
            if assignment.fallback_model:
                async for token in provider.generate_stream(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=assignment.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield token
            else:
                raise

    def get_usage_stats(self) -> dict:
        """Get per-agent usage statistics."""
        return dict(self._usage_stats)

    def get_total_cost(self) -> float:
        """Get total estimated cost across all agents."""
        return round(sum(s["cost"] for s in self._usage_stats.values()), 6)

    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        for stats in self._usage_stats.values():
            stats["calls"] = 0
            stats["input_tokens"] = 0
            stats["output_tokens"] = 0
            stats["cost"] = 0.0
            stats["errors"] = 0

    # ================================================================
    # Internal
    # ================================================================

    def _get_assignment(self, agent_type: str) -> ModelAssignment:
        """Get model assignment for an agent type. Falls back to 'orchestrator' if unknown."""
        if agent_type not in self._assignments:
            logger.warning(f"Unknown agent_type '{agent_type}', using orchestrator assignment")
            return self._assignments.get("orchestrator", list(self._assignments.values())[0])
        return self._assignments[agent_type]

    def _get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get or lazily initialize a provider instance."""
        if provider_name not in self._providers:
            self._providers[provider_name] = self._init_provider(provider_name)
        return self._providers[provider_name]

    def _init_provider(self, provider_name: str) -> BaseLLMProvider:
        """Initialize a provider instance based on name."""
        settings = get_settings()

        if provider_name == "anthropic":
            if not settings.has_anthropic:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            return AnthropicProvider(api_key=settings.anthropic_api_key)

        elif provider_name == "openai":
            if not settings.has_openai:
                raise ValueError("OPENAI_API_KEY not configured")
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return OpenAIProvider(**kwargs)

        elif provider_name in ("modelscope", "dashscope"):
            # OpenAI-compatible providers — use the same API format
            if not settings.has_openai:
                raise ValueError(
                    f"OPENAI_API_KEY not configured for {provider_name}. "
                    f"Set OPENAI_API_KEY and OPENAI_BASE_URL in .env"
                )
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return OpenAIProvider(**kwargs)

        elif provider_name == "openrouter":
            if not settings.has_openrouter:
                raise ValueError("OPENROUTER_API_KEY not configured")
            return OpenAIProvider(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )

        elif provider_name == "ollama":
            return OpenAIProvider(
                api_key="ollama",  # Not used but required by OpenAI client
                base_url=f"{settings.ollama_base_url}/v1",
            )

        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def _track_success(self, agent_type: str, response: LLMResponse) -> None:
        """Track successful API call."""
        stats = self._usage_stats.setdefault(agent_type, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "errors": 0,
        })
        stats["calls"] += 1
        stats["input_tokens"] += response.input_tokens
        stats["output_tokens"] += response.output_tokens
        stats["cost"] += response.cost

    def _track_error(self, agent_type: str) -> None:
        """Track failed API call."""
        stats = self._usage_stats.setdefault(agent_type, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "errors": 0,
        })
        stats["errors"] += 1
