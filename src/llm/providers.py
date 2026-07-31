"""LLM provider implementations — Anthropic, OpenAI, OpenRouter, Ollama."""

import json
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ValidationError

from .client import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ============================================================
# Shared helpers for structured output
# ============================================================


def _extract_json(text: str) -> str:
    """Extract JSON from text that may have markdown code fences or // comments."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines:
            lines = lines[1:]  # Drop ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Strip // line comments (Qwen3 copies these from schema examples)
    # Only strip // that appear at the start of a line or after a field value,
    # NOT // inside strings. A simple heuristic: strip from // to end-of-line
    # when // appears outside of quoted strings.
    cleaned = []
    for line in text.split("\n"):
        in_string = False
        escaped = False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                # Found // outside a string — rest of line is a comment
                line = line[:i].rstrip()
                break
        if line.strip():  # Skip empty lines from comment removal
            cleaned.append(line)
    text = "\n".join(cleaned)

    return text.strip()


def _build_field_map(response_model: type[BaseModel]) -> dict[str, str]:
    """Build mapping from Chinese description → English field name."""
    mapping: dict[str, str] = {}
    for name, field in response_model.model_fields.items():
        desc = (field.description or "").strip()
        if desc:
            mapping[desc] = name
    return mapping


def _translate_keys(data: dict, cn_to_en: dict[str, str]) -> dict:
    """Recursively translate Chinese dict keys to English field names."""
    if not isinstance(data, dict):
        return data
    result: dict = {}
    for key, value in data.items():
        en_key = cn_to_en.get(key, key)
        if isinstance(value, dict):
            value = _translate_keys(value, cn_to_en)
        elif isinstance(value, list):
            value = [_translate_keys(item, cn_to_en) if isinstance(item, dict) else item for item in value]
        result[en_key] = value
    return result


def _coerce_field_types(data: dict, response_model: type[BaseModel]) -> dict:
    """Coerce field values to match the expected Pydantic types.

    Local models often return nested objects for str fields (e.g. magic_system)
    or JSON-encoded strings for list/dict fields.  This helper fixes both.
    """
    for name, field in response_model.model_fields.items():
        if name not in data:
            continue
        value = data[name]
        if value is None:
            continue
        ann = field.annotation
        if ann is None:
            continue

        origin = getattr(ann, "__origin__", None)
        args = getattr(ann, "__args__", ())

        # Field expects plain str → just the class itself
        wants_str = ann is str
        # Field expects Optional[str] = Union[str, NoneType]
        wants_optional_str = origin is not None and str in args and type(None) in args
        if (wants_str or wants_optional_str) and isinstance(value, (dict, list)):
            data[name] = json.dumps(value, ensure_ascii=False)
            continue

        # Field expects list but got a string → try json.loads, or wrap
        wants_list = (ann is list) or (origin is list)
        if wants_list and isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        data[name] = parsed
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            # Last resort: wrap the string in a single-element list
            data[name] = [stripped]
            continue

        # Field expects dict but got a string → try json.loads
        wants_dict = (ann is dict) or (origin is dict)
        if wants_dict and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    data[name] = parsed
            except (json.JSONDecodeError, TypeError):
                pass
            continue

        # Field expects list but got bool/int/float → wrap
        # MUST run before the str-coercion block below, otherwise bool values
        # like True get str()'d into 'True' instead of being wrapped in a list.
        if wants_list and isinstance(value, (bool, int, float)):
            data[name] = [value] if value is not True else []
            continue

        # Field expects list but got dict → merge all list values
        if wants_list and isinstance(value, dict):
            merged = []
            for v in value.values():
                if isinstance(v, list):
                    merged.extend(v)
                elif isinstance(v, str):
                    merged.append(v)
            if merged:
                data[name] = merged
            continue

        # Field expects str but got int/float/bool → convert
        wants_str = (ann is str) or (origin is not None and str in args and type(None) in args)
        if wants_str and isinstance(value, (int, float, bool)):
            data[name] = str(value)
            continue
        # Also handle non-Optional str fields
        if ann is str and isinstance(value, (int, float, bool)):
            data[name] = str(value)
            continue

    return data


def _build_chinese_schema_prompt(response_model: type[BaseModel]) -> str:
    """Build a Chinese-language schema description for local models.

    Instead of dumping raw JSON Schema (which confuses Chinese-native models
    with its English field names + Chinese descriptions), we build an explicit
    template showing exactly what keys to use.
    """
    lines = [
        "请严格按照以下 JSON 格式输出，**字段名必须使用英文**（括号内为中文说明），不要使用中文作为 JSON key：",
        "",
        "```json",
        "{",
    ]

    for name, field in response_model.model_fields.items():
        desc = field.description or name
        ann = field.annotation

        # Determine a sample value based on the field type
        if ann is str:
            sample = '"..."'
        elif ann is int:
            sample = "0"
        elif ann is float:
            sample = "0.0"
        elif ann is bool:
            sample = "false"
        elif ann is not None:
            origin = getattr(ann, "__origin__", None)
            args = getattr(ann, "__args__", ())
            if origin is list:
                sample = '["..."]'
            elif origin is dict:
                sample = "{}"
            elif type(None) in args:
                sample = "null"
            else:
                sample = '"..."'
        else:
            sample = '"..."'

        required = field.is_required()
        required_note = " (required)" if required else ""
        comma = ","
        lines.append(f'  "{name}": {sample}{comma}  # {desc}{required_note}')

    lines.extend(
        [
            "}",
            "```",
            "",
            "**重要提醒**：",
            "- JSON 的 key 必须使用英文（如上面的 name, world_type），绝对不能使用中文作为 key",
            "- 值的内容用中文",
            "- 不要在上面的 JSON 中添加注释（// 或 # 等），输出必须是纯 JSON",
        ]
    )

    return "\n".join(lines)


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""

    provider_name = "anthropic"

    def __init__(self, api_key: str):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        sys_prompt = system_prompt
        if response_format is not None:
            schema_json = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
            sys_prompt += (
                f"\n\nYou MUST respond with valid JSON matching this schema exactly. "
                f"Do not include any text outside the JSON object:\n{schema_json}"
            )

        message = await self.client.messages.create(
            model=model,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = message.content[0].text if message.content else ""
        input_tokens = message.usage.input_tokens if message.usage else 0
        output_tokens = message.usage.output_tokens if message.usage else 0

        if response_format is not None:
            content = _extract_json(content)

        cost = self._estimate_cost(model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> BaseModel:
        response = await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_model,
        )
        return response_model.model_validate_json(response.content)

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on Anthropic pricing."""
        # Approximate pricing per 1M tokens (as of 2025)
        pricing = {
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-haiku-4-5-20251001": (0.8, 4.0),
        }
        input_price, output_price = pricing.get(model, (3.0, 15.0))
        cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
        return round(cost, 6)


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs (including Ollama, OpenRouter).

    For local models (Ollama) that may not follow English field-name instructions,
    we add three layers of defence:
    1. Chinese-language schema prompt with explicit field-name requirements
    2. Markdown code-fence extraction
    3. Fallback Chinese→English key translation using Field descriptions
    """

    provider_name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self._base_url = base_url

    @property
    def _is_ollama(self) -> bool:
        """Detect whether this provider is backed by Ollama."""
        if self._base_url:
            return "localhost" in self._base_url or "ollama" in self._base_url
        return False

    @property
    def _is_third_party(self) -> bool:
        """Detect third-party OpenAI-compatible APIs (ModelScope, DashScope, etc.).

        These often don't support ``response_format: {"type": "json_object"}``
        correctly, so we use prompt-based schema injection instead.
        """
        if not self._base_url:
            return False  # Default OpenAI
        if self._is_ollama:
            return False  # Handled separately
        return "api.openai.com" not in self._base_url

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        sys_prompt = system_prompt
        use_json_mode = False  # Whether to use API-level response_format

        if response_format is not None:
            # For Ollama and third-party APIs (ModelScope etc.), inject the
            # schema into the prompt because they don't reliably support
            # response_format: {"type": "json_object"}.
            if self._is_ollama or self._is_third_party:
                sys_prompt += "\n\n" + _build_chinese_schema_prompt(response_format)
            else:
                # Real OpenAI or OpenRouter — these support json_object mode
                schema_json = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
                sys_prompt += (
                    f"\n\nYou MUST respond with valid JSON matching this schema exactly. "
                    f"Do not include any text outside the JSON object:\n{schema_json}"
                )
                use_json_mode = True

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if response_format is not None:
            content = _extract_json(content)

        cost = self._estimate_cost(model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> BaseModel:
        response = await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_model,
        )

        # Layer 1: direct validation
        try:
            return response_model.model_validate_json(response.content)
        except (ValidationError, ValueError) as e:
            logger.debug(f"Direct validation failed, trying Chinese-key fallback: {e}")

        # Layer 2: try Chinese→English key translation
        try:
            from .scheduler import _try_parse_json

            data = _try_parse_json(response.content)
            cn_to_en = _build_field_map(response_model)

            # If the top-level object has a single Chinese key whose value
            # is a dict, unwrap it (model wrapped everything, e.g.
            # {"世界观设定": {...actual data...}})
            if isinstance(data, dict) and len(data) == 1:
                sole_key = next(iter(data))
                sole_val = data[sole_key]
                if sole_key not in cn_to_en and isinstance(sole_val, dict):
                    logger.debug(f"Unwrapping nested key '{sole_key}'")
                    data = sole_val

            translated = _translate_keys(data, cn_to_en)
            coerced = _coerce_field_types(translated, response_model)
            return response_model.model_validate(coerced)
        except Exception:
            raise  # Re-raise the original error if fallback also fails

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM as they are generated.

        Uses OpenAI-compatible streaming (works with Ollama, OpenRouter, etc.).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        response = await self.client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on OpenAI pricing. Ollama models are free."""
        if self._is_ollama:
            return 0.0
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
        }
        input_price, output_price = pricing.get(model, (2.5, 10.0))
        cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
        return round(cost, 6)
