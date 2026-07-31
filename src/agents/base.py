"""Base agent class with prompt rendering and structured LLM generation.

Supports both inline prompts (for backward compat) and Jinja2 template-based
prompts (for externalized, editable prompt management).
"""

import logging
from abc import ABC
from typing import Any, AsyncGenerator, Optional

from pydantic import BaseModel

from ..config.prompts.prompt_loader import PromptLoader, get_prompt_loader
from ..llm.client import LLMResponse
from ..llm.scheduler import ModelScheduler

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all novel-writing agents.

    Each agent has:
    - An agent_type string that maps to a ModelAssignment
    - A reference to the ModelScheduler for LLM calls
    - Prompt rendering via Jinja2 templates (with inline fallback)
    """

    agent_type: str = "base"

    def __init__(self, scheduler: ModelScheduler, prompt_loader: Optional[PromptLoader] = None):
        self.scheduler = scheduler
        self._prompt_loader = prompt_loader

    # ================================================================
    # Core generation methods
    # ================================================================

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a text response from the assigned LLM.

        Args:
            system_prompt: System-level instructions for the LLM.
            user_prompt: The actual task content.
            temperature_override: Override the default temperature.
            max_tokens_override: Override the default max tokens.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        return await self.scheduler.generate(
            agent_type=self.agent_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        )

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> BaseModel:
        """Generate a structured response and parse it into a Pydantic model.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The actual task content.
            response_model: Pydantic model class to parse the response into.
            temperature_override: Override the default temperature.
            max_tokens_override: Override the default max tokens.

        Returns:
            An instance of response_model populated from the LLM output.
        """
        return await self.scheduler.generate_structured(
            agent_type=self.agent_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        )

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a text response from the assigned LLM, yielding tokens.

        For structured (JSON) output, use generate_structured() instead.
        """
        async for token in self.scheduler.generate_stream(
            agent_type=self.agent_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        ):
            yield token

    # ================================================================
    # Template-based prompt rendering
    # ================================================================

    @property
    def prompt_loader(self) -> PromptLoader:
        """Get the prompt loader, lazily initializing from singleton if needed."""
        if self._prompt_loader is None:
            self._prompt_loader = get_prompt_loader()
        return self._prompt_loader

    def render_prompt(
        self,
        method: str,
        prompt_type: str = "system",
        default: str = "",
        **variables: Any,
    ) -> str:
        """Render a prompt from a Jinja2 template, with inline fallback.

        Template path: {agent_type}/{method}_{prompt_type}.j2

        Args:
            method: Method name (e.g. 'generate_chapter').
            prompt_type: 'system' or 'user'.
            default: Fallback string if template doesn't exist.
            **variables: Variables passed to the template.

        Returns:
            Rendered prompt string (or default if template not found).
        """
        result = self.prompt_loader.render_or_none(
            agent=self.agent_type,
            method=method,
            prompt_type=prompt_type,
            **variables,
        )
        if result is not None:
            return result
        # Template not found — log once and use default
        logger.debug(f"No template for {self.agent_type}/{method}_{prompt_type}.j2 — using inline prompt")
        return default

    def has_template(self, method: str) -> bool:
        """Check if both system+user templates exist for a method."""
        return self.prompt_loader.has_template(self.agent_type, method)

    # ================================================================
    # Prompt helpers (inline fallback — kept for backward compat)
    # ================================================================

    @staticmethod
    def build_system_prompt(role: str, expertise: str, constraints: str = "", inspiration: str = "") -> str:
        """Build a standard system prompt for an agent.

        Args:
            role: The agent's role description.
            expertise: What the agent is expert at.
            constraints: Any specific constraints or rules.
            inspiration: Optional human-provided inspiration/ideas.

        Returns:
            Formatted system prompt string.
        """
        prompt = f"""你是一位{role}。

你的专长：{expertise}

工作原则：
1. 严格遵循给定的格式要求，输出结构化数据
2. 基于提供的上下文信息进行创作和判断，不要凭空编造
3. 保持一致性，注意前后逻辑连贯
4. 用中文思考和输出（除非上下文指定了其他语言）
"""
        if constraints:
            prompt += f"\n特别约束：\n{constraints}"

        if inspiration:
            prompt += f"\n\n【用户灵感/要求】\n{inspiration}\n请参考以上用户提供的灵感和要求进行创作。"

        return prompt

    @staticmethod
    def build_context_block(
        project_config: Optional[dict] = None,
        bible: Optional[dict] = None,
        characters: Optional[dict] = None,
        outline: Optional[dict] = None,
        memory: Optional[dict] = None,
    ) -> str:
        """Build a context block string from available artifacts.

        Pass dicts obtained from model_dump() to avoid coupling to model imports.
        """
        parts = []

        if project_config:
            parts.append("=== 项目配置 ===")
            parts.append(f"题材: {project_config.get('genre', [])}")
            parts.append(f"目标读者: {project_config.get('target_readers', '')}")
            parts.append(f"语调: {project_config.get('tone', '')}")
            parts.append(f"语言: {project_config.get('language', 'zh-CN')}")
            parts.append("")

        if bible:
            parts.append("=== 世界观设定 ===")
            world = bible.get("world", {})
            if world:
                parts.append(f"世界名称: {world.get('name', '')}")
                parts.append(f"世界类型: {world.get('world_type', '')}")
                parts.append(f"地理: {world.get('geography', '')}")
                parts.append(f"历史: {world.get('history', '')}")
                parts.append(f"文化: {world.get('culture', '')}")
                if world.get("magic_system"):
                    parts.append(f"魔法/力量体系: {world['magic_system']}")

            style = bible.get("style_contract", {})
            if style:
                parts.append(f"文风语调: {style.get('tone', '')}")
                parts.append(f"句式风格: {style.get('sentence_style', '')}")

            themes = bible.get("themes", [])
            if themes:
                theme_names = [t.get("name", "") for t in themes]
                parts.append(f"主题: {', '.join(theme_names)}")

            parts.append("")

        if characters:
            parts.append("=== 角色档案 ===")
            for char_id, char in characters.items():
                if isinstance(char, dict):
                    parts.append(f"[{char_id}] {char.get('name', '')} ({char.get('role', '')})")
                    parts.append(f"  性格: {char.get('personality', '')}")
                    parts.append(f"  动机: {char.get('motivation', '')}")
                    parts.append(f"  缺陷: {char.get('flaw', '')}")
                    state = char.get("current_state", {})
                    if state:
                        parts.append(f"  当前状态: {state}")
            parts.append("")

        if outline:
            parts.append("=== 大纲参考 ===")
            parts.append(f"主线: {outline.get('logline', '')}")
            volumes = outline.get("volumes", [])
            if volumes:
                for v in volumes:
                    if isinstance(v, dict):
                        parts.append(f"  卷{v.get('number', '?')}: {v.get('title', '')}")
            parts.append("")

        if memory:
            st = memory.get("short_term", {})
            if st:
                parts.append("=== 近期记忆 ===")
                parts.append(f"上一章摘要: {st.get('current_chapter_summary', '')}")
                parts.append(f"未解决钩子: {st.get('unresolved_hooks', [])}")
                parts.append(f"活跃伏笔: {st.get('active_foreshadowing', [])}")
            parts.append("")

        return "\n".join(parts)
