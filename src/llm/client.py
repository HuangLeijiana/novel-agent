"""LLM client abstraction — unified interface for multiple providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: str = Field(..., description="Raw text response")
    model: str = Field(..., description="Model ID used")
    provider: str = Field(..., description="Provider name")
    input_tokens: int = Field(default=0, description="Input token count")
    output_tokens: int = Field(default=0, description="Output token count")
    cost: float = Field(default=0.0, description="Estimated cost in USD")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User message / content to process.
            model: Model identifier for this provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional Pydantic model for structured output.
                If provided, the response is parsed into this model.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        ...

    @abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> BaseModel:
        """Generate a response and parse it into a Pydantic model.

        This is a convenience wrapper around generate() with response_format.
        """
        ...

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming text response, yielding tokens as they arrive.

        Structured output (JSON mode) is NOT supported in streaming mode —
        use generate_structured() for that.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User message / content to process.
            model: Model identifier for this provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Yields:
            Text chunks from the LLM as they are generated.
        """
        # Default implementation: call generate() and yield full content
        # Providers that support true streaming should override this
        response = await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.content
