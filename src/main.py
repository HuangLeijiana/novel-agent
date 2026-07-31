"""Novel Agent — Entry point."""

import logging
import sys

import uvicorn

from .api.server import create_app
from .config.settings import get_settings


def main():
    """Start the Novel Agent server."""
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Novel Agent v0.1.0")
    logger.info(f"Workspace: {settings.workspace_path}")

    # Check API key availability (provider-agnostic)
    configs = []
    if settings.has_anthropic:
        configs.append("Anthropic")
    if settings.has_openai:
        configs.append("OpenAI")
    if settings.has_openrouter:
        configs.append("OpenRouter")
    if settings.openai_base_url:
        # Detect ModelScope / DashScope from base URL
        if "modelscope" in settings.openai_base_url:
            configs.append("ModelScope")
        elif "dashscope" in settings.openai_base_url or "aliyuncs" in settings.openai_base_url:
            configs.append("DashScope")
        elif "deepseek" in settings.openai_base_url:
            configs.append("DeepSeek")
        else:
            configs.append(f"OpenAI-compatible ({settings.openai_base_url})")

    if configs:
        logger.info(f"API providers configured: {', '.join(configs)}")
        logger.info(f"Active provider: {settings.active_provider}")
    else:
        logger.warning(
            "No API keys configured! Set at least one of:\n"
            "  - ANTHROPIC_API_KEY + DEFAULT_PROVIDER=anthropic\n"
            "  - OPENAI_API_KEY + DEFAULT_PROVIDER=openai\n"
            "  - OPENROUTER_API_KEY + DEFAULT_PROVIDER=openrouter\n"
            "  - OPENAI_API_KEY + OPENAI_BASE_URL (ModelScope/DashScope/DeepSeek)\n"
            "  - Or use local Ollama: DEFAULT_PROVIDER=ollama"
        )

    # Create app and serve
    app = create_app()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
