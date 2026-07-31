# Contributing to Novel Agent

Thanks for your interest in contributing! This guide will help you get set up and make changes.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/novel-agent.git
   cd novel-agent
   ```
3. **Install dependencies**:
   ```bash
   uv sync --dev
   ```
4. **Configure** your LLM provider:
   ```bash
   cp .env.example .env
   # Edit .env with your API key
   ```

## Development Workflow

```bash
# Run unit tests (no API key needed)
uv run pytest tests/ -v -k "not integration"

# Run all tests (needs API key)
uv run pytest tests/ -v

# Run the app
uv run python -m src.main
```

## Project Structure

```
src/
├── agents/       # 12 AI agents — each has one responsibility
├── api/          # FastAPI server, routes, WebSocket
├── config/       # Settings, model assignments, prompt templates
├── frontend/     # Web UI (vanilla JS)
├── graph/        # LangGraph pipeline
├── llm/          # Model scheduler, providers, JSON repair
├── models/       # Pydantic data models
├── storage/      # File I/O for workspaces
└── utils/        # Content safety, cost calculator, Jinja2 helpers
```

## Code Style

- **Python**: Follow PEP 8. Use type hints on all function signatures.
- **Pydantic models**: Define in `src/models/`. Every new model should have `model_validator` defenses for Chinese LLM output quirks (field name aliases, string coercion).
- **Agents**: Extend `BaseAgent`. Use `generate_structured()` for JSON output, `generate()` for free text.
- **Commit messages**: Use conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

## Adding a New LLM Provider

1. Add provider detection in `src/config/settings.py` (`PROVIDER_PRESETS` and auto-detect logic)
2. The `OpenAIProvider` in `src/llm/providers.py` handles all OpenAI-compatible APIs (ModelScope, DeepSeek, DashScope, Ollama) — new compatible APIs need no code changes, just `.env` configuration
3. For non-OpenAI-compatible APIs, add a new provider class in `src/llm/providers.py`

## Testing

- **Unit tests** (`tests/test_unit_*.py`) — No API key needed. Test Pydantic models, prompt building, JSON repair, content safety.
- **Integration tests** (`tests/test_*_modelscope.py`) — Require a real LLM. Mark with `-k "modelscope"` to run selectively.

Before submitting a PR, run:
```bash
uv run pytest tests/ -v
```

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] New code has type hints
- [ ] New Pydantic models have `model_validator` for Chinese LLM compat
- [ ] No new dependencies without discussion first
- [ ] Commit history is clean (squash WIP commits)

## Questions?

Open an issue with the `question` label — we'll get back to you.
