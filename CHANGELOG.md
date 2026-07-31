# Changelog

## [0.1.0] — 2026-07-31

### Added
- Initial public release
- 12-agent LangGraph pipeline: Topic Scout → Architect → Character Manager → Plot Planner → Writer → Editor → Continuity Checker → Reader Simulator → Reviewer → Refiner → Memory Manager → Orchestrator
- Provider-agnostic LLM layer supporting ModelScope, DeepSeek, DashScope, Anthropic, OpenAI, OpenRouter, Ollama
- Structured planning: world bible, character profiles, master outline, scene-level chapter plans
- Multi-pass review system: editorial → continuity → reader simulation → adversarial review
- Memory system: world facts, character states, foreshadowing tracking
- Web UI with real-time WebSocket progress updates
- JSON repair pipeline for Chinese LLM robustness (Qwen3 compatible)
- Two-pass chapter expansion for hitting word count targets
- Docker support (Dockerfile + docker-compose.yml)
- GitHub Actions CI: unit tests (3 Python versions) + optional integration tests
- `.env.example` with 5 provider presets
