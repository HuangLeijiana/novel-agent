# Novel Agent

**Multi-agent AI novel writing system** — 12 specialized agents collaborate through a LangGraph pipeline to produce complete, coherent web novels from idea to finished chapters.

> 🤖 Powered by LLMs (ModelScope, DeepSeek, Claude, OpenAI, Ollama) | 🎨 Frontend control panel | 📋 Structured planning pipeline

---

## Features

- **12-Agent Pipeline** — Topic Scout → Architect → Character Manager → Plot Planner → Writer → Editor → Continuity Checker → Reader Simulator → Reviewer → Refiner → Memory Manager → Orchestrator
- **Provider Agnostic** — Works with ModelScope (free), DeepSeek, DashScope, Claude, OpenAI, OpenRouter, or local Ollama
- **Structured Planning** — World bible, character profiles, master outline, scene-level chapter plans before a single word is written
- **Quality Control** — Multi-pass review: editorial feedback → continuity checks → reader simulation → adversarial review → refinement
- **Memory System** — Tracks world facts, character states, foreshadowing, and continuity across the entire novel
- **Web UI** — Real-time control panel with WebSocket progress updates
- **Zero-cost Start** — ModelScope free tier (2000 calls/day) lets you generate a complete novel for free

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Clone & Install

```bash
git clone https://github.com/HuangLeijiana/novel-agent.git
cd novel-agent
uv sync
```

### 2. Configure (30 seconds)

```bash
cp .env.example .env
```

Edit `.env` and choose your LLM provider. **To start for free with ModelScope:**

```bash
# 1. Register at https://modelscope.cn → get your SDK Token
# 2. Uncomment these three lines in .env:
OPENAI_API_KEY=ms-your-token-here
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
DEFAULT_PROVIDER=modelscope
DEFAULT_MODEL=Qwen/Qwen3-235B-A22B
```

Other providers: DeepSeek (¥5-10/novel), DashScope (¥4/M tokens), Claude ($15/M output), Ollama (free, local GPU).

### 3. Launch

```bash
uv run novel-agent
# Or: uv run python -m src.main
```

Open **http://127.0.0.1:8000** in your browser.

### 4. Write Your Novel

1. Enter your novel idea, genre, and target audience
2. Click through the pipeline phases — each phase builds on the previous one
3. Watch as agents collaborate: outline → characters → chapters → review → polish
4. Export finished chapters as `.docx`

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Web UI (Vanilla JS)                    │
│              Real-time WebSocket progress                 │
└──────────────────────┬───────────────────────────────────┘
                       │ FastAPI + WebSocket
┌──────────────────────▼───────────────────────────────────┐
│                   Orchestrator Agent                      │
│              Coordinates 12-agent pipeline                │
└──────┬───────┬───────┬───────┬───────┬──────────────────┘
       │       │       │       │       │
  ┌────▼──┐ ┌─▼──┐ ┌──▼──┐ ┌─▼───┐ ┌─▼──────┐
  │Topic  │ │Arch│ │Char │ │Plot │ │Writer   │   Creative
  │Scout  │ │itect│ │Mgr │ │Plan │ │(2-pass) │   Tier
  └───────┘ └────┘ └─────┘ └─────┘ └────────┘
  ┌───────┐ ┌────┐ ┌─────┐ ┌─────┐ ┌────────┐
  │Editor │ │Cont│ │Read │ │Rev  │ │Refiner │   Review
  │       │ │inuity│ │Sim  │ │iewer│ │        │   Tier
  └───────┘ └────┘ └─────┘ └─────┘ └────────┘
                       │
              ┌────────▼────────┐
              │  Memory Manager  │   State
              │ (facts, states,  │   Tier
              │  foreshadowing)  │
              └─────────────────┘
```

### Project Structure

```
novel-agent/
├── src/
│   ├── agents/          # 12 AI agents (writer, editor, reviewer, etc.)
│   ├── api/             # FastAPI server, WebSocket, routes
│   ├── config/          # Settings, model assignments
│   ├── frontend/        # Web UI (index.html + vanilla JS + CSS)
│   ├── graph/           # ★ LangGraph workflow — the real runtime engine
│   │   ├── workflow.py  #   StateGraph: Phase 0 → bible → characters → outline → chapter loop
│   │   ├── nodes.py     #   Phase nodes (executors + persistence + progress broadcasts)
│   │   └── edges.py     #   Conditional routing: review accept/revise/rewrite, next-chapter/done
│   ├── llm/             # Model scheduler, providers, JSON repair
│   ├── models/          # Pydantic data models (outline, bible, chapter, etc.)
│   ├── storage/         # Workspace file I/O
│   └── utils/           # Jinja2 templates, helpers
├── tests/               # Unit + integration tests
├── .env.example         # Configuration template with 5 provider presets
├── pyproject.toml
└── README.md
```

### How the graph runs

The workflow is a **real LangGraph StateGraph** (`src/graph/workflow.py`) — the
FastAPI layer invokes it directly; there is no parallel hand-written loop.

- **Routing is live**: `edges.py` decides review accept/revise/rewrite (with
  `MAX_REVIEW_ITERATIONS` forcing acceptance) and the next-chapter/done
  transition; chapters loop until the outline's chapter count is reached.
- **Human-in-the-loop is `interrupt()`-based**: after each top-level phase
  (scan → topic → mini-arc → bible → characters → outline) the graph pauses;
  the API layer broadcasts `phase_blocked` and resumes with
  `Command(resume=...)` when you confirm — persisted by the checkpointer
  (MemorySaver, or AsyncSqliteSaver via `build_async_workflow(db_path)`).
- **Resume & skip are preserved**: completed artifacts are detected from disk,
  so re-running a project skips finished phases and resumes mid-chapter.


## Pipeline Phases

| Phase | Agent | What It Does |
|-------|-------|-------------|
| 1 | **Topic Scout** | Market analysis, trending topics, title generation |
| 2 | **Architect** | World-building bible (geography, magic system, factions) |
| 3 | **Character Manager** | Character profiles with arcs, flaws, motivations |
| 4 | **Plot Planner** | Master outline with volumes, chapters, turning points |
| 5 | **Writer** | Chapter drafts with scene-level planning + auto-expansion |
| 6 | **Editor** | Structural feedback, pacing, dialogue review |
| 7 | **Continuity Checker** | Fact consistency, timeline validation |
| 8 | **Reader Simulator** | Reader engagement scoring, "boring" detection |
| 9 | **Reviewer** | Adversarial review — finds blind spots |
| 10 | **Refiner** | Applies edits based on review feedback |
| ∞ | **Memory Manager** | Runs continuously: tracks world facts, character states, foreshadowing |

## Configuration

### Provider Presets

| Provider | Model (Quality) | Cost | Setup |
|----------|----------------|------|-------|
| **ModelScope** | Qwen3-235B | Free (2000 calls/day) | SDK Token |
| **DeepSeek** | deepseek-chat | ~¥5-10/novel | API Key |
| **DashScope** | qwen-max | ¥4/M output tokens | API Key |
| **Claude** | claude-sonnet-4 | $15/M output | API Key |
| **OpenAI** | gpt-4o | $10/M output | API Key |
| **OpenRouter** | 200+ models | Varies | API Key |
| **Ollama** | qwen3:32b | Free (local GPU) | None |

### Per-Agent Overrides

You can assign different models to different agents for cost optimization:

```bash
# Writer needs the best model — use Claude
WRITER_MODEL=anthropic:claude-sonnet-4-20250514
# Reviewers can use cheaper models
EDITOR_MODEL=deepseek-chat
CONTINUITY_CHECKER_MODEL=deepseek-chat
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CHAPTER_WORD_COUNT` | 3000 | Target words per chapter |
| `MAX_REVIEW_ITERATIONS` | 3 | Max edit-review cycles |
| `MIN_REVIEW_SCORE_ACCEPT` | 6.5 | Minimum score to pass review |
| `MAX_CONTEXT_CHAPTERS` | 5 | Recent chapters in context window |
| `WORKSPACE_ROOT` | ./workspace | Where novels are saved |

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests (no API key needed, runs in < 1s)
uv run pytest tests/test_unit.py -v

# Run integration tests (needs API key configured in .env)
uv run pytest tests/ -v -k "modelscope"

# Run all tests
uv run pytest tests/ -v

# Run the app in dev mode
uv run python -m src.main
```

### Docker

```bash
# Build and run with docker-compose
cp .env.example .env   # edit with your API key first
docker compose up -d

# Or build manually
docker build -t novel-agent .
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env:ro -v $(pwd)/workspace:/app/workspace novel-agent
```

### Testing

| Type | Command | API Key | What It Tests |
|------|---------|---------|---------------|
| **Unit** | `pytest tests/test_unit.py -v` | No | Pydantic models, JSON repair, content safety, prompts, settings (50+ tests) |
| **Integration** | `pytest tests/ -v -k "modelscope"` | Yes | End-to-end agent pipeline against real LLMs |

CI runs unit tests on Python 3.11/3.12/3.13 on every push and PR via GitHub Actions.

## 记忆检索评测（可复现）

RAG 记忆检索质量由 `eval/eval_rag_retrieval.py` 量化：

```bash
# 1. 安装 rag 依赖（chromadb + sentence-transformers）
uv sync --extra rag

# 2. 运行评测（数据使用仓库内 eval/fixtures/memory 的 42 条设定事实与 30 条时间线事件）
uv run python eval/eval_rag_retrieval.py
```

脚本走项目自身的 `NovelVectorStore + Embedder` 代码路径（BGE-small-zh 512 维）；每条记忆由本地 Ollama qwen3:8b 改写为自然中文检索问题后查询 top-k，统计 recall@k / MRR（Ollama 不可用时自动退化为原文查询，可用 `--bge-model` 指定本地模型路径）。

结果写入 `eval/rag_retrieval_results.json`（当前结果：42 条事实 recall@1=100%、30 条事件 recall@5=86.7%、MRR=0.78）。

## FAQ

**Q: Can I really write a complete novel for free?**
A: Yes. ModelScope provides 2000 free API calls per day. A 30-chapter novel uses ~150 calls (5 phases + 30 chapters × ~5 agent calls each).

**Q: What genre does it support?**
A: All genres. The system works with any setting — xianxia, sci-fi, romance, urban, horror. You define the genre and tone in the project config.

**Q: How much manual editing is needed?**
A: The pipeline produces complete, coherent drafts. Most users do light editing (dialogue polish, pacing tweaks) rather than rewrites. The multi-pass review system catches most issues automatically.

**Q: Does it support non-Chinese languages?**
A: The pipeline is language-aware. Set `language` in your project config. Prompt templates are in Chinese but the system works with any language the underlying LLM supports.

## License

MIT © [Huang Lei](https://github.com/HuangLeijiana)
