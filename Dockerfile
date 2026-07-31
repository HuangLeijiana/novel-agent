# ── Build stage: install dependencies with uv ──
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ── Runtime stage ──
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Use the virtualenv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Default: start with .env.example — user should mount their own .env
CMD ["python", "-m", "src.main"]
