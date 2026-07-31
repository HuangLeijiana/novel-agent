"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..config.settings import get_settings
from .routes import router
from .websocket import ws_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    logger.info("Novel Agent starting...")
    settings = get_settings()
    settings.workspace_path.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Novel Agent shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Novel Agent",
        description="Multi-agent novel writing system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for MVP
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST API routes
    app.include_router(router, prefix="/api")

    # WebSocket endpoint
    @app.websocket("/ws/{project_id}")
    async def websocket_endpoint(ws: WebSocket, project_id: str):
        await ws_manager.connect(project_id, ws)
        try:
            while True:
                data = await ws.receive_text()
                # Client can send messages if needed
                logger.debug(f"WS message from {project_id}: {data[:100]}")
        except WebSocketDisconnect:
            await ws_manager.disconnect(project_id, ws)

    # Serve frontend static files
    import os

    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/")
        async def serve_frontend():
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
            return {"message": "Novel Agent API is running. Frontend not found."}

        # Middleware to disable caching for all static files
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        class NoCacheStaticMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                response = await call_next(request)
                if request.url.path.startswith("/static/"):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                return response

        app.add_middleware(NoCacheStaticMiddleware)

    return app
