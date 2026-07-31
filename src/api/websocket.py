"""WebSocket manager for real-time workflow updates."""

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections per project for real-time UI updates."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, project_id: str, ws: WebSocket) -> None:
        """Accept a WebSocket connection and register it for a project."""
        await ws.accept()
        if project_id not in self._connections:
            self._connections[project_id] = []
        self._connections[project_id].append(ws)
        logger.info(f"WebSocket connected for project {project_id} "
                     f"(total: {len(self._connections[project_id])})")

    async def disconnect(self, project_id: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if project_id in self._connections:
            self._connections[project_id] = [
                c for c in self._connections[project_id] if c != ws
            ]
            if not self._connections[project_id]:
                del self._connections[project_id]
        logger.info(f"WebSocket disconnected for project {project_id}")

    async def broadcast(self, project_id: str, event: dict[str, Any]) -> None:
        """Send an event to all WebSocket connections for a project."""
        if project_id not in self._connections:
            return
        message = json.dumps(event, ensure_ascii=False, default=str)
        dead = []
        for ws in self._connections[project_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(project_id, ws)

    async def broadcast_phase_update(
        self, project_id: str, phase: str, progress: float, message: str,
    ) -> None:
        """Broadcast a workflow phase update."""
        await self.broadcast(project_id, {
            "type": "phase_update",
            "phase": phase,
            "progress": progress,
            "message": message,
        })

    async def broadcast_agent_message(
        self, project_id: str, agent: str, content: str,
    ) -> None:
        """Broadcast an agent output message."""
        await self.broadcast(project_id, {
            "type": "agent_message",
            "agent": agent,
            "content": content,
        })

    async def broadcast_token(
        self, project_id: str, agent: str, token: str,
    ) -> None:
        """Broadcast a single streaming token for real-time text display."""
        await self.broadcast(project_id, {
            "type": "token",
            "agent": agent,
            "token": token,
        })

    async def broadcast_stream_end(
        self, project_id: str, agent: str, full_text: str = "",
    ) -> None:
        """Signal end of a streaming response."""
        await self.broadcast(project_id, {
            "type": "stream_end",
            "agent": agent,
            "full_text": full_text,
        })

    async def broadcast_phase_complete(
        self, project_id: str, phase: str, data: dict = None,
    ) -> None:
        """Broadcast that a workflow phase has completed and is waiting for confirmation."""
        await self.broadcast(project_id, {
            "type": "phase_complete",
            "phase": phase,
            "data": data or {},
        })

    async def broadcast_phase_blocked(
        self, project_id: str, phase: str,
    ) -> None:
        """Broadcast that a phase is blocked waiting for human confirmation."""
        await self.broadcast(project_id, {
            "type": "phase_blocked",
            "phase": phase,
            "human_input_required": True,
        })

    async def broadcast_error(
        self, project_id: str, phase: str, error: str,
    ) -> None:
        """Broadcast an error."""
        await self.broadcast(project_id, {
            "type": "error",
            "phase": phase,
            "error": error,
        })

    async def broadcast_chapter_complete(
        self, project_id: str, chapter_number: int, scores: dict,
    ) -> None:
        """Broadcast chapter completion with scores."""
        await self.broadcast(project_id, {
            "type": "chapter_complete",
            "chapter": chapter_number,
            "scores": scores,
        })

    async def broadcast_human_input_required(
        self, project_id: str, chapter_number: int, context: dict,
    ) -> None:
        """Notify the UI that human input is needed."""
        await self.broadcast(project_id, {
            "type": "human_input_required",
            "chapter": chapter_number,
            "context": context,
        })


# Global singleton
ws_manager = WebSocketManager()
