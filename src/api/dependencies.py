"""FastAPI dependency injection — provides shared services to route handlers."""

from ..config.settings import get_settings, Settings
from ..llm.scheduler import ModelScheduler
from ..storage.file_manager import ProjectFileManager

# Singleton instances
_settings: Settings | None = None
_scheduler: ModelScheduler | None = None


def get_settings_instance() -> Settings:
    """Get the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_scheduler() -> ModelScheduler:
    """Get the global ModelScheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ModelScheduler()
    return _scheduler


def get_file_manager(project_id: str) -> ProjectFileManager:
    """Create a ProjectFileManager for a specific project."""
    settings = get_settings_instance()
    return ProjectFileManager(settings.workspace_path, project_id)
