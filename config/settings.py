"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings for AXIS."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AXIS"
    app_version: str = "0.2.0"
    debug: bool = Field(default=False, validation_alias="DEBUG")
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    base_dir: Path = Field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")
    cache_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / ".cache")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/axis"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    camera_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480

    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    ws_enabled: bool = True
    ws_heartbeat: int = 30


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.data_dir.mkdir(exist_ok=True)
    settings.models_dir.mkdir(exist_ok=True)
    settings.cache_dir.mkdir(exist_ok=True)
    return settings
