"""Application settings and configuration."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for AXIS."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "AXIS"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, validation_alias="DEBUG")
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Path("/workspace/project/AXIS/data")
    models_dir: Path = Path("/workspace/project/AXIS/models")
    cache_dir: Path = Path("/workspace/project/AXIS/.cache")

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/axis"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Camera & SLAM
    camera_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    slam_enabled: bool = True
    reconstruction_interval: int = 10

    # Models
    sam_model: str = "sam2.1_hiera_large"
    dino_model: str = "dinov2_vitl14"
    clip_model: str = "ViT-L/14"

    # LLM
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # WebSocket
    ws_enabled: bool = True
    ws_heartbeat: int = 30

    # GPU
    cuda_enabled: bool = True
    gpu_device: int = 0


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()