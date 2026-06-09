"""Main entry point for AXIS backend."""

import sys
from pathlib import Path

import uvicorn
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings


def setup_logging() -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.add(
        str(logs_dir / "axis_{time}.log"),
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )


def main() -> None:
    """Run AXIS server."""
    setup_logging()
    settings = get_settings()

    logger.info(f"Starting AXIS v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    uvicorn.run(
        "backend.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
