"""Main entry point for AXIS backend."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from loguru import logger

from config import get_settings


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    logger.add(
        "logs/axis_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )


def main():
    """Run AXIS server."""
    setup_logging()
    settings = get_settings()

    logger.info(f"Starting AXIS v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="debug" if settings.debug else "info"
    )


if __name__ == "__main__":
    main()