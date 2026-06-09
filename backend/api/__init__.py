"""API module for FastAPI endpoints and WebSocket."""

from .server import app, router, lifespan

__all__ = ["app", "router", "lifespan"]
