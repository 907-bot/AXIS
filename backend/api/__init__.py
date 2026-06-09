"""API module for FastAPI endpoints and WebSocket."""
from .server import app, router, lifespan
from .routes import setup_routes

__all__ = ["app", "router", "lifespan", "setup_routes"]