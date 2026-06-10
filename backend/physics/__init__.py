"""Physics package for AXIS — rigid body simulation and interaction."""
from loguru import logger

try:
    from .simulation import PhysicsEngine, sim_force_vectors
    logger.info("Using MuJoCo physics engine")
    __all__ = ["PhysicsEngine", "sim_force_vectors"]
except Exception as e:
    logger.info(f"MuJoCo not available ({e}), using custom physics engine")
    from .engine import PhysicsEngine, RigidBody, MaterialDatabase, sim_force_vectors
    __all__ = ["PhysicsEngine", "RigidBody", "MaterialDatabase", "sim_force_vectors"]
