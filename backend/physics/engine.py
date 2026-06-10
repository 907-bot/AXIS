"""Lightweight rigid-body physics engine for AXIS Phase 5.

Provides basic dynamics, collision detection, and material estimation
for interactive object simulation in the browser.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

GRAVITY = np.array([0.0, -9.81, 0.0], dtype=float)
GROUND_Y = 0.0
MAX_BODIES = 50


@dataclass
class MaterialProperties:
    """Estimated physical material properties for an object class."""
    density: float = 500.0        # kg/m³
    restitution: float = 0.3      # bounciness 0-1
    static_friction: float = 0.5
    dynamic_friction: float = 0.3
    label: str = "default"


class MaterialDatabase:
    """Maps class names to estimated material properties."""

    DEFAULTS: Dict[str, MaterialProperties] = {
        "bottle":           MaterialProperties(900, 0.25, 0.6, 0.4, "plastic"),
        "cup":              MaterialProperties(800, 0.20, 0.5, 0.3, "ceramic"),
        "book":             MaterialProperties(700, 0.10, 0.8, 0.6, "paper"),
        "chair":            MaterialProperties(400, 0.30, 0.7, 0.5, "wood"),
        "table":            MaterialProperties(600, 0.25, 0.7, 0.5, "wood"),
        "laptop":           MaterialProperties(1200, 0.05, 0.4, 0.2, "metal"),
        "cell phone":       MaterialProperties(1500, 0.10, 0.3, 0.2, "glass"),
        "person":           MaterialProperties(1000, 0.0,  0.9, 0.7, "organic"),
        "tv":               MaterialProperties(800, 0.05, 0.5, 0.3, "electronics"),
        "remote":           MaterialProperties(900, 0.20, 0.5, 0.3, "plastic"),
        "vase":             MaterialProperties(2000, 0.10, 0.6, 0.4, "ceramic"),
        "plant":            MaterialProperties(500, 0.15, 0.7, 0.5, "organic"),
        "bowl":             MaterialProperties(1800, 0.15, 0.6, 0.4, "ceramic"),
        "clock":            MaterialProperties(1000, 0.10, 0.5, 0.3, "metal"),
        "backpack":         MaterialProperties(300, 0.30, 0.8, 0.6, "fabric"),
        "umbrella":         MaterialProperties(400, 0.20, 0.6, 0.4, "fabric"),
        "sports ball":      MaterialProperties(600, 0.80, 0.7, 0.5, "rubber"),
        "baseball glove":   MaterialProperties(500, 0.10, 0.8, 0.6, "leather"),
        "skateboard":       MaterialProperties(700, 0.20, 0.6, 0.4, "wood"),
        "surfboard":        MaterialProperties(400, 0.15, 0.6, 0.4, "foam"),
        "suitcase":         MaterialProperties(600, 0.10, 0.7, 0.5, "plastic"),
    }

    @classmethod
    def get(cls, class_name: str) -> MaterialProperties:
        """Look up material for a class name; fall back to defaults."""
        key = class_name.lower().strip()
        return cls.DEFAULTS.get(key, MaterialProperties(label=key))


@dataclass
class RigidBody:
    """A single rigid body in the physics simulation."""
    object_id: str
    class_name: str
    mass: float = 1.0
    radius: float = 0.3
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    force_accum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    restitution: float = 0.3
    friction: float = 0.5
    pinned: bool = False
    active: bool = True

    def apply_force(self, force: np.ndarray) -> None:
        if not self.pinned:
            self.force_accum += force

    def apply_impulse(self, impulse: np.ndarray) -> None:
        if not self.pinned and self.mass > 0:
            self.velocity += impulse / self.mass

    def reset_force(self) -> None:
        self.force_accum = np.zeros(3, dtype=float)


SIMULATION_STEP = 1.0 / 60.0  # 60 Hz physics tick


class PhysicsEngine:
    """Manages rigid-body simulation, collisions, and state."""

    def __init__(self) -> None:
        self.bodies: Dict[str, RigidBody] = {}
        self.time: float = 0.0
        self.running: bool = False
        self._last_step: float = time.time()
        self._collision_count: int = 0
        self._step_count: int = 0

    def init_from_objects(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create physics bodies from scene objects. Returns body list."""
        self.bodies.clear()
        self.time = 0.0
        self._step_count = 0
        self._collision_count = 0

        for obj in objects[:MAX_BODIES]:
            oid = obj.get("object_id") or obj.get("class_name", "") + "_" + str(random.randint(0, 9999))
            cls_name = obj.get("class_name", "default")
            pos = obj.get("position", {})
            scale = obj.get("size") or obj.get("scale") or {}
            sx = scale.get("x", scale.get("width", 0.3)) if isinstance(scale, dict) else 0.3
            sy = scale.get("y", scale.get("height", 0.3)) if isinstance(scale, dict) else 0.3
            radius = max(0.15, (sx + sy) / 4.0)

            mat = MaterialDatabase.get(cls_name)
            volume = 4.0 / 3.0 * math.pi * (radius ** 3)
            mass = max(0.1, mat.density * volume)

            body = RigidBody(
                object_id=oid,
                class_name=cls_name,
                mass=mass,
                radius=radius,
                position=np.array([pos.get("x", 0), max(pos.get("y", 0) + radius, radius), pos.get("z", 0)], dtype=float),
                velocity=np.zeros(3, dtype=float),
                restitution=mat.restitution,
                friction=mat.dynamic_friction,
                pinned=cls_name.lower() in ("wall", "floor", "ceiling", "ground"),
            )
            self.bodies[oid] = body

        return self.get_state()

    def step(self, dt: float = SIMULATION_STEP) -> Dict[str, Any]:
        """Run one simulation tick."""
        if not self.bodies:
            return self.get_state()

        dt = min(dt, 0.05)
        self._step_count += 1

        for body in self.bodies.values():
            if body.pinned or not body.active:
                continue

            body.reset_force()
            # Gravity
            body.apply_force(GRAVITY * body.mass)
            # Drag (simple linear damping, mass-independent)
            body.apply_force(-body.velocity * 0.5)

            # Semi-implicit Euler
            body.acceleration = body.force_accum / max(body.mass, 0.001)
            body.velocity += body.acceleration * dt
            body.position += body.velocity * dt

        # Collisions
        collisions = self._detect_and_resolve_collisions()

        # Ground plane
        for body in self.bodies.values():
            if body.pinned or not body.active:
                continue
            if body.position[1] - body.radius < GROUND_Y:
                body.position[1] = GROUND_Y + body.radius
                if body.velocity[1] < 0:
                    body.velocity[1] = -body.velocity[1] * body.restitution
                    body.velocity[0] *= (1.0 - body.friction)
                    body.velocity[2] *= (1.0 - body.friction)
                    if abs(body.velocity[1]) < 0.05:
                        body.velocity[1] = 0.0

        self.time += dt
        return {**self.get_state(), "collisions_this_step": len(collisions)}

    def _detect_and_resolve_collisions(self) -> List[Dict[str, Any]]:
        """Detect sphere-sphere collisions and resolve with restitution."""
        collisions = []
        body_list = [b for b in self.bodies.values() if b.active]
        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                a, b = body_list[i], body_list[j]
                diff = a.position - b.position
                dist = float(np.linalg.norm(diff))
                min_dist = a.radius + b.radius
                if dist < min_dist and dist > 1e-6:
                    self._collision_count += 1
                    normal = diff / dist
                    overlap = min_dist - dist
                    # Separate (handle pinned bodies)
                    total_mass = a.mass + b.mass
                    if a.pinned:
                        b.position -= normal * overlap
                    elif b.pinned:
                        a.position += normal * overlap
                    else:
                        a.position += normal * overlap * (b.mass / total_mass)
                        b.position -= normal * overlap * (a.mass / total_mass)
                    # Relative velocity along normal
                    rel_v = a.velocity - b.velocity
                    vn = float(np.dot(rel_v, normal))
                    if vn < 0:
                        restitution = min(a.restitution, b.restitution)
                        j_impulse = -(1.0 + restitution) * vn / (1.0 / a.mass + 1.0 / b.mass + 1e-6)
                        impulse = normal * j_impulse
                        if not a.pinned:
                            a.velocity += impulse / a.mass
                        if not b.pinned:
                            b.velocity -= impulse / b.mass
                        collisions.append({
                            "body_a": a.object_id,
                            "body_b": b.object_id,
                            "restitution": restitution,
                        })
        return collisions

    def apply_force_to(self, object_id: str, force: List[float]) -> Optional[Dict[str, Any]]:
        """Apply a force vector to a specific body."""
        body = self.bodies.get(object_id)
        if body is None:
            return None
        body.apply_force(np.array(force, dtype=float))
        return {"object_id": object_id, "force": force}

    def apply_impulse_to(self, object_id: str, impulse: List[float]) -> Optional[Dict[str, Any]]:
        """Apply an instantaneous impulse to a specific body."""
        body = self.bodies.get(object_id)
        if body is None:
            return None
        body.apply_impulse(np.array(impulse, dtype=float))
        return {"object_id": object_id, "impulse": impulse}

    def push(self, object_id: str, direction: str = "forward", strength: float = 10.0) -> Optional[Dict[str, Any]]:
        """Apply a directional push impulse."""
        dir_map = {
            "forward":  [0.0, 0.0, -strength],
            "backward": [0.0, 0.0, strength],
            "left":     [-strength, 0.0, 0.0],
            "right":    [strength, 0.0, 0.0],
            "up":       [0.0, strength, 0.0],
            "down":     [0.0, -strength, 0.0],
        }
        impulse = dir_map.get(direction, [0.0, 0.0, -strength])
        return self.apply_impulse_to(object_id, impulse)

    def drop(self, object_id: str, height_drop: float = 2.0) -> Optional[Dict[str, Any]]:
        """Teleport object upward and let gravity take over."""
        body = self.bodies.get(object_id)
        if body is None:
            return None
        body.position[1] += height_drop
        body.velocity = np.zeros(3, dtype=float)
        return {"object_id": object_id, "dropped_from": body.position[1]}

    def rotate(self, object_id: str, axis: str = "y", degrees: float = 45.0) -> Optional[Dict[str, Any]]:
        """Apply an angular impulse (simplified — translate to linear)."""
        body = self.bodies.get(object_id)
        if body is None:
            return None
        rad = math.radians(degrees)
        tangent_map = {
            "y": [rad * body.radius, 0.0, 0.0],
            "x": [0.0, rad * body.radius, 0.0],
            "z": [0.0, 0.0, rad * body.radius],
        }
        impulse = tangent_map.get(axis, [rad * body.radius, 0.0, 0.0])
        body.apply_impulse(np.array(impulse, dtype=float))
        return {"object_id": object_id, "axis": axis, "degrees": degrees}

    def get_state(self) -> Dict[str, Any]:
        """Return full physics simulation state."""
        bodies_list = []
        for body in self.bodies.values():
            bodies_list.append({
                "object_id": body.object_id,
                "class_name": body.class_name,
                "mass": round(body.mass, 3),
                "radius": round(body.radius, 3),
                "position": {
                    "x": round(float(body.position[0]), 4),
                    "y": round(float(body.position[1]), 4),
                    "z": round(float(body.position[2]), 4),
                },
                "velocity": {
                    "x": round(float(body.velocity[0]), 4),
                    "y": round(float(body.velocity[1]), 4),
                    "z": round(float(body.velocity[2]), 4),
                },
                "pinned": body.pinned,
                "active": body.active,
            })

        return {
            "time": round(self.time, 3),
            "body_count": len(self.bodies),
            "bodies": bodies_list,
            "steps": self._step_count,
            "total_collisions": self._collision_count,
            "running": self.running,
        }

    def reset(self) -> None:
        self.bodies.clear()
        self.time = 0.0
        self._step_count = 0
        self._collision_count = 0
        self.running = False


def sim_force_vectors() -> Dict[str, List[float]]:
    """Named force presets for the frontend UI."""
    return {
        "gentle_push":      [0.0, 0.0, -50.0],
        "hard_push":        [0.0, 0.0, -200.0],
        "upward":           [0.0, 150.0, 0.0],
        "left":             [-100.0, 0.0, 0.0],
        "right":            [100.0, 0.0, 0.0],
        "backward":         [0.0, 0.0, 100.0],
    }
