"""MuJoCo-based rigid-body physics engine for AXIS Phase 5.

Provides real physics simulation using DeepMind MuJoCo when available.
Falls back gracefully if mujoco package is not installed.
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
SIMULATION_STEP = 1.0 / 60.0

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco = None


@dataclass
class MaterialProperties:
    density: float = 500.0
    restitution: float = 0.3
    static_friction: float = 0.5
    dynamic_friction: float = 0.3
    label: str = "default"


class MaterialDatabase:
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
        key = class_name.lower().strip()
        return cls.DEFAULTS.get(key, MaterialProperties(label=key))


class PhysicsEngine:
    """Rigid-body physics using MuJoCo with fallback to custom Euler integration."""

    def __init__(self) -> None:
        self.bodies: Dict[str, Dict[str, Any]] = {}
        self.time: float = 0.0
        self.running: bool = False
        self._last_step: float = time.time()
        self._collision_count: int = 0
        self._step_count: int = 0
        self._model = None
        self._data = None
        self._body_id_map: Dict[str, int] = {}
        self._use_mujoco = False

        if HAS_MUJOCO:
            try:
                self._init_mujoco_scene()
                self._use_mujoco = True
                logger.info("MuJoCo physics engine initialized")
            except Exception as e:
                logger.warning(f"MuJoCo init failed ({e}), using custom physics")

    def _init_mujoco_scene(self) -> None:
        xml = (
            '<mujoco model="axis_scene">'
            '  <option gravity="0 0 -9.81" />'
            '  <worldbody>'
            '    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.3 0.3 0.3 1" friction="0.5" />'
            '  </worldbody>'
            '</mujoco>'
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

    def _ensure_mujoco_body(self, object_id: str, radius: float, pos: np.ndarray,
                             restitution: float, friction: float) -> None:
        if object_id in self._body_id_map:
            bid = self._body_id_map[object_id]
            self._data.qpos[bid * 7: bid * 7 + 3] = pos
            return

        nq = self._model.nq
        body_xml = (
            f'<body name="{object_id}" pos="{pos[0]} {pos[1]} {pos[2]}">'
            f'  <joint type="free" damping="0.1" />'
            f'  <geom type="sphere" size="{radius}" rgba="0.5 0.6 0.8 0.9" '
            f'        friction="{friction}" solref="0.02 1" solimp="0.9 0.95 0.001" />'
            f'</body>'
        )
        xml = self._model.get_xml().decode() if hasattr(self._model, 'get_xml') else ''
        if '</worldbody>' in xml:
            new_xml = xml.replace('</worldbody>', body_xml + '</worldbody>')
            self._model = mujoco.MjModel.from_xml_string(new_xml)
            self._data = mujoco.MjData(self._model)
            self._body_id_map[object_id] = self._model.nbody - 1

    def init_from_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.bodies.clear()
        self._body_id_map.clear()
        self.time = 0.0
        self._step_count = 0
        self._collision_count = 0

        if self._use_mujoco:
            self._init_mujoco_scene()

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
            position = np.array([pos.get("x", 0), max(pos.get("y", 0) + radius, radius), pos.get("z", 0)], dtype=float)

            if self._use_mujoco:
                self._ensure_mujoco_body(oid, radius, position, mat.restitution, mat.dynamic_friction)

            self.bodies[oid] = {
                "object_id": oid,
                "class_name": cls_name,
                "mass": mass,
                "radius": radius,
                "position": position.copy(),
                "velocity": np.zeros(3, dtype=float),
                "restitution": mat.restitution,
                "friction": mat.dynamic_friction,
                "pinned": cls_name.lower() in ("wall", "floor", "ceiling", "ground"),
            }

        return self.get_state()

    def step(self, dt: float = SIMULATION_STEP) -> Dict[str, Any]:
        if not self.bodies:
            return self.get_state()
        dt = min(dt, 0.05)
        self._step_count += 1

        if self._use_mujoco:
            try:
                for oid, body in self.bodies.items():
                    if body["pinned"]:
                        continue
                    pos = body["position"]
                    bid = self._body_id_map.get(oid)
                    if bid is not None:
                        self._data.qpos[bid * 7: bid * 7 + 3] = pos

                mujoco.mj_step(self._model, self._data, 1)

                for oid, body in self.bodies.items():
                    if body["pinned"]:
                        continue
                    bid = self._body_id_map.get(oid)
                    if bid is not None:
                        qvel_start = self._model.nq + bid * 6
                        body["velocity"] = self._data.qvel[qvel_start:qvel_start + 3].copy()
                        body["position"] = self._data.qpos[bid * 7: bid * 7 + 3].copy()
                        if body["position"][1] < GROUND_Y:
                            body["position"][1] = GROUND_Y + body["radius"]
                            if body["velocity"][1] < 0:
                                body["velocity"][1] = -body["velocity"][1] * body["restitution"]
                                if abs(body["velocity"][1]) < 0.05:
                                    body["velocity"][1] = 0.0
            except Exception:
                self._euler_step(dt)
        else:
            self._euler_step(dt)

        self.time += dt
        return {**self.get_state(), "collisions_this_step": 0}

    def _euler_step(self, dt: float) -> None:
        for body in self.bodies.values():
            if body["pinned"]:
                continue
            force = GRAVITY * body["mass"]
            force -= body["velocity"] * 0.5
            acc = force / max(body["mass"], 0.001)
            body["velocity"] += acc * dt
            body["position"] += body["velocity"] * dt

            if body["position"][1] - body["radius"] < GROUND_Y:
                body["position"][1] = GROUND_Y + body["radius"]
                if body["velocity"][1] < 0:
                    body["velocity"][1] = -body["velocity"][1] * body["restitution"]
                    body["velocity"][0] *= (1.0 - body["friction"])
                    body["velocity"][2] *= (1.0 - body["friction"])
                    if abs(body["velocity"][1]) < 0.05:
                        body["velocity"][1] = 0.0

    def apply_force_to(self, object_id: str, force: List[float]) -> Optional[Dict[str, Any]]:
        body = self.bodies.get(object_id)
        if body is None or body["pinned"]:
            return None
        f = np.array(force, dtype=float)
        if self._use_mujoco:
            bid = self._body_id_map.get(object_id)
            if bid is not None:
                xfrc_start = bid * 6
                self._data.xfrc_applied[xfrc_start:xfrc_start + 3] = f
        else:
            body["velocity"] += f / max(body["mass"], 0.001) * SIMULATION_STEP
        return {"object_id": object_id, "force": force}

    def apply_impulse_to(self, object_id: str, impulse: List[float]) -> Optional[Dict[str, Any]]:
        body = self.bodies.get(object_id)
        if body is None or body["pinned"]:
            return None
        body["velocity"] += np.array(impulse, dtype=float) / max(body["mass"], 0.001)
        return {"object_id": object_id, "impulse": impulse}

    def push(self, object_id: str, direction: str = "forward", strength: float = 10.0) -> Optional[Dict[str, Any]]:
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
        body = self.bodies.get(object_id)
        if body is None:
            return None
        body["position"][1] += height_drop
        body["velocity"] = np.zeros(3, dtype=float)
        if self._use_mujoco:
            bid = self._body_id_map.get(object_id)
            if bid is not None:
                self._data.qpos[bid * 7: bid * 7 + 3] = body["position"]
                qvel_start = self._model.nq + bid * 6
                self._data.qvel[qvel_start:qvel_start + 6] = 0.0
        return {"object_id": object_id, "dropped_from": body["position"][1]}

    def rotate(self, object_id: str, axis: str = "y", degrees: float = 45.0) -> Optional[Dict[str, Any]]:
        body = self.bodies.get(object_id)
        if body is None or body["pinned"]:
            return None
        rad = math.radians(degrees)
        tangent = {
            "y": [0.0, 0.0, rad * body["radius"]],
            "x": [rad * body["radius"], 0.0, 0.0],
            "z": [0.0, rad * body["radius"], 0.0],
        }.get(axis, [rad * body["radius"], 0.0, 0.0])
        return self.apply_impulse_to(object_id, tangent)

    def get_state(self) -> Dict[str, Any]:
        bodies_list = []
        for body in self.bodies.values():
            bodies_list.append({
                "object_id": body["object_id"],
                "class_name": body["class_name"],
                "mass": round(body["mass"], 3),
                "radius": round(body["radius"], 3),
                "position": {
                    "x": round(float(body["position"][0]), 4),
                    "y": round(float(body["position"][1]), 4),
                    "z": round(float(body["position"][2]), 4),
                },
                "velocity": {
                    "x": round(float(body["velocity"][0]), 4),
                    "y": round(float(body["velocity"][1]), 4),
                    "z": round(float(body["velocity"][2]), 4),
                },
                "pinned": body["pinned"],
            })
        return {
            "time": round(self.time, 3),
            "body_count": len(self.bodies),
            "bodies": bodies_list,
            "steps": self._step_count,
            "total_collisions": self._collision_count,
            "running": self.running,
            "engine": "mujoco" if self._use_mujoco else "custom",
        }

    def reset(self) -> None:
        self.bodies.clear()
        self._body_id_map.clear()
        self.time = 0.0
        self._step_count = 0
        self._collision_count = 0
        self.running = False
        if self._use_mujoco:
            try:
                self._init_mujoco_scene()
            except Exception:
                pass


def sim_force_vectors() -> Dict[str, List[float]]:
    return {
        "gentle_push":      [0.0, 0.0, -50.0],
        "hard_push":        [0.0, 0.0, -200.0],
        "upward":           [0.0, 150.0, 0.0],
        "left":             [-100.0, 0.0, 0.0],
        "right":            [100.0, 0.0, 0.0],
        "backward":         [0.0, 0.0, 100.0],
    }
