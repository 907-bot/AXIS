"""Human motion analytics — joint angles, balance, stability metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from ..core.types import Vector3


JOINT_CONNECTIONS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]


def calculate_joint_angle(
    a: Vector3, b: Vector3, c: Vector3
) -> float:
    """Angle at joint b between segments a-b and b-c (degrees)."""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def compute_all_joint_angles(
    keypoints: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute all detectable joint angles from keypoints."""
    angles = {}

    def kp(name: str) -> Optional[Vector3]:
        p = keypoints.get(name)
        if p and p.get("confidence", 0) > 0.3:
            return Vector3(x=p["x"], y=p["y"], z=p.get("z", 0))
        return None

    left_shoulder = kp("left_shoulder")
    left_elbow = kp("left_elbow")
    left_wrist = kp("left_wrist")
    right_shoulder = kp("right_shoulder")
    right_elbow = kp("right_elbow")
    right_wrist = kp("right_wrist")
    left_hip = kp("left_hip")
    left_knee = kp("left_knee")
    left_ankle = kp("left_ankle")
    right_hip = kp("right_hip")
    right_knee = kp("right_knee")
    right_ankle = kp("right_ankle")

    if left_shoulder and left_elbow and left_wrist:
        angles["left_elbow"] = calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
    if right_shoulder and right_elbow and right_wrist:
        angles["right_elbow"] = calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
    if left_hip and left_knee and left_ankle:
        angles["left_knee"] = calculate_joint_angle(left_hip, left_knee, left_ankle)
    if right_hip and right_knee and right_ankle:
        angles["right_knee"] = calculate_joint_angle(right_hip, right_knee, right_ankle)
    if left_shoulder and left_elbow:
        up = Vector3(x=left_shoulder.x, y=left_shoulder.y - 1.0, z=left_shoulder.z)
        angles["left_shoulder"] = calculate_joint_angle(up, left_shoulder, left_elbow)
    if right_shoulder and right_elbow:
        up = Vector3(x=right_shoulder.x, y=right_shoulder.y - 1.0, z=right_shoulder.z)
        angles["right_shoulder"] = calculate_joint_angle(up, right_shoulder, right_elbow)

    return angles


def compute_balance_score(
    keypoints: Dict[str, Dict[str, float]],
    frame_width: float = 640.0,
    frame_height: float = 480.0,
) -> float:
    """Estimate balance (0-1). 1 = perfectly stable."""
    left_ankle = keypoints.get("left_ankle")
    right_ankle = keypoints.get("right_ankle")
    nose = keypoints.get("nose")

    if not (left_ankle and right_ankle and nose):
        return 0.5

    com_x = nose["x"]
    feet_center_x = (left_ankle["x"] + right_ankle["x"]) / 2.0
    offset = abs(com_x - feet_center_x) / max(frame_width, 1.0)
    balance = max(0.0, 1.0 - offset * 0.3)
    return round(min(1.0, balance), 3)


def compute_stability(
    pose_history: List[Dict[str, Dict[str, float]]]
) -> float:
    """Stability based on nose position variance over recent frames."""
    if len(pose_history) < 3:
        return 1.0
    recent = pose_history[-10:]
    positions = []
    for p in recent:
        nose = p.get("nose")
        if nose:
            positions.append([nose["x"], nose["y"]])
    if len(positions) < 3:
        return 1.0
    arr = np.array(positions)
    variance = float(np.mean(np.var(arr, axis=0)))
    stability = max(0.0, 1.0 - variance * 0.5)
    return round(min(1.0, stability), 3)


def compute_motion_velocity(
    pose_history: List[Dict[str, Dict[str, float]]],
    frame_width: float = 640.0,
    frame_height: float = 480.0,
) -> Dict[str, float]:
    """Compute velocity from nose position over time."""
    if len(pose_history) < 2:
        return {"vx": 0, "vy": 0, "speed": 0}
    prev = pose_history[-2].get("nose")
    curr = pose_history[-1].get("nose")
    if not prev or not curr:
        return {"vx": 0, "vy": 0, "speed": 0}
    dt = 0.033
    vx = ((curr["x"] - prev["x"]) / max(frame_width, 1.0)) / dt
    vy = ((curr["y"] - prev["y"]) / max(frame_height, 1.0)) / dt
    speed = math.sqrt(vx * vx + vy * vy)
    return {"vx": round(vx, 3), "vy": round(vy, 3), "speed": round(speed, 3)}


def summarize_analytics(
    keypoints: Dict[str, Dict[str, float]],
    pose_history: List[Dict[str, Dict[str, float]]],
    frame_width: float = 640.0,
    frame_height: float = 480.0,
) -> Dict:
    """Full analytics summary for a single person."""
    joint_angles = compute_all_joint_angles(keypoints)
    balance = compute_balance_score(keypoints, frame_width, frame_height)
    stability = compute_stability(pose_history)
    velocity = compute_motion_velocity(pose_history, frame_width, frame_height)
    return {
        "joint_angles": joint_angles,
        "balance": balance,
        "stability": stability,
        "velocity": velocity,
        "is_moving": velocity["speed"] > 0.15,
    }
