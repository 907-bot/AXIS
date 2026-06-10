"""Prediction engine using trajectory extrapolation and action forecasting."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

PREDICTED_ACTIONS = ["sitting", "walking", "standing", "reaching", "stationary"]
# Normalized velocity thresholds (fraction of frame width per second)
WALKING_FAST_THRESHOLD = 0.012
WALKING_SLOW_THRESHOLD = 0.005
REACHING_THRESHOLD = 0.002


class TrajectoryBuffer:
    """Tracks position history for a single entity and fits a trajectory."""

    def __init__(self, max_len: int = 60) -> None:
        self.observations: List[Dict[str, float]] = []
        self.max_len = max_len
        self._coefficients: Optional[Dict[str, np.ndarray]] = None

    def add(self, x: float, y: float, z: float) -> None:
        self.observations.append({"x": x, "y": y, "z": z, "t": time.time()})
        if len(self.observations) > self.max_len:
            self.observations.pop(0)
        self._fit()

    def _fit(self) -> None:
        if len(self.observations) < 3:
            self._coefficients = None
            return
        times = np.array([o["t"] for o in self.observations])
        t0 = times[0]
        times_norm = times - t0
        X = np.column_stack([np.ones_like(times_norm), times_norm])
        pos = np.array([[o["x"], o["y"], o["z"]] for o in self.observations])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, pos, rcond=None)
            self._coefficients = {"offset": t0, "coeffs": coeffs}
        except np.linalg.LinAlgError:
            self._coefficients = None

    def predict(self, seconds_ahead: float) -> Optional[Dict[str, float]]:
        if not self._coefficients or len(self.observations) < 2:
            return None
        offset = self._coefficients["offset"]
        c = self._coefficients["coeffs"]
        t_pred = time.time() + seconds_ahead - offset
        pred = c[0] + c[1] * t_pred
        return {"x": float(pred[0]), "y": float(pred[1]), "z": float(pred[2])}

    def confidence(self) -> float:
        if len(self.observations) < 3:
            return max(0.0, len(self.observations) / 3.0 * 0.3)
        recent = self.observations[-5:]
        if len(recent) < 2:
            return 0.3
        positions = np.array([[o["x"], o["y"], o["z"]] for o in recent])
        diffs = np.diff(positions, axis=0)
        step_sizes = np.linalg.norm(diffs, axis=1)
        mean_step = float(np.mean(step_sizes))
        step_std = float(np.std(step_sizes)) if len(step_sizes) > 1 else 0.0
        smoothness = 1.0 - min(1.0, step_std / max(0.01, mean_step + 1e-6))
        data_confidence = min(1.0, len(self.observations) / 20.0)
        return round(0.4 * data_confidence + 0.6 * smoothness, 3)


class HumanActionPredictor:
    """Forecasts human actions from pose history."""

    def __init__(self) -> None:
        self.pose_history: List[Dict[str, Dict[str, float]]] = []
        self.max_len = 90

    def feed(self, keypoints: Dict[str, Dict[str, float]]) -> None:
        self.pose_history.append(keypoints)
        if len(self.pose_history) > self.max_len:
            self.pose_history.pop(0)

    def predict_action(self) -> Tuple[str, float]:
        if len(self.pose_history) < 5:
            return ("stationary", 0.3)

        recent = self.pose_history[-10:]
        velocities = []
        for i in range(1, len(recent)):
            prev = recent[i - 1].get("nose", {})
            curr = recent[i].get("nose", {})
            if "x" in prev and "x" in curr:
                v = math.sqrt(
                    ((curr["x"] - prev["x"]) / 640.0) ** 2 + ((curr["y"] - prev["y"]) / 480.0) ** 2
                )
                velocities.append(v)

        avg_vel = float(np.mean(velocities)) if velocities else 0.0

        hip_kps = [p.get("left_hip", {}) for p in recent[-3:] if p.get("left_hip")]
        hip_y = float(np.mean([h.get("y", 0) for h in hip_kps])) if hip_kps else 0.0
        knee_kps = [p.get("left_knee", {}) for p in recent[-3:] if p.get("left_knee")]
        knee_y = float(np.mean([k.get("y", 0) for k in knee_kps])) if knee_kps else 0.0
        torso_compress = abs(hip_y - knee_y) if hip_y and knee_y else 0.0

        if avg_vel > WALKING_FAST_THRESHOLD:
            return ("walking", min(0.85, 0.5 + avg_vel * 30.0))
        if avg_vel > WALKING_SLOW_THRESHOLD:
            return ("walking", 0.6)
        if torso_compress < 15.0 and hip_y > 0.3:
            return ("sitting", 0.7)
        if avg_vel > REACHING_THRESHOLD:
            return ("reaching", 0.55)

        return ("standing", 0.65)

    def predict_pose(
        self, seconds_ahead: float
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if len(self.pose_history) < 3:
            return None
        keys = list(self.pose_history[-1].keys())
        predicted: Dict[str, Dict[str, float]] = {}
        for k in keys:
            values = []
            for p in self.pose_history[-5:]:
                v = p.get(k)
                if v and "x" in v:
                    values.append((v["x"], v["y"]))
            if len(values) >= 3:
                xs = np.array([v[0] for v in values])
                ys = np.array([v[1] for v in values])
                dx = float(np.mean(xs[-3:] - xs[-4:-1])) if len(xs) >= 4 else 0.0
                dy = float(np.mean(ys[-3:] - ys[-4:-1])) if len(ys) >= 4 else 0.0
                last = values[-1]
                predicted[k] = {
                    "x": last[0] + dx * seconds_ahead,
                    "y": last[1] + dy * seconds_ahead,
                    "z": 0.0,
                    "confidence": max(0.1, 1.0 - seconds_ahead * 0.1),
                }
        return predicted if predicted else None


class PredictionEngine:
    """Main prediction engine — aggregates object + human forecasting."""

    def __init__(self) -> None:
        self.object_buffers: Dict[str, TrajectoryBuffer] = defaultdict(TrajectoryBuffer)
        self.object_labels: Dict[str, str] = {}
        self.human_predictor = HumanActionPredictor()
        self._last_prune = time.time()

    def feed_objects(
        self, objects: List[Dict[str, Any]]
    ) -> None:
        for obj in objects:
            oid = obj.get("object_id", "")
            if not oid:
                continue
            pos = obj.get("position", {})
            x, y, z = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)
            self.object_buffers[oid].add(x, y, z)
            self.object_labels[oid] = obj.get("class_name", "unknown")
        self._prune_stale()

    def feed_human_pose(
        self, keypoints: Dict[str, Dict[str, float]]
    ) -> None:
        self.human_predictor.feed(keypoints)

    def _prune_stale(self, max_age: float = 10.0) -> None:
        now = time.time()
        stale = []
        for oid, buf in self.object_buffers.items():
            if not buf.observations:
                stale.append(oid)
            else:
                last_t = buf.observations[-1].get("t", 0)
                if now - last_t > max_age:
                    stale.append(oid)
        for oid in stale:
            del self.object_buffers[oid]
            self.object_labels.pop(oid, None)

    def predict_objects(
        self, seconds_ahead: float
    ) -> List[Dict[str, Any]]:
        results = []
        for oid, buf in self.object_buffers.items():
            pred = buf.predict(seconds_ahead)
            conf = buf.confidence() * max(0.1, 1.0 - seconds_ahead * 0.08)
            if pred is not None:
                results.append(
                    {
                        "object_id": oid,
                        "class_name": self.object_labels.get(oid, "unknown"),
                        "predicted_position": pred,
                        "confidence": round(min(1.0, conf), 3),
                        "seconds_ahead": seconds_ahead,
                    }
                )
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    def predict_human_action(self) -> Dict[str, Any]:
        action, confidence = self.human_predictor.predict_action()
        return {"action": action, "confidence": confidence}

    def predict_human_pose(
        self, seconds_ahead: float
    ) -> Optional[Dict[str, Dict[str, float]]]:
        return self.human_predictor.predict_pose(seconds_ahead)

    def get_timeline(
        self, deltas: List[float] = None
    ) -> Dict[str, Any]:
        if deltas is None:
            deltas = [1.0, 3.0, 5.0]
        current_time = time.time()
        current_objects = [
            {
                "object_id": oid,
                "class_name": self.object_labels.get(oid, "unknown"),
                "position": {
                    "x": buf.observations[-1]["x"],
                    "y": buf.observations[-1]["y"],
                    "z": buf.observations[-1]["z"],
                },
                "last_observed": buf.observations[-1].get("t", current_time),
            }
            for oid, buf in self.object_buffers.items()
            if buf.observations
        ]
        frames: Dict[str, Any] = {}
        for dt in deltas:
            preds = self.predict_objects(dt)
            act = self.predict_human_action()
            pose = self.predict_human_pose(dt)
            frames[f"+{int(dt)}s"] = {
                "objects": preds,
                "human_action": act,
                "human_pose": pose,
                "timestamp": current_time + dt,
            }
        return {
            "current": {"objects": current_objects},
            "timeline": frames,
            "generated_at": current_time,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tracked_objects": len(self.object_buffers),
            "human_pose_frames": len(self.human_predictor.pose_history),
            "current_action": self.predict_human_action(),
        }

    def reset(self) -> None:
        self.object_buffers.clear()
        self.object_labels.clear()
        self.human_predictor.pose_history.clear()
