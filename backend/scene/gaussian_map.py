"""Gaussian splatting scene representation with semantic features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math
import numpy as np
from loguru import logger

from ..core.types import Vector3


@dataclass
class SemanticGaussian:
    """A single 3D Gaussian with semantic features.

    Each Gaussian stores:
    - Position (x, y, z)
    - Color (RGB)
    - Opacity (alpha)
    - Scale (3-axis standard deviations)
    - Rotation (quaternion)
    - CLIP embedding (text semantic feature)
    - DINO embedding (visual semantic feature)
    """
    gaussian_id: str
    class_name: str

    # Gaussian parameters
    position: Vector3
    color: Tuple[float, float, float]  # RGB 0-1
    opacity: float = 0.85
    scale: Vector3 = field(default_factory=lambda: Vector3(0.3, 0.3, 0.3))
    rotation: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))  # Euler angles (simplified)

    # Semantic features
    clip_embedding: Optional[np.ndarray] = None
    dino_embedding: Optional[np.ndarray] = None

    # Metadata
    confidence: float = 1.0
    first_observed: float = field(default_factory=lambda: datetime.now().timestamp())
    last_observed: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def covariance_3d(self) -> np.ndarray:
        """Compute 3x3 covariance matrix from scale and rotation."""
        sx, sy, sz = self.scale.x, self.scale.y, self.scale.z
        rx, ry, rz = self.rotation.x, self.rotation.y, self.rotation.z

        # Rotation matrix from Euler angles (XYZ order)
        cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
        sx_r, sy_r, sz_r = math.sin(rx), math.sin(ry), math.sin(rz)

        rot = np.array([
            [cy*cz, -cy*sz, sy_r],
            [sx_r*sy_r*cz + cx*sz, -sx_r*sy_r*sz + cx*cz, -sx_r*cy],
            [-cx*sy_r*cz + sx_r*sz, cx*sy_r*sz + sx_r*cz, cx*cy]
        ], dtype=np.float32)

        scale_mat = np.diag([sx, sy, sz])
        cov = rot @ (scale_mat @ scale_mat.T) @ rot.T
        return cov

    def gaussian_weight(self, point: Vector3) -> float:
        """Evaluate the Gaussian at a given 3D point (un-normalized)."""
        diff = np.array([
            point.x - self.position.x,
            point.y - self.position.y,
            point.z - self.position.z
        ], dtype=np.float32)
        cov = self.covariance_3d
        try:
            inv_cov = np.linalg.inv(cov + np.eye(3) * 1e-6)
            exponent = -0.5 * diff @ inv_cov @ diff
            return self.opacity * math.exp(max(-20.0, exponent))
        except np.linalg.LinAlgError:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gaussian_id": self.gaussian_id,
            "class_name": self.class_name,
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "color": {"r": self.color[0], "g": self.color[1], "b": self.color[2]},
            "opacity": self.opacity,
            "scale": {"x": self.scale.x, "y": self.scale.y, "z": self.scale.z},
            "rotation": {"x": self.rotation.x, "y": self.rotation.y, "z": self.rotation.z},
            "confidence": self.confidence,
            "embedding_dim": len(self.clip_embedding) if self.clip_embedding is not None else 0,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
        }


class GaussianMap:
    """Manages a collection of semantic 3D Gaussians.

    Provides:
    - Gaussian storage and lifecycle
    - Semantic search via CLIP/DINO embeddings
    - Spatial queries
    - Conversion from MapObject detections
    """

    def __init__(self, max_gaussians: int = 50000):
        self.max_gaussians = max_gaussians
        self.gaussians: Dict[str, SemanticGaussian] = {}
        self.class_counts: Dict[str, int] = {}

    def add_gaussian(
        self,
        gaussian_id: str,
        class_name: str,
        position: Vector3,
        color: Tuple[float, float, float],
        opacity: float = 0.85,
        scale: Optional[Vector3] = None,
        clip_embedding: Optional[np.ndarray] = None,
        dino_embedding: Optional[np.ndarray] = None,
        confidence: float = 1.0,
    ) -> SemanticGaussian:
        """Add a new Gaussian."""
        if len(self.gaussians) >= self.max_gaussians:
            oldest = min(self.gaussians.keys(), key=lambda k: self.gaussians[k].last_observed)
            del self.gaussians[oldest]

        g = SemanticGaussian(
            gaussian_id=gaussian_id,
            class_name=class_name,
            position=position,
            color=color,
            opacity=opacity,
            scale=scale or Vector3(0.3, 0.3, 0.3),
            clip_embedding=clip_embedding,
            dino_embedding=dino_embedding,
            confidence=confidence,
        )
        self.gaussians[gaussian_id] = g
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        logger.debug(f"Added Gaussian {gaussian_id} ({class_name}) at ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})")
        return g

    def update_gaussian(
        self,
        gaussian_id: str,
        position: Optional[Vector3] = None,
        confidence: Optional[float] = None,
        opacity: Optional[float] = None,
        clip_embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """Update an existing Gaussian."""
        if gaussian_id not in self.gaussians:
            return False
        g = self.gaussians[gaussian_id]
        if position is not None:
            g.position = position
        if confidence is not None:
            g.confidence = confidence
        if opacity is not None:
            g.opacity = opacity
        if clip_embedding is not None:
            g.clip_embedding = clip_embedding
        g.last_observed = datetime.now().timestamp()
        return True

    def remove_gaussian(self, gaussian_id: str) -> bool:
        """Remove a Gaussian."""
        if gaussian_id not in self.gaussians:
            return False
        g = self.gaussians[gaussian_id]
        self.class_counts[g.class_name] = max(0, self.class_counts.get(g.class_name, 0) - 1)
        del self.gaussians[gaussian_id]
        return True

    def remove_stale(self, max_age_seconds: float = 8.0) -> int:
        """Remove Gaussians not updated recently."""
        now = datetime.now().timestamp()
        stale = [
            gid for gid, g in self.gaussians.items()
            if (now - g.last_observed) > max_age_seconds
        ]
        for gid in stale:
            self.remove_gaussian(gid)
        return len(stale)

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[SemanticGaussian, float]]:
        """Search Gaussians by CLIP embedding similarity."""
        results: List[Tuple[SemanticGaussian, float]] = []
        for g in self.gaussians.values():
            if g.clip_embedding is not None:
                sim = self._cosine_similarity(query_embedding, g.clip_embedding)
                results.append((g, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def spatial_search(
        self,
        center: Vector3,
        radius: float,
        class_filter: Optional[List[str]] = None,
    ) -> List[SemanticGaussian]:
        """Search Gaussians by spatial proximity."""
        results = []
        for g in self.gaussians.values():
            if class_filter and g.class_name not in class_filter:
                continue
            dx = g.position.x - center.x
            dy = g.position.y - center.y
            dz = g.position.z - center.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= radius:
                results.append(g)
        results.sort(key=lambda g: (
            (g.position.x - center.x)**2 +
            (g.position.y - center.y)**2 +
            (g.position.z - center.z)**2
        ))
        return results

    def get_gaussians_for_class(self, class_name: str) -> List[SemanticGaussian]:
        """Get all Gaussians for a given class."""
        return [g for g in self.gaussians.values() if g.class_name == class_name]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_gaussians": len(self.gaussians),
            "class_counts": dict(self.class_counts),
            "embedding_dim": next(
                (len(g.clip_embedding) for g in self.gaussians.values() if g.clip_embedding is not None),
                0
            ),
        }

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in self.gaussians.values()]

    def clear(self) -> None:
        self.gaussians.clear()
        self.class_counts.clear()
        logger.info("Gaussian map cleared")
