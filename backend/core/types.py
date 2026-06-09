"""Core data types for AXIS."""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Vector3":
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3":
        mag = self.magnitude()
        if mag > 0:
            return self * (1.0 / mag)
        return Vector3()


@dataclass
class Quaternion:
    """Quaternion for 3D rotation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Quaternion":
        return cls(w=float(arr[0]), x=float(arr[1]), y=float(arr[2]), z=float(arr[3]))

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])


@dataclass
class Point3D:
    """3D point with optional features."""
    position: Vector3
    color: Optional[Tuple[int, int, int]] = None
    normal: Optional[Vector3] = None
    semantic_embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    timestamp: float = 0.0
    object_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "color": self.color,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "object_id": self.object_id
        }


@dataclass
class Pose:
    """Camera/object pose in 3D space."""
    position: Vector3
    orientation: Quaternion
    timestamp: float = 0.0
    confidence: float = 1.0

    def to_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        pos = self.position.to_numpy()
        rot = self.orientation.to_rotation_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = pos
        return matrix

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, timestamp: float = 0.0) -> "Pose":
        """Create pose from 4x4 transformation matrix."""
        pos = Vector3.from_numpy(matrix[:3, 3])
        rot_matrix = matrix[:3, :3]
        # Extract quaternion from rotation matrix
        trace = np.trace(rot_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        else:
            if rot_matrix[0, 0] > rot_matrix[1, 1]:
                s = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                s = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                z = 0.25 * s
        return cls(position=pos, orientation=Quaternion(w=w, x=x, y=y, z=z), timestamp=timestamp)


@dataclass
class BoundingBox3D:
    """3D bounding box."""
    center: Vector3
    size: Vector3  # width, height, depth
    orientation: Quaternion = field(default_factory=Quaternion)
    label: str = ""
    confidence: float = 1.0

    def volume(self) -> float:
        return self.size.x * self.size.y * self.size.z

    def contains_point(self, point: Vector3) -> bool:
        # Simplified check - assumes axis-aligned
        dx = abs(point.x - self.center.x)
        dy = abs(point.y - self.center.y)
        dz = abs(point.z - self.center.z)
        return dx <= self.size.x / 2 and dy <= self.size.y / 2 and dz <= self.size.z / 2


@dataclass
class TrackPoint:
    """Tracked point across frames."""
    point_id: str
    positions: List[Point3D] = field(default_factory=list)
    trajectory: List[Vector3] = field(default_factory=list)
    velocity: Optional[Vector3] = None

    def add_position(self, position: Point3D):
        self.positions.append(position)
        if len(self.positions) >= 2:
            prev = self.positions[-2].position
            curr = position.position
            dt = position.timestamp - self.positions[-2].timestamp
            if dt > 0:
                self.velocity = (curr - prev) * (1.0 / dt)