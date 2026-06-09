"""Core module with optional runtime dependencies."""

from .types import Point3D, Pose, Quaternion, Vector3

try:
    from .camera import CameraCapture
except ImportError:  # pragma: no cover - optional local camera dependency
    CameraCapture = None

try:
    from .frame import Frame, FrameProcessor
except ImportError:  # pragma: no cover - optional torch dependency
    Frame = None
    FrameProcessor = None

__all__ = [
    "Point3D",
    "Pose",
    "Vector3",
    "Quaternion",
    "CameraCapture",
    "Frame",
    "FrameProcessor",
]
