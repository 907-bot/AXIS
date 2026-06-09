"""Core module - shared utilities and base classes."""
from .camera import CameraCapture
from .frame import Frame, FrameProcessor
from .types import Point3D, Pose, Vector3, Quaternion

__all__ = [
    "CameraCapture",
    "Frame",
    "FrameProcessor",
    "Point3D",
    "Pose",
    "Vector3",
    "Quaternion"
]