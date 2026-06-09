"""Frame processing utilities."""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from loguru import logger


@dataclass
class Frame:
    """Processed frame with semantic features."""
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    features: Optional[Dict[str, np.ndarray]] = None
    timestamp: float = 0.0
    frame_id: int = 0

    def to_tensor(self) -> torch.Tensor:
        """Convert RGB to normalized tensor."""
        tensor = torch.from_numpy(self.rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor


@dataclass
class SegmentationResult:
    """Object segmentation result."""
    masks: np.ndarray  # N, H, W, bool
    labels: List[str]
    confidences: np.ndarray
    bounding_boxes: List[List[int]]  # [x, y, w, h]


@dataclass
class SemanticFeatures:
    """Extracted semantic features from frame."""
    clip_embeddings: Optional[torch.Tensor] = None
    dino_embeddings: Optional[torch.Tensor] = None
    object_classes: List[str] = None
    scene_labels: List[str] = None


class FrameProcessor:
    """Base frame processor interface."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def process(self, frame: Frame) -> Frame:
        """Process a frame."""
        return frame

    def to(self, device: str):
        """Move processor to device."""
        self.device = device
        return self


class FrameBuffer:
    """Circular buffer for frame sequences."""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._frames: List[Frame] = []

    def add(self, frame: Frame):
        """Add frame to buffer."""
        self._frames.append(frame)
        if len(self._frames) > self.max_size:
            self._frames.pop(0)

    def get_recent(self, n: int = 5) -> List[Frame]:
        """Get n most recent frames."""
        return self._frames[-n:] if len(self._frames) >= n else self._frames

    def clear(self):
        """Clear buffer."""
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_empty(self) -> bool:
        return len(self._frames) == 0