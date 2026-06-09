"""Camera capture module for webcam and depth sensor input."""
import asyncio
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty

import cv2
import numpy as np
from loguru import logger


@dataclass
class CameraFrame:
    """Captured camera frame with metadata."""
    rgb: np.ndarray  # H, W, 3, uint8
    depth: Optional[np.ndarray] = None  # H, W, float32 (meters)
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    intrinsics: Optional[np.ndarray] = None  # 3x3 camera matrix
    extrinsics: Optional[np.ndarray] = None  # 4x4 pose matrix

    @property
    def h(self) -> int:
        return self.rgb.shape[0]

    @property
    def w(self) -> int:
        return self.rgb.shape[1]


class CameraCapture:
    """Real-time camera capture with frame queuing."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
        buffer_size: int = 10
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.buffer_size = buffer_size

        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: Queue[CameraFrame] = Queue(maxsize=buffer_size)
        self._callbacks: list[Callable[[CameraFrame], None]] = []
        self._frame_count = 0
        self._intrinsics = self._compute_intrinsics()

    def _compute_intrinsics(self) -> np.ndarray:
        """Compute camera intrinsic matrix based on resolution."""
        fx = self.width * 0.5 / np.tan(np.radians(30))  # ~60 degree FOV
        fy = self.height * 0.5 / np.tan(np.radians(30))
        cx = self.width / 2
        cy = self.height / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def start(self) -> bool:
        """Start camera capture."""
        if self._running:
            logger.warning("Camera already running")
            return True

        self._capture = cv2.VideoCapture(self.device_id)
        if not self._capture.isOpened():
            logger.error(f"Failed to open camera {self.device_id}")
            return False

        # Configure capture properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Camera {self.device_id} started at {self.width}x{self.height} @ {self.fps}fps")
        return True

    def _capture_loop(self):
        """Background capture loop."""
        while self._running:
            if self._capture is None:
                break

            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create camera frame
            camera_frame = CameraFrame(
                rgb=rgb,
                depth=None,  # Depth requires additional setup
                timestamp=time.time(),
                frame_id=self._frame_count,
                intrinsics=self._intrinsics.copy()
            )

            self._frame_count += 1

            # Add to queue (drop if full)
            try:
                self._frame_queue.put_nowait(camera_frame)
            except:
                pass  # Queue full, drop frame

            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(camera_frame)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def get_frame(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """Get next frame from queue."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def register_callback(self, callback: Callable[[CameraFrame], None]):
        """Register a frame callback."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[CameraFrame], None]):
        """Unregister a frame callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._capture:
            self._capture.release()
            self._capture = None
        logger.info("Camera stopped")

    @property
    def intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix."""
        return self._intrinsics.copy()

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DepthCameraCapture(CameraCapture):
    """Extended camera capture with depth sensor support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._depth_enabled = kwargs.get("enable_depth", False)

    def _capture_loop(self):
        """Enhanced capture loop with depth data."""
        while self._running:
            if self._capture is None:
                break

            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # For depth cameras (Intel RealSense, Azure Kinect, etc.)
            # This would require SDK integration
            depth = None
            if self._depth_enabled:
                # Placeholder for depth data
                depth = np.zeros((self.height, self.width), dtype=np.float32)

            camera_frame = CameraFrame(
                rgb=rgb,
                depth=depth,
                timestamp=time.time(),
                frame_id=self._frame_count,
                intrinsics=self._intrinsics.copy()
            )

            self._frame_count += 1

            try:
                self._frame_queue.put_nowait(camera_frame)
            except:
                pass

            for callback in self._callbacks:
                try:
                    callback(camera_frame)
                except Exception as e:
                    logger.error(f"Callback error: {e}")