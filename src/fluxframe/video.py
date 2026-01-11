"""Video I/O utilities for frame matching."""

from pathlib import Path

import cv2
import numpy as np

from .models import VideoInfo


class VideoReader:
    """Handles video file reading and frame extraction."""

    def __init__(
        self,
        video_path: Path,
        fps_override: float | None = None,
        demo_mode: bool = False,
        demo_seconds: int = 20,
    ):
        """Initialize video reader.

        Args:
            video_path: Path to video file
            fps_override: Override video FPS
            demo_mode: If True, limit to demo_seconds
            demo_seconds: Number of seconds in demo mode
        """
        self.video_path = video_path
        self.fps_override = fps_override
        self.demo_mode = demo_mode
        self.demo_seconds = demo_seconds

        # Get video info
        self.video_info = self._get_video_info()

    def _get_video_info(self) -> VideoInfo:
        """Extract video information.

        Returns:
            VideoInfo with fps, total_frames, width, height
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps_override:
            fps = self.fps_override

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if self.demo_mode:
            max_frames = int(self.demo_seconds * fps)
            total_frames = min(total_frames, max_frames)

        return VideoInfo(
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )

    def open(self) -> cv2.VideoCapture:
        """Open video for reading.

        Returns:
            OpenCV VideoCapture object
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return cap

    def get_info(self) -> VideoInfo:
        """Get video information.

        Returns:
            VideoInfo object
        """
        return self.video_info

    def skip_to(self, cap: cv2.VideoCapture, frame_number: int) -> bool:
        """Skip to specific frame number.

        Args:
            cap: OpenCV VideoCapture object
            frame_number: Frame index to skip to

        Returns:
            True if successful, False otherwise
        """
        return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    @staticmethod
    def frame_to_lab(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to LAB vector.

        Args:
            frame: BGR frame from cv2

        Returns:
            Flattened LAB vector (64*64*3,) as float32
        """
        # Resize to 64x64 (matching database)
        resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

        # Convert to LAB
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        # Flatten and convert to float32
        return lab.reshape(-1).astype(np.float32)
