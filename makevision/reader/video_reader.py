import cv2
import numpy as np
from typing import Tuple
import time

from makevision.core import Reader, FrameData
from makevision.types import ColorSpace


class VideoFrameData(FrameData):
    """Class for video frame data."""

    def __init__(self, frame: np.ndarray, color_space: ColorSpace) -> None:
        super().__init__(frame, color_space)


class VideoReader(Reader):
    """Video reader class for reading video files."""

    def __init__(self, video_path: str, loop: bool = False, cap_fps: bool = True, fps: int = 30, frame_type: FrameData = VideoFrameData) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.loop = loop
        self.fps = fps
        self.cap_fps = cap_fps
        self.time_per_frame = 1 / fps if cap_fps else 0
        self.last_frame_time = time.time()
        self.frame_type = frame_type

    def read(self, *args, **kwargs) -> Tuple[bool, FrameData]:
        """Read a frame from the video."""
        if self.cap_fps:
            # Calculate time since last frame
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            # If not enough time has passed, return the previous frame
            if elapsed < self.time_per_frame:
                time.sleep(self.time_per_frame - elapsed)

            # Update last frame time
            self.last_frame_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.reset()
                return self.read()
            return False, None
        return True, self.frame_type(frame, ColorSpace.BGR, *args, **kwargs)

    def release(self) -> None:
        """Release the video capture object."""
        self.cap.release()

    def reset(self) -> None:
        """Reset the video capture to the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
