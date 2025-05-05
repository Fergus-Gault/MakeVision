from makevision.core import Reader, FrameData
from typing import Tuple
import cv2
import numpy as np

class VideoFrameData(FrameData):
    """Class for video frame data."""
    
    def __init__(self, frame: np.ndarray):
        self._frame = frame

    @property
    def frame(self) -> np.ndarray:
        """Get the frame data."""
        return self._frame
    
    @frame.setter
    def frame(self, value: np.ndarray):
        self._frame = value


class VideoReader(Reader):
    """Video reader class for reading video files."""
    
    def __init__(self, video_path: str, loop: bool = False) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.loop = loop
        

    def read(self):
        """Read a frame from the video."""
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.reset()
                return self.read()
            return False, None
        return True, VideoFrameData(frame)

    def release(self) -> None:
        """Release the video capture object."""
        self.cap.release()

    def reset(self) -> None:
        """Reset the video capture to the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)