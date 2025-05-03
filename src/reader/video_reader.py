from src.core import Reader, FrameData
from typing import Tuple
import cv2

class VideoReader(Reader):
    """Video reader class for reading video files."""
    
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def read(self) -> Tuple[bool, FrameData]:
        """Read a frame from the video."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, FrameData(frame=frame)

    def release(self) -> None:
        """Release the video capture object."""
        self.cap.release()

    def reset(self) -> None:
        """Reset the video capture to the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)