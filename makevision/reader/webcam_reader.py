import cv2
import numpy as np
from typing import Tuple

from makevision.core import Reader, FrameData
from makevision.types import ColorSpace
from makevision.exceptions import InvalidWebcamSourceError


class WebcamFrameData(FrameData):
    """Class for webcam frame data."""

    def __init__(self, frame: np.ndarray, color_space: ColorSpace) -> None:
        super().__init__(frame, color_space)


class WebcamReader(Reader):
    def __init__(self, source: int = 0, backend: int = cv2.CAP_MSMF, dimensions: Tuple[int, int] = (1920, 1080), fps: int = 30, frame_type: FrameData = WebcamFrameData) -> None:
        self.source = source
        self.cap = cv2.VideoCapture(self.source, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimensions[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimensions[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise InvalidWebcamSourceError(source)
        self.frame_type = frame_type

    def read(self, *args, **kwargs) -> Tuple[bool, FrameData]:
        """Reads a frame from the webcam."""
        success, frame = self.cap.read()
        if not success:
            return False, None

        return True, self.frame_type(frame, ColorSpace.BGR, *args, **kwargs)

    def release(self):
        """Releases the webcam."""
        if self.cap.isOpened():
            self.cap.release()

    def reset(self):
        """Not applicable to webcam."""
        pass
