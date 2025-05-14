from typing import Any, Tuple

import cv2
import numpy as np

from makevision.core import FrameData, Reader


class ImageFrameData(FrameData):
    """Class for image frame data."""

    def __init__(self, frame: np.ndarray[Any, np.dtype[Any]]):
        self._frame = frame

    @property
    def frame(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Get the frame data."""
        return self._frame

    @frame.setter
    def frame(self, value: np.ndarray[Any, np.dtype[Any]]):
        self._frame = value


class ImageReader(Reader):
    """Image reader class for reading image files."""

    def __init__(self, image_file: str, frame_type: FrameData = ImageFrameData) -> None:
        self.image_file = image_file
        self.frame_type = frame_type

    def read(self) -> Tuple[bool, FrameData]:
        """Read an image from the file."""
        frame = cv2.imread(self.image_file)
        return True, self.frame_type(frame)

    def release(self) -> None:
        pass

    def reset(self) -> None:
        pass
