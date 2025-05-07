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

    def read(self, image_file: str) -> Tuple[bool, ImageFrameData]:
        """Read an image from the file."""
        frame = cv2.imread(image_file)
        return True, ImageFrameData(frame)

    def release(self) -> None:
        pass

    def reset(self) -> None:
        pass
