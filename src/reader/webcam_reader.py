from src.core import Reader, FrameData
from src.core.exceptions import InvalidWebcamSourceError
import cv2
import numpy as np

class WebcamFrameData(FrameData):
    """Class for webcam frame data."""
    
    def __init__(self, frame: np.ndarray):
        self._frame = frame

    @property
    def frame(self) -> np.ndarray:
        """Get the frame data."""
        return self._frame
    
    @frame.setter
    def frame(self, value: np.ndarray):
        self._frame = value


class WebcamReader(Reader):
    def __init__(self, source=0, backend=cv2.CAP_MSMF, dimensions=(1920, 1080), fps=30):
        self.source = source
        self.cap = cv2.VideoCapture(self.source, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimensions[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimensions[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise InvalidWebcamSourceError(source)
        
    def read(self):
        """Reads a frame from the webcam."""
        success, frame =  self.cap.read()
        if not success:
            return False, None
        
        return True, WebcamFrameData(frame)

    def release(self):
        """Releases the webcam."""
        if self.cap.isOpened():
            self.cap.release()

    def reset(self):
        """Not applicable to webcam."""
        pass