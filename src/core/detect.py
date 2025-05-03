from abc import ABC, abstractmethod
from typing import List
from .reader import FrameData

class Detector(ABC):
    """Abstract base class for detectors."""

    @abstractmethod
    def detect(self, frame: FrameData) -> List:
        """
        Detect objects in the given frame.
        Args:
            frame (np.ndarray): The frame to detect objects in. 
        Returns:
            result (List): A list of detections.
        """
        pass

    @abstractmethod
    def visualize(self, results: List) -> None:
        """
        Visualize the detection results.
        Args:
            results (List): Detection results to visualize.
        """
        pass
