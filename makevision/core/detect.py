from abc import ABC, abstractmethod
from typing import List
from .reader import FrameData
from .model import Model

class Detector(ABC):
    """Abstract base class for detectors."""
    def __init__(self, model: Model) -> None:
        """
        Initialize the detector with a model.
        Args:
            model (Model): The model to use for detection.
        """
        self.model = model

    @abstractmethod
    def detect(self, frame: FrameData, *args, **kwargs) -> List:
        """
        Detect objects in the given frame.
        Args:
            frame (np.ndarray): The frame to detect objects in. 
        Returns:
            result (List): A list of detections.
        """
        pass

    @abstractmethod
    def visualize(self, frame: FrameData, results: List, *args, **kwargs) -> None:
        """
        Visualize the detection results.
        Args:
            frame (FrameData): The frame to visualize.
            results (List): Detection results to visualize.
        """
        pass
