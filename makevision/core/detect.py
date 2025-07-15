from abc import ABC, abstractmethod
from typing import List, Optional

from .model import Model
from makevision.types import FrameData


class Detector(ABC):
    """Abstract base class for detectors."""

    def __init__(self, model: Optional[Model], streaming: Optional[bool]) -> None:
        """
        Initialize the detector with a model.
        Args:
            model (Optional[Model]): The model to use for detection. If None, detector will not perform model inference.
            streaming (Optional[bool]): Whether to use streaming mode, which may affect performance optimization.
        """
        self.model = model
        self.streaming = streaming

    @abstractmethod
    def detect(self, frame: FrameData, *args, **kwargs) -> List:
        """
        Detect objects in the given frame.
        Args:
            frame (FrameData): The frame object to detect objects in. 
        Returns:
            result (List): A list of detections.
        """
        pass

    @abstractmethod
    def visualize(self, frame: FrameData, results: Optional[List], *args, **kwargs) -> None:
        """
        Visualize the detection results.
        Args:
            frame (FrameData): The frame to visualize.
            results (Optional[List]): Detection results to visualize.
        """
        pass
