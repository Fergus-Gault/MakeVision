from abc import ABC, abstractmethod
from src.core import FrameData
from typing import List

class ObstructionDetector(ABC):
    """Abstract base class for obstruction detection algorithms."""

    @abstractmethod
    def detect_obstruction(self, frame: FrameData) -> bool:
        """
        Detects obstructions in the given image.
        Args:
            image (FrameData): The input image to process.
        Returns:
            obstruction (bool): A boolean indicating whether an obstruction was detected.
        """
        pass

    @abstractmethod
    def get_obstruction_coordinates(self, frame: FrameData) -> List:
        """
        Returns the coordinates of the detected obstruction.
        Args:
            image (FrameData): The input image to process.
        Returns:
            obstructions (List): A list of coordinates representing the obstruction's location.
        """
        pass