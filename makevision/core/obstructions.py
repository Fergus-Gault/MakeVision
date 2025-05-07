from abc import ABC, abstractmethod
from typing import List

from makevision.core import FrameData


class ObstructionDetector(ABC):
    """Abstract base class for obstruction detection algorithms."""

    @abstractmethod
    def detect_obstruction(self, frame: FrameData, *args, **kwargs) -> bool:
        """
        Detects obstructions in the given image.
        Args:
            image (FrameData): The input image to process.
        Returns:
            obstruction (bool): A boolean indicating whether an obstruction was detected.
        """
        pass

    @abstractmethod
    def get_obstruction_coordinates(self, frame: FrameData, *args, **kwargs) -> List:
        """
        Returns the coordinates of the detected obstruction.
        Args:
            image (FrameData): The input image to process.
        Returns:
            obstructions (List): A list of coordinates representing the obstruction's location.
        """
        pass
