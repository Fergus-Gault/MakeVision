from abc import ABC, abstractmethod
from .reader import FrameData


class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: FrameData, **kwargs) -> FrameData:
        """
        Process the given frame. Could be used to adjust color, apply filters, etc.
        This method should be overridden by subclasses to implement specific processing logic.

        Args:
            frame (FrameData): The frame to process.
        Returns:
            frame (FrameData): The processed frame.
        """
        pass
