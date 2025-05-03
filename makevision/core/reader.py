from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class FrameData(ABC):
    """Generic class for frame data."""
    @property
    @abstractmethod
    def frame(self) -> np.ndarray:
        """Get the frame data."""
        pass

class Reader(ABC):
    """Abstract base class for reading data from a source."""

    @abstractmethod
    def read(self) -> Tuple[bool, FrameData]:
        """Read data from the source."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Close the reader."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the reader."""
        pass

    