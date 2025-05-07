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
    def read(self, *args, **kwargs) -> Tuple[bool, FrameData]:
        """
        Read a frame from the source.

        Returns:
            Tuple[bool, FrameData]: A tuple containing a boolean 
            indicating success and the frame data.
            If the read was unsuccessful, the frame data will be None.
        """
        pass

    @abstractmethod
    def release(self, *args, **kwargs) -> None:
        """Release the resources used by the reader."""
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset the reader."""
        pass
