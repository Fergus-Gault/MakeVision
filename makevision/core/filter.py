from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class Filter(ABC):
    """Abstract base class for detection filtering logic."""
    def __init__(self, model) -> None:
        self.model = model
        
    @abstractmethod
    def apply(self, results: List, *args, **kwargs) -> List:
        """
        Apply filtering logic to the detection results.

        Args:
            results (List): List of detection results to filter.

        Returns:
            List: Filtered detection results.
        """
        pass