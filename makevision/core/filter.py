from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class Filter(ABC):
    """Abstract base class for detection filtering logic."""
    @abstractmethod
    def apply(self, results: List, labels: Dict[int, str]) -> List:
        """
        Apply filtering logic to the detection results.

        Args:
            results (List): List of detection results to filter.
            labels (Dict[int, str]): Mapping of class IDs to class names.

        Returns:
            List: Filtered detection results.
        """
        pass