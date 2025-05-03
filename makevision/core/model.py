from abc import ABC, abstractmethod
from typing import Any

class Model(ABC):
    """Abstract base class for detection models."""

    @abstractmethod
    def load_model(self) -> Any:
        """Loads the model of choice."""
        pass