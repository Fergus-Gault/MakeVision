from abc import ABC, abstractmethod
from typing import Any

class Model(ABC):
    """Abstract base class for detection models."""
    def __init__(self, model_path: str):
        """Initialize the model with the given path."""
        self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Loads the model of choice."""
        pass