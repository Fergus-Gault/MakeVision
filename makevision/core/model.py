from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Abstract base class for detection models."""

    def __init__(self, model_path: str) -> None:
        """Initialize the model with the given path."""
        self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, path: str, *args, **kwargs) -> Any:
        """
        Loads the model from the specified path.

        Args:
            path (str): The path to the model file.

        Returns:
            Any: The loaded model.
        """
        pass
