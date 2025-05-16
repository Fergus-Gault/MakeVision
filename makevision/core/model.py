from abc import ABC, abstractmethod
from typing import Any, Optional


class Model(ABC):
    """Abstract base class for detection models."""

    def __init__(self, model_path: str, task: str) -> None:
        """Initialize the model with the given path."""
        self.model = self.load_model(model_path, task)

    @abstractmethod
    def load_model(self, path: Optional[str], task: Optional[str], *args, **kwargs) -> Any:
        """
        Loads the model from the specified path.

        Args:
            path (str): The path to the model file.

        Returns:
            Any: The loaded model.
        """
        pass
