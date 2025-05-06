from abc import ABC, abstractmethod
from typing import Any
class State(ABC):
    """Abstract base class for the state of the program."""
    @abstractmethod
    def update(self, state: Any, *args, **kwargs) -> None:
        """
        Update the state of the program.
        This method should be implemented by plugins.

        Args:
            state (Any): The state to update.
        """
        pass