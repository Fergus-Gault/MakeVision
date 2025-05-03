from abc import ABC, abstractmethod
from typing import Any
class State(ABC):
    """Abstract base class for the state of the game."""
    @abstractmethod
    def update(self, state: Any) -> None:
        """Update the state of the game."""
        pass