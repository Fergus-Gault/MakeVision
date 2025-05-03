from abc import ABC, abstractmethod

class State(ABC):
    """Abstract base class for the state of the game."""
    @abstractmethod
    def update(self, state) -> None:
        """Update the state of the game."""
        pass