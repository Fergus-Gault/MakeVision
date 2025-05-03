from abc import ABC, abstractmethod
from typing import Any, Dict

class Data(ABC):
    """Generic data class."""
    @abstractmethod
    @property
    def data(self) -> Any:
        """Get the data."""
        pass
    
    @abstractmethod
    def convert(self) -> Dict:
        """Converts data to a dict. Dictionary is then converted to specified format."""
        pass


class FileManager(ABC):
    """Abstract base class for file managers."""
    @abstractmethod
    def save(self, path: str, data: Data) -> None:
        """Save data to a file."""
        pass
    @abstractmethod
    def load(self, path: str) -> Data:
        """Load data from a file."""
        pass