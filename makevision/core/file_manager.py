from abc import ABC, abstractmethod
from typing import Any, Dict

class Data(ABC):
    """Generic data class."""
    @property
    @abstractmethod
    def data(self) -> Any:
        """Get the data."""
        pass
    
    @abstractmethod
    def convert(self, *args, **kwargs) -> Dict:
        """
        Convert the data to a dictionary format.

        Returns:
            Dict: A dictionary representation of the data.
        """
        pass


class FileManager(ABC):
    """Abstract base class for file managers."""
    @abstractmethod
    def save(self, path: str, data: Data, *args, **kwargs) -> None:
        """
        Save the data to a file.

        Args:
            path (str): The path to save the file.
            data (Data): The data to save.
        """
        pass

    @abstractmethod
    def load(self, path: str, *args, **kwargs) -> Data:
        """
        Load data from a file.

        Args:
            path (str): The path to the file.

        Returns:
            Data: The loaded data.
        """
        pass