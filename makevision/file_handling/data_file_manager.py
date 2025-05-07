import os
from abc import ABC, abstractmethod
from typing import Dict

from makevision.core import Data, FileManager
from .json_file_manager import JsonFileManager
from .numpy_file_manager import NumpyFileManager
from .yaml_file_manager import YamlFileManager


class FileManagerFactory(ABC):
    @abstractmethod
    def create_file_manager(self, path: str) -> FileManager:
        """
        Abstract method to create a file manager based on the file extension.

        Args:
            path (str): The path to the file.

        Returns:
            FileManager: An instance of the appropriate file manager.
        """
        pass


class DefaultFileManagerFactory(FileManagerFactory):
    """Factor class to create file managers based on file extension."""

    def create_file_manager(self, path: str) -> FileManager:
        if not path:
            return None
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension == '.json':
            return JsonFileManager()
        elif file_extension == '.yaml' or file_extension == '.yml':
            return YamlFileManager()
        elif file_extension == '.npy' or file_extension == '.npz':
            return NumpyFileManager()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


class DataFileManager(FileManager):
    """Abstract file manager for specific data types."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        self.path = path
        self.file_manager = file_manager_factory.create_file_manager(path)

    def load(self) -> Data:
        data = self.file_manager.load(self.path)
        return self._create_data_object(data)

    def save(self, data: Data) -> None:
        self.file_manager.save(self.path, data)

    @abstractmethod
    def _create_data_object(self, data: Dict) -> Data:
        """
        Creates a data object from the loaded data.

        Args:
            data (Dict): The loaded data.

        Returns:
            Data: An instance of the data object.
        """
        pass
