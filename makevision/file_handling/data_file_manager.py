from abc import ABC, abstractmethod
from .json_file_manager import JsonFileManager
from .yaml_file_manager import YamlFileManager
from .numpy_file_manager import NumpyFileManager
from makevision.core import FileManager, Data

import os

class FileManagerFactory(ABC):
    @abstractmethod
    def create_file_manager(self, path: str) -> FileManager:
        pass

class DefaultFileManagerFactory(FileManagerFactory):
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
    def _create_data_object(self, data: dict) -> Data:
        """Abstract method to create a Data object from a dictionary."""
        pass
    