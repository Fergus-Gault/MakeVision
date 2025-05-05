from abc import ABC, abstractmethod
from .json_file_manager import JsonFileManager
from .yaml_file_manager import YamlFileManager
from makevision.core import FileManager

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
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    