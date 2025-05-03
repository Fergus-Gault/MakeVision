from abc import ABC, abstractmethod
from .json_file_manager import JsonFileManager
from .yaml_file_manager import YamlFileManager
from makevision.core import FileManager

class FileManagerFactory(ABC):
    @abstractmethod
    def create_file_manager(self, path: str) -> FileManager:
        pass

class JsonFileManagerFactory(FileManagerFactory):
    def create_file_manager(self, path: str) -> FileManager:
        return JsonFileManager()
    
class YamlFileManagerFactory(FileManagerFactory):
    def create_file_manager(self, path: str) -> FileManager:
        return YamlFileManager()
    