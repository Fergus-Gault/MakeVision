from makevision.core import CalibrationData, FileManager
from .file_manager_factory import FileManagerFactory

class CalibrationDataFileManager(FileManager):
    """File manager for calibration data."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        self.path = path
        self.file_manager = file_manager_factory.create_file_manager(path)

    def load(self) -> CalibrationData:
        data = self.file_manager.load(self.path)
        if data == {}:
            return None
        return CalibrationData(data)

    def save(self, data: CalibrationData) -> None:
        self.file_manager.save(self.path, data)