from makevision.core import CalibrationData
from .file_manager_factory import FileManagerFactory

class CalibrationDataFileManager:
    """File manager for calibration data."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        self.path = path
        self.file_manager = file_manager_factory.create_file_manager(path)

    def load(self) -> CalibrationData:
        data = self.file_manager.load(self.path)
        return CalibrationData(data=data)

    def save(self, data: CalibrationData) -> None:
        self.file_manager.save(self.path, data)