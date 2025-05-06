from makevision.core import CalibrationData
from .data_file_manager import FileManagerFactory, DataFileManager
from .json_file_manager import JsonFileManager

class CalibrationDataFileManager(DataFileManager):
    """File manager for calibration data."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)
    
class CalibrationDataJsonFileManager(CalibrationDataFileManager, JsonFileManager):
    """File manager for calibration data in JSON format."""
    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)
    
class CalibrationDataYamlFileManager(CalibrationDataFileManager, JsonFileManager):
    """File manager for calibration data in YAML format."""
    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)
    
class CalibrationDataNumpyFileManager(CalibrationDataFileManager, JsonFileManager):
    """File manager for calibration data in Numpy format."""
    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)