from typing import Dict

from makevision.core import CalibrationData
from .data_file_manager import DataFileManager, FileManagerFactory


class CalibrationDataFileManager(DataFileManager):
    """File manager for calibration data."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: Dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)


class CalibrationDataJsonFileManager(CalibrationDataFileManager):
    """File manager for calibration data in JSON format."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: Dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)


class CalibrationDataYamlFileManager(CalibrationDataFileManager):
    """File manager for calibration data in YAML format."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: Dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)


class CalibrationDataNumpyFileManager(CalibrationDataFileManager):
    """File manager for calibration data in Numpy format."""

    def __init__(self, file_manager_factory: FileManagerFactory, path: str) -> None:
        super().__init__(file_manager_factory, path)

    def _create_data_object(self, data: Dict) -> CalibrationData:
        if not data:
            return None
        return CalibrationData(data)
