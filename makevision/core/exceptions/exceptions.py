class TrackerError(Exception):
    """Base class for all exceptions raised by the tracker."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidWebcamSourceError(TrackerError):
    """Raised when the webcam source is invalid."""

    def __init__(self, source: int):
        super().__init__(f"Webcam source {source} is invalid.")


class FileNotJsonError(TrackerError):
    """Raised when the file type is incorrect."""

    def __init__(self):
        super().__init__("File type is not JSON. Please provide a .json file.")


class FileNotYamlError(TrackerError):
    """Raised when the file type is incorrect."""

    def __init__(self):
        super().__init__("File type is not YAML. Please provide a .yaml file.")
