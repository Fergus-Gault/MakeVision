from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from cv2 import aruco
from cv2.aruco import ArucoDetector, CharucoBoard, DetectorParameters

from makevision.core import Data, FrameData

@dataclass
class ArucoBoardDef:
    """Holds the ArUco board parameters."""
    aruco_size: tuple = (6, 8)
    square_length: int = 30
    marker_length: int = 25
    aruco_dict: int = aruco.DICT_4X4_50 

@dataclass
class ArucoBoard:
    board: CharucoBoard
    params: DetectorParameters
    detector: ArucoDetector

class CalibrationData(Data):
    """Holds the calibration data."""
    
    def __init__(self, data: dict):
        """
        Initialize the calibration data.
        
        Args:
            camera_matrix: Camera matrix
            dist_coeffs: Distortion coefficients
            newcamera_mtx: New camera matrix
            roi: Region of interest
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.newcamera_mtx = None
        self.roi = None
        self.convert(data)
    
    @property
    def data(self) -> dict:
        """Get the calibration data."""
        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "newcamera_mtx": self.newcamera_mtx,
            "roi": self.roi,
        }
    
    def convert(self, data: dict) -> None:
        """Convert the data to a numpy array."""
        try:
            self.camera_matrix = np.array(data.get("mtx")) \
                if data.get("mtx") is not None else None
            self.dist_coeffs = np.array(data.get("dist")) \
                if data.get("dist") is not None else None
            self.newcamera_mtx = np.array(data.get("newcamera_mtx")) \
                if data.get("newcamera_mtx") is not None else None
            self.roi = np.array(data.get("roi")) \
                if data.get("roi") is not None else None
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid calibration data format: {e}") from e



class Calibrator(ABC):
    @abstractmethod
    def calibrate(self, images_path: str, aruco_board: ArucoBoardDef) -> CalibrationData:
        """Calibrate the camera."""
        pass

    @abstractmethod
    def undistort(self, frame: FrameData) -> FrameData:
        """Undistort the frame."""
        pass