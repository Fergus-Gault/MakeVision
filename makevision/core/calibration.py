from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from cv2 import aruco
from cv2.aruco import ArucoDetector, CharucoBoard, DetectorParameters
from typing import Dict, Any
import cv2

from makevision.core import Data, FrameData

@dataclass
class ArucoBoardDef:
    """Holds the ArUco board parameters."""
    aruco_size: tuple = (6, 8)
    square_length: int = 34
    marker_length: int = 27
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
            data: Dictionary containing calibration parameters.
        """
        self.camera_mtx = np.array(data.get("camera_mtx")) \
            if data.get("camera_mtx") is not None else None
        self.dist_coeffs = np.array(data.get("dist_coeffs")) \
            if data.get("dist_coeffs") is not None else None
        self.newcamera_mtx = np.array(data.get("newcamera_mtx")) \
            if data.get("newcamera_mtx") is not None else None
        self.roi = np.array(data.get("roi")) \
            if data.get("roi") is not None else None
        self.img_size = tuple(data.get("img_size")) \
            if data.get("img_size") is not None else None

    @property
    def data(self) -> dict:
        """Get the calibration data."""
        return {
            "camera_mtx": self.camera_mtx,
            "dist_coeffs": self.dist_coeffs,
            "newcamera_mtx": self.newcamera_mtx,
            "roi": self.roi,
            "img_size": self.img_size
        }

    def convert(self) -> Dict[str, Any]:
        """Convert data to a dictionary suitable for saving."""
        data_dict = {
            "camera_mtx": self.camera_mtx.tolist() if self.camera_mtx is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "newcamera_mtx": self.newcamera_mtx.tolist() if self.newcamera_mtx is not None else None,
            "roi": self.roi.tolist() if self.roi is not None else None,
            "img_size": list(self.img_size) if self.img_size is not None else None,
        }
        return data_dict
    
    def calculate_undistort_maps(self):
        """Calculate the undistort maps."""
        return cv2.initUndistortRectifyMap(
            self.camera_mtx,
            self.dist_coeffs,
            None,
            self.newcamera_mtx,
            self.img_size,
            cv2.CV_32FC1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert data to a dictionary."""
        data_dict = {
            "camera_mtx": self.camera_mtx.tolist() if self.camera_mtx is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "newcamera_mtx": self.newcamera_mtx.tolist() if self.newcamera_mtx is not None else None,
            "roi": self.roi.tolist() if self.roi is not None else None,
            "img_size": list(self.img_size) if self.img_size is not None else None,
        }
        return data_dict

    def to_numpy(self) -> Dict[str, Any]:
        """Convert data to a numpy array."""
        data_dict = {
            "camera_mtx": self.camera_mtx if self.camera_mtx is not None else None,
            "dist_coeffs": self.dist_coeffs if self.dist_coeffs is not None else None,
            "newcamera_mtx": self.newcamera_mtx if self.newcamera_mtx is not None else None,
            "roi": self.roi if self.roi is not None else None,
            "img_size": np.array(self.img_size) if self.img_size is not None else None,
        }
        return data_dict



class Calibrator(ABC):
    @abstractmethod
    def calibrate(self, images_path: str, aruco_board: ArucoBoardDef, *args, **kwargs) -> None:
        """Calibrate the camera."""
        pass

    @abstractmethod
    def undistort(self, frame: FrameData, *args, **kwargs) -> FrameData:
        """Undistort the frame."""
        pass