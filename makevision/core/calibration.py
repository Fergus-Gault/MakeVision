from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from cv2 import aruco
from cv2.aruco import ArucoDetector, CharucoBoard, DetectorParameters

from makevision.core import Data, FrameData


@dataclass
class ArucoBoardDef:
    """Holds the ArUco board parameters."""
    aruco_size: Tuple = (6, 8)
    square_length: int = 34
    marker_length: int = 27
    aruco_dict: int = aruco.DICT_4X4_50


@dataclass
class ArucoBoard:
    """Holds the ArUco board and its parameters."""
    board: CharucoBoard
    params: DetectorParameters
    detector: ArucoDetector


class CalibrationData(Data):
    """Holds the calibration data."""

    def __init__(self, data: Dict) -> None:
        """
        Initialize the calibration data.

        Args:
            data (Dict): Dictionary containing calibration parameters.
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
    def data(self) -> Dict:
        """
        Get the calibration data as a dictionary.

        Returns:
            Dict: A dictionary containing the calibration parameters.
        """
        return {
            "camera_mtx": self.camera_mtx,
            "dist_coeffs": self.dist_coeffs,
            "newcamera_mtx": self.newcamera_mtx,
            "roi": self.roi,
            "img_size": self.img_size
        }

    def convert(self) -> Dict:
        """
        Convert the calibration data to a dictionary format.

        Returns:
            Dict: A dictionary representation of the calibration data.
        """
        try:
            data_dict = {
                "camera_mtx": self.camera_mtx.tolist() if self.camera_mtx is not None else None,
                "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                "newcamera_mtx": self.newcamera_mtx.tolist() if self.newcamera_mtx is not None else None,
                "roi": self.roi.tolist() if self.roi is not None else None,
                "img_size": list(self.img_size) if self.img_size is not None else None,
            }
            return data_dict
        except Exception as e:
            raise ValueError(
                f"Error converting calibration data to dictionary: {e}")

    def calculate_undistort_maps(self) -> Tuple:
        """
        Calculate the undistortion maps for the camera.
        This crops the image to the ROI obtained from calibration
        and generates the undistortion maps.

        This is more efficient than generating maps for the entire image.
        The maps are used for remapping the image to undistort it.

        Returns:
            Tuple: A tuple containing the x and y undistortion maps.
        """
        if (self.camera_mtx is None or self.dist_coeffs is None or
                self.newcamera_mtx is None or self.img_size is None):
            raise ValueError(
                "Calibration data is incomplete. Cannot calculate undistort maps.")

        R = np.eye(3)

        # Extract ROI coordinates
        x, y, w, h = self.roi

        # Create a specialized camera matrix for the ROI
        roi_camera_mtx = self.newcamera_mtx.copy()
        # Adjust the principal point (cx, cy) to account for the ROI offset
        roi_camera_mtx[0, 2] = roi_camera_mtx[0, 2] - x
        roi_camera_mtx[1, 2] = roi_camera_mtx[1, 2] - y

        # Generate maps only for the ROI region
        maps = cv2.initUndistortRectifyMap(
            self.camera_mtx,
            self.dist_coeffs,
            R,
            roi_camera_mtx,
            (w, h),  # Only map the ROI size
            cv2.CV_16SC2
        )

        return maps

    def to_dict(self) -> Dict:
        """
        Convert the calibration data to a dictionary format.

        Returns:
            Dict: A dictionary representation of the calibration data.
        """
        try:
            data_dict = {
                "camera_mtx": self.camera_mtx.tolist() if self.camera_mtx is not None else None,
                "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                "newcamera_mtx": self.newcamera_mtx.tolist() if self.newcamera_mtx is not None else None,
                "roi": self.roi.tolist() if self.roi is not None else None,
                "img_size": list(self.img_size) if self.img_size is not None else None,
            }
            return data_dict
        except Exception as e:
            raise ValueError(
                f"Error converting calibration data to dictionary: {e}")

    def to_numpy(self) -> Dict:
        """
        Convert the calibration data to a numpy format.

        Returns:
            Dict: A dictionary representation of the calibration data in numpy format.
        """
        try:
            data_dict = {
                "camera_mtx": self.camera_mtx if self.camera_mtx is not None else None,
                "dist_coeffs": self.dist_coeffs if self.dist_coeffs is not None else None,
                "newcamera_mtx": self.newcamera_mtx if self.newcamera_mtx is not None else None,
                "roi": self.roi if self.roi is not None else None,
                "img_size": np.array(self.img_size) if self.img_size is not None else None,
            }
            return data_dict
        except Exception as e:
            raise ValueError(
                f"Error converting calibration data to numpy format: {e}")


class Calibrator(ABC):
    @abstractmethod
    def calibrate(self, images_path: str, aruco_board: ArucoBoardDef, *args, **kwargs) -> None:
        """
        Calibrate the camera using the images in the specified path.

        Args:
            images_path (str): Path to the images for calibration.
            aruco_board (ArucoBoardDef): Aruco board definition.
        """
        pass

    @abstractmethod
    def undistort(self, frame: FrameData, *args, **kwargs) -> FrameData:
        """
        Undistorts the given frame using the calibration parameters

        Args:
            frame (FrameData): The frame to undistort.

        Returns:
            FrameData: The undistorted frame.
        """
        pass
