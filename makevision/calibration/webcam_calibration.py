import glob
from typing import List, Tuple

import cv2
from cv2 import aruco

from makevision.core import ArucoBoard, ArucoBoardDef, Calibrator, CalibrationData, FrameData
from makevision.file_handling import CalibrationDataFileManager, DefaultFileManagerFactory


class WebcamCalibrator(Calibrator):
    """Calibrates a webcam using ChArUco boards."""

    def __init__(self, path: str) -> None:
        file_manager_factory = DefaultFileManagerFactory()
        self.file_manager = CalibrationDataFileManager(
            file_manager_factory, path)
        self.calibration_data = None
        self.undistort_maps = None

    def calibrate(self, images_path: str, aruco_board_def: ArucoBoardDef = ArucoBoardDef()) -> None:
        self.calibration_data = self.file_manager.load()
        if self.calibration_data:
            self.undistort_maps = self.calibration_data.calculate_undistort_maps()
            return

        if not images_path:
            raise ValueError("No images path provided for calibration.")

        # Define ChArUco board
        aruco_board = self._get_aruco_board(aruco_board_def)

        # Load images
        images, img_size = self._load_images(images_path)
        all_charuco_ids, all_charuco_corners = self._get_charuco_corners_and_ids(
            images, aruco_board)

        # Calibrate the camera
        _, camera_mtx, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids,
            aruco_board.board, img_size,
            None, None)

        h, w = img_size[:2]
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_mtx, dist_coeffs, (w, h), 1, (w, h))

        # Create CalibrationData object
        calibration_data = {"camera_mtx": camera_mtx,
                            "dist_coeffs": dist_coeffs,
                            "newcamera_mtx": newcamera_mtx,
                            "roi": roi,
                            "img_size": (w, h)}
        self.calibration_data = CalibrationData(calibration_data)

        # Calculate undistort maps
        self.undistort_maps = self.calibration_data.calculate_undistort_maps()

        # Attempt to save the calibration data
        self.file_manager.save(self.calibration_data)

    def undistort(self, frame: FrameData) -> FrameData:
        if frame is None or frame.frame is None or \
                self.calibration_data is None or self.undistort_maps is None:
            return frame

        # Remap using ROI optimized maps
        undistorted_frame = cv2.remap(
            frame.frame,
            self.undistort_maps[0],
            self.undistort_maps[1],
            cv2.INTER_LINEAR,
        )

        frame.frame = undistorted_frame

        return frame

    def _get_charuco_corners_and_ids(
        self, images: List[str], aruco_board: ArucoBoard
    ) -> Tuple[List, List]:
        """
        Get ChArUco corners and IDs from the images.

        Args:
            images (List[str]): List of image file paths for calibration.
            aruco_board (ArucoBoard): Aruco board object containing the board and detector.

        Returns:
            Tuple[List, List]: List of ChArUco corners and IDs.
        """
        all_charuco_ids = []
        all_charuco_corners = []

        for image_file in images:
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = aruco_board.detector.detectMarkers(
                gray)

            if len(marker_ids) > 0:
                ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, aruco_board.board)
                if ret > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)

        return all_charuco_ids, all_charuco_corners

    def _get_aruco_board(self, aruco_board: ArucoBoardDef) -> ArucoBoard:
        """
        Create an ArucoBoard object from the given ArucoBoardDef.

        Args:
            aruco_board (ArucoBoardDef): Aruco board definition object.

        Returns:
            ArucoBoard: Aruco board object containing the board and detector.
        """
        def_aruco_dict = aruco.getPredefinedDictionary(aruco_board.aruco_dict)
        board = aruco.CharucoBoard(aruco_board.aruco_size,
                                   aruco_board.square_length,
                                   aruco_board.marker_length,
                                   def_aruco_dict)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(def_aruco_dict, params)
        return ArucoBoard(board, params, detector)

    def _load_images(self, images_path: str) -> Tuple[list, tuple]:
        """
        Load images from the specified path.

        Args:
            images_path (str): Path to the directory containing images.

        Raises:
            ValueError: If no images path is provided or if no images are found.
            ValueError: If no images are found for calibration.

        Returns:
            Tuple[list, tuple]: List of image file paths and the size of the first image.
        """
        if not images_path:
            raise ValueError("No images path provided for calibration.")
        images = glob.glob(images_path + "*.jpg")
        if not images:
            raise ValueError("No images found for calibration.")
        image = cv2.imread(images[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_size = image.shape
        return images, img_size
