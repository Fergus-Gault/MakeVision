import glob
from typing import List, Tuple

import cv2
from cv2 import aruco

from src.core import ArucoBoard, ArucoBoardDef, CalibrationData, Calibrator, FrameData
from src.file_handling import CalibrationDataFileManager

class WebcamCalibrator(Calibrator):
    def __init__(self, path: str) -> None:
        self.file_manager = CalibrationDataFileManager(path)
        self.calibration_data = None

    def calibrate(self, images_path: str, aruco_board: ArucoBoardDef) -> None:
        self.calibration_data = self.file_manager.load()
        if self.calibration_data:
            return self.calibration_data
        
        # Define ChArUco board
        aruco_board = self._get_aruco_board(aruco_board)

        # Load images
        images, img_size = self._load_images(images_path)
        all_charuco_ids, all_charuco_corners = self._get_charuco_corners_and_ids(images, aruco_board)

        # Calibrate the camera
        _, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, 
            aruco_board.board, img_size, 
            None, None)
        
        w, h = img_size[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.calibration_data = CalibrationData(mtx, dist, new_mtx, roi)
        
        # Save the generated calibration data
        self.file_manager.save(self.calibration_data)


    def undistort(self, frame: FrameData) -> FrameData:
        if frame is None or frame.frame is None or self.calibration_data is None:
            return frame
        
        undistorted_frame = cv2.undistort(
            frame.frame, 
            self.calibration_data.camera_matrix, 
            self.calibration_data.dist_coeffs, 
            None,
            self.calibration_data.newcamera_mtx)
        
        # Replace the frame with the undistorted one
        x,y,w,h = self.calibration_data.roi
        new_frame = undistorted_frame[y:y+h, x:x+w]
        frame.frame = new_frame
        
        return frame
    

    def _get_charuco_corners_and_ids(images: List[str], aruco_board: ArucoBoard) -> Tuple[List, List]:
        all_charuco_ids = []
        all_charuco_corners = []

        for image_file in images:
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = aruco_board.detector.detectMarkers(gray)

            if len(marker_ids) > 0:
                charuco_corners, charuco_ids = aruco_board.detector.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, aruco_board.board)
                if charuco_corners is not None and charuco_ids is not None:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)

        return all_charuco_ids, all_charuco_corners
    

    def _get_aruco_board(self, aruco_board: ArucoBoardDef) -> ArucoBoard:
        def_aruco_dict = aruco.getPredefinedDictionary(aruco_board.aruco_dict)
        board = aruco.CharucoBoard(aruco_board.aruco_dict, 
                                   aruco_board.square_length, 
                                   aruco_board.marker_length, 
                                   def_aruco_dict)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(def_aruco_dict, params)
        return ArucoBoard(board, params, detector)
    

    def _load_images(self, images_path: str) -> Tuple[list, tuple]:
        if not images_path:
            raise ValueError("No images path provided for calibration.")
        images = glob.glob(images_path + "*.jpg")
        if not images:
            raise ValueError("No images found for calibration.")
        image = cv2.imread(images[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_size = image.shape
        return images, img_size

