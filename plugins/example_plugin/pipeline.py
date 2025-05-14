from makevision.core import Calibrator, Reader, Detector, Filter, State, Pipeline, ArucoBoardDef
from makevision.utils import Timer
import logging
import cv2

from makevision.model import ColorModel
from makevision.detection import ColorDetector

import numpy as np


class ExamplePipeline(Pipeline):
    def run(self, calibrator: Calibrator, reader: Reader):

        logging.basicConfig(level=logging.INFO)
        colors = {
            # Yellow in HSV
            "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
            # Red in HSV (single range)
            "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        }

        model = ColorModel(colors=colors)
        detector = ColorDetector(model=model, streaming=True)

        calibrator.calibrate(
            "./plugins/example_plugin/wide_angle_cam/", aruco_board_def=ArucoBoardDef())

        with Timer("Main Loop"):
            while True:
                success, frame = reader.read()
                if not success:
                    break

                with Timer("undistort", accumulate=True):
                    calibrator.undistort(frame)

                with Timer("detect", accumulate=True):
                    detections = detector.detect(frame)

                with Timer("visualize", accumulate=True):
                    detector.visualize(frame, detections)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        Timer.summary()
        reader.release()
        cv2.destroyAllWindows()
