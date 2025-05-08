from makevision.core import Calibrator, Reader, Detector, Filter, State, Pipeline, ArucoBoardDef
from makevision.utils import Timer
import logging
import cv2


class ExamplePipeline(Pipeline):
    def run(self, reader: Reader, detector: Detector,
            filter: Filter, state: State, calibrator: Calibrator):

        logging.basicConfig(level=logging.INFO)

        calibrator.calibrate(
            "./plugins/example_plugin/wide_angle_cam/", aruco_board_def=ArucoBoardDef())

        with Timer("main loop"):
            while True:
                success, frame = reader.read()
                if not success:
                    break

                calibrator.undistort(frame)

                results = detector.detect(frame)
                results = filter.apply(results)
                state.update(results)

                detector.visualize(frame, results)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        reader.release()
        cv2.destroyAllWindows()
