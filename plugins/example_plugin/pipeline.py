from makevision.core import Calibrator, Reader, Detector, Filter, ObstructionDetector, State, Network, Pipeline
import cv2

class ExamplePipeline(Pipeline):
    def run(self, reader: Reader, detector: Detector, 
            filter: Filter, state: State, calibrator: Calibrator):

        calibrator.calibrate("./plugins/example_plugin/wide_angle_cam/")

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