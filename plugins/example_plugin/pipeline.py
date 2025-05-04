from makevision.core import Calibrator, Reader, Detector, Filter, ObstructionDetector, State, Network, Pipeline
import cv2

class ExamplePipeline(Pipeline):
    def run(self, calibrator: Calibrator, 
            reader: Reader, detector: Detector, 
            filter: Filter, obstruction_detector: ObstructionDetector, 
            state: State, network: Network):
        
        calibrator.calibrate("path/to/calibration/data")

        while True:
            success, frame = reader.read()
            if not success:
                break

            results = detector.detect(frame)
            results = filter.apply(results)
            state.update(results)

            detector.visualize(frame, results)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
        cv2.destroyAllWindows()