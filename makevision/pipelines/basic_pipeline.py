from makevision.core import (Detector, Network,
                      Pipeline, State, Calibrator,
                      Reader, Filter, ObstructionDetector,
                      ArucoBoardDef)
import cv2

class BasicPipeline(Pipeline):
    def run(calibrator: Calibrator, reader: Reader, 
            detector: Detector, filter: Filter, 
            obstruction: ObstructionDetector, 
            state: State, network: Network,
            calibration_path: str = "./data/images/",
            aruco_board: ArucoBoardDef = ArucoBoardDef()) -> None:
        
        """Run the pipeline."""
        calibrator.calibrate(calibration_path, aruco_board)

        while True:
            success, frame = reader.read()
            if not success:
                break

            calibrator.undistort(frame)

            obstruction_detected = obstruction.detect_obstruction(frame)

            detections = detector.detect(frame)
            filtered_detections = filter.apply(detections)

            state.update(filtered_detections, obstruction_detected)

            network.send_data(filtered_detections)

            detector.visualize(frame, detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
        network.disconnect()
        cv2.destroyAllWindows()
