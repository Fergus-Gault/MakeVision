import cv2
import logging

from makevision import Calibrator, Detector, Pipeline, Reader


# Usage example:
# python pipeline.py --input test_video.mp4 --calibration-data camera_calibration.json --model detection_model.pt


class ExamplePipeline(Pipeline):
    def run(self, reader: Reader, calibrator: Calibrator, detector: Detector) -> None:

        logging.basicConfig(level=logging.INFO)

        calibrator.calibrate("./wide_angle_cam/")

        while True:
            success, frame = reader.read()
            if not success:
                break

            calibrator.undistort(frame)

            detections = detector.detect(frame)

            detector.visualize(frame, detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    makevision.start()
