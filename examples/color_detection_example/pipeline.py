import cv2
import numpy as np

from makevision import Pipeline
from makevision.detection import ColorDetector
from makevision.model import ColorModel
from makevision.reader import VideoReader

# Example usage:
# python pipeline.py


class ColorDetectionPipeline(Pipeline):
    def run(self):
        colors = {
            # Yellow in HSV
            "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
            # Red in HSV
            "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        }
        reader = VideoReader("test_video.mp4")
        model = ColorModel(colors)
        detector = ColorDetector(model)

        while True:
            ret, frame = reader.read()
            if not ret:
                break

            # Process the image
            detections = detector.detect(frame)

            # Draw the detections on the frame
            detector.visualize(frame, detections)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = ColorDetectionPipeline()
    pipeline.run()
