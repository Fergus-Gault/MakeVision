from makevision.core import Pipeline, Reader, Detector
import cv2

# Usage:
# python pipeline.py --input webcam --model hand_pose_model.pt


class HandPosePipeline(Pipeline):
    def run(self, reader: Reader, detector: Detector) -> None:
        while True:
            success, frame = reader.read()
            if not success:
                break

            # Detect hand poses
            detections = detector.detect(frame)

            # Visualize the detections
            detector.visualize(frame, detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import makevision
    makevision.start()
