from makevision.core import Detector
from ultralytics import YOLO
import cv2
import torch
from makevision.core import Model

class YoloDetector(Detector):
    def __init__(self, model: Model):
        self._model = model
        self.model = model.model
        self.use_half = True if self.model.device == 'cuda' else False

    def detect(self, frame):
        """Detect objects in the given frame using the YOLO model."""
    
        results = self.model(
            frame.frame,                # FrameData object containing the frame
            verbose=False,              # Suppress verbose output
            conf=0.5,                   # Confidence threshold for detections
            iou=0.4,                    # IOU threshold for non-max suppression 
            device=self.model.device,   # Use GPU if available (0), or 'cpu'
            half=self.use_half,         # Use FP16 half-precision inference
            stream=True,                # Enable streaming mode for real-time processing
        )
        
        # Convert results from generator to list if streaming
        results = list(results)

        return results
    
    def visualize(self, frame, detections):
        """Visualize the detection results on the frame."""
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"Class {self._model.labels[int(cls)]}: {conf:.2f}"
                cv2.rectangle(frame.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow("Detection", frame.frame)