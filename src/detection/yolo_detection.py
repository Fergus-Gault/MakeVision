from src.core import Detector
from ultralytics import YOLO

class YoloDetector(Detector):
    def __init__(self, model: YOLO):
        self.model = model

    def detect(self, frame):
        """Detect objects in the given frame using the YOLO model."""
        results = self.model(
            frame, 
            verbose=False, 
            conf=0.5, 
            iou=0.4, 
            agnostic_nms=True, 
            device=self.model.device,
        )

        return results