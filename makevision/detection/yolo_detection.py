from makevision.core import Detector
from ultralytics import YOLO
import torch

class YoloDetector(Detector):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect") 

    def detect(self, frame):
        """Detect objects in the given frame using the YOLO model."""
        results = self.model(
            frame.frame, 
            verbose=False, 
            conf=0.5, 
            iou=0.4, 
            agnostic_nms=True, 
            device = self.model.device,
        )

        return results
    
    def visualize(self, frame):
        pass