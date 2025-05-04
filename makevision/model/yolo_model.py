from makevision.core import Model
from ultralytics import YOLO
import os
import torch

class YoloModel(Model):
    def __init__(self, model_path: str):
        """Initialize the YOLO model with the given model path."""
        super().__init__(model_path)
        self.device = self.model.device
        self.labels = self.model.names
        
    def load_model(self, model_path: str):
        """Load the YOLO model from the specified path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(model_path, task="detect")
            model.to(device)
            return model

