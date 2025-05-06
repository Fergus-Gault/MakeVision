import os
import torch
from makevision.core import Model
from ultralytics import YOLO


class YoloModel(Model):
    """Yolo model class for loading and managing YOLO models."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.device = self.model.device
        self.labels = self.model.names

    def load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model from the specified path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(model_path, task="detect")
            model.to(device)
            return model
