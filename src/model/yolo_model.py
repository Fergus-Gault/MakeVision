from src.core import Model
from ultralytics import YOLO
import os
import torch

class YoloModel(Model):
    def load_model(self, model_path: str):
        """Load the YOLO model from the specified path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_device = torch.device(device)
            model = YOLO(model_path, task="detect").to(torch_device)
            return model

