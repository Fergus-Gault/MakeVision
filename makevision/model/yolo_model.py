import os
from makevision.core import Model


class YoloModel(Model):
    """Yolo model class for loading and managing YOLO models."""

    def __init__(self, model_path: str, task: str = None) -> None:
        super().__init__(model_path, task)
        self.device = self.model.device
        self.labels = self.model.names

    def load_model(self, model_path: str, task: str):
        """Load the YOLO model from the specified path."""
        from ultralytics import YOLO
        import torch

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(model_path, task=task)
            model.to(device)

            return model
