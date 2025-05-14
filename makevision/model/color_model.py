from makevision.core import Model
from typing import Dict
import numpy as np


class ColorModel(Model):
    def __init__(self, colors: Dict[str, np.ndarray] = None) -> None:
        self.colors = colors or {}

    def load_model(self) -> None:
        return self
