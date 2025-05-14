from makevision.core import Detector
from typing import List, Tuple
import cv2
import numpy as np
from makevision.core import Model, FrameData


class ColorDetector(Detector):
    def __init__(self, model: Model, streaming: bool = False) -> None:
        """
        Initialise the color detector with a model.

        Args:
            model (Model): The model to use for color detection.
        """
        self._model = model
        self.colors = model.colors

    def detect(self, frame: FrameData) -> List:
        """
        Detects colors in the given frame.

        Args:
            frame (FrameData): The frame to detect colors in.

        Returns:
            List: A list of color names and their corresponding masks.
        """
        masks = []
        frame = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2HSV)
        for name, (lower_bound, upper_bound) in self.colors.items():
            # Create a mask for each color range
            mask = cv2.inRange(frame, lower_bound, upper_bound)
            masks.append((name, mask))

        return masks

    def visualize(self, frame: FrameData, masks: List) -> None:
        """
        Visualize the detected colors on the image.

        Args:
            image (np.ndarray): The image to visualize the detections on.
            masks (List): A list containing the name and mask of each detected color.
        """
        for (name, mask) in masks:
            masked_frame = cv2.bitwise_and(frame.frame, frame.frame, mask=mask)
            cv2.imshow(f"{name} Detection", masked_frame)

        cv2.imshow("Original Frame", frame.frame)
