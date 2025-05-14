import cv2
from typing import List

from makevision.core import Detector, FrameData, Model
import numpy as np


class YoloDetector(Detector):
    def __init__(self, model: Model, streaming: bool) -> None:
        self._model = model
        self.model = model.model
        self.use_half = True if self.model.device == 'cuda' else False
        self.streaming = streaming

    def detect(self, frame: FrameData, verbose: bool = False, conf: float = 0.5, iou: float = 0.45, imgsz: int = 640) -> List:
        """Detect objects in the given frame using the YOLO model."""

        # Optimize for inference speed in video processing
        results = self.model(
            frame.frame,                        # FrameData object containing the frame
            verbose=verbose,                    # Suppress verbose output
            conf=conf,                          # Lower confidence threshold for faster processing
            iou=iou,                            # Adjusted IOU threshold
            # Use GPU if available (0), or 'cpu'
            device=self.model.device,
            stream=self.streaming,              # Enable streaming mode for real-time processing
            imgsz=imgsz,                        # Resize images to 640x640 for faster processing
            stream_buffer=not self.streaming,   # Buffer for streaming mode
            half=self.use_half,                 # Use half precision for faster inference on GPU
            agnostic_nms=True,                  # Enable class-agnostic NMS for faster processing
        )

        return list(results)

    def visualize(self, frame: FrameData, detections: List) -> None:
        """Visualize the detection results on the frame."""
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)

            for _, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                label = f"{self._model.labels[cls_id]}: {conf:.2f}"

                cv2.rectangle(frame.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame.frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detection", frame.frame)
