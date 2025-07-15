import cv2
from typing import List, Any

from makevision.core import Detector, FrameData, Model
import numpy as np


class YoloDetector(Detector):
    def __init__(self, model: Model, streaming: bool) -> None:
        self._model = model
        self.model = model.model
        self.use_half = True if self.model.device == 'cuda' else False
        self.streaming = streaming

    def detect(self, frame: FrameData, **kwargs) -> List:
        """
        Detect objects in the given frame using the YOLO model.

        Args:
            frame: FrameData object containing the frame
            **kwargs: Any parameters to pass to the YOLO model
                      Common options include: verbose, conf, iou, imgsz, agnostic_nms, etc.

        Returns:
            List of detection results
        """
        # Set default parameters with instance defaults
        params = {
            'verbose': False,           # Display detection information
            'conf': 0.5,                # Confidence threshold
            'iou': 0.45,                # IoU threshold for NMS
            'device': self.model.device,  # Device to run on (cuda or cpu)
            'stream': self.streaming,   # Stream mode (True/False)
            'imgsz': 640,               # Input image size
            'stream_buffer': not self.streaming,  # Buffer all streaming frames
            'half': self.use_half,      # Use FP16 half-precision inference
            'agnostic_nms': True,       # Class-agnostic NMS
            'max_det': 300,             # Maximum detections per image
            'classes': None,            # Filter by class (None = all classes)
            'augment': False,           # Augmented inference
            'retina_masks': False,      # Use high-resolution segmentation masks
            'vid_stride': 1,            # Video frame-rate stride
            'visualize': False,         # Visualize model features
            'show': False,              # Show results if visualize
            'save': False,              # Save results to *.txt
            'save_conf': False,         # Save confidences in --save-txt labels
            'save_crop': False,         # Save cropped prediction boxes
            'line_width': None,         # The line width of the bounding boxes
            'show_labels': False,        # Show labels
            'show_conf': False,          # Show confidences
        }

        # Override defaults with any provided kwargs
        params.update(kwargs)

        # Perform detection with provided parameters
        results = self.model(frame.frame, **params)

        return list(results)

    def visualize(self, frame: FrameData, detections: List) -> None:
        """Visualize the detection results on the frame."""
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)

            # Visualise keypoints if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                self._visualise_keypoints(frame, result)

            for _, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                label = f"{self._model.labels[cls_id]}: {conf:.2f}"

                cv2.rectangle(frame.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame.frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detection", frame.frame)

    def _visualise_keypoints(self, frame: FrameData, result: Any) -> None:
        """Visualize keypoints on the frame."""
        for i, kpts in enumerate(result.keypoints.data):
            if kpts is not None:
                kpts = kpts.cpu().numpy()
                # Draw each keypoint
                for x, y, conf in kpts:
                    if conf > 0.5:
                        cv2.circle(
                            frame.frame, (int(x), int(y)), 5, (0, 0, 255), -1)
