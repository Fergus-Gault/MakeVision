# MakeVision

MakeVision is a flexible, modular computer vision framework designed to simplify the development of vision-based applications. It provides a structured approach to common computer vision tasks by abstracting away boilerplate code and offering seamless integration into any Python project.

## Features

- **Simple Integration**: Easily include MakeVision in any project
- **Vision Pipeline**: Built-in pipeline components for reading, processing, and analyzing images/video
- **Camera Calibration**: Tools for camera calibration using ArUco markers
- **Multiple Input Sources**: Support for images, videos, and webcam input
- **Model Integration**: Support for various model types (YOLO, TensorFlow, ONNX)
- **Detection Utilities**: Easy-to-use object detection and filtering
- **Performance Monitoring**: Built-in timing utilities to measure performance
- **Data Management**: Simplified file handling for various data formats
- **Networking**: Components for sending processed data over network connections

## Installation

```bash
# Install from PyPI
pip install makevision

# Or install the development version directly from GitHub
pip install git+https://github.com/fergus-gault/MakeVision.git
```

## Core Components

MakeVision consists of several core abstractions:

- **Reader**: Handles input sources (images, videos, webcam)
- **Detector**: Processes frames to detect objects/features
- **Model**: Wraps machine learning models for inference
- **Pipeline**: Orchestrates the flow of data through components
- **Calibrator**: Handles camera calibration and image correction
- **Filter**: Processes detection results to filter/transform outputs
- **Network**: Handles data transmission to external systems
- **State**: Manages application state and transitions
- **ObstructionDetector**: Detects obstructions in the view

## Quick Start

Creating a computer vision pipeline with MakeVision is straightforward:

1. Import the necessary components
2. Define your pipeline class
3. Call `makevision.start()`

```python
import cv2
import makevision
from makevision.core import Pipeline, Reader, Detector, Calibrator

class MyPipeline(Pipeline):
    def run(self, reader: Reader, detector: Detector, calibrator: Calibrator):
        # Your processing logic here
        while True:
            success, frame = reader.read()
            if not success:
                break
                
            # Process frame
            detections = detector.detect(frame)
            detector.visualize(frame, detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        reader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    makevision.start()
```

When you run this script, MakeVision will:

1. Automatically detect your pipeline class
2. Initialize components based on command-line arguments
3. Inject dependencies into your pipeline
4. Run your pipeline

Run with command-line options:

```bash
# Basic usage with webcam
python my_cv_script.py --input webcam

# With a video file
python my_cv_script.py --input ./videos/sample.mp4

# With a specific model
python my_cv_script.py --input webcam --model ./models/yolov8n.pt
```

## Manual Component Configuration

If you prefer not to use command-line arguments, you can manually create and configure the components within your pipeline:

```python
import cv2
import makevision
import numpy as np
from makevision.core import Pipeline, Reader, Detector
from makevision.reader import VideoReader
from makevision.model import ColorModel
from makevision.detection import ColorDetector
from makevision.calibration import WebcamCalibrator

class CustomConfigPipeline(Pipeline):
    def run(self):
        # Manually create components instead of receiving them as parameters
        reader = VideoReader("./videos/sample.mp4", loop=True, fps=30)
        
        # Define custom color ranges for detection
        colors = {
            "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
            "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        }
        model = ColorModel(colors)
        detector = ColorDetector(model)
        calibrator = WebcamCalibrator("./calibration/camera_params.yaml")
        
        # Rest of your pipeline logic using these components
        calibrator.calibrate("./calibration_images/")
        
        while True:
            success, frame = reader.read()
            if not success:
                break
                
            calibrator.undistort(frame)
            detections = detector.detect(frame)
            detector.visualize(frame, detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        reader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the pipeline - no command line args needed
    pipeline = CustomConfigPipeline()
    pipeline.run()
```

This approach gives you complete control over the configuration of each component and doesn't rely on the automatic component detection and instantiation provided by `makevision.start()`.

## Component Customization

### Creating a Custom Detector

```python
from makevision.core import Detector, FrameData, Model
import cv2
import numpy as np

class MyCustomDetector(Detector):
    def __init__(self, model: Model, streaming: bool = False) -> None:
        self._model = model
        self.streaming = streaming
        
    def detect(self, frame: FrameData) -> list:
        # Your detection logic here
        # Process the frame.frame (numpy array)
        results = []
        # ... detection code ...
        return results
        
    def visualize(self, frame: FrameData, detections: list) -> None:
        # Visualization code
        # Draw bounding boxes, labels, etc.
        cv2.imshow("Detection", frame.frame)
```

## Performance Measurement

MakeVision includes utilities for measuring performance:

```python
from makevision.utils import Timer

# Use as a context manager
with Timer("detection_time"):
    results = detector.detect(frame)

# Or manually
timer = Timer("processing_time", accumulate=True)
timer.start()
# Do some processing
processed_frame = process_frame(frame)
timer.stop()

# Get summary of all timings
Timer.summary()
```

## Project Structure

```
makevision/
├── calibration/       # Camera calibration utilities
├── core/              # Core interfaces and base classes
├── detection/         # Detection implementations
├── file_handling/     # File I/O utilities
├── model/             # Model implementations
├── network/           # Network communication
├── pipelines/         # Pipeline implementations
├── reader/            # Input readers
└── utils/             # Utility functions and classes
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

GNU General Public License v2.0
