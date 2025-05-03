from src.pipelines import BasicPipeline
from src.calibration import WebcamCalibrator
from src.core import (ArucoBoardDef, CalibrationData,
                       ArucoBoard, Detector, Network,
                       State, Reader, Filter, ObstructionDetector,
                       Pipeline)

import os
import json
import argparse

# A "plugin" is a folder containing a __init__.py file and other modules.
# Modules used in the plugin are used, otherwise using a predefined module.
def detect_plugin(plugin_name: str) -> object:
    raise NotImplementedError("Plugin detection not yet implemented.")

def detect_source(input_path: str) -> Reader:
    """Determine the detector based on the input type."""
    if input_path == "webcam":
        from src.reader import WebcamReader
        return WebcamReader()
    elif os.path.isfile(input_path):
        from src.reader import VideoReader
        return VideoReader(input_path)
    else:
        raise ValueError("Invalid input type. Use 'webcam' or a video file path.")

def detect_model(model_path: str) -> Detector:
    """Determine the detector based on the model path."""
    if model_path is None or not os.path.isfile(model_path):
        raise ValueError("Invalid model path.")
    
    # Try to determine model type from file extension
    file_ext = os.path.splitext(model_path)[1].lower()
    
    if file_ext in ['.pt', '.pth']:  # YOLO typical extensions
        from src.detection import YoloDetector
        return YoloDetector(model_path)
    elif file_ext in ['.pb', '.tflite']:  # TensorFlow extensions
        raise NotImplementedError("TensorFlow model not yet supported.")
    elif file_ext in ['.onnx']:  # ONNX format
        raise NotImplementedError("ONNX model not yet supported.")
    else:
        from src.detection import YoloDetector
        return YoloDetector(model_path)
    
def detect_pipeline(pipeline_name: str) -> Pipeline:
    raise NotImplementedError("Pipeline detection not yet implemented.")

def detect_filter(filter_name: str) -> Filter:
    raise NotImplementedError("Filter detection not yet implemented.")

def detect_state(state_config: str) -> State:
    raise NotImplementedError("State detection not yet implemented.")

def detect_obstruction_detector(config: str) -> ObstructionDetector:
    raise NotImplementedError("Obstruction detector detection not yet implemented.")
    

def detect_network(network_path: str) -> Network:
    """Determine the network type based on the configuration path."""
    if not os.path.isfile(network_path):
        raise ValueError(f"Network configuration file not found: {network_path}")
    
    try:
        with open(network_path, 'r') as f:
            config = json.load(f)
            
        network_type = config.get("type", "").lower()
        
        if network_type == "socketio":
            from src.network import SocketIONetwork
            return SocketIONetwork(config)
        elif network_type == "database":
            raise NotImplementedError("Database network not yet supported.")
        elif network_type == "file":
            raise NotImplementedError("File network not yet supported.")
        elif network_type == "log":
            raise NotImplementedError("Log network not yet supported.")
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in network configuration file: {network_path}")

def main():
    parser = argparse.ArgumentParser(description="Run the basic pipeline.")
    parser.add_argument("input", help="Type of input (webcam or video file).")
    parser.add_argument("--pipeline", required=True, help="Specify the pipeline to use.")
    parser.add_argument("--model", required=True, help="Path to the YOLO model file.")
    parser.add_argument("--network", required=True, help="Path to the network configuration file.")
    args = parser.parse_args()

    # Check what modules are within the plugin, use them.
    # If not found, check other arguments for remaining modules.
    plugin = detect_plugin(args.plugin)

    # Initialize components
    reader = detect_source(args.input)
    calibrator = WebcamCalibrator("./data/images/")
    detector = detect_model(args.model)
    network = detect_network(args.network)
    filter = detect_filter(args.filter)
    obstruction_detector = detect_obstruction_detector(args.obstruction_detector)
    state = detect_state(args.state)

    # Get the pipeline
    pipeline = detect_pipeline(args.pipeline)

    # Run the pipeline
    pipeline.run(calibrator, reader, detector, filter, obstruction_detector, state, network)
