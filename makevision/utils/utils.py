import inspect
import json
import os
from types import ModuleType
from typing import Tuple, Dict
import importlib.util
import logging

from makevision.calibration import WebcamCalibrator
from makevision.core import (
    Calibrator,
    Detector,
    Filter,
    Model,
    Network,
    ObstructionDetector,
    Pipeline,
    Reader,
    State,
)
from makevision.model import OnnxModel, TfModel, YoloModel

logger = logging.getLogger(__name__)


def detect_pipeline() -> Pipeline:
    """
    Detects and returns the first Pipeline subclass found in the current directory.

    Returns:
        Pipeline: The first found Pipeline subclass, or None if none found.
    """
    for module_file in [f for f in os.listdir(os.getcwd()) if f.endswith('.py') and f != '__init__.py']:
        try:
            module = load_module(module_file, module_file[:-3])
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Pipeline) and obj != Pipeline:
                    return obj()
        except Exception as e:
            logger.error(f"Error analyzing module {module_file}: {e}")

    return None


def load_module(module_path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_source(input_path: str, loop: bool) -> Tuple[bool, Reader]:
    """Determine the detector based on the input type."""
    if input_path == "webcam":
        from makevision.reader import WebcamReader
        return True, WebcamReader()
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        from makevision.reader import ImageReader
        return False, ImageReader(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        from makevision.reader import VideoReader
        return False, VideoReader(input_path, loop)
    else:
        return False, None


def detect_model(model_path: str) -> Model:
    """Determine the model type based on the model path."""

    # Try to determine model type from file extension
    file_ext = os.path.splitext(model_path)[1].lower()

    if file_ext in ['.pt', '.pth']:  # YOLO typical extensions
        from makevision.model import YoloModel
        return YoloModel(model_path)
    elif file_ext in ['.pb', '.tflite']:  # TensorFlow extensions
        raise NotImplementedError("TensorFlow model not yet supported.")
    elif file_ext in ['.onnx']:  # ONNX format
        raise NotImplementedError("ONNX model not yet supported.")
    else:
        return None


def detect_detector(model: Model, streaming: bool) -> Detector:
    """Determine the detector based on the model path."""
    if isinstance(model, YoloModel):
        from makevision.detection import YoloDetector
        return YoloDetector(model, streaming)
    elif isinstance(model, TfModel):  # TensorFlow extensions
        raise NotImplementedError("TensorFlow model not yet supported.")
    elif isinstance(model, OnnxModel):  # ONNX format
        raise NotImplementedError("ONNX model not yet supported.")
    else:
        return None


def detect_filter(filter_name: str) -> Filter:
    return None


def detect_state(state_config: str) -> State:
    return None


def detect_obstruction_detector(config: str) -> ObstructionDetector:
    return None


def detect_calibrator(calibration_path: str) -> Calibrator:
    return WebcamCalibrator(calibration_path)


def detect_network(network_path: str) -> Network:
    """Determine the network type based on the configuration path."""

    try:
        with open(network_path, 'r') as f:
            config = json.load(f)

        network_type = config.get("type", "").lower()

        if network_type == "socketio":
            from makevision.network import SocketIONetwork
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
        raise ValueError(
            f"Invalid JSON format in network configuration file: {network_path}")


def inject_and_run(pipeline: Pipeline, available_components: Dict):
    """
    Inject the components into the pipeline and run it.

    Args:
        pipeline (Pipeline): The pipeline to run.
        available_components (Dict): A dictionary of available components 
                                    to inject into the pipeline.
    """
    # Determine the parameters of the pipeline's run method
    sig = inspect.signature(pipeline.run)
    param_names = [key for key in sig.parameters.keys() if key != "self"]

    kwargs = {}

    for param_name in param_names:
        param = sig.parameters[param_name]

        if param_name in available_components:
            kwargs[param_name] = available_components[param_name]

        elif param.default is param.empty:
            raise ValueError(
                f"Missing required parameter '{param_name}' for pipeline run method.")

    # Run the pipeline with the initialized components
    pipeline.run(**kwargs)
