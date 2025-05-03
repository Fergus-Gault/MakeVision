from makevision.core import (
    Reader,
    Detector,
    Network,
    Filter,
    ObstructionDetector,
    State,
    Pipeline
)
import os
import inspect
from types import ModuleType
import importlib.util
import json

def detect_plugin_components(plugin_name: str) -> object:
    """Detects a plugin in the plugin folder based on the name."""
    plugin_path = os.path.join("plugins", plugin_name)
    if not os.path.isdir(plugin_path):
        raise ValueError(f"Plugin {plugin_name} not found in plugins directory.")
    
    plugin_components = {
        "reader": None,
        "detector": None,
        "network": None,
        "filter": None,
        "obstruction_detector": None,
        "state": None,
        "pipeline": None
    }
    
    for module_file in os.listdir(plugin_path):
        if not module_file.endswith(".py") or module_file == "__init__.py":
            continue
            
        module_name = module_file[:-3]
        module_path = os.path.join(plugin_path, module_file)

        try:
            module = load_module(module_path, module_name)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the class inherits from any of our core types
                if issubclass(obj, Reader) and obj != Reader:
                    plugin_components["reader"] = obj
                elif issubclass(obj, Detector) and obj != Detector:
                    plugin_components["detector"]= obj
                elif issubclass(obj, Network) and obj != Network:
                    plugin_components["network"]= obj
                elif issubclass(obj, Filter) and obj != Filter:
                    plugin_components["filter"] = obj
                elif issubclass(obj, ObstructionDetector) and obj != ObstructionDetector:
                    plugin_components["obstruction_detector"] = obj
                elif issubclass(obj, State) and obj != State:
                    plugin_components["state"] = obj
                elif issubclass(obj, Pipeline) and obj != Pipeline:
                    plugin_components["pipeline"] = obj
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not analyze module {module_name}: {e}")

    # If we found components, return the plugin_components dictionary
    if any(component for component in plugin_components.values()):
        return plugin_components
    return None

def load_module(module_path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_source(input_path: str) -> Reader:
    """Determine the detector based on the input type."""
    if input_path == "webcam":
        from makevision.reader import WebcamReader
        return WebcamReader()
    elif os.path.isfile(input_path):
        from makevision.reader import VideoReader
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
        from makevision.detection import YoloDetector
        return YoloDetector(model_path)
    elif file_ext in ['.pb', '.tflite']:  # TensorFlow extensions
        raise NotImplementedError("TensorFlow model not yet supported.")
    elif file_ext in ['.onnx']:  # ONNX format
        raise NotImplementedError("ONNX model not yet supported.")
    else:
        from makevision.detection import YoloDetector
        return YoloDetector(model_path)
    
def detect_pipeline(pipeline_name: str) -> Pipeline:
    return None

def detect_filter(filter_name: str) -> Filter:
    return None

def detect_state(state_config: str) -> State:
    return None

def detect_obstruction_detector(config: str) -> ObstructionDetector:
    return None
    

def detect_network(network_path: str) -> Network:
    """Determine the network type based on the configuration path."""
    if not os.path.isfile(network_path):
        raise ValueError(f"Network configuration file not found: {network_path}")
    
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
        raise ValueError(f"Invalid JSON format in network configuration file: {network_path}")