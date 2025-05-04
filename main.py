from makevision.calibration import WebcamCalibrator
from makevision.utils import *

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the basic pipeline.")
    parser.add_argument("input", help="Type of input (webcam or video file).")
    parser.add_argument("--plugin", required=False, help="Specify the plugin to use.")
    parser.add_argument("--pipeline", required=False, help="Specify the pipeline to use.")
    parser.add_argument("--calibration-data", required=False, help="Path to the calibration data file.")
    parser.add_argument("--model", required=False, help="Path to the YOLO model file.")
    parser.add_argument("--network", required=False, help="Path to the network configuration file.")
    parser.add_argument("--filter", required=False, help="Specify the filter to use.")
    parser.add_argument("--obstruction_detector", required=False, help="Specify the obstruction detector to use.")
    parser.add_argument("--state", required=False, help="Specify the state to use.")

    args = parser.parse_args()

    # Check what modules are within the plugin, use them.
    # If not found, check other arguments for remaining modules.
    plugin_components = detect_plugin_components(args.plugin)

    # Initialize components from plugin or use the ones from args
    reader = plugin_components["reader"]() \
        if plugin_components["reader"] is not None else detect_source(args.input)
    calibrator = plugin_components["calibrator"](args.calibration_data) \
        if plugin_components["calibrator"] is not None else detect_calibrator(args.calibration_data)
    
    # Detect the model based on the provided model path
    model = detect_model(args.model)
    # Create a detector based on the model type
    detector = plugin_components["detector"](model) \
        if plugin_components["detector"] is not None else detect_detector(model)
    
    network = plugin_components["network"]() \
        if plugin_components["network"] is not None else detect_network(args.network)
    filter = plugin_components["filter"]() \
        if plugin_components["filter"] is not None else detect_filter(args.filter)
    
    obstruction_detector = plugin_components["obstruction_detector"]() \
        if plugin_components["obstruction_detector"] is not None \
            else detect_obstruction_detector(args.obstruction_detector)
    state = plugin_components["state"]() \
        if plugin_components["state"] is not None else detect_state(args.state)

    # Get the pipeline
    pipeline = plugin_components["pipeline"]() \
        if plugin_components["pipeline"] is not None else detect_pipeline(args.pipeline)

    # Run the pipeline
    pipeline.run(calibrator, reader, detector, filter, obstruction_detector, state, network)

if __name__ == "__main__":
    main()