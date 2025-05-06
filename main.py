from makevision.utils import *

import argparse

def main():
    """
    This is the main function, the entry point for the program.
    It sets up the argument parser and runs the pipeline.
    It will check any arguments provided by the user and use those.
    If arguments are not provided, it will check the plugin for the modules.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the basic pipeline.")
    parser.add_argument("input", help="Type of input (webcam or video file).")
    parser.add_argument("--loop", action="store_true", help="Loop the video input.")
    parser.add_argument("--plugin", required=True, help="Specify the plugin to use.")
    parser.add_argument("--pipeline", required=False, help="Specify the pipeline to use.")
    parser.add_argument("--calibration-data", required=False, help="Path to the calibration data file.")
    parser.add_argument("--model", required=False, help="Path to the YOLO model file.")
    parser.add_argument("--network", required=False, help="Path to the network configuration file.")
    parser.add_argument("--filter", required=False, help="Specify the filter to use.")
    parser.add_argument("--obstruction-detector", required=False, help="Specify the obstruction detector to use.")
    parser.add_argument("--state", required=False, help="Specify the state to use.")

    args = parser.parse_args()

    # Check what modules are within the plugin, use them.
    # If not found, check other arguments for remaining modules.
    plugin_components = detect_plugin_components(args.plugin)

    # Initialize components based on provided arguments first, then from plugin if not provided
    reader = detect_source(args.input, args.loop) if args.input else plugin_components["reader"]()
    
    calibrator = detect_calibrator(args.calibration_data, args.plugin) \
        if args.calibration_data else \
        plugin_components["calibrator"]() if plugin_components["calibrator"] else None
    
    # For model and detector
    model = detect_model(args.model, args.plugin) if args.model else None
    detector = detect_detector(model) if args.model else \
        plugin_components["detector"]() if plugin_components["detector"] else None
    
    network = detect_network(args.network, args.plugin) if args.network else \
        plugin_components["network"]() if plugin_components["network"] else None
    
    filter = detect_filter(args.filter) if args.filter else \
        plugin_components["filter"](model) if plugin_components["filter"] else None
    
    obstruction_detector = detect_obstruction_detector(args.obstruction_detector) \
        if args.obstruction_detector else \
        plugin_components["obstruction_detector"]() if plugin_components["obstruction_detector"] else None
    
    state = detect_state(args.state) if args.state else \
        plugin_components["state"]() if plugin_components["state"] else None
    
    # Get the pipeline
    pipeline = detect_pipeline(args.pipeline) if args.pipeline else \
        plugin_components["pipeline"]() if plugin_components["pipeline"] else None

    # Inject dependencies and run the pipeline
    inject_and_run(pipeline, {
        "reader": reader,
        "calibrator": calibrator,
        "detector": detector,
        "network": network,
        "filter": filter,
        "obstruction_detector": obstruction_detector,
        "state": state
    })



if __name__ == "__main__":
    main()