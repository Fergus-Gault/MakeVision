from makevision.utils import *

import argparse
import logging


def start():
    """
    This is the main function, the entry point for the program.
    It sets up the argument parser and runs the pipeline.
    It will check any arguments provided by the user and use those.
    If arguments are not provided, it will check the plugin for the modules.
    """

    # Set up logging
    logging.getLogger('makevision').addHandler(logging.NullHandler())

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the basic pipeline.")
    parser.add_argument("--input", required=False,
                        help="Type of input (webcam or video file).")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the video input.")
    parser.add_argument("--pipeline", required=False,
                        help="Specify the pipeline to use.")
    parser.add_argument("--calibration-data", required=False,
                        help="Path to the calibration data file.")
    parser.add_argument("--model", required=False,
                        help="Path to the YOLO model file.")
    parser.add_argument("--network", required=False,
                        help="Path to the network configuration file.")
    parser.add_argument("--filter", required=False,
                        help="Specify the filter to use.")
    parser.add_argument("--obstruction-detector", required=False,
                        help="Specify the obstruction detector to use.")
    parser.add_argument("--state", required=False,
                        help="Specify the state to use.")

    args = parser.parse_args()

    # Check what modules are within the plugin, use them.
    # If not found, check other arguments for remaining modules.
    pipeline = detect_pipeline()

    # Detect the source or assume user defines within pipeline
    if args.input:
        streaming, reader = detect_source(args.input, args.loop)
    else:
        streaming, reader = False, None

    calibrator = detect_calibrator(args.calibration_data) \
        if args.calibration_data else None

    # For model and detector
    if args.model:
        model = detect_model(args.model)
    else:
        model = None

    if args.model:
        detector = detect_detector(model, streaming)
    else:
        detector = None

    network = detect_network(args.network) if args.network else None

    filter = detect_filter(args.filter) if args.filter else None

    obstruction_detector = detect_obstruction_detector(args.obstruction_detector) \
        if args.obstruction_detector else None

    state = detect_state(args.state) if args.state else None

    # Build components dictionary with non-None items
    components = {
        "pipeline": pipeline,
        "reader": reader,
        "calibrator": calibrator,
        "detector": detector,
        "network": network,
        "filter": filter,
        "obstruction_detector": obstruction_detector,
        "state": state
    }
    # Remove None values
    components = {k: v for k, v in components.items() if v is not None}

    # Inject dependencies and run the pipeline
    inject_and_run(pipeline, components)
