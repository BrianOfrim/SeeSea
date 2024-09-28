"Training script for the SeeSea model."

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import seesea.utils as utils
from observation import Observation

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train the SeeSea model")
    arg_parser.add_argument("--input", help="The directory containing the training data")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="data")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
    arg_parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=10)
    arg_parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    arg_parser.add_argument("--learning-rate", type=float, help="The learning rate to use for training", default=0.001)
    arg_parser.add_argument("--model", type=str, help="The model to use for training", default="resnet")
    arg_parser.add_argument("--model-path", type=str, help="The path to save the trained model", default="model.pth")
    input_args = arg_parser.parse_args()

    # setup the loggers
    LOGGER.setLevel(input_args.log)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    if input_args.log_file is not None:
        file_logging_handler = logging.FileHandler(input_args.log_file)
        file_logging_handler.setFormatter(log_formatter)
        LOGGER.addHandler(file_logging_handler)

    # load the training data
    observations = []
    for file_name in os.listdir(input_args.input):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_args.input, file_name)
            LOGGER.info("Loading observations from %s", file_path)
            observations.extend([Observation(**obs) for obs in utils.load_json(file_path)])

    # train the model
    LOGGER.info("Training the model")

    # create the model
