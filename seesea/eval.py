"""Run inference on a trained model."""

import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import seesea.utils as utils
from seesea.observation import ImageObservation

from seesea.train import TrainingDetails
from seesea.seesea_dataset import SeeSeaDataset

LOGGER = logging.getLogger(__name__)


def main(args):

    # Verify the input directory exists, contains the training, details and the model weights

    if not os.path.exists(args.model_dir):
        LOGGER.error(f"Input directory {args.model_dir} does not exist.")
        return

    training_details_path = os.path.join(args.model_dir, "training_details.json")
    if not os.path.exists(training_details_path):
        LOGGER.error(f"Training details file {training_details_path} does not exist.")
        return

    # Load the training details and the model weights
    training_details_json = utils.load_json(training_details_path)
    if training_details_json is None:
        LOGGER.error(f"Failed to load training details from {training_details_path}")
        return

    model_weights_path = os.path.join(args.model_dir, "model.pth")
    if not os.path.exists(model_weights_path):
        LOGGER.error(f"Model weights file {model_weights_path} does not exist.")
        return

    if not os.path.exists(args.input):
        LOGGER.error(f"Image observation file {args.input} does not exist.")
        return

    training_details = utils.from_dict(TrainingDetails, training_details_json)

    LOGGER.debug("Loaded training details: %s", training_details)

    model, transform = utils.continuous_single_output_model_factory(training_details.model, model_weights_path)

    dataset = SeeSeaDataset(
        args.input,
        observation_key=training_details.output_name,
        transform=transform,
        max_length=args.num_samples,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    criterion = nn.MSELoss()

    # Run inference on the dataset
    model.eval()

    LOGGER.info("Running inference on the dataset to classify %s", training_details.output_name)

    with torch.no_grad():
        for idx in range(max(len(dataset), args.num_samples)):
            image, value = dataset[idx]
            np_image = image.numpy().transpose((1, 2, 0))

            image = image.unsqueeze(0).to(device)
            value = value.unsqueeze(0).to(device)

            output = model(image)
            output = output.view(-1)

            loss = criterion(output, value.view(-1))
            LOGGER.info("Loss: %f", loss.item())
            LOGGER.info("Expected: %f, Predicted: %f", value.item(), output.item())

            # Plot the image and the prediction
            plt.imshow(np_image)
            plt.title(f"Expected: {value.item()}, Predicted: {output.item()}")
            plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--input", help="Path to the directory containing the image observation data.")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)
    parser.add_argument("--num-samples", type=int, help="Number of samples to run inference on", default=10)

    return parser


if __name__ == "__main__":

    parser = get_args_parser()

    args = parser.parse_args()

    # setup the loggers
    LOGGER.setLevel(args.log)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    if args.log_file is not None:
        file_logging_handler = logging.FileHandler(args.log_file)
        file_logging_handler.setFormatter(log_formatter)
        LOGGER.addHandler(file_logging_handler)

    main(args)
