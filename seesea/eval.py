"""Run inference on a trained model."""

import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import seesea.utils as utils
from seesea.observation import from_huggingface_dataset

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

    image_observations = from_huggingface_dataset(args.input)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # Run inference on the dataset
    model.eval()

    LOGGER.info("Running inference on the dataset to classify %s", training_details.output_name)

    # shuffle the data
    np.random.shuffle(image_observations)
    num_to_display = min(len(image_observations), args.num_samples)

    with torch.no_grad():
        for idx in range(num_to_display):

            image_observation = image_observations[idx]
            observation = image_observation.observation

            label = getattr(observation, training_details.output_name, None)
            if label is None:
                LOGGER.error(
                    "Observation %s:%s does not have a valid value for %s",
                    observation.station_id,
                    observation.timestamp,
                    observation.output_name,
                )
                continue

            # original image as it is in the dataset
            original_image = utils.load_image(image_observation.image_path)

            # trainsform the image to the model input
            transformed_image = transform(original_image)

            input_image = transformed_image.unsqueeze(0).to(device)

            label = np.array([label]).astype("float32")
            label = torch.from_numpy(label)
            label = label.to(device)

            output = model(input_image)

            LOGGER.info(
                "Predicted: %f, Expected: %f, Diff: %f", output.item(), label.item(), output.item() - label.item()
            )

            # Show the orignial image, the transformed image and the expected and predicted values
            plt.figure()
            # plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

            # plt.subplot(1, 2, 2)
            # convert the image to the correct format for display
            # transformed_image = transformed_image.permute(1, 2, 0)

            # # normalize the pixel values from -2 to 2 to 0 to 1
            # transformed_image = (transformed_image + 2) / 4

            # get original image brightness
            brightness = utils.get_brightness(original_image)

            # plt.imshow(transformed_image)
            # plt.title("Transformed Image")
            # plt.axis("off")

            plt.suptitle(
                f"Expected: {label.item():.3f}, Predicted: {output.item():.3f}, Diff:"
                f" {output.item() - label.item():.3f}, Brightness: {brightness:.2f}"
            )

            plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--input", help="Path to the directory containing the image observation data.")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)
    parser.add_argument("--num-samples", type=int, help="Number of samples to run inference on", default=None)

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
