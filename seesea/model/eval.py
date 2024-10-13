"""Run inference on a trained model."""

import os
import logging

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

import seesea.common.utils as utils
from seesea.model.train import TrainingDetails

LOGGER = logging.getLogger(__name__)


def main(args):

    # Verify the input directory exists, contains the training, details and the model weights

    if not os.path.exists(args.model_dir):
        LOGGER.error("Input directory %s does not exist.", args.model_dir)
        return

    training_details_path = os.path.join(args.model_dir, "training_details.json")
    if not os.path.exists(training_details_path):
        LOGGER.error("Training details file %s does not exist.", training_details_path)
        return

    # Load the training details and the model weights
    training_details_json = utils.load_json(training_details_path)
    if training_details_json is None:
        LOGGER.error("Failed to load training details from %s", training_details_path)
        return

    model_weights_path = os.path.join(args.model_dir, "model.pth")
    if not os.path.exists(model_weights_path):
        LOGGER.error("Model weights file %s does not exist.", model_weights_path)
        return

    training_details = utils.from_dict(TrainingDetails, training_details_json)

    LOGGER.debug("Loaded training details: %s", training_details)

    model, transform = utils.continuous_single_output_model_factory(training_details.model, model_weights_path)

    eval_ds = load_dataset("webdataset", data_dir=args.input, split=args.split, streaming=True)
    eval_ds = eval_ds.shuffle()

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

    count = 0

    with torch.no_grad():
        for samlple in eval_ds:

            image = samlple["jpg"]
            label = samlple["json"]
            name = samlple["__key__"]
            tranformed_image = transform(image).unsqueeze(0).to(device)

            target = label[training_details.output_name]

            output = model(tranformed_image)

            # get original image brightness
            brightness = utils.get_brightness(image)
            sharpness = utils.get_sharpness(image)
            dropplets = utils.detect_water_droplets(image)

            LOGGER.info(
                "%s: Predicted: %f, Expected: %f, Diff: %f", name, output.item(), target, output.item() - target
            )

            LOGGER.debug("Brightness: %f, Sharpness: %f, Dropplets %s", brightness, sharpness, dropplets)

            plt.imshow(image)
            plt.title(f"Image: {name} {training_details.output_name}")
            plt.axis("off")

            plt.suptitle(
                f"target: {target:.3f}, out: {output.item():.3f}, diff:"
                f" {output.item() - target:.3f}, bright: {brightness:.2f}, sharp: {sharpness:.2f}"
            )

            plt.show()

            count += 1

            if count == args.num_samples:
                break


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--input", help="Path to the directory containing the dataset to load")
    parser.add_argument("--split", help="Which dataset split to load.", default="test")
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
