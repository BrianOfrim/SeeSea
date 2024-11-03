"""
Visualize the output of a trained model using the Hugging Face Transformers library.
"""

import os
import logging

from datasets import load_dataset
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt

from seesea.common import utils

LOGGER = logging.getLogger(__name__)


def main(args):

    model = AutoModelForImageClassification.from_pretrained(os.path.join(args.model_dir))
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(args.model_dir))

    # Get the output name from the output_name.text file in the model directory

    with open(os.path.join(args.model_dir, "output_name.txt"), "r", encoding="utf-8") as f:
        output_name = f.read().strip()

    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    model.eval()

    LOGGER.info("Running inference on the dataset to classify %s", output_name)

    count = 0

    for sample in dataset:
        image = sample["jpg"]
        label = sample["json"][output_name]
        name = sample["__key__"]

        transformed_image = image_processor(image, return_tensors="pt")
        transformed_image = transformed_image["pixel_values"]
        transformed_image = transformed_image.to(device)

        with torch.no_grad():
            output = model(transformed_image)

        output = output.logits.squeeze()

        error = output.item() - label

        if args.min_error is not None and abs(error) < args.min_error:
            continue

        # get original image brightness
        brightness = utils.get_brightness(image)
        sharpness = utils.get_sharpness(image)
        dropplets = utils.detect_water_droplets(image)

        LOGGER.info("%s: Predicted: %f, Expected: %f, Diff: %f", name, output.item(), label, output.item() - label)

        LOGGER.debug("Brightness: %f, Sharpness: %f, Dropplets %s", brightness, sharpness, dropplets)

        plt.imshow(image)
        plt.title(f"Image: {name} {output_name}")
        plt.axis("off")

        plt.suptitle(
            f"target: {label:.3f}, out: {output.item():.3f}, diff:"
            f" {error:.3f}, bright: {brightness:.2f}, sharp: {sharpness:.2f}"
        )

        plt.show()

        count += 1

        if args.num_samples is not None and count == args.num_samples:
            break


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--dataset", help="Path to the directory containing the dataset to load")
    parser.add_argument("--split", help="Which dataset split to load.", default="test")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)
    parser.add_argument("--num-samples", type=int, help="Number of samples to run inference on", default=None)
    parser.add_argument(
        "--min-error", type=float, help="Only show images with an error greater than this value", default=None
    )

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
