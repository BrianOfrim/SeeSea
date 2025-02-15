"""
Visualize the outputs of the multihead model
"""

import argparse
import os
import logging

from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt

from seesea.common import utils

from seesea.model.multihead.multihead_model import MultiHeadModel  # Import the model class

LOGGER = logging.getLogger(__name__)


def main(args):
    # Load model and processor
    model = torch.load(os.path.join(args.model_dir, "model.pt"))
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(args.model_dir))

    # Get output names
    with open(os.path.join(args.model_dir, "output_names.txt"), "r", encoding="utf-8") as f:
        output_names = f.read().strip().split("\n")

    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    LOGGER.info("Running inference on the dataset to classify %s", output_names)

    count = 0

    for sample in dataset:
        image = sample["jpg"]
        labels = [sample["json"].get(name) for name in output_names]
        image_name = sample["__key__"]

        transformed_image = image_processor(image, return_tensors="pt")
        transformed_image = transformed_image["pixel_values"]
        transformed_image = transformed_image.to(device)

        # Create labels tensor with NaN for missing values
        label_tensor = torch.tensor([[float(l) if l is not None else float("nan") for l in labels]]).to(device)

        with torch.no_grad():
            _, outputs = model(transformed_image, label_tensor)

        outputs = outputs.squeeze().cpu().numpy()
        labels = label_tensor.squeeze().cpu().numpy()

        LOGGER.info("%s:", image_name)
        for name, pred, target in zip(output_names, outputs, labels):
            LOGGER.info("  %s: Predicted: %f, Expected: %f, Diff: %f", name, pred, target, pred - target)

        # Get image quality metrics
        brightness = utils.get_brightness(image)
        sharpness = utils.get_sharpness(image)
        droplets = utils.detect_water_droplets(image)

        LOGGER.debug("Brightness: %f, Sharpness: %f, Droplets: %s", brightness, sharpness, droplets)

        plt.imshow(image)
        plt.title(f"Image: {image_name}")
        plt.axis("off")

        # Create subtitle with all predictions
        subtitle = ""
        for name, pred, target in zip(output_names, outputs, labels):
            subtitle += f"{name}: target={target:.3f}, pred={pred:.3f}, diff={pred - target:.3f}\n"

        plt.suptitle(subtitle)

        # Set the window title
        plt.gcf().canvas.manager.set_window_title(image_name)

        # Check if image_name matches a file in readme_assets and save if it does
        # readme_assets_dir = "readme_assets/regression_samples"
        # if os.path.exists(os.path.join(readme_assets_dir, image_name + "_vis.png")):
        #     plt.savefig(os.path.join(readme_assets_dir, image_name + "_vis_new.png"), bbox_inches="tight", dpi=300)

        plt.show()

        count += 1

        if args.num_samples is not None and count == args.num_samples:
            break


def get_args_parser():

    parser = argparse.ArgumentParser(description="Run inference on a trained multihead model.")
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
