"""
Run experiments on a trained model.
"""

import os
import logging

from datasets import load_dataset
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    stats = {}

    stats_path = os.path.join(args.output, "exp_stats.json")

    if os.path.exists(stats_path):
        stats = utils.load_json(stats_path)
    else:
        for sample in tqdm(dataset, leave=False, desc="exp", disable=LOGGER.level > logging.INFO):
            image = sample["jpg"]
            label = sample["json"][output_name]
            name = sample["__key__"]
            station_id = sample["json"]["station_id"]

            transformed_image = image_processor(image, return_tensors="pt")
            transformed_image = transformed_image["pixel_values"]
            transformed_image = transformed_image.to(device)

            with torch.no_grad():
                output = model(transformed_image)

            output = output.logits.squeeze()

            error = output.item() - label

            # get original image brightness
            brightness = utils.get_brightness(image)
            sharpness = utils.get_sharpness(image)

            stats.setdefault(station_id, []).append(
                {
                    "label": label,
                    "output": output.item(),
                    "error": error,
                    "brightness": brightness,
                    "sharpness": sharpness,
                }
            )

        # Save the stats to a file
        utils.save_json(stats, stats_path)

    # Plot sharpness vs. absolute error
    sharpness = []
    error = []
    for station_id, data in stats.items():
        for d in data:
            if d["sharpness"] < 50 and abs(d["error"]) > 0.25 and abs(d["error"]) < 2.0:
                sharpness.append(d["sharpness"])
                error.append(abs(d["error"]))

    plt.scatter(x=sharpness, y=error, s=2)
    plt.xlabel("Sharpness")
    plt.ylabel("Absolute Error")
    plt.title("Sharpness vs. Absolute Error")
    plt.savefig(os.path.join(args.output, "sharpness_vs_abs_error.png"))
    plt.close()

    # Bin the data
    # num_bins = 100  # Adjust based on desired resolution

    # Plot the 2D histogram
    plt.figure(figsize=(10, 8))
    plt.hist2d(sharpness, error, bins=(50, 10), cmap="inferno")
    plt.colorbar(label="Count")
    plt.title("2D Histogram of Sharpness vs. Prediction Error")
    plt.xlabel("Sharpness")
    plt.ylabel("Error")
    plt.savefig(os.path.join(args.output, "sharpness_vs_error_2d_hist.png"))
    plt.close()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--output", help="Path to the directory to save the output files")
    parser.add_argument("--dataset", help="Path to the directory containing the dataset to load")
    parser.add_argument("--split", help="Which dataset split to load.", default="test")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)

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
