"""
Visualize the outputs of the discrete model
"""

import os
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset
from transformers import AutoImageProcessor

from seesea.model.discrete.train import NUM_BINS, mps_to_bin

LOGGER = logging.getLogger(__name__)


def main(args):

    # Load the model and the preprocessor
    model = torch.load(os.path.join(args.model_dir, "model.pt"))
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)

    # Load the dataset
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    for sample in dataset:
        image = sample["jpg"]
        wind_speed = sample["json"]["wind_speed_mps"]
        image_name = sample["__key__"]
        bin = mps_to_bin(wind_speed)

        transformed_image = image_processor(image, return_tensors="pt")
        transformed_image = transformed_image["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(transformed_image)

        logits = outputs.logits
        predictions = logits.squeeze().cpu().numpy()
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=0).numpy()

        predicted_bin = np.argmax(predictions)
        confidence = probabilities[predicted_bin]

        LOGGER.info(
            "Wind speed: %s, Bin: %s, Predicted: %s, Confidence: %s", wind_speed, bin, predicted_bin, confidence
        )

        plt.imshow(image)
        # Set the figure name
        figure_name = f"{image_name}"

        plt.title(f"Image: {image_name}")
        subtitle = f"Label: bin={bin}, ({wind_speed:.2f}m/s)\n"
        subtitle += f"Predicted: bin={predicted_bin}, confidence={confidence:.2f}"
        plt.suptitle(subtitle)
        plt.axis("off")

        # Set the window title
        plt.gcf().canvas.manager.set_window_title(figure_name)

        plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained multihead model.")
    parser.add_argument("--dataset", help="Path to the directory containing the dataset to load")
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
