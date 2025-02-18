"""
See how well the Multihead model can predict the Beaufort scale from the image.
"""

import argparse
from typing import Callable
from functools import partial
import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator

from seesea.model.multihead.modeling_multihead import MultiHeadModel  # Import the model class
from seesea.model.beaufort.beaufort_utils import mps_to_beaufort

LOGGER = logging.getLogger(__name__)


def preprocess_batch(transform: Callable, output_names: list, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]
    samples["labels"] = [[obj[key] for key in output_names] for obj in samples["json"]]
    return samples


def main(args):
    # Load the model
    model = torch.load(os.path.join(args.model_dir, "model.pt"))
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(args.model_dir))

    with open(os.path.join(args.model_dir, "output_names.txt"), "r", encoding="utf-8") as f:
        output_names = f.read().strip().split("\n")

    if "wind_speed_mps" not in output_names:
        raise ValueError("wind_speed_mps is not in the output names")

    wind_speed_idx = output_names.index("wind_speed_mps")
    # Setup dataset
    map_fn = partial(preprocess_batch, image_processor, output_names)
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True)
    dataset = dataset.map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    # Setup dataloader
    collator = DefaultDataCollator()
    loader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size)

    # Get the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    beaufort_predictions = []
    beaufort_labels = []

    for batch in tqdm(loader, desc="Testing", disable=LOGGER.level > logging.INFO):

        beaufort_labels += [mps_to_beaufort(obj[wind_speed_idx]) for obj in batch["labels"]]
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            _, logits = model(**batch)

        # Get the predictions at the wind speed index
        predictions = logits.cpu().numpy()
        predictions_mps = predictions[:, wind_speed_idx]
        predictions_mps = np.maximum(predictions_mps, 0)
        beaufort_predictions += [mps_to_beaufort(pred.item()) for pred in predictions_mps]

    # Compute the accuracy of the predictions
    accuracy = sum(1 for pred, label in zip(beaufort_predictions, beaufort_labels) if pred == label) / len(
        beaufort_predictions
    )
    LOGGER.info("Accuracy: %f", accuracy)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run inference on a regression model to predict the wind speed on the Beaufort scale."
    )
    parser.add_argument("--dataset", help="Path to the directory containing the dataset to load")
    parser.add_argument("--split", help="Which dataset split to load.", default="test")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--batch-size", type=int, help="Batch size for testing", default=32)
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
