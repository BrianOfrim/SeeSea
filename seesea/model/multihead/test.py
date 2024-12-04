"""
Test the multihead model
"""

import os
from typing import Callable
from functools import partial
import logging
import datetime
import json

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, DefaultDataCollator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from seesea.model.multihead.multihead_model import MultiHeadModel  # Import the model class

LOGGER = logging.getLogger(__name__)


def preprocess_batch(transform: Callable, output_names: list, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]
    samples["labels"] = [[obj[key] for key in output_names] for obj in samples["json"]]
    return samples


def save_results(labels, predictions, output_dir):
    """Save test results and plots"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate metrics for each output
    test_results = {}
    for name in labels.keys():
        residuals = labels[name] - predictions[name]
        test_results[name] = {
            "mse": np.mean(np.square(residuals)).item(),
            "mae": np.mean(np.abs(residuals)).item(),
        }

    # Overall metrics
    all_residuals = np.concatenate([labels[name] - predictions[name] for name in labels.keys()])
    test_results["overall"] = {
        "mse": np.mean(np.square(all_residuals)).item(),
        "mae": np.mean(np.abs(all_residuals)).item(),
    }

    LOGGER.info("Test results: %s", test_results)

    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    # Generate plots for each output
    for name in labels.keys():
        inputs = labels[name]
        outputs = predictions[name]
        residuals = inputs - outputs

        # Inputs vs Outputs scatter plot
        plt.figure()
        plt.scatter(inputs, outputs, alpha=0.5, s=1)
        plt.plot([np.min(inputs), np.max(inputs)], [np.min(inputs), np.max(inputs)], "r--")
        plt.title(f"{name}: Inputs vs Outputs")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.savefig(os.path.join(output_dir, f"{name}_inputs_vs_outputs.png"))
        plt.close()

        # Residuals histogram
        plt.figure()
        plt.hist(residuals, bins="auto")
        plt.title(f"{name}: Residuals Histogram")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, f"{name}_residuals_hist.png"))
        plt.close()


def main(args):
    """Run test on multihead model"""

    # Load model and processor
    model = torch.load(os.path.join(args.model_dir, "model.pt"))
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(args.model_dir))

    # Get output names
    with open(os.path.join(args.model_dir, "output_names.txt"), "r", encoding="utf-8") as f:
        output_names = f.read().strip().split("\n")

    LOGGER.info("Testing model for outputs: %s", output_names)

    # Setup dataset
    map_fn = partial(preprocess_batch, image_processor, output_names)
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True)
    dataset = dataset.map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    # Setup dataloader
    collator = DefaultDataCollator()
    loader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # Run test
    all_outputs = {name: [] for name in output_names}
    all_inputs = {name: [] for name in output_names}

    test_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    model.eval()
    for batch in tqdm(loader, desc="Testing", disable=LOGGER.level > logging.INFO):
        # Split batch labels into separate arrays for each output
        batch_labels = batch["labels"].numpy()
        for i, name in enumerate(output_names):
            all_inputs[name].append(batch_labels[:, i])

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _, logits = model(**batch)

        # Split predictions into separate arrays for each output
        predictions = logits.cpu().numpy()
        for i, name in enumerate(output_names):
            all_outputs[name].append(predictions[:, i])

    test_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    LOGGER.info("Test duration: %s", test_end_time - test_start_time)

    # Concatenate results
    all_inputs = {name: np.concatenate(arrays) for name, arrays in all_inputs.items()}
    all_outputs = {name: np.concatenate(arrays) for name, arrays in all_outputs.items()}

    # Save results
    save_results(all_inputs, all_outputs, args.output)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Test the SeeSea multihead model")
    parser.add_argument("--model-dir", help="Directory containing the trained model", required=True)
    parser.add_argument("--dataset", help="Directory containing the test dataset", required=True)
    parser.add_argument("--output", help="Directory to save test results", required=True)
    parser.add_argument("--split", help="Dataset split to use", default="test")
    parser.add_argument("--batch-size", type=int, help="Batch size for testing", default=32)
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
