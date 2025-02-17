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

from seesea.model.multihead.multihead_model import MultiHeadModel, MultiHeadConfig  # Import the model class

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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

    if "wind_speed_mps" in labels.keys():
        rounded_predictions = np.round(predictions["wind_speed_mps"])
        test_results["wind_speed_mps"] |= {
            "rounded_mse": np.mean(np.square(rounded_predictions - labels["wind_speed_mps"])).item(),
            "rounded_mae": np.mean(np.abs(rounded_predictions - labels["wind_speed_mps"])).item(),
        }
        # Use isclose() instead of direct equality comparison
        test_results["wind_speed_mps"] |= {
            "rounded_accuracy": np.mean(
                np.isclose(rounded_predictions, labels["wind_speed_mps"], rtol=1e-5, atol=1e-8)
            ).item(),
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

        # Inputs vs Outputs density scatter plot
        plt.figure(figsize=(10, 8))
        plt.hist2d(inputs, outputs, bins=50, cmap="viridis", norm=plt.cm.colors.LogNorm())
        plt.colorbar(label="Count")
        plt.plot([np.min(inputs), np.max(inputs)], [np.min(inputs), np.max(inputs)], "r--")
        plt.title(f"{name}: Inputs vs Outputs (with density)")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.savefig(os.path.join(output_dir, f"{name}_inputs_vs_outputs_density.png"))
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

    config = MultiHeadConfig.from_pretrained(args.model_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = MultiHeadModel.from_pretrained(args.model_dir, config=config)

    LOGGER.info("Testing model for outputs: %s", config.output_head_names)

    # Setup dataset
    map_fn = partial(preprocess_batch, image_processor, config.output_head_names)
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
    all_outputs = {name: [] for name in config.output_head_names}
    all_inputs = {name: [] for name in config.output_head_names}

    test_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    model.eval()
    for batch in tqdm(loader, desc="Testing", disable=LOGGER.level > logging.INFO):
        # Split batch labels into separate arrays for each output
        batch_labels = batch["labels"].numpy()
        for i, name in enumerate(config.output_head_names):
            all_inputs[name].append(batch_labels[:, i])

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        # Split predictions into separate arrays for each output
        predictions = outputs["logits"].cpu().numpy()
        for i, name in enumerate(config.output_head_names):
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
