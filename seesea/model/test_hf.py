"""
Test a trained model using the Hugging Face Transformers library.
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
from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def save_results(inputs, outputs, output_dir):
    residuals = inputs - outputs

    test_results = {
        "mse": np.mean(np.square(residuals)).item(),
        "mae": np.mean(np.abs(residuals)).item(),
    }

    LOGGER.info("Test results: %s", test_results)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    # Plot the inputs histogram
    plt.hist(inputs, bins="auto")
    plt.title("Inputs histogram")
    plt.xlabel("Inputs")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "inputs_hist.png"))
    plt.close()

    # Plot the residuals histogram
    plt.hist(residuals, bins="auto")
    plt.title("Residuals histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "residuals_hist.png"))
    plt.close()

    # Plot the absolute residuals histogram
    plt.hist(np.abs(residuals), bins="auto")
    plt.title("Absolute residuals histogram")
    plt.xlabel("Absolute residuals")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "abs_residuals_hist.png"))
    plt.close()

    # Plot the residuals vs. outputs
    plt.scatter(outputs, residuals)
    plt.title("Residuals vs. Outputs")
    plt.xlabel("Outputs")
    plt.ylabel("Residuals")
    plt.savefig(os.path.join(output_dir, "residuals_vs_outputs.png"))
    plt.close()

    # Plot inputs vs. outputs
    plt.scatter(inputs, outputs)
    # Plot the line y = x
    plt.plot([np.min(inputs), np.max(inputs)], [np.min(inputs), np.max(inputs)], color="red")
    plt.title("Inputs vs. Outputs")
    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    plt.savefig(os.path.join(output_dir, "inputs_vs_outputs.png"))
    plt.close()


def preprocess_batch(transform: Callable, label_key: str, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]
    samples["label"] = [obj[label_key] for obj in samples["json"]]
    return samples


def compute_metrics(metric, eval_pred):
    """Compute the metrics for the evaluation"""
    logits, labels = eval_pred
    predictions = logits.squeeze()  # Get predictions from logits
    return metric.compute(predictions=predictions, references=labels)


def main(args):

    model = AutoModelForImageClassification.from_pretrained(os.path.join(args.model_dir))
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(args.model_dir))

    # Get the output name from the output_name.text file in the model directory

    with open(os.path.join(args.model_dir, "output_name.txt"), "r", encoding="utf-8") as f:
        output_name = f.read().strip()

    map_fn = partial(preprocess_batch, image_processor, output_name)
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True)

    dataset = dataset.map(map_fn, batched=True).select_columns(["label", "pixel_values"])

    collator = DefaultDataCollator()

    loader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    all_outputs = np.array([])
    all_inputs = np.array([])

    test_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    model.eval()
    for batch in tqdm(loader, leave=False, desc="Eval", disable=LOGGER.level > logging.INFO):
        all_inputs = np.append(all_inputs, batch["labels"].numpy())
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.squeeze()  # Get predictions from logits
        all_outputs = np.append(all_outputs, predictions.cpu().numpy())

    test_end_time = datetime.datetime.now(tz=datetime.timezone.utc)

    LOGGER.info("Test duration: %s", test_end_time - test_start_time)

    save_results(all_inputs, all_outputs, args.output)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--dataset", help="Path to the directory containing the dataset to load")
    parser.add_argument("--output", help="Path to the directory to save the results")
    parser.add_argument("--batch-size", type=int, help="The batch size to use for evaluation", default=128)
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
