"""Test the model."""

import os
import logging
from functools import partial
import datetime
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from seesea.common import utils
from seesea.model.training_results import TrainingResults
from seesea.model import engine

LOGGER = logging.getLogger(__name__)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--input", help="Path to the directory containing the dataset to load")
    parser.add_argument("--batch-size", type=int, help="The batch size to use for evaluation", default=128)
    parser.add_argument("--split", help="Which dataset split to load.", default="test")
    parser.add_argument("--model-dir", help="Path to the directory containing the trained model.")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)

    return parser


def save_results(inputs, outputs, output_dir):
    residuals = inputs - outputs

    test_results = {
        "mse": np.mean(np.square(residuals)).item(),
        "mae": np.mean(np.abs(residuals)).item(),
    }

    LOGGER.info("Test results: %s", test_results)

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


@torch.no_grad()
def test_model(model, loader, device, output_dir):

    all_outputs = np.array([])
    all_inputs = np.array([])

    test_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    model.eval()
    for inputs, label in tqdm(loader, leave=False, desc="Eval", disable=LOGGER.level > logging.INFO):
        all_inputs = np.append(all_inputs, label.numpy())
        inputs = inputs.to(device)
        label = label.to(device)

        outputs = model(inputs)
        outputs = outputs.view(-1)
        all_outputs = np.append(all_outputs, outputs.cpu().numpy())

    test_end_time = datetime.datetime.now(tz=datetime.timezone.utc)

    LOGGER.info("Test duration: %s", test_end_time - test_start_time)
    save_results(all_inputs, all_outputs, output_dir)


def main(args):
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    training_details = utils.from_dict(TrainingResults, training_details_json)

    LOGGER.debug("Loaded training details: %s", training_details)

    model, transform = engine.continuous_single_output_model_factory(training_details.model, model_weights_path)

    map_fn = partial(engine.preprocess, transform, training_details.output_name)
    test_ds = load_dataset("webdataset", data_dir=args.input, split=args.split, streaming=True).map(map_fn)

    test_loader = DataLoader(test_ds, collate_fn=engine.collate, batch_size=args.batch_size)

    model = model.to(device)

    test_model(model, test_loader, device, args.model_dir)


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
