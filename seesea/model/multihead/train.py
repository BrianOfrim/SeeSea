"Training script for multiheaded regression model"

import os
import logging
from functools import partial
import datetime
from typing import Callable
import json
import argparse
import numpy as np

from torchvision import transforms
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
import evaluate
import torch

from seesea.model.multihead.multihead_model import MultiHeadModel

LOGGER = logging.getLogger(__name__)


def augment_batch(augmentation: Callable, samples):
    """Preprocess a batch of samples"""
    samples["jpg"] = [augmentation(jpg) for jpg in samples["jpg"]]
    return samples


def preprocess_batch(transform: Callable, label_keys: list, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]

    # Create labels array with NaN for missing values
    labels = []
    for obj in samples["json"]:
        sample_labels = []
        # Add main labels
        for key in label_keys:
            value = obj.get(key)
            sample_labels.append(float(value) if value is not None else float("nan"))

        # Add image direction as the last column for each sample
        image_direction = obj.get("image_direction_deg")
        sample_labels.append(float(image_direction) if image_direction is not None else float("nan"))

        labels.append(sample_labels)

    samples["labels"] = labels
    return samples


def compute_angle_from_sincos(predictions_slice):
    # Convert numpy array to tensor if needed
    if isinstance(predictions_slice, np.ndarray):
        predictions_slice = torch.from_numpy(predictions_slice)

    # predictions_slice has shape [batch_size, 2], where:
    # predictions_slice[:, 0] is the sin component and predictions_slice[:, 1] is the cos component.
    predicted_radians = torch.atan2(predictions_slice[:, 0], predictions_slice[:, 1])
    # Convert to degrees
    predicted_degrees = torch.rad2deg(predicted_radians)
    return predicted_degrees


def circular_error(predictions, targets):
    # Convert numpy arrays to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Compute the absolute difference and then take the circular minimum
    diff = torch.abs(predictions - targets) % 360.0
    return torch.minimum(diff, 360.0 - diff)


def compute_metrics(metric, output_names, metric_stats, eval_pred):
    """Compute the metrics for the evaluation"""
    logits, labels = eval_pred
    results = {}
    col = 0

    # Collect all predictions and references
    all_predictions = []
    all_references = []

    for name in output_names:
        # Get the range for this parameter from metric_stats
        # Find the stat entry for this name
        stat_entry = next((stat for stat in metric_stats if stat.get("name") == name), {})
        param_range = stat_entry.get("max", 1.0) - stat_entry.get("min", 0.0)

        # Avoid division by zero
        param_range = max(param_range, 1.0)

        if name.endswith("_deg"):  # Angle head
            predictions_slice = logits[:, col : col + 2]
            predictions_angles = compute_angle_from_sincos(predictions_slice)

            valid_mask = ~np.isnan(labels[:, col]) & ~np.isnan(labels[:, -1])

            if valid_mask.any():
                predictions = predictions_angles[valid_mask] + torch.from_numpy(labels[valid_mask, -1])
                predictions = predictions % 360.0
                errors = circular_error(predictions, labels[valid_mask, col])
                # Store unnormalized MAE for individual metric
                mae_circular = errors.mean().item()
                results[f"mae_{name}"] = mae_circular

                # Add normalized values to overall metrics (normalize by 180 for circular metrics)
                all_predictions.append(errors.cpu().numpy() / 180.0)
                all_references.append(torch.zeros_like(errors).cpu().numpy() / 180.0)
            else:
                results[f"mae_{name}"] = float("nan")

            col += 2
        else:
            predictions = logits[:, col]
            valid_mask = ~np.isnan(labels[:, col])

            if valid_mask.any():
                # Store unnormalized MAE for individual metric
                mae = metric.compute(predictions=predictions[valid_mask], references=labels[valid_mask, col])["mae"]
                results[f"mae_{name}"] = mae

                # Add normalized values to overall metrics
                norm_pred = predictions[valid_mask] / param_range
                norm_ref = labels[valid_mask, col] / param_range
                all_predictions.append(norm_pred)
                all_references.append(norm_ref)
            else:
                results[f"mae_{name}"] = float("nan")

            col += 1

    # Compute overall normalized MAE
    if all_predictions and all_references:
        all_predictions = np.concatenate([p.flatten() for p in all_predictions])
        all_references = np.concatenate([r.flatten() for r in all_references])
        results["mae"] = metric.compute(predictions=all_predictions, references=all_references)["mae"]
    else:
        results["mae"] = float("nan")

    return results


def main(args):
    """train the model"""

    LOGGER.info("Training the model to classify %s", args.output_names)

    os.makedirs(args.output, exist_ok=True)

    training_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    timestamp_str = training_start_time.strftime("%Y_%m_%d_%H%M")

    if args.checkpoint:
        # Get the parent directory of the checkpoint
        output_dir = os.path.join(args.checkpoint, os.pardir)
    else:
        output_dir = os.path.join(args.output, timestamp_str)

    split_sizes = None
    if os.path.exists(os.path.join(args.input, "split_sizes.json")):
        with open(os.path.join(args.input, "split_sizes.json"), "r", encoding="utf-8") as f:
            split_sizes = json.load(f)

    if args.num_training_samples is not None:
        num_training_samples = args.num_training_samples
        LOGGER.info(
            "Using %d training samples, from command line argument --num-training-samples", num_training_samples
        )
    elif split_sizes is not None:
        num_training_samples = split_sizes["training"]
        LOGGER.info("Loaded split sizes from split_sizes.json. Using %d training samples", num_training_samples)
    else:
        num_training_samples = 65536
        LOGGER.info("No split sizes found, using %d training samples", num_training_samples)

    if args.num_validation_samples is not None:
        num_validation_samples = args.num_validation_samples
        LOGGER.info(
            "Using %d validation samples, from command line argument --num-validation-samples", num_validation_samples
        )
    elif split_sizes is not None:
        num_validation_samples = split_sizes["validation"]
        LOGGER.info("Loaded split sizes from split_sizes.json. Using %d validation samples", num_validation_samples)
    else:
        num_validation_samples = 4096
        LOGGER.info("No split sizes found, using %d validation samples", num_validation_samples)

    steps_per_epoch = num_training_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    if args.checkpoint is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model)
        # Save the image processor to the output directory
        image_processor.save_pretrained(output_dir)

        base_model = AutoModelForImageClassification.from_pretrained(args.model, ignore_mismatched_sizes=True)
        model = MultiHeadModel(base_model, args.output_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=steps_per_epoch,
            save_strategy="steps",
            save_steps=steps_per_epoch,
            load_best_model_at_end=True,
            save_total_limit=1,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_strategy="steps",
            logging_steps=50,
            max_steps=total_steps,
        )
    else:
        image_processor = AutoImageProcessor.from_pretrained(output_dir)
        model = torch.load(os.path.join(args.checkpoint))
        training_args = torch.load(os.path.join(args.checkpoint, "training_args.bin"))
        training_args.ignore_data_skip = True

    augmentation = None
    if args.rotation is not None:
        augmentation = transforms.RandomRotation(args.rotation, interpolation=transforms.InterpolationMode.BILINEAR)
        LOGGER.info("Using random rotation of %.2f degrees for data augmentation", args.rotation)

    map_fn = partial(preprocess_batch, image_processor, args.output_names)

    full_dataset = load_dataset("webdataset", data_dir=args.input, streaming=True)

    train_ds = full_dataset["train"].take(num_training_samples).shuffle()

    if augmentation is not None:
        train_ds = train_ds.map(partial(augment_batch, augmentation), batched=True)

    train_ds = train_ds.map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    val_ds = (
        full_dataset["validation"]
        .take(num_validation_samples)
        .map(map_fn, batched=True)
        .select_columns(["labels", "pixel_values"])
    )

    data_collator = DefaultDataCollator()

    metric_stats = {}
    if os.path.exists(os.path.join(args.input, "statistics.json")):
        with open(os.path.join(args.input, "statistics.json"), "r", encoding="utf-8") as f:
            metric_stats = json.load(f)

    mae_metric = evaluate.load("mae")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=partial(compute_metrics, mae_metric, args.output_names, metric_stats),
    )

    trainer.train(resume_from_checkpoint=args.checkpoint)

    # save the model
    torch.save(model, os.path.join(output_dir, "model.pt"))
    LOGGER.info("Model saved successfully to %s", output_dir)

    # save the output name
    output_names_path = os.path.join(output_dir, "output_names.txt")
    with open(output_names_path, "w", encoding="utf-8") as output_names_file:
        output_names_file.write("\n".join(args.output_names))

    # run the test set
    test_ds = full_dataset["test"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    test_result = trainer.predict(test_ds)

    LOGGER.info("Test results: %s", test_result)


def get_args_parser():

    parser = argparse.ArgumentParser(description="Train the SeeSea multiheaded regression model")

    parser.add_argument("--input", help="The directory containing the training data", default="data")

    parser.add_argument(
        "--rotation", type=float, help="The random rotation angle to use for data augmentation", default=None
    )

    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--checkpoint", help="Path to a saved model checkpoint to load", default=None)

    group.add_argument("--model", type=str, help="The model to use for training", default="resnet18")

    # The following are only used if no checkpoint path was provided
    parser.add_argument(
        "--output",
        help="The directory to write the output files to.",
        default="data/train",
    )
    parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=30)
    parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    parser.add_argument("--learning-rate", type=float, help="The learning rate to use for training", default=0.001)
    parser.add_argument("--warmup-ratio", type=float, help="The ratio of steps to use for warmup", default=0.1)
    parser.add_argument(
        "--output-names",
        type=str,
        nargs="+",
        help="The observation variable(s) to train the network to classify",
    )
    parser.add_argument("--num-training-samples", type=int, help="The number of training samples to use", default=None)
    parser.add_argument(
        "--num-validation-samples", type=int, help="The number of validation samples to use", default=None
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
