"Training script for the SeeSea model."

import os
import logging
from functools import partial
import datetime
from typing import Callable

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
from torch import nn
import torch

LOGGER = logging.getLogger(__name__)


def augment_batch(augmentation: Callable, samples):
    """Preprocess a batch of samples"""
    samples["jpg"] = [augmentation(jpg) for jpg in samples["jpg"]]
    return samples


def preprocess_batch(transform: Callable, label_keys: list, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]
    samples["labels"] = [[obj[key] for key in label_keys] for obj in samples["json"]]
    return samples


def compute_metrics(metric, eval_pred):
    """Compute the metrics for the evaluation"""
    logits, labels = eval_pred
    predictions = logits
    results = {}
    # Individual MAEs for each output
    for i in range(predictions.shape[1]):
        results[f"mae_{i}"] = metric.compute(predictions=predictions[:, i], references=labels[:, i])["mae"]
    # Overall MAE across all outputs
    results["mae_overall"] = metric.compute(predictions=predictions.flatten(), references=labels.flatten())["mae"]
    return results


class MultiHeadModel(nn.Module):
    def __init__(self, base_model, num_outputs):
        super().__init__()
        if num_outputs < 1:
            raise ValueError("num_outputs must be at least 1")

        self.base_model = base_model
        hidden_size = base_model.classifier.in_features

        # Add global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_outputs)])

    def forward(self, pixel_values, labels):
        if labels is None:
            raise ValueError("labels cannot be None")

        features = self.base_model.base_model(pixel_values)[0]  # Shape: [batch_size, channels, height, width]
        # Apply pooling and reshape
        features = self.pool(features)  # Shape: [batch_size, channels, 1, 1]
        features = features.flatten(1)  # Shape: [batch_size, channels]
        logits = torch.cat([head(features) for head in self.heads], dim=1)

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels.float())

        return loss, logits


def main(args):
    """train the model"""

    LOGGER.info("Training the model to classify %s", args.output_names)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_processor = AutoImageProcessor.from_pretrained(args.model)

    base_model = AutoModelForImageClassification.from_pretrained(args.model, ignore_mismatched_sizes=True)
    model = MultiHeadModel(base_model, len(args.output_names))

    augmentation = None
    if args.rotation is not None:
        augmentation = transforms.RandomRotation(args.rotation, interpolation=transforms.InterpolationMode.BILINEAR)
        LOGGER.info("Using random rotation of %.2f degrees for data augmentation", args.rotation)

    map_fn = partial(preprocess_batch, image_processor, args.output_names)

    num_training_samples = 65536
    steps_per_epoch = num_training_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    LOGGER.debug(
        "Training for %d epochs with %d steps per epoch. Batch size: %d,  Total steps: %d",
        args.epochs,
        steps_per_epoch,
        args.batch_size,
        total_steps,
    )

    full_dataset = load_dataset("webdataset", data_dir=args.input, streaming=True)

    train_ds = full_dataset["train"].take(num_training_samples).shuffle()

    if augmentation is not None:
        train_ds = train_ds.map(partial(augment_batch, augmentation), batched=True)

    train_ds = train_ds.map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    val_ds = full_dataset["validation"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    data_collator = DefaultDataCollator()

    mae_metric = evaluate.load("mae")

    training_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    timestamp_str = training_start_time.strftime("%Y_%m_%d_%H%M")
    model_dir = os.path.join(args.output, timestamp_str)

    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="steps",
        logging_steps=50,
        max_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=partial(compute_metrics, mae_metric),
    )

    trainer.train()

    # run the test set
    test_ds = full_dataset["test"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    test_result = trainer.predict(test_ds)

    LOGGER.info("Test results: %s", test_result)

    # save the model
    model_ouput_dir = os.path.join(model_dir, "model")
    if not os.path.exists(model_ouput_dir):
        os.makedirs(model_ouput_dir)
    model.save_pretrained(model_ouput_dir)
    # save the image processor
    image_processor.save_pretrained(model_ouput_dir)

    # save the output name
    output_names_path = os.path.join(model_ouput_dir, "output_names.txt")
    with open(output_names_path, "w", encoding="utf-8") as output_names_file:
        output_names_file.write("\n".join(args.output_names))


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train the SeeSea model")
    parser.add_argument("--input", help="The directory containing the training data", default="data")
    parser.add_argument("--output", help="The directory to write the output files to", default="data/train")
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)
    parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=30)
    parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    parser.add_argument("--learning-rate", type=float, help="The learning rate to use for training", default=0.001)
    parser.add_argument("--model", type=str, help="The model to use for training", default="resnet18")
    parser.add_argument(
        "--output-names",
        type=str,
        nargs="+",
        help="The observation variable(s) to train the network to classify",
        required=True,
    )
    parser.add_argument(
        "--rotation", type=float, help="The random rotation angle to use for data augmentation", default=None
    )
    parser.add_argument("--warmup-ratio", type=float, help="The ratio of steps to use for warmup", default=0.1)
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
