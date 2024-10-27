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

import seesea.common.utils as utils


LOGGER = logging.getLogger(__name__)


def augment_batch(augmentation: Callable, samples):
    """Preprocess a batch of samples"""
    samples["jpg"] = [augmentation(jpg) for jpg in samples["jpg"]]
    return samples


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
    """train the model"""

    LOGGER.info("Training the model to classify %s", args.output_name)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_processor = AutoImageProcessor.from_pretrained(args.model)

    model = AutoModelForImageClassification.from_pretrained(args.model, ignore_mismatched_sizes=True, num_labels=1)

    augmentation = None
    if args.rotation is not None:
        augmentation = transforms.RandomRotation(args.rotation, interpolation=transforms.InterpolationMode.BILINEAR)
        LOGGER.info("Using random rotation of %.2f degrees for data augmentation", args.rotation)

    map_fn = partial(preprocess_batch, image_processor, args.output_name)

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

    train_ds = train_ds.map(map_fn, batched=True).select_columns(["label", "pixel_values"])

    val_ds = full_dataset["validation"].map(map_fn, batched=True).select_columns(["label", "pixel_values"])

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
        warmup_ratio=0.15,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="steps",
        logging_steps=50,
        max_steps=total_steps,
        save_safetensors=False,
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
    test_ds = full_dataset["test"].map(map_fn, batched=True).select_columns(["label", "pixel_values"])

    test_result = trainer.predict(test_ds)

    LOGGER.info("Test results: %s", test_result)


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
        "--output-name",
        type=str,
        help="The observation variable to train the netowrk to classify",
        default="wind_speed_mps",
    )
    parser.add_argument(
        "--rotation", type=float, help="The random rotation angle to use for data augmentation", default=None
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
