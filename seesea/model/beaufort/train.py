"""Train a model to classify Beaufort sea states"""

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
from seesea.model.beaufort.beaufort_utils import id2label_beaufort, label2id_beaufort, preprocess_batch_beaufort

LOGGER = logging.getLogger(__name__)


def augment_batch(augmentation: Callable, samples):
    """Preprocess a batch of samples"""
    samples["jpg"] = [augmentation(jpg) for jpg in samples["jpg"]]
    return samples


def compute_metrics(metric, eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main(args):
    """train the model"""

    os.makedirs(args.output, exist_ok=True)

    training_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    timestamp_str = training_start_time.strftime("%Y_%m_%d_%H%M")

    if args.checkpoint:
        model_name_or_path = args.checkpoint
        # Get the parent directory of the checkpoint
        output_dir = os.path.join(args.checkpoint, os.pardir)
    else:
        model_name_or_path = args.model
        output_dir = os.path.join(args.output, timestamp_str)

    if os.path.exists(os.path.join(args.input, "split_sizes.json")):
        with open(os.path.join(args.input, "split_sizes.json"), "r", encoding="utf-8") as f:
            split_sizes = json.load(f)
        num_training_samples = split_sizes["training"]
        LOGGER.info(
            "Loaded split sizes from %s. Using %d training samples",
            os.path.join(args.input, "split_sizes.json"),
            num_training_samples,
        )
    else:
        num_training_samples = 65536
        LOGGER.info("No split sizes found, using %d training samples", num_training_samples)

    steps_per_epoch = num_training_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    model = AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        id2label=id2label_beaufort,
        label2id=label2id_beaufort,
        num_labels=len(id2label_beaufort),
        ignore_mismatched_sizes=True,
    )

    if args.checkpoint is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model)
        # Save the image processor to the output directory
        image_processor.save_pretrained(output_dir)
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
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
    else:
        image_processor = AutoImageProcessor.from_pretrained(output_dir)
        training_args = torch.load(os.path.join(args.checkpoint, "training_args.bin"))
        training_args.ignore_data_skip = True
        print(training_args)

    augmentation = None
    if args.rotation is not None:
        augmentation = transforms.RandomRotation(args.rotation, interpolation=transforms.InterpolationMode.BILINEAR)
        LOGGER.info("Using random rotation of %.2f degrees for data augmentation", args.rotation)

    map_fn = partial(preprocess_batch_beaufort, image_processor)

    full_dataset = load_dataset("webdataset", data_dir=args.input, streaming=True)

    train_ds = full_dataset["train"].take(num_training_samples).shuffle()

    if augmentation is not None:
        train_ds = train_ds.map(partial(augment_batch, augmentation), batched=True)

    train_ds = train_ds.map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    val_ds = full_dataset["validation"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    data_collator = DefaultDataCollator()

    accuracy = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=accuracy,
    )

    trainer.train(resume_from_checkpoint=args.checkpoint)

    # run the test set
    test_ds = full_dataset["test"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    test_result = trainer.predict(test_ds)

    LOGGER.info("Test results: %s", test_result)

    # save the model

    torch.save(model, os.path.join(output_dir, "model.pt"))
    LOGGER.info("Model saved successfully to %s", output_dir)


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train the SeeSea model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", help="Path to a saved model checkpoint to load", default=None)

    group.add_argument(
        "--model",
        type=str,
        help="The model architecture to use for training.",
        default="resnet18",
    )

    parser.add_argument("--input", help="The directory containing the training data", default="data")
    parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=30)
    parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    parser.add_argument("--log", type=str, help="Log level", default="INFO")
    parser.add_argument("--log-file", type=str, help="Log file", default=None)
    parser.add_argument(
        "--rotation", type=float, help="The random rotation angle to use for data augmentation", default=None
    )
    parser.add_argument("--learning-rate", type=float, help="The learning rate to use for training", default=0.001)
    parser.add_argument("--warmup-ratio", type=float, help="The ratio of steps to use for warmup", default=0.1)

    parser.add_argument(
        "--output",
        help="The directory to write the output files to. (only used if no checkpoint is provided)",
        default="data/train",
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

    logging.getLogger("transformers.trainer").setLevel(logging.DEBUG)
    main(args)
