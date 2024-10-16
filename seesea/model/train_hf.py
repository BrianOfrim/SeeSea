"Training script for the SeeSea model."

import os
import logging
from functools import partial
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

LOGGER = logging.getLogger(__name__)


def preprocess_batch(transform: Callable, label_key: str, samples):
    samples["pixel_values"] = [transform(jpg) for jpg in samples["jpg"]]
    samples["label"] = [obj[label_key] for obj in samples["json"]]
    return samples


def main(args):
    """train the model"""

    LOGGER.info("Training the model to classify %s", args.output_name)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_processor = AutoImageProcessor.from_pretrained(args.model)

    normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    base_transform = transforms.Compose([transforms.RandomResizedCrop(size), transforms.ToTensor(), normalize])

    model = AutoModelForImageClassification.from_pretrained(args.model, ignore_mismatched_sizes=True, num_labels=1)

    train_transform = base_transform

    if args.rotation is not None:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(args.rotation, interpolation=transforms.InterpolationMode.BILINEAR),
                train_transform,
            ]
        )
        LOGGER.info("Using random rotation of %.2f degrees for data augmentation", args.rotation)

    train_map_fn = partial(preprocess_batch, train_transform, args.output_name)
    validation_map_fn = partial(preprocess_batch, base_transform, args.output_name)

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

    train_ds = (
        full_dataset["train"]
        .map(train_map_fn, batched=True)
        .take(num_training_samples)
        .shuffle()
        .select_columns(["label", "pixel_values"])
    )

    val_ds = full_dataset["validation"].map(validation_map_fn, batched=True).select_columns(["label", "pixel_values"])

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="/Volumes/external/tmp",
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="steps",
        logging_steps=64,
        max_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
    )

    trainer.train()


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
    parser.add_argument("--model-path", type=str, help="The path to save the trained model", default="model.pth")
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
