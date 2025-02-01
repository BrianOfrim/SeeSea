"""
Fine-tuning a segmentation model for the Seesea dataset
"""

import os
import json
import logging
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    DefaultDataCollator,
    pipeline,
    AutoConfig,
)
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import datetime

LOGGER = logging.getLogger(__name__)


def main(args):
    """
    Fine-tune a segmentation model for the Seesea dataset
    Segmentation models seem to confuse water with mountains or sand.
    We will fine-tune the model to mitigate this.
    """

    # Load a teacher and student model, we will use the teacher to generate labels for the student
    # We will add the teacher's mountain and sand masks to the water mask

    os.makedirs(args.output_dir, exist_ok=True)

    training_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    timestamp_str = training_start_time.strftime("%Y_%m_%d_%H%M")

    run_output_dir = os.path.join(args.output_dir, timestamp_str)

    num_training_samples = None
    num_validation_samples = None
    num_test_samples = None

    if args.num_training_samples is not None:
        num_training_samples = args.num_training_samples

    elif os.path.exists(os.path.join(args.dataset, "split_sizes.json")):
        with open(os.path.join(args.dataset, "split_sizes.json"), "r", encoding="utf-8") as f:
            split_sizes = json.load(f)
        num_training_samples = split_sizes["training"]
        num_validation_samples = split_sizes["validation"]
        num_test_samples = split_sizes["test"]
        LOGGER.info(
            "Loaded split sizes from %s. Using %d training samples, %d validation samples, %d test samples",
            os.path.join(args.dataset, "split_sizes.json"),
            num_training_samples,
            num_validation_samples,
            num_test_samples,
        )

    if num_training_samples is None:
        num_training_samples = 65536

    if num_validation_samples is None:
        num_validation_samples = num_training_samples // 10

    if num_test_samples is None:
        num_test_samples = num_training_samples // 10

    LOGGER.info(
        "Using %d training samples, %d validation samples, %d test samples",
        num_training_samples,
        num_validation_samples,
        num_test_samples,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)
    model = model.to(device)

    image_processor = SegformerImageProcessor.from_pretrained(args.model_name)

    config = AutoConfig.from_pretrained(args.model_name)

    id2label = config.id2label
    label2id = config.label2id

    # Load the dataset
    full_dataset = load_dataset("webdataset", data_dir=args.dataset, streaming=True)

    def map_fn(samples):
        # Process images with the image processor and convert to tensor
        pixel_values = image_processor(samples["image.jpg"], return_tensors="pt")["pixel_values"]

        # Convert PNG mask to numpy array first, then to tensor
        labels = torch.tensor(np.array(samples["mask.png"]), dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}

    # Save the image processor to the output directory
    image_processor.save_pretrained(run_output_dir)

    # train the model
    train_ds = (
        full_dataset["train"]
        .take(num_training_samples)
        .shuffle(buffer_size=512)
        .map(map_fn, batched=True)
        .select_columns(["pixel_values", "labels"])
    )

    val_ds = (
        full_dataset["validation"]
        .take(num_validation_samples)
        .map(map_fn, batched=True)
        .select_columns(["pixel_values", "labels"])
    )

    steps_per_epoch = num_training_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()

    # run the test set
    test_ds = (
        full_dataset["test"].take(num_test_samples).map(map_fn, batched=True).select_columns(["pixel_values", "labels"])
    )

    test_result = trainer.predict(test_ds)

    LOGGER.info("Test results: %s", test_result)

    # save the model

    torch.save(model, os.path.join(run_output_dir, "model.pt"))
    LOGGER.info("Model saved successfully to %s", run_output_dir)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a segmentation model for the Seesea dataset")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--model-name", type=str, help="Name of the pretrained model", required=True)
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--output-dir", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, help="Warmup ratio", default=0.1)
    parser.add_argument("--num-training-samples", type=int, help="Number of training samples", default=None)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
