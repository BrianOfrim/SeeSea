"""
Fine-tuning a segmentation model for the Seesea dataset
"""

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

LOGGER = logging.getLogger(__name__)


def main(args):
    """
    Fine-tune a segmentation model for the Seesea dataset
    Segmentation models seem to confuse water with mountains or sand.
    We will fine-tune the model to mitigate this.
    """

    # Load a teacher and student model, we will use the teacher to generate labels for the student
    # We will add the teacher's mountain and sand masks to the water mask

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    teacher_pipe = pipeline("image-segmentation", model=args.model_name, device=device)
    student_model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)

    image_processor = SegformerImageProcessor.from_pretrained(args.model_name)

    config = AutoConfig.from_pretrained(args.model_name)

    id2label = config.id2label
    label2id = config.label2id

    # Load the dataset
    full_dataset = load_dataset("webdataset", data_dir=args.dataset, streaming=True)

    def map_fn(samples):
        samples["pixel_values"] = image_processor(samples["jpg"])["pixel_values"]
        teacher_outputs = teacher_pipe(samples["jpg"])
        samples["masks"] = []
        for teacher_output in teacher_outputs:
            # for each sample merge all masks into a single maske where the pixel values are the label ids
            first_output = teacher_output[0]
            first_mask = first_output["mask"]
            label_mask = torch.zeros_like(torch.tensor(np.array(first_mask)))
            for output in teacher_output:
                id = label2id[output["label"]]
                mask_array = torch.tensor(np.array(output["mask"]))
                label_mask[mask_array != 0] = id
            samples["masks"].append(label_mask)

        return samples

    # train the
    train_ds = full_dataset["train"].take(8).shuffle().map(map_fn, batched=True)

    for sample in train_ds:
        print(sample)
        break

    # user the teacher to generate labels
    # val_ds = full_dataset["validation"].map(map_fn, batched=True).select_columns(["labels", "pixel_values"])

    # data_collator = DefaultDataCollator()

    # accuracy = evaluate.load("accuracy")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a segmentation model for the Seesea dataset")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--model-name", type=str, help="Name of the pretrained model", required=True)
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--output-dir", type=str, help="Path to the output directory", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
