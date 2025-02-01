import os
import logging
import torch
from datasets import load_dataset
from transformers import pipeline, AutoConfig
import numpy as np
import webdataset as wds
from tqdm import tqdm
from PIL import Image

LOGGER = logging.getLogger(__name__)

# the model sometimes confuses sea for mountains, sand, etc``
label_as_sea = ["earth", "mountain", "sand", "water"]

# Threshold for the fraction of a mask to relabel
threashold_to_relabel = 0.01


def process_dataset(original_dataset, name, model, config):
    output_dir = os.path.join(args.output, name)
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(os.path.join(output_dir, "%06d.tar"), maxsize=1e9) as sink:
        for sample in tqdm(original_dataset):

            image = sample["jpg"]
            output = model(image)

            first_output = output[0]
            first_mask = first_output["mask"]
            label_mask = torch.zeros_like(torch.tensor(np.array(first_mask)))

            # Create a mask for each label with different colors
            for o in output:
                label = o["label"]
                mask = np.array(o["mask"])
                mask_array = torch.tensor(mask, dtype=torch.long)
                if label in label_as_sea:
                    fraction_of_mask = torch.sum(mask_array != 0) / torch.numel(mask_array)
                    if fraction_of_mask > threashold_to_relabel:
                        label = "sea"

                    label_id = config.label2id[label]
                    label_mask[mask_array != 0] = label_id

            # Convert tensor to numpy array and then to PIL Image for PNG saving
            label_mask_np = label_mask.numpy().astype(np.uint8)
            mask_image = Image.fromarray(label_mask_np)

            key = sample["__key__"]

            # Save both image and mask as their respective formats
            sink.write({"image.jpg": image, "mask.png": mask_image, "__key__": key})


def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Create output pattern for tar files

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(args.model)
    model = pipeline("image-segmentation", model=args.model, device=device)

    dataset = load_dataset("webdataset", data_dir=args.input, streaming=True)

    train_ds = dataset["train"]
    validation_ds = dataset["validation"]
    test_ds = dataset["test"]

    process_dataset(train_ds, "train", model, config)
    process_dataset(validation_ds, "validation", model, config)
    process_dataset(test_ds, "test", model, config)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Make a dataset for the Seesea dataset")
    parser.add_argument("--input", type=str, help="Path to the input directory", required=True)
    parser.add_argument("--output", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--model", type=str, help="Name of the model to use to generate the dataset", required=True)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
