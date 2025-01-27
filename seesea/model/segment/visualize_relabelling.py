import os
import logging
import torch
from datasets import load_dataset
from transformers import pipeline, AutoConfig
import numpy as np
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(args.model)

    id2label = config.id2label
    label2id = config.label2id

    # seed the random number generator
    np.random.seed(42)

    # create a map between the label and the color
    label_to_color = {}
    for label, id in label2id.items():
        color = plt.cm.rainbow(np.random.random())
        label_to_color[label] = color

    dataset = load_dataset("webdataset", data_dir=args.input, streaming=True)

    train_ds = dataset["train"]
    model = pipeline("image-segmentation", model=args.model)

    # the model sometimes confuses sea for mountains and sand

    label_as_sea = ["earth", "mountain", "sand", "water"]

    threashold_to_relabel = 0.01

    for sample in train_ds:

        image = sample["jpg"]
        output = model(image)

        first_output = output[0]
        first_mask = first_output["mask"]
        label_mask = torch.zeros_like(torch.tensor(np.array(first_mask)))

        # Plot original masks
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)

        # Create a mask for each label with different colors
        legend_elements = []

        for o in output:
            label = o["label"]
            mask = np.array(o["mask"])
            masked = np.ma.masked_where(mask == 0, mask)
            color = label_to_color[label]
            plt.imshow(masked, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=label))

        plt.title("Original Masks")
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center")

        # Create legend elements for relabeled masks
        relabeled_legend_elements = []

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        for o in output:
            label = o["label"]
            mask = np.array(o["mask"])
            mask_array = torch.tensor(mask, dtype=torch.long)
            if label in label_as_sea:
                fraction_of_mask = torch.sum(mask_array != 0) / torch.numel(mask_array)
                if fraction_of_mask > threashold_to_relabel:
                    label = "sea"

            label_id = label2id[label]
            label_mask[mask_array != 0] = label_id

            color = label_to_color[label]

            # Add to legend if it is not already in the legend
            if label not in [e.get_label() for e in relabeled_legend_elements]:
                relabeled_legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=label))

            masked = np.ma.masked_where(mask == 0, mask)
            plt.imshow(masked, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))

        plt.title("Relabeled Masks")
        plt.legend(handles=relabeled_legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center")
        plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Make a dataset for the Seesea dataset")
    parser.add_argument("--input", type=str, help="Path to the input directory", required=True)
    parser.add_argument("--model", type=str, help="Name of the model to use to generate the dataset", required=True)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
