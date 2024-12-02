"""Load the dataset and run a pretrained image segmentation model on it."""

import os
import argparse

from datasets import load_dataset
from transformers import pipeline
import matplotlib.pyplot as plt
import torch
import numpy as np

# Use a pipeline as a high-level helper
from transformers import pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pipe = pipeline("image-segmentation", model="nvidia/segformer-b4-finetuned-ade-512-512", device=device)
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True)

    for sample in dataset:
        image = sample["jpg"]
        outputs = pipe(image)
        print(outputs)

        # Display all image alongside all the masks if there are more than one, label the masks with the class name
        if len(outputs) > 1:
            fig, axs = plt.subplots(1, len(outputs) + 1, figsize=(20, 4))
            axs[0].imshow(image)
            axs[0].set_title("Original")
            axs[0].axis("off")
            for i, output in enumerate(outputs):
                axs[i + 1].imshow(output["mask"])
                axs[i + 1].set_title(f"{output['label']}")
                axs[i + 1].axis("off")

            plt.show()

        # if the output has more than one class then display it
        # if len(outputs) > 1:
        #     # display all
        #     plt.imshow(image)
        #     plt.imshow(outputs[1]["mask"], alpha=0.5)
        #     plt.show()

        # display the image and the segmentation mask
        # plt.imshow(image)
        # plt.imshow(outputs[0]["mask"], alpha=0.5)
        # plt.show()
