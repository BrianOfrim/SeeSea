"""Load the dataset and run a pretrained image segmentation model on it."""

import os
import argparse

from datasets import load_dataset
from transformers import pipeline, AutoConfig
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

    model_name = "nvidia/segformer-b1-finetuned-ade-512-512"

    water_body_labels = ["sea", "water", "river", "lake"]

    config = AutoConfig.from_pretrained(model_name)

    # Print the labels
    # if hasattr(config, "id2label"):
    #     labels = config.id2label
    #     print("Model Labels:")
    #     for label_id, label_name in labels.items():
    #         print(f"{label_id}: {label_name}")
    # else:
    #     print("The model configuration does not contain label information.")

    water_body_ids = [config.label2id[label] for label in water_body_labels]

    pipe = pipeline("image-segmentation", model=model_name, device=device)
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True)

    for sample in dataset:
        image = sample["jpg"]
        outputs = pipe(image)

        # Convert PIL Image to numpy array for shape access
        image_array = np.array(image)
        print(image_array.shape)

        if len(outputs) == 0:
            continue

        # Continue if the output does not contain at least two types of water body labels
        # water_outputs = [output for output in outputs if output["label"] in water_body_labels]
        # if len(water_outputs) < 2:
        #     continue

        binary_mask = np.zeros(np.array(outputs[0]["mask"]).shape)
        for output in outputs:
            if output["label"] in water_body_labels:
                # Convert PIL Image mask to numpy array before comparison
                mask_array = np.array(output["mask"])
                binary_mask[mask_array != 0] = 1

        percent_water = np.sum(binary_mask) / (binary_mask.shape[0] * binary_mask.shape[1])

        if percent_water > 0.1:
            print(f"Water body detected with {percent_water:.2f} coverage")
            continue

        # Display the image and the binary mask side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(binary_mask)
        axs[1].set_title(f"Water Body Mask, {percent_water:.2f}")
        axs[1].axis("off")
        plt.show()

        # Display all image alongside all the masks if there are more than one, label the masks with the class name
        if len(outputs) > 1:
            fig, axs = plt.subplots(1, len(outputs) + 1, figsize=(20, 4))
            axs[0].imshow(image)
            axs[0].set_title("Original")
            axs[0].axis("off")
            for i, output in enumerate(outputs):
                mask_array = np.array(output["mask"])
                axs[i + 1].imshow(output["mask"])
                # Convert mask_array to a binary mask where all non-zero values are set to 1
                binary_mask = (mask_array != 0).astype(int)

                # Calculate the percentage of non-zero elements
                percentage = np.sum(binary_mask) / (binary_mask.shape[0] * binary_mask.shape[1])
                axs[i + 1].set_title(f"{output['label']} ({percentage:.2f})")
                axs[i + 1].axis("off")

            plt.show()

        # make a binary mask for the water body labels
        # mask_array = np.array(outputs[0]["mask"])
        # binary_mask = np.zeros(mask_array.shape)
        # for water_body_id in water_body_ids:
        #     binary_mask[mask_array == water_body_id] = 1

        # Display the image and the water body mask side by side
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(image)
        # axs[0].set_title("Original")
        # axs[0].axis("off")
        # axs[1].imshow(binary_mask, alpha=0.5)
        # axs[1].set_title("Water Body Mask")
        # axs[1].axis("off")
        # plt.show()

        # Display all image alongside all the masks if there are more than one, label the masks with the class name
        # if len(outputs) > 1:
        #     fig, axs = plt.subplots(1, len(outputs) + 1, figsize=(20, 4))
        #     axs[0].imshow(image)
        #     axs[0].set_title("Original")
        #     axs[0].axis("off")
        #     for i, output in enumerate(outputs):
        #         mask_array = np.array(output["mask"])
        #         axs[i + 1].imshow(output["mask"])
        #         # Convert mask_array to a binary mask where all non-zero values are set to 1
        #         binary_mask = (mask_array != 0).astype(int)

        #         # Calculate the percentage of non-zero elements
        #         percentage = np.sum(binary_mask) / (binary_mask.shape[0] * binary_mask.shape[1])
        #         axs[i + 1].set_title(f"{output['label']} ({percentage:.2f})")
        #         axs[i + 1].axis("off")

        #     plt.show()

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
