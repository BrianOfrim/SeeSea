"""
Visualize the output of the fine-tuned model
"""

import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoImageProcessor, AutoConfig, AutoModelForSemanticSegmentation
from datasets import load_dataset
import matplotlib.pyplot as plt


def main(args):
    # model = torch.load(os.path.join(args.model_dir, "model.pt"))

    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    config = AutoConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    id2label = config.id2label
    label2id = config.label2id

    # seed the random number generator
    np.random.seed(42)

    # create a map between the label and the color - convert to RGB by dropping alpha channel
    id_to_color = [plt.cm.rainbow(np.random.random())[:3] for _ in range(len(id2label))]

    model.eval()
    for sample in dataset:
        image = sample["image.jpg"]
        label_mask = sample["mask.png"]
        name = sample["__key__"]

        inputs = image_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # PIL size is (width, height)
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

        # Convert colors to numpy array, shape (num_classes, 3) for RGB
        palette = np.array(id_to_color)  # Colors already 0-1 from plt.cm.rainbow

        legend_elements = []
        for label_id, color in enumerate(palette):
            if pred_seg[pred_seg == label_id].shape[0] > 0:
                color_seg[pred_seg == label_id] = color * 255  # Scale to 0-255 for image
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, fc=color, label=id2label[label_id])
                )  # Keep 0-1 for matplotlib

        color_seg = color_seg[..., ::-1]  # convert to BGR

        masked_image = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
        masked_image = masked_image.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(masked_image)
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center")
        plt.show()
        continue
        legend_elements = []

        # Plot the original image with the predicted mask overlaid next to the original image with the label mask
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Predicted Masks")

        plt.imshow(image)

        for o in outputs:
            label = o["label"]
            mask = np.array(o["mask"])
            masked = np.ma.masked_where(mask == 0, mask)
            color = id_to_color[label2id[label]]
            plt.imshow(masked, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=label))

        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center")

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Label Masks")

        label_mask = np.array(label_mask)
        plt.imshow(label_mask, alpha=0.5, cmap=plt.cm.colors.ListedColormap(id_to_color))

        plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the output of the fine-tuned model")
    parser.add_argument("--model-dir", type=str, help="Path to the model directory", required=True)
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--split", type=str, help="Split to use", default="test")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
