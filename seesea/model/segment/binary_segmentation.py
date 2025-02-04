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

consider_as_sea = ["sea", "water", "river", "lake"]
prevent_relabel_if_top = ["sky"]  # Categories that prevent relabeling if they're the top prediction
MIN_SEA_PERCENTAGE = 0.10  # 10% minimum sea requirement


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
    palette = np.array(id_to_color)

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

        # Compute original argmax segmentation mask and keep a copy for display
        original_pred_seg = upsampled_logits.argmax(dim=1)[0].clone()

        # Specify the number of top predictions to display
        n_top = 10
        # Compute top n predictions for each pixel. Shape: (n_top, height, width)
        top_preds = upsampled_logits.topk(n_top, dim=1)[1][0]

        # Apply softmax to logits to get probabilities
        probabilities = torch.nn.functional.softmax(upsampled_logits[0], dim=0)

        # Create an updated segmentation from the top n predictions
        pred_seg = original_pred_seg.clone()

        # --- Updated logic with confidence sum threshold ---
        CONFIDENCE_THRESHOLD = 0.001  # 0.1% threshold

        # Get top n probabilities and their indices
        top_probs, top_indices = probabilities.topk(n_top, dim=0)

        # Create mask for pixels where top prediction is in prevent_relabel_if_top list
        prevent_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in prevent_relabel_if_top:
            if label in label2id:
                prevent_mask |= original_pred_seg == label2id[label]

        # Calculate sum of probabilities for sea-related classes within top n
        sea_confidence_sum = torch.zeros_like(pred_seg, dtype=torch.float)
        for label in consider_as_sea:
            if label in label2id:
                label_id = label2id[label]
                # Create a mask for where this label appears in top n
                label_in_top_n = top_indices == label_id
                # Add the corresponding probabilities where the label appears
                sea_confidence_sum += (top_probs * label_in_top_n).sum(dim=0)

        # Create mask where sum of sea-related confidences exceeds threshold
        # AND the pixel is not in prevent_relabel_if_top categories
        mask = (sea_confidence_sum > CONFIDENCE_THRESHOLD) & ~prevent_mask

        # Relabel any pixel where the criteria are met
        pred_seg[mask] = label2id["sea"]

        print(f"Number of pixels that will be relabeled: {mask.sum()}")

        # Determine image shape
        height, width = original_pred_seg.shape

        # --- Create overlay for the original argmax segmentation ---
        color_seg_original = np.zeros((height, width, 3), dtype=np.uint8)
        legend_elements_original = []
        for label_id, color in enumerate(palette):
            if (original_pred_seg == label_id).sum() > 0:
                color_seg_original[original_pred_seg == label_id] = (np.array(color[:3]) * 255).astype(np.uint8)
                legend_elements_original.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=id2label[label_id]))

        # --- Create overlay for the top n relabeled segmentation ---
        color_seg_updated = np.zeros((height, width, 3), dtype=np.uint8)
        legend_elements_updated = []
        for label_id, color in enumerate(palette):
            if (pred_seg == label_id).sum() > 0:
                color_seg_updated[pred_seg == label_id] = (np.array(color[:3]) * 255).astype(np.uint8)
                legend_elements_updated.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=id2label[label_id]))

        image_np = np.array(image)
        masked_image_original = (image_np * 0.5 + color_seg_original * 0.5).astype(np.uint8)
        masked_image_updated = (image_np * 0.5 + color_seg_updated * 0.5).astype(np.uint8)

        # Plot the overlays side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # Changed to 2 subplots
        axes[0].imshow(masked_image_original)
        axes[0].set_title("Original Argmax Segmentation Overlay")
        axes[0].axis("off")
        axes[0].legend(
            handles=legend_elements_original,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_original),
        )

        # Create binary mask
        binary_mask = torch.zeros_like(pred_seg, dtype=torch.bool)

        # Add areas that were originally sea-related
        for label in consider_as_sea:
            if label in label2id:
                binary_mask |= original_pred_seg == label2id[label]

        # Add areas that were relabeled to sea
        binary_mask |= mask  # mask from earlier contains the relabeled pixels

        # Create binary overlay
        color_seg_binary = np.zeros((height, width, 3), dtype=np.uint8)
        # Sea areas in blue
        color_seg_binary[binary_mask] = [0, 0, 255]  # Blue for sea
        # Non-sea areas in green
        color_seg_binary[~binary_mask] = [0, 255, 0]  # Green for non-sea

        # Create the binary overlay image
        masked_image_binary = (image_np * 0.7 + color_seg_binary * 0.3).astype(np.uint8)

        # Calculate sea percentage
        total_pixels = binary_mask.numel()
        sea_pixels = binary_mask.sum().item()
        sea_percentage = sea_pixels / total_pixels

        # Plot binary overlay with border based on sea percentage
        axes[1].imshow(masked_image_binary)
        axes[1].set_title(f"Binary Sea/Non-Sea Overlay\n{sea_percentage:.1%} sea")
        axes[1].axis("off")

        # Add border based on sea percentage threshold
        if sea_percentage >= MIN_SEA_PERCENTAGE:
            # Green rectangle for valid images
            rect = plt.Rectangle(
                (-20, -20), width + 40, height + 40, fill=False, color="green", linewidth=5, transform=axes[1].transData
            )
            axes[1].add_patch(rect)
        else:
            # Red circle for invalid images
            circle = plt.Circle(
                (width / 2, height / 2),
                radius=min(width, height) / 1.8,
                fill=False,
                color="red",
                linewidth=5,
                transform=axes[1].transData,
            )
            axes[1].add_patch(circle)

        # Add legend for binary mask
        legend_elements_binary = [
            plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1), label="Sea"),
            plt.Rectangle((0, 0), 1, 1, fc=(0, 1, 0), label="Non-Sea"),
        ]
        axes[1].legend(
            handles=legend_elements_binary,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_binary),
        )

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
