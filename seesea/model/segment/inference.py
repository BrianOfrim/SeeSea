"""
Inference for sea/non-sea binary segmentation.

This script loads a semantic segmentation model and an image dataset,
computes a segmentation mask, and then outputs a binary decision (True/False)
depending on whether the percentage of pixels labeled as sea exceeds min_sea_percentage.
"""

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoConfig, AutoModelForSemanticSegmentation
from datasets import load_dataset
from PIL import Image

# Constants used to determine which labels are considered sea-related.
consider_as_sea = ["sea", "lake"]
prevent_relabel_if_top = ["sky"]  # Categories that prevent relabeling if they're the top prediction


def load_image_dataset(path, split, image_key, streaming=True):
    """
    Load a dataset from a directory of images or from a Hugging Face dataset name.
    """
    if os.path.isdir(path):
        image_files = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if image_files:

            def image_dataset():
                for img_file in image_files:
                    img_path = os.path.join(path, img_file)
                    yield {"image": Image.open(img_path), "__key__": img_file}

            return image_dataset()
    # Fall back to loading by name/path via Hugging Face datasets.
    return load_dataset(path, split=split, streaming=streaming).shuffle()


def get_image_from_sample(sample, image_key):
    """
    Extract an image from a dataset sample based on a given key.
    """
    if isinstance(sample, dict):
        if image_key in sample:
            return sample[image_key]
        for key in ["image", "img", "jpg", "jpeg", "png"]:
            if key in sample:
                return sample[key]
    if hasattr(sample, "convert") and callable(sample.convert):
        return sample
    raise ValueError(f"Could not find image in sample using key '{image_key}'")


def process_outputs(outputs):
    processed = {}

    for name, output in outputs.items():
        if name.endswith("_sin"):
            # Get the base name without _sin
            base_name = name[:-4]
            if base_name.endswith("deg"):
                # Find corresponding cos output
                cos_name = f"{base_name}_cos"
                if cos_name in outputs:
                    # Convert sin/cos back to degrees
                    sin_val = outputs[name]
                    cos_val = outputs[cos_name]
                    angle_rad = torch.atan2(sin_val, cos_val)
                    processed[base_name] = torch.rad2deg(angle_rad)
        elif not name.endswith("_cos"):  # Skip cos components as they're handled with sin
            processed[name] = output

    return processed


def main(args):
    # Load model, image processor, and config (which contains label mappings)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    id2label = config.id2label
    label2id = config.label2id

    # Choose appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load dataset and process each sample
    dataset = load_image_dataset(args.dataset, args.split, args.image_key, streaming=args.streaming)
    for sample in dataset:
        try:
            image = get_image_from_sample(sample, args.image_key)
            name = sample.get("__key__", "unknown")
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

        # Preprocess image and run model inference
        inputs = image_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu()
        # Upsample logits to match the original image size (PIL images have size (width, height))
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Compute the original prediction segmentation mask (argmax across class channels)
        original_pred_seg = upsampled_logits.argmax(dim=1)[0].clone()

        # Use top n predictions to possibly relabel pixels as sea
        n_top = 10
        probabilities = torch.nn.functional.softmax(upsampled_logits[0], dim=0)
        top_probs, top_indices = probabilities.topk(n_top, dim=0)

        # Copy the segmentation mask for potential updates
        pred_seg = original_pred_seg.clone()

        # Use the confidence threshold loaded from the command line
        confidence_threshold = args.confidence_threshold

        # Create mask for pixels that should not be relabeled based on prevent_relabel_if_top
        prevent_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in prevent_relabel_if_top:
            if label in label2id:
                prevent_mask |= original_pred_seg == label2id[label]

        # Calculate sum of probabilities for sea-related classes within the top n predictions
        sea_confidence_sum = torch.zeros_like(pred_seg, dtype=torch.float)
        for label in consider_as_sea:
            if label in label2id:
                label_id = label2id[label]
                label_in_top_n = top_indices == label_id
                sea_confidence_sum += (top_probs * label_in_top_n).sum(dim=0)

        # Create mask where sum of sea-related confidences exceeds threshold and is not in prevent_mask
        mask = (sea_confidence_sum > confidence_threshold) & ~prevent_mask
        pred_seg[mask] = label2id["sea"]

        # Build a binary mask: pixel is sea if either:
        # a) It was originally predicted as any sea-related label OR
        # b) It was relabeled based on high sea confidence
        binary_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in consider_as_sea:
            if label in label2id:
                binary_mask |= original_pred_seg == label2id[label]
        binary_mask |= mask

        # Compute sea percentage in the image
        total_pixels = binary_mask.numel()
        sea_pixels = binary_mask.sum().item()
        sea_percentage = min(max(sea_pixels / total_pixels, 0.0), 1.0)  # Clamp between 0 and 1

        # Output a binary decision: True if sea_percentage exceeds threshold, else False.
        decision = sea_percentage > args.min_sea_fraction
        print(f"Sample: {name} | Sea Percentage: {sea_percentage:.1%} | Decision: {decision}")

        # Display the image with mask overlay if requested
        if args.show_overlay:
            # Create overlay with red tint for sea pixels
            overlay = np.array(image)
            mask_np = binary_mask.numpy()
            overlay[mask_np] = overlay[mask_np] * 0.5 + np.array([255, 0, 0]) * 0.5  # Red tint for sea

            # Setup interactive figure with hover annotation for top n probabilities
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(overlay)
            ax.set_title(f"Sea Mask Overlay (Sea: {sea_percentage:.1%})")
            ax.axis("off")

            # Annotation box for hover details (initially hidden)
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)

            # Convert top probabilities and indices to numpy arrays for interactive lookup
            top_probs_np = top_probs.numpy()  # shape: (n_top, height, width)
            top_indices_np = top_indices.numpy()  # shape: (n_top, height, width)

            # Callback to update annotation based on mouse hover
            def update_annot(event):
                if event.inaxes == ax:
                    x, y = int(round(event.xdata)), int(round(event.ydata))
                    h, w = overlay.shape[:2]
                    if x < 0 or y < 0 or x >= w or y >= h:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                        return

                    # Build annotation text from top n predictions for pixel (y, x)
                    text_lines = []
                    for i in range(n_top):
                        class_id = int(top_indices_np[i, y, x])
                        prob = float(top_probs_np[i, y, x])
                        label_name = id2label.get(class_id, f"id_{class_id}")
                        text_lines.append(f"{i+1}: {label_name} ({prob:.2f})")
                    annot.xy = (x, y)
                    annot.set_text("\n".join(text_lines))
                    annot.get_bbox_patch().set_alpha(0.8)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

            # Connect the event handler for mouse motion
            fig.canvas.mpl_connect("motion_notify_event", update_annot)
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal binary inference script for sea segmentation")
    parser.add_argument("--model", type=str, help="Path or ID of the segmentation model", required=True)
    parser.add_argument("--dataset", type=str, help="Path or name of the dataset", required=True)
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument(
        "--image-key",
        type=str,
        default="image",
        help="Key in the dataset example for retrieving the image",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for dataset loading (if supported)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.02,
        help="Confidence threshold for relabeling pixels as sea based on sea-related probabilities",
    )
    parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Show the image with the sea mask overlay",
    )
    parser.add_argument(
        "--min-sea-fraction",
        type=float,
        default=0.2,
        help="If sea pixels exceed this fraction (0.0-1.0) of the image, output True",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
