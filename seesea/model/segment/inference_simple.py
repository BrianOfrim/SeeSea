"""
Inference for sea/non-sea binary segmentation.

This script loads a semantic segmentation model and an image dataset,
computes a segmentation mask, and then outputs a binary decision (True/False)
depending on whether the percentage of pixels labeled as sea exceeds min_sea_percentage.
"""

import os
import argparse
import time

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


def main(args):
    # Load model, image processor, and config (which contains label mappings)
    processor_kwargs = {}

    # Apply custom input size if specified
    if args.input_size is not None:
        processor_kwargs["size"] = {"height": args.input_size, "width": args.input_size}
        print(f"Using custom input size: {args.input_size}x{args.input_size}")

    # Load the image processor with optional custom size
    image_processor = AutoImageProcessor.from_pretrained(args.model, **processor_kwargs)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
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

    # Track total and per-image timing
    total_time = 0
    image_count = 0

    for sample in dataset:
        try:
            start_time = time.time()  # Start timing

            image = get_image_from_sample(sample, args.image_key)
            name = sample.get("__key__", "unknown")
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

        # Preprocess image and run model inference
        inputs = image_processor(image, return_tensors="pt").to(device)

        # Time just the model inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - inference_start

        logits = outputs.logits.cpu()

        # Time the binary mask computation
        mask_start = time.time()

        # calculate the softmax of the logits
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        # calculate a binary mask where the top prediction is in prevent_relabel_if_top
        prevent_mask = torch.zeros((probabilities.shape[1], probabilities.shape[2]), dtype=torch.bool)

        # get the arg max for each pixel
        top_predictions = probabilities.argmax(dim=0)

        # set the mask to True for the pixels where the top prediction is in prevent_relabel_if_top
        for label in prevent_relabel_if_top:
            if label in label2id:
                prevent_mask |= top_predictions == label2id[label]

        # calculate the sum of the probabilities of the classes in consider_as_sea
        sea_confidence_sum = torch.zeros((probabilities.shape[1], probabilities.shape[2]), dtype=torch.float)
        for label in consider_as_sea:
            if label in label2id:
                sea_confidence_sum += probabilities[label2id[label]]

        # create a binary mask where the sum of the probabilities of the classes in consider_as_sea is greater than confidence_threshold
        mask = (sea_confidence_sum > args.confidence_threshold) & ~prevent_mask

        # Calculate time for mask computation
        mask_time = time.time() - mask_start

        # calculate the percentage of sea pixels
        sea_percentage = mask.sum() / mask.numel()

        # Calculate total processing time for this image
        process_time = time.time() - start_time
        total_time += process_time
        image_count += 1

        # Output the percentage of sea pixels and timing information
        print(
            f"{name}: {sea_percentage:.1%} | Inference: {inference_time:.3f}s | Mask: {mask_time:.3f}s | Total:"
            f" {process_time:.3f}s"
        )

        # Display the image with mask overlay if requested
        if args.show_overlay:
            # Convert the binary mask to numpy for visualization
            mask_np = mask.numpy()

            # Get original image dimensions
            original_height, original_width = image.size[::-1]  # PIL image size is (width, height)

            # Resize the mask to match the original image dimensions if they're different
            if mask_np.shape != (original_height, original_width):
                # Convert to PIL Image for resizing (nearest neighbor to preserve binary values)
                mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((original_width, original_height), Image.NEAREST)
                mask_np = np.array(mask_pil) > 0  # Convert back to boolean mask

            # Create overlay with red tint for sea pixels
            overlay = np.array(image)
            overlay[mask_np] = overlay[mask_np] * 0.5 + np.array([255, 0, 0]) * 0.5  # Red tint for sea

            # Setup figure for display
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(overlay)
            ax.set_title(f"Sea Mask Overlay (Sea: {sea_percentage:.1%})")
            ax.axis("off")

            plt.show()

    # Print summary timing statistics if processed any images
    if image_count > 0:
        avg_time = total_time / image_count
        print(f"\nProcessed {image_count} images in {total_time:.2f}s (avg: {avg_time:.3f}s per image)")


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
        "--input-size",
        type=int,
        default=None,
        help="Size to resize input images (both height and width). If not specified, uses model default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
