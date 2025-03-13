#!/usr/bin/env python
"""
CoreML inference script for base sea segmentation model.

This script loads a CoreML base segmentation model, runs inference on images,
and applies post-processing to determine the percentage of sea in each image.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoConfig

import coremltools as ct
import matplotlib.pyplot as plt


def apply_sea_post_processing(logits, config, sea_config):
    """
    Apply post-processing to the model logits to determine sea percentage.

    Args:
        logits: Model output logits with shape [1, num_classes, height, width]
        config: Model configuration with label mappings
        sea_config: Sea-specific configuration

    Returns:
        sea_percentage: Percentage of the image that is sea
        binary_mask: Binary mask indicating sea pixels
    """
    # Extract configuration
    id2label = config.id2label
    label2id = config.label2id
    consider_as_sea = sea_config["consider_as_sea"]
    prevent_relabel_if_top = sea_config["prevent_relabel_if_top"]
    confidence_threshold = sea_config["confidence_threshold"]

    # Get dimensions
    _, num_classes, height, width = logits.shape

    # Apply softmax to get probabilities
    # Subtract max for numerical stability
    exp_logits = np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    # Get the argmax for each pixel (top prediction)
    top_predictions = np.argmax(probabilities, axis=0)

    # Create mask for pixels that should not be relabeled based on prevent_relabel_if_top
    prevent_mask = np.zeros((height, width), dtype=bool)
    for label in prevent_relabel_if_top:
        if label in label2id:
            prevent_mask |= top_predictions == label2id[label]

    # Calculate sum of probabilities for sea-related classes
    sea_confidence_sum = np.zeros((height, width), dtype=np.float32)
    for label in consider_as_sea:
        if label in label2id:
            sea_confidence_sum += probabilities[label2id[label]]

    # Create mask where sea confidence exceeds threshold and is not in prevent_mask
    binary_mask = (sea_confidence_sum > confidence_threshold) & ~prevent_mask

    return binary_mask


def main(args):
    """
    Run inference on images using the CoreML model.
    """
    # Load the CoreML model
    print(f"Loading CoreML model from {args.model}")
    model = ct.models.MLModel(args.model)

    # Load configuration
    model_dir = os.path.dirname(args.model)
    config = AutoConfig.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    # Load sea-specific configuration
    sea_config_path = os.path.join(model_dir, "sea_config.json")
    with open(sea_config_path, "r") as f:
        sea_config = json.load(f)

    print(f"Loaded sea config: {sea_config}")

    # Print model input and output details
    print("Model input description:")
    for input_desc in model.get_spec().description.input:
        print(f"  Name: {input_desc.name}")
        if input_desc.type.WhichOneof("Type") == "multiArrayType":
            shape = [dim for dim in input_desc.type.multiArrayType.shape]
            print(f"  Shape: {shape}")

    print("\nModel output description:")
    for output_desc in model.get_spec().description.output:
        print(f"  Name: {output_desc.name}")
        if output_desc.type.WhichOneof("Type") == "multiArrayType":
            shape = [dim for dim in output_desc.type.multiArrayType.shape]
            print(f"  Shape: {shape}")

    # Load dataset
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    # Process each image
    for sample in dataset:
        image = sample["jpg"]
        image_name = sample["__key__"]

        print(f"\nProcessing image: {image_name}")
        print(f"Original image size: {image.size}")

        # Store original image size for later resizing
        original_width, original_height = image.size

        # Apply preprocessing
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs["pixel_values"]

        # Get the model's input size
        print(f"Model input shape: {pixel_values.shape}")

        # Run inference
        try:
            prediction = model.predict({"pixel_values": pixel_values})
            logits = prediction["logits"]

            print(f"Logits shape: {logits.shape}")

            # Apply post-processing
            binary_mask = apply_sea_post_processing(logits, config, sea_config)

            # Compute sea percentage
            total_pixels = binary_mask.size
            sea_pixels = np.sum(binary_mask)
            sea_percentage = min(max(sea_pixels / total_pixels, 0.0), 1.0)

            # Make decision based on sea percentage
            decision = sea_percentage > args.min_sea_fraction

            print(f"Sea percentage: {sea_percentage:.1%}")
            print(f"Contains sea: {decision}")

            # Display the image with result if requested
            if args.show_images:
                # Convert the binary mask to numpy for visualization
                mask_np = binary_mask

                # Get original image dimensions
                original_height, original_width = image.size[::-1]  # PIL image size is (width, height)

                # Resize the mask to match the original image dimensions if they're different
                if mask_np.shape != (original_height, original_width):
                    # Convert to PIL Image for resizing (nearest neighbor to preserve binary values)
                    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize((original_width, original_height), Image.Resampling.NEAREST)
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

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with CoreML base sea segmentation model")
    parser.add_argument("--model", type=str, help="Path to the CoreML model (.mlpackage)", required=True)
    parser.add_argument(
        "--min-sea-fraction",
        type=float,
        default=0.2,
        help="Minimum fraction of sea to classify as containing sea (0.0-1.0)",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Show images with inference results",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
