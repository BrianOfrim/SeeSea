#!/usr/bin/env python
"""
CoreML inference script for sea segmentation model.

This script loads a CoreML model and runs inference on images to determine
the percentage of sea in each image.
"""

import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoConfig

import coremltools as ct
import matplotlib.pyplot as plt


def main(args):
    """
    Run inference on images using the CoreML model.
    """
    # Load the CoreML model
    print(f"Loading CoreML model from {args.model}")
    model = ct.models.MLModel(args.model)

    model_dir = os.path.dirname(args.model)

    # Get model metadata
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    # Print model input and output details for debugging
    print("Model input description:")
    for input_desc in model.get_spec().description.input:
        print(f"  Name: {input_desc.name}")
        print(f"  Type: {input_desc.type.WhichOneof('Type')}")
        if input_desc.type.WhichOneof("Type") == "multiArrayType":
            shape = [dim for dim in input_desc.type.multiArrayType.shape]
            print(f"  Shape: {shape}")

    print("\nModel output description:")
    for output_desc in model.get_spec().description.output:
        print(f"  Name: {output_desc.name}")
        print(f"  Type: {output_desc.type.WhichOneof('Type')}")

    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    # Process each image
    for sample in dataset:
        image = sample["jpg"]
        image_name = sample["__key__"]

        print(f"\nProcessing image: {image_name}")

        # Print original image info
        print(f"Original image size: {image.size}")

        # Apply preprocessing
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs["pixel_values"]

        # Print transformed image info
        print(f"Transformed image shape: {pixel_values.shape}")
        print(f"Pixel value range: min={pixel_values.min():.4f}, max={pixel_values.max():.4f}")

        # Ensure the pixel values are in the expected format
        # CoreML typically expects float32 in range [0,1] or [-1,1]
        if pixel_values.dtype != np.float32:
            pixel_values = pixel_values.astype(np.float32)

        # Run inference with explicit input dictionary
        try:
            prediction = model.predict({"pixel_values": pixel_values})

            # Print raw prediction for debugging
            print("Raw prediction output:")
            for k, v in prediction.items():
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v}")

            # Extract sea percentage
            sea_percentage = float(prediction["sea_percentage"][0])

            # Make decision based on sea percentage
            decision = sea_percentage > args.min_sea_fraction

            print(f"Sea percentage: {sea_percentage:.1%}")
            print(f"Contains sea: {decision}")

            # Display the image with result if requested
            if args.show_images:
                # Create figure
                plt.figure(figsize=(8, 6))
                plt.imshow(image)

                # Add text with result
                result_text = f"Sea: {sea_percentage:.1%} ({'YES' if decision else 'NO'})"
                plt.title(result_text)

                # Add colored border based on decision
                border_color = "green" if decision else "red"
                plt.gca().spines["top"].set_color(border_color)
                plt.gca().spines["bottom"].set_color(border_color)
                plt.gca().spines["left"].set_color(border_color)
                plt.gca().spines["right"].set_color(border_color)
                plt.gca().spines["top"].set_linewidth(5)
                plt.gca().spines["bottom"].set_linewidth(5)
                plt.gca().spines["left"].set_linewidth(5)
                plt.gca().spines["right"].set_linewidth(5)

                plt.axis("off")
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error during inference: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with CoreML sea segmentation model")
    parser.add_argument("--model", type=str, help="Path to the CoreML model (.mlpackage or .mlmodel)", required=True)
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
