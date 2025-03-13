#!/usr/bin/env python
"""
Convert the base sea segmentation model to CoreML format.

This script loads a semantic segmentation model, converts it to CoreML format,
and saves both the model and configuration for later post-processing.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoImageProcessor, AutoConfig, AutoModelForSemanticSegmentation


class BaseSegmentationWrapper(nn.Module):
    """Simple wrapper that extracts logits from the model output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # Run the model and extract just the logits
        outputs = self.model(pixel_values)
        return outputs.logits


def main(args):
    # Apply custom input size if specified
    processor_kwargs = {}
    if args.image_size is not None:
        processor_kwargs["size"] = {"height": args.image_size, "width": args.image_size}
        print(f"Using custom input size: {args.image_size}x{args.image_size}")

    # Load the image processor with optional custom size
    image_processor = AutoImageProcessor.from_pretrained(args.model, **processor_kwargs)

    # Load model, image processor, and config
    print(f"Loading model from {args.model}")
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    # Create a simple wrapper that returns just the logits
    wrapped_model = BaseSegmentationWrapper(model)
    wrapped_model.eval()

    # Get input image size from the image processor
    input_size = image_processor.size
    if isinstance(input_size, dict):
        height = input_size.get("height")
        width = input_size.get("width")
    else:
        height, width = input_size if isinstance(input_size, tuple) else (input_size, input_size)

    print(f"Input size: {height}x{width}")

    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 3, height, width)

    # Trace the model with strict=False to handle the dictionary output
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, dummy_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="pixel_values", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS15,
    )

    # Set model metadata
    mlmodel.author = "SeeSea"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Base segmentation model for sea detection"
    mlmodel.version = "1.0"

    # Make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.output_dir, "sea_segmentation_base.mlpackage")
    mlmodel.save(model_path)
    print(f"Base model saved to {model_path}")

    # Save the config (contains label mappings) and image processor
    config.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

    # Save the sea-related configuration
    sea_config = {
        "consider_as_sea": ["sea", "lake"],
        "prevent_relabel_if_top": ["sky"],
        "confidence_threshold": args.confidence_threshold,
    }

    config_path = os.path.join(args.output_dir, "sea_config.json")
    with open(config_path, "w") as f:
        json.dump(sea_config, f, indent=2)

    print(f"Configuration saved to {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert base sea segmentation model to CoreML")
    parser.add_argument("--model", type=str, help="Path or ID of the segmentation model", required=True)
    parser.add_argument("--output-dir", type=str, help="Output directory for the CoreML model", required=True)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.02,
        help="Confidence threshold for relabeling pixels as sea (saved in config)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Size to resize input images (both height and width). If not specified, uses model default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
