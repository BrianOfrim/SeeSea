"""Inference script for the multihead model in CoreML format"""

import argparse
import os
from datasets import load_dataset
from transformers import AutoImageProcessor
from seesea.model.multihead.modeling_multihead import MultiHeadConfig
import matplotlib.pyplot as plt
import coremltools as ct
import numpy as np
from PIL import Image


def main(args):
    """Main function for the inference script"""

    config = MultiHeadConfig.from_pretrained(args.model_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)

    model_path = os.path.join(args.model_dir, "model.mlpackage")

    # Load the Core ML model
    coreml_model = ct.models.MLModel(model_path)

    # Load image
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    for sample in dataset:
        image = sample["jpg"]
        labels = [sample["json"].get(name) for name in config.output_head_names]
        image_name = sample["__key__"]

        print(f"\nProcessing image: {image_name}")
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")  # Check if RGB, RGBA, etc.

        # Get raw pixel values of first few pixels before any processing
        raw_pixels = list(image.getdata())[:5]
        print(f"First 5 raw pixels (RGB): {raw_pixels}")

        # Print first pixel in detail
        r, g, b = raw_pixels[0]
        print(f"First pixel raw values (RGB):")
        print(f"R: {r}")
        print(f"G: {g}")
        print(f"B: {b}")

        # Get the actual transformation steps
        print("\nApplying transformations...")
        transformed_image = image_processor(image)

        # Create a copy for logging
        debug_image = np.array(transformed_image["pixel_values"])
        print(f"\nTransformed shape: {debug_image.shape}")
        print("\nFirst pixel normalized values (RGB):")
        print(f"R: {debug_image[0,0,0,0]:.6f}")
        print(f"G: {debug_image[0,1,0,0]:.6f}")
        print(f"B: {debug_image[0,2,0,0]:.6f}")

        # Print first few values of each channel
        print("\nFirst few values of each channel:")
        for c in range(3):
            channel = ["R", "G", "B"][c]
            values = debug_image[0, c, 0, :5]
            print(f"{channel}: {[f'{v:.6f}' for v in values]}")

        # Calculate the normalized values manually to verify
        print("\nManually calculated normalized values for first pixel:")
        r_norm = ((r / 255.0) - 0.485) / 0.229
        g_norm = ((g / 255.0) - 0.456) / 0.224
        b_norm = ((b / 255.0) - 0.406) / 0.225
        print(f"R: {r_norm:.6f}")
        print(f"G: {g_norm:.6f}")
        print(f"B: {b_norm:.6f}")

        predictions = coreml_model.predict({"pixel_values": transformed_image["pixel_values"]})

        outputs = predictions["predictions"].squeeze()
        print(f"\nPredictions:")
        for name, pred, target in zip(config.output_head_names, outputs, labels):
            print(f"  {name}: Predicted: {pred:.6f}, Expected: {target:.6f}, Diff: {pred - target:.6f}")

        if args.show_images:
            # show the image with the predicted and expected values
            plt.imshow(image)
            plt.title(f"Image: {image_name}")
            plt.axis("off")

            # Create subtitle with all predictions
            subtitle = ""
            for name, pred, target in zip(config.output_head_names, outputs, labels):
                subtitle += f"{name}: target={target:.3f}, pred={pred:.3f}, diff={pred - target:.3f}\n"

            plt.suptitle(subtitle)

            # Set the window title
            plt.gcf().canvas.manager.set_window_title(image_name)

            plt.show()


def get_args_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--show-images", action="store_true")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
