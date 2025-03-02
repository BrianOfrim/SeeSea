"""
Convert the sea/non-sea binary segmentation model to CoreML format.

This script loads a semantic segmentation model, converts it to CoreML format,
and adds post-processing to output the percentage of the image that contains sea.
"""

import argparse
import os
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoImageProcessor, AutoConfig, AutoModelForSemanticSegmentation

# Constants used to determine which labels are considered sea-related.
consider_as_sea = ["sea", "lake"]
prevent_relabel_if_top = ["sky"]  # Categories that prevent relabeling if they're the top prediction


class SeaSegmentationWrapper(nn.Module):
    """Wrapper class that adds post-processing to the segmentation model."""

    def __init__(self, model, config, confidence_threshold=0.02):
        super().__init__()
        self.model = model
        self.id2label = config.id2label
        self.label2id = config.label2id
        self.confidence_threshold = confidence_threshold

    def forward(self, pixel_values):
        # Run the model
        outputs = self.model(pixel_values)
        logits = outputs.logits

        # Get image dimensions
        _, _, height, width = pixel_values.shape

        # Upsample logits to match the input image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # Compute the original prediction segmentation mask
        original_pred_seg = upsampled_logits.argmax(dim=1)

        # Use top n predictions to possibly relabel pixels as sea
        n_top = 10
        probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1)
        top_probs, top_indices = probabilities.topk(n_top, dim=1)

        # Create mask for pixels that should not be relabeled based on prevent_relabel_if_top
        prevent_mask = torch.zeros_like(original_pred_seg, dtype=torch.bool)
        for label in prevent_relabel_if_top:
            if label in self.label2id:
                label_match = original_pred_seg == self.label2id[label]
                prevent_mask = torch.logical_or(prevent_mask, label_match)

        # Calculate sum of probabilities for sea-related classes within the top n predictions
        sea_confidence_sum = torch.zeros_like(original_pred_seg, dtype=torch.float)
        for label in consider_as_sea:
            if label in self.label2id:
                label_id = self.label2id[label]
                for i in range(n_top):
                    label_match = top_indices[:, i] == label_id
                    sea_confidence_sum = sea_confidence_sum + (top_probs[:, i] * label_match)

        # Create mask where sum of sea-related confidences exceeds threshold and is not in prevent_mask
        mask = (sea_confidence_sum > self.confidence_threshold) & ~prevent_mask

        # Build a binary mask: pixel is sea if either:
        # a) It was originally predicted as any sea-related label OR
        # b) It was relabeled based on high sea confidence
        binary_mask = torch.zeros_like(original_pred_seg, dtype=torch.bool)
        for label in consider_as_sea:
            if label in self.label2id:
                label_match = original_pred_seg == self.label2id[label]
                binary_mask = torch.logical_or(binary_mask, label_match)

        binary_mask = torch.logical_or(binary_mask, mask)

        # Compute sea percentage in the image
        total_pixels = binary_mask.numel()
        sea_pixels = binary_mask.sum().float()
        sea_percentage = torch.clamp(sea_pixels / total_pixels, 0.0, 1.0)

        return sea_percentage.unsqueeze(0)  # Return as a single value


def convert_to_coreml(args):
    # Load model, image processor, and config
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    # Create the wrapper model with post-processing
    wrapped_model = SeaSegmentationWrapper(model, config, args.confidence_threshold)
    wrapped_model.eval()

    # Get input image size from the image processor
    input_size = image_processor.size
    if isinstance(input_size, dict):
        height = input_size.get("height")
        width = input_size.get("width")
    else:
        height, width = input_size if isinstance(input_size, tuple) else (input_size, input_size)

    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 3, height, width)

    # Trace the model
    traced_model = torch.jit.trace(wrapped_model, dummy_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="pixel_values", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="sea_percentage")],
        minimum_deployment_target=ct.target.iOS15,
    )

    # Set model metadata
    mlmodel.author = "SeeSea"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Sea segmentation model that outputs sea percentage"
    mlmodel.version = "1.0"

    # Make output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Save the model
    mlmodel.save(os.path.join(args.output_dir, "sea_segmentation.mlpackage"))
    print(f"Model saved to {args.output_dir}")

    # Save the image processor
    image_processor.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert sea segmentation model to CoreML")
    parser.add_argument("--model", type=str, help="Path or ID of the segmentation model", required=True)
    parser.add_argument("--output-dir", type=str, help="Output directory for the CoreML model", required=True)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.02,
        help="Confidence threshold for relabeling pixels as sea based on sea-related probabilities",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_coreml(args)
