"""Convert a model from safetensors format to pytorch format"""

import os
import logging
import torch
from transformers import AutoModelForImageClassification

LOGGER = logging.getLogger(__name__)


def main(args):
    """Load safetensors model and save as pytorch"""

    # Load the model from safetensors format
    model = AutoModelForImageClassification.from_pretrained(args.input)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Save model in pytorch format
    output_path = os.path.join(args.output, "model.pt")
    torch.save(model, output_path)
    LOGGER.info("Model saved to %s", output_path)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert model from safetensors to pytorch format")
    parser.add_argument("--input", help="Directory containing the safetensors model", required=True)
    parser.add_argument("--output", help="Directory to save the pytorch model", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
