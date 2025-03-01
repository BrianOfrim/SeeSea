"""A script to convert a model to CoreML format."""

import argparse
import os
import torch
import coremltools as ct
import torch.nn as nn
import numpy as np

from seesea.model.multihead.modeling_multihead import MultiHeadModel, MultiHeadConfig
from transformers import AutoImageProcessor


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs["logits"]


def main(args):

    config = MultiHeadConfig.from_pretrained(args.model_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = MultiHeadModel.from_pretrained(args.model_dir, config=config)
    model.eval()

    # get the model input shape
    input_shape = image_processor.size

    # Convert image_mean list to numpy array to get dtype
    data_type = np.array(image_processor.image_mean).dtype

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, input_shape["height"], input_shape["width"], dtype=torch.float32)

    model_wrapper = ModelWrapper(model)
    model_wrapper.eval()

    traced_model = torch.jit.trace(model_wrapper, dummy_input)

    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="pixel_values", shape=dummy_input.shape, dtype=data_type)],
        outputs=[ct.TensorType(name="predictions")],
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Save the model
    output_path = os.path.join(args.output_dir, "model.mlpackage")
    coreml_model.save(output_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description="Convert a model to CoreML format")
    parser.add_argument("--model-dir", type=str, help="The path to the model to convert", required=True)
    parser.add_argument("--output-dir", type=str, help="The path to save the CoreML model", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
