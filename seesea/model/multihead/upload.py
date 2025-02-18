"""Load a model from a directory and upload it to the Hugging Face Hub"""

import argparse
import os
import torch
from seesea.model.multihead.modeling_multihead import MultiHeadModel, MultiHeadConfig
from transformers import AutoImageProcessor, AutoConfig, AutoModel


def main(args):
    """Upload a model to the Hugging Face Hub"""

    # Register the model and config

    AutoConfig.register("multihead-regression", MultiHeadConfig)
    AutoModel.register(MultiHeadConfig, MultiHeadModel)

    MultiHeadConfig.register_for_auto_class()
    MultiHeadModel.register_for_auto_class("AutoModel")

    config = MultiHeadConfig.from_pretrained(args.model_dir)
    model = MultiHeadModel.from_pretrained(args.model_dir, config=config)

    model.push_to_hub(args.repo_id, use_temp_dir=True)


def get_args_parser():
    """Get the arguments for the script"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", help="Path to the saved model file (model.pt)", required=True)
    parser.add_argument("--repo-id", help="The repository ID to upload the model to", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
