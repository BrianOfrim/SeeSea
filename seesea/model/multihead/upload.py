"""A script to upload a trained model to the Hugging Face Hub."""

import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoImageProcessor, AutoConfig


def main(args):
    # The model directory should now contain:
    # - config.json
    # - pytorch_model.bin
    # - preprocessor_config.json
    
    api = HfApi()
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=repo_id,
        repo_type="model"
    )

    # Upload the image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    api.upload_file(file_path=image_processor, repo_id=repo_id, repo_type="model")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Upload a trained model to the Hugging Face Hub")
    parser.add_argument("--model-dir", help="Path to the saved model file (model.pt)", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
