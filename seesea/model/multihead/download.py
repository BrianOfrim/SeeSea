"""A script to download a model from the Hugging Face Hub."""

import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoImageProcessor, AutoModel


def main(args):
    api = HfApi()
    # api.download_folder(repo_id=args.repo_id, local_dir=args.local_dir)

    # Download the model
    model = AutoModel.from_pretrained(args.repo_id)
    # image_processor = AutoImageProcessor.from_pretrained(args.repo_id)

    return model


def get_args_parser():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, help="The repository ID to download the model from.")
    parser.add_argument("--local-dir", type=str, help="The local directory to save the model to.")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
