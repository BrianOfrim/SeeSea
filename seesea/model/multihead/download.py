"""A script to download a model from the Hugging Face Hub."""

import argparse
from seesea.model.multihead.modeling_multihead import (
    MultiHeadModel,
    MultiHeadConfig,
)


def main(args):
    config = MultiHeadConfig.from_pretrained(args.repo_id, trust_remote_code=True)
    model = MultiHeadModel.from_pretrained(args.repo_id, config=config, trust_remote_code=True)
    print(model)


def get_args_parser():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, help="The repository ID to download the model from.", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
