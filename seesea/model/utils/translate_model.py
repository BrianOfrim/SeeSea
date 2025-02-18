"""Load the model and save it again"""

import logging

from seesea.model.multihead.modeling_multihead import MultiHeadModel, MultiHeadConfig

LOGGER = logging.getLogger(__name__)


def main(args):
    """
    Load the model and save it again.
    There is some logig in the loader that will translate the state dict to the new format
    """

    config = MultiHeadConfig.from_pretrained(args.input)

    model = MultiHeadModel.from_pretrained(args.input, config=config)
    model.save_pretrained(args.output)

    print(f"Model saved to {args.output}")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Directory containing the safetensors model", required=True)
    parser.add_argument("--output", help="Directory to save the pytorch model", required=True)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
