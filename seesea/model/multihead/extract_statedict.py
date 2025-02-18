import torch
import argparse
import seesea.model.multihead.modeling_multihead
import os


def extract_statedict(input_path: str, output_path: str):
    """
    Load a full model and save just its state dictionary.

    Args:
        input_path: Path to the full model file
        output_path: Path where the state dictionary should be saved
    """
    # Load the full model
    print(f"Loading model from {input_path}")
    model = torch.load(input_path, map_location="cpu")

    # make the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract the state dictionary
    state_dict = model.state_dict()

    # Save just the state dictionary
    print(f"Saving state dictionary to {output_path}")
    torch.save(state_dict, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Extract state dictionary from a full model")
    parser.add_argument("--input", help="Path to the full model file")
    parser.add_argument("--output", help="Path where the state dictionary should be saved")

    args = parser.parse_args()
    extract_statedict(args.input, args.output)


if __name__ == "__main__":
    main()
