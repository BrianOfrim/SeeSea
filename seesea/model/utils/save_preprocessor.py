"""Load a preprocessor from a checkpoint"""

from transformers import AutoImageProcessor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to load")
    parser.add_argument("--output", type=str, help="The output directory")

    args = parser.parse_args()

    preprocessor = AutoImageProcessor.from_pretrained(args.model)

    preprocessor.save_pretrained(args.output)

    print(f"Saved preprocessor to {args.output}")
