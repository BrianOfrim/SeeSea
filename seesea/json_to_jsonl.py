import json
import argparse


def json_to_jsonl(input_file, output_file):
    """
    Convert a JSON file containing an array of objects to a JSONL file.

    Args:
    input_file (str): Path to the input JSON file.
    output_file (str): Path to the output JSONL file.
    """
    # Open the input JSON file and load the data
    with open(input_file, "r") as infile:
        data = json.load(infile)

    # Ensure the data is a list (array) of objects
    if isinstance(data, list):
        # Open the output file for writing JSONL data
        with open(output_file, "w") as outfile:
            for obj in data:
                # Write each object as a single line in JSONL format
                json.dump(obj, outfile)
                outfile.write("\n")
        print(f"Successfully converted {input_file} to {output_file}")
    else:
        print(f"Error: {input_file} does not contain a list of objects")


def main(args):
    # Convert the JSON to JSONL
    json_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    # Create argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Convert JSON file to JSONL format.")
    parser.add_argument("--input", type=str, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, help="Path to the output JSONL file")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
