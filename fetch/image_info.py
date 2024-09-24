"""
A script to explore the images in the dataset
"""

import os
import numpy as np
from PIL import Image
import logging
from typing import List
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def get_all_files(directory, extension):
    # List to store found files
    found_files = []

    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                # Get the full path of the file
                full_path = os.path.join(os.path.abspath(root), file)
                found_files.append(full_path)

    return found_files


def sort_image_by_brightness(image_paths):
    brightness = []
    for i, image_path in enumerate(image_paths):
        # load image
        if i % 1000 == 0:
            LOGGER.debug("Read image %d", i)
        try:
            img = Image.open(image_path)
        except Exception as e:
            LOGGER.warning("\ttError loading image %s: %s", image_path, e)
            continue
        img_array = np.array(img, dtype=np.uint8)
        mean = np.mean(img_array)
        brightness.append((image_path, float(mean)))

    return sorted(brightness, key=lambda x: x[1])


def get_brighntesses(base_dir, cache_file):

    if os.path.exists(cache_file):
        # open cahce file
        with open(cache_file, "r") as file:
            # Read each line, strip newline characters, and convert the string back to a tuple using eval
            return [eval(line.strip()) for line in file]

    observation_paths = get_all_files(base_dir, "observation.json")

    LOGGER.info("Found %d observation.json files", len(observation_paths))

    image_paths = []
    for obs in observation_paths:
        paths = get_all_files(os.path.dirname(obs), "jpg")
        # exculte images that have the pattern *full.jpg
        image_paths.extend([p for p in paths if "full" not in p.lower()])

    LOGGER.info("Found %d, jpg files", len(image_paths))
    image_brightnesses = sort_image_by_brightness(image_paths)

    # write to cache file
    with open(cache_file, "w") as file:
        # Write each tuple to a new line in the file
        for item in image_brightnesses:
            file.write(f"{item}\n")

    LOGGER.info("Wrote image brightnesses to cache file to %s", cache_file)

    return image_brightnesses


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Count the number of images in a dataset")
    arg_parser.add_argument("--dir", help="The directory containing the dataset")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
    arg_parser.add_argument("--max-images", type=int, help="The maximum number of images to process", default=None)
    arg_parser.add_argument(
        "--min-brightness", type=int, help="The minimum average image brightness to include in the dataset", default=0
    )
    arg_parser.add_argument(
        "--max-brightness", type=int, help="The maximum average image brightness to include in the dataset", default=255
    )
    input_args = arg_parser.parse_args()

    # setup the loggers
    LOGGER.setLevel(input_args.log)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    if input_args.log_file is not None:
        file_logging_handler = logging.FileHandler(input_args.log_file)
        file_logging_handler.setFormatter(log_formatter)
        LOGGER.addHandler(file_logging_handler)

    image_brightnesses = get_brighntesses(input_args.dir, "/tmp/brightnesscache.txt")

    brightnesses = [
        b[1]
        for b in image_brightnesses
        if b[1] >= input_args.min_brightness
        if input_args.min_brightness is not None
        if b[1] <= input_args.max_brightness
        if input_args.max_brightness is not None
    ]

    LOGGER.info(
        "Number of images within bounds: %d, total images found: %d (%d %%)",
        len(brightnesses),
        len(image_brightnesses),
        (len(brightnesses) / len(image_brightnesses)) * 100,
    )

    bellow_min = sum(1 for b in image_brightnesses if b[1] < input_args.min_brightness)
    LOGGER.info(
        "Number of images below min brightness: %d",
        bellow_min,
    )
    above_max = sum(1 for b in image_brightnesses if b[1] > input_args.max_brightness)
    LOGGER.info(
        "Number of images above max brightness: %d",
        above_max,
    )

    plt.hist(brightnesses, bins=256, edgecolor="black", alpha=0.7)

    # Add titles and labels
    plt.title("Distribution of Brightnesses")
    plt.xlabel("Number")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

    # Get the image brightnesses as user input
    input_brighntess = input("Enter the brightness value to start from: ")
    input_brighntess = int(input_brighntess)

    # Get the index of the input brightness

    # find the first index that is equal to or greater than the input brightness
    start_index = next(i for i, b in enumerate(image_brightnesses) if b[1] >= input_brighntess)

    # Allow the user to go back and forth in the brightnesses list and show the image
    current_index = start_index
    while True:
        # Get the image path
        image_path = image_brightnesses[current_index][0]
        print(
            f"Image {current_index + 1} of {len(image_brightnesses)}: {image_path}, brightness:"
            f" {image_brightnesses[current_index][1]}"
        )

        # Load the image
        img = Image.open(image_path)
        img.show()

        # Ask the user for input
        user_input = input(
            "Enter 'b' <brightness> to jump to brightness 'b', 'n' for next, 'p' for previous, 'q' to quit: "
        )

        # Check the user input
        if user_input == "n":
            current_index += 1
        elif user_input == "p":
            current_index -= 1
        elif user_input == "q":
            break
        elif user_input.startswith("b"):
            brightness = int(user_input[1:])
            # find the first index that is equal to or greater than the input brightness
            current_index = next(i for i, b in enumerate(image_brightnesses) if b[1] >= brightness)
        else:
            print("Invalid input")
            continue

        # Check if the index is out of bounds
        if current_index < 0:
            print("At the beginning of the list")
            current_index = 0
        elif current_index >= len(image_brightnesses):
            print("At the end of the list")
            current_index = len(image_brightnesses) - 1
