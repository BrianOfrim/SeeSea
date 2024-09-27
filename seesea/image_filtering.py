"""
A script to read images from the buoycam dataset and filter them based on brightness
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import seesea.utils as utils

LOGGER = logging.getLogger(__name__)


def show_brightness_histogram(brightness_values):
    """Show a histogram of the brightness values"""
    plt.hist(brightness_values, bins=255, color="blue", alpha=0.7)
    plt.title("Image Brightness Histogram")
    plt.xlabel("Brightness")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Count the number of images in a dataset")
    arg_parser.add_argument("--input", help="The directory containing the dataset")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="data")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
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

    # Create the output directory if it doesn't exist
    if not os.path.exists(input_args.output):
        os.makedirs(input_args.output)

    image_paths = utils.get_image_paths(input_args.input)

    LOGGER.info("Found %d image files under %s", len(image_paths), input_args.input)

    # Get the brightnesses of all images in the dataset
    brightnesses = []

    for i, image_path in enumerate(image_paths):
        # load image
        if i % 1000 == 0:
            LOGGER.debug("Read image %d", i)
        img = utils.load_image(image_path)
        if img is None:
            continue
        brightness = utils.get_brightness(img)
        brightnesses.append((image_path, brightness))

    # Print how many images were less than the min and how many were greater than the max
    num_images_below_min = sum(brightness < input_args.min_brightness for _, brightness in brightnesses)
    num_images_above_max = sum(brightness > input_args.max_brightness for _, brightness in brightnesses)
    LOGGER.info("Number of images with brightness below %d: %d", input_args.min_brightness, num_images_below_min)
    LOGGER.info("Number of images with brightness above %d: %d", input_args.max_brightness, num_images_above_max)

    # Filter the images based on brightness
    filtered_images = [
        brightness
        for brightness in brightnesses
        if input_args.min_brightness <= brightness[1] <= input_args.max_brightness
    ]

    # show_brightness_histogram([brightness[1] for brightness in filtered_images])

    # Write all filtered image paths to a file
    filtered_path = os.path.join(input_args.output, "filtered_images.txt")
    with open(filtered_path, "w", encoding="utf-8") as file:
        for path, _ in filtered_images:
            file.write(f"{path}\n")
    LOGGER.info("Wrote all filtered image paths to %s", filtered_path)

    # Break up the filtered images into test, validation, and training sets
    np.random.shuffle(filtered_images)

    num_images = len(filtered_images)
    test_size = int(num_images * 0.1)
    validation_size = int(num_images * 0.1)
    training_size = num_images - test_size - validation_size

    test_images = filtered_images[:test_size]
    validation_images = filtered_images[test_size : test_size + validation_size]
    training_images = filtered_images[test_size + validation_size :]
    LOGGER.info("Number of test images: %d", len(test_images))
    LOGGER.info("Number of validation images: %d", len(validation_images))
    LOGGER.info("Number of training images: %d", len(training_images))

    test_path = os.path.join(input_args.output, "test_images.txt")
    # Write the test, validation, and training image paths to files
    with open(test_path, "w", encoding="utf-8") as file:
        for path, _ in test_images:
            file.write(f"{path}\n")

    validation_path = os.path.join(input_args.output, "validation_images.txt")
    with open(validation_path, "w", encoding="utf-8") as file:
        for path, _ in validation_images:
            file.write(f"{path}\n")

    training_path = os.path.join(input_args.output, "training_images.txt")
    with open(training_path, "w", encoding="utf-8") as file:
        for path, _ in training_images:
            file.write(f"{path}\n")

    LOGGER.info("Wrote test image paths to %s", test_path)
    LOGGER.info("Wrote validation image paths to %s", validation_path)
    LOGGER.info("Wrote training image paths to %s", training_path)
