"""
A script to read images from the buoycam dataset and filter them based on brightness
"""

import os
import logging
import json
from typing import List, Tuple

import numpy as np
from seesea.common import utils
from seesea.common.observation import (
    Observation,
    ImageObservation,
    to_webdataset,
    get_all_image_observations,
    get_statistics,
)
from json import JSONEncoder


LOGGER = logging.getLogger(__name__)


def filter_by_observation_keys(
    image_observations: List[ImageObservation], observation_keys: List[str]
) -> List[ImageObservation]:
    """Filter the image observations based on the keys that must be present in the observation"""
    filtered_image_observations = []
    for image_observation in image_observations:
        if utils.attributes_exist(image_observation.observation, observation_keys):
            filtered_image_observations.append(image_observation)
    return filtered_image_observations


def calculate_image_brightnesses(image_observations: List[ImageObservation]) -> List[Tuple[ImageObservation, float]]:
    """Calculate the brightness of each image in the dataset"""
    image_brighnesses: List[(ImageObservation, float)] = []

    for i, image_observation in enumerate(image_observations):
        # load image
        if i % 5000 == 0:
            LOGGER.debug("Read image %d", i)
        img = utils.load_image(image_observation.image_path)
        if img is None:
            LOGGER.warning("Failed to load image %s", image_observation.image_path)
            continue
        image_brighnesses.append((image_observation, utils.get_brightness(img)))

    return image_brighnesses


def filter_by_brightness(image_observations: List[ImageObservation], min: float, max: float) -> List[ImageObservation]:
    """Filter the images based on their brightness"""
    image_brighnesses = calculate_image_brightnesses(image_observations)

    # Print how many images were less than the min and how many were greater than the max
    if min is not None:
        num_images_below_min = sum(brightness < min for _, brightness in image_brighnesses)
        LOGGER.info("Number of images with brightness below %d: %d", min, num_images_below_min)

    if max is not None:
        num_images_above_max = sum(brightness > max for _, brightness in image_brighnesses)
        LOGGER.info("Number of images with brightness above %d: %d", max, num_images_above_max)

    # Filter the images based on brightness
    filtered = []
    for image_observation, brightness in image_brighnesses:
        if (min is None or brightness >= min) and (max is None or brightness <= max):
            filtered.append(image_observation)

    return filtered


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Count the number of images in a dataset")
    arg_parser.add_argument("--input", help="The directory containing the dataset")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="data")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
    arg_parser.add_argument(
        "--min-brightness",
        type=int,
        help="The minimum average image brightness to include in the dataset",
        default=None,
    )
    arg_parser.add_argument(
        "--max-brightness",
        type=int,
        help="The maximum average image brightness to include in the dataset",
        default=None,
    )
    arg_parser.add_argument(
        "--observation-keys",
        nargs="+",
        type=str,
        help="Values that must be present in the observation in order to include it's images in the dataset",
        default=None,
    )
    arg_parser.add_argument("--test-split", type=float, help="The fraction of images to use for testing", default=0.1)
    arg_parser.add_argument(
        "--validation-split", type=float, help="The fraction of images to use for validation", default=0.1
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

    image_observations: List[ImageObservation] = get_all_image_observations(input_args.input)
    LOGGER.info("Number of images found: %d", len(image_observations))

    if input_args.observation_keys is not None:
        image_observations = filter_by_observation_keys(image_observations, input_args.observation_keys)
        LOGGER.info("Number of images after filtering for valid observation keys: %d", len(image_observations))

    if input_args.min_brightness is not None or input_args.max_brightness is not None:
        image_observations = filter_by_brightness(
            image_observations, input_args.min_brightness, input_args.max_brightness
        )
        LOGGER.info("Number of images after filtering for brightness: %d", len(image_observations))

    # Save the statistics to a file
    statistics = get_statistics(image_observations)
    with open(os.path.join(input_args.output, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump([stat.to_dict() for stat in statistics], f, indent=4)

    LOGGER.info("Wrote statistics to %s", os.path.join(input_args.output, "statistics.json"))

    # Break up the filtered images into test, validation, and training sets
    np.random.shuffle(image_observations)

    num_images = len(image_observations)
    test_size = int(num_images * input_args.test_split)
    validation_size = int(num_images * input_args.validation_split)
    training_size = num_images - test_size - validation_size

    test_image_observations = image_observations[:test_size]
    validation_image_observations = image_observations[test_size : test_size + validation_size]
    training_image_observaitons = image_observations[test_size + validation_size :]

    test_path = os.path.join(input_args.output, "test")
    to_webdataset(test_image_observations, test_path)
    LOGGER.info("Wrote %d test image observations to %s", len(test_image_observations), test_path)

    validation_path = os.path.join(input_args.output, "validation")
    to_webdataset(validation_image_observations, validation_path)
    LOGGER.info("Wrote %d validation image observations to %s", len(validation_image_observations), validation_path)

    training_path = os.path.join(input_args.output, "train")
    to_webdataset(training_image_observaitons, training_path)
    LOGGER.info("Wrote %d training image observations to %s", len(training_image_observaitons), training_path)

    # Save split sizes to a file
    with open(os.path.join(input_args.output, "split_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({"test": test_size, "validation": validation_size, "training": training_size}, f, indent=4)
