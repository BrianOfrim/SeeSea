"""A script to visualize various statistics about the dataset"""

import os
import logging

import matplotlib.pyplot as plt
import numpy as np

import seesea.utils as utils
from seesea.observation import ImageObservation, Observation

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Visualize statistics about the dataset")
    arg_parser.add_argument("--input", help="The json file containing the dataset image observation information")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="data/stats")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
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

    # get the observations from the image observation json file
    LOGGER.info("Getting image observation data from %s", input_args.input)

    image_observations_json = utils.load_json(input_args.input)
    if image_observations_json is None:
        LOGGER.error("Failed to load image observation data from %s", input_args.input)
        exit(1)

    if len(image_observations_json) == 0:
        LOGGER.error("No image observation data found in %s", input_args.input)
        exit(1)

    image_observations = [utils.from_dict(ImageObservation, obs) for obs in image_observations_json]

    # Get all keys in the observation data that have float or int values
    number_keys = [
        key
        for key in Observation.__dataclass_fields__.keys()
        if Observation.__dataclass_fields__[key].type == float or Observation.__dataclass_fields__[key].type == int
    ]

    # filter out keys that are degrees
    # float_keys = [key for key in float_keys if not key.endswith("_deg")]

    # Get statistics for each float key
    for key in number_keys:
        values = [getattr(obs.observation, key) for obs in image_observations]
        # remove None values
        values = [value for value in values if value is not None]

        if len(values) == 0:
            LOGGER.warning("No values found for %s", key)
            continue
        LOGGER.info("%s statistics:", key)

        values.sort()

        LOGGER.info("\tMin: %f", np.min(values))
        LOGGER.info("\tMax: %f", np.max(values))
        LOGGER.info("\tMean: %f", np.mean(values))
        LOGGER.info("\tMedian: %f", np.median(values))
        LOGGER.info("\tStandard Deviation: %f", np.std(values))

        # get all unique values
        unique_values = np.unique(values)
        LOGGER.info("\tTotal values %d, Unique values: %d", len(values), len(unique_values))

        LOGGER.info(
            "\tUnique values (first 100): %s%s", unique_values[0:100], " ..." if len(unique_values) > 100 else ""
        )

        # plot histogram
        plt.hist(values, bins="auto")
        plt.title(f"{key} Histogram")
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(input_args.output, f"{key}_histogram.png"))
        plt.close()
