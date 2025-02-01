"""A script to visualize various statistics about the dataset"""

import os
import logging
import math
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt
import numpy as np

from seesea.common.observation import Observation, get_all_image_observations, ObservationStatistics

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Visualize statistics about the dataset")
    arg_parser.add_argument("--input", help="The json file containing the dataset image observation information")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="stats")
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

    image_observations = get_all_image_observations(input_args.input)

    if len(image_observations) == 0:
        LOGGER.error("No image observation data found in %s", input_args.input)
        exit(1)

    # Get all keys in the observation data that have float or int values
    number_keys = [
        key
        for key in Observation.__dataclass_fields__.keys()
        if Observation.__dataclass_fields__[key].type == float or Observation.__dataclass_fields__[key].type == int
    ]

    # filter out keys that are degrees
    # float_keys = [key for key in float_keys if not key.endswith("_deg")]

    statistics = []

    # Get statistics for each float key
    for key in number_keys:
        values = [getattr(obs.observation, key) for obs in image_observations]
        # remove None values
        values = [value for value in values if value is not None]
        values = [value for value in values if not math.isnan(value)]

        if len(values) == 0:
            LOGGER.warning("No values found for %s", key)
            continue

        LOGGER.info("%s statistics:", key)

        values.sort()

        stats = ObservationStatistics(
            key=key,
            min=np.min(values),
            max=np.max(values),
            mean=np.mean(values),
            median=np.median(values),
            std=np.std(values),
        )

        LOGGER.info(stats)

        statistics.append(stats)

        # get all unique values
        unique_values = np.unique(values)

        if len(unique_values) == 0:
            LOGGER.warning("No unique values found for %s", key)
            continue

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

    # write the statistics to a json file
    with open(os.path.join(input_args.output, "statistics.json"), "w") as f:
        json.dump([stat.__dict__ for stat in statistics], f)

    # Create a dict of observations for each id
    id_observations = {}
    for obs in image_observations:
        id_observations.setdefault(obs.observation.station_id, []).append(obs)

    # Get the historgram for each id

    for id, obs_list in id_observations.items():
        for key in number_keys:
            values = [getattr(obs.observation, key) for obs in obs_list]
            # remove None values
            values = [value for value in values if value is not None]
            values = [value for value in values if not math.isnan(value)]

            if len(values) == 0:
                continue

            # plot histogram
            plt.hist(values, bins="auto")
            plt.title(f"{key} Histogram for ID {id}")
            plt.xlabel(key)
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(input_args.output, f"{key}_histogram_id_{id}.png"))
            plt.close()
