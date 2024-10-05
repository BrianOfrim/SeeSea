import os
import logging
import re
import json

import seesea.utils as utils
from seesea.observation import Observation

LOGGER = logging.getLogger(__name__)


def kts_to_mps(kts: float) -> float:
    """Convert knots to meters per second"""
    return kts / 1.94384


if __name__ == "__main__":
    print("Transforming observation files")
    import argparse

    arg_parser = argparse.ArgumentParser(description="Create a dataset from the observation files")
    arg_parser.add_argument("--input", help="The directory containing the observation files")

    input_args = arg_parser.parse_args()

    LOGGER.setLevel("INFO")

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    # get all observation files
    observation_file_paths = utils.get_all_files(input_args.input, re.compile(r"observation.json", re.IGNORECASE))

    LOGGER.info("Found %s observation files", len(observation_file_paths))
    # transsform the observation files
    for observation_file_path in observation_file_paths:

        observation = utils.load_json(observation_file_path)
        LOGGER.debug("Transforming %s", observation_file_path)
        if observation is None:
            LOGGER.error("Failed to load observations from %s", observation_file_path)
            continue

        if "id" in observation:

            observation["station_id"] = observation["id"]
            del observation["id"]

        # write the transformed observation the file
        observation_obj = utils.from_dict(Observation, observation)

        with open(observation_file_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(observation_obj.to_dict(), indent=4))
