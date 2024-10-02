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

        # check if it has the key "wind_speed_kts"
        if "wind_speed_kts" in observation:
            val_in_mps = None
            if observation["wind_speed_kts"] is not None:
                val_in_mps = kts_to_mps(observation["wind_speed_kts"])
                # Verify that the value is close to an integer
                if abs(val_in_mps - round(val_in_mps)) > 0.00001:
                    LOGGER.warning("Wind speed in mps is not close to an integer")
                # if val in mps is very close to an integer, round it to the nearest integer
                # if abs(val_in_mps - round(val_in_mps)) < 0.0001:
                #     val_in_mps = round(val_in_mps)
                # else:
                #     LOGGER.warning("Wind speed in mps is not close to an integer")

            # add the new key "wind_speed_mps"
            observation["wind_speed_mps"] = val_in_mps
            # remove the old key "wind_speed_kts"
            del observation["wind_speed_kts"]

        if "gust_speed_kts" in observation:
            val_in_mps = None
            if observation["gust_speed_kts"] is not None:
                val_in_mps = kts_to_mps(observation["gust_speed_kts"])
                if abs(val_in_mps - round(val_in_mps)) > 0.00001:
                    LOGGER.warning("Gust speed in mps is not close to an integer")

                # if val in mps is very close to an integer, round it to the nearest integer
                # if abs(val_in_mps - round(val_in_mps)) < 0.0001:
                #     val_in_mps = round(val_in_mps)
                # else:
                #     LOGGER.warning("Gust speed in mps is not close to an integer")

            # add the new key "gust_speed_mps"
            observation["gust_speed_mps"] = val_in_mps
            # remove the old key "gust_speed_kts"
            del observation["gust_speed_kts"]

        if "visibility_m" in observation:
            val_in_nmi = None
            if observation["visibility_m"] is not None:
                LOGGER.info("Visibility in meters: %s", observation["visibility_m"])
                val_in_nmi = observation["visibility_m"] / 1852
                # Verify that the value is close to an integer
                if abs(val_in_nmi - round(val_in_nmi)) > 0.00001:
                    LOGGER.warning("Visibility in nmi is not close to an integer")

                # if val in nmi is very close to an integer, round it to the nearest integer
                # if abs(val_in_nmi - round(val_in_nmi)) < 0.0001:
                #     val_in_nmi = round(val_in_nmi)
                # else:
                #     LOGGER.warning("Visibility in nmi is not close to an integer")

            # add the new key "visibility_nmi"
            observation["visibility_nmi"] = val_in_nmi
            # remove the old key "visibility_m"
            del observation["visibility_m"]

        if "bearing_of_first_image" in observation:
            observation["bearing_of_first_image_deg"] = observation["bearing_of_first_image"]
            del observation["bearing_of_first_image"]

        # write the transformed observation the file
        observation_obj = utils.from_dict(Observation, observation)

        with open(observation_file_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(observation_obj.to_dict(), indent=4))
