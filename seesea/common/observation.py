"""Data classes and functions for working with buoy observations"""

import os
import math
from dataclasses import dataclass, asdict
from typing import List
import re
import copy

from seesea.common import utils
import webdataset as wds

DEGREES_PER_IMAGE = 360 / 6


@dataclass
class Observation:
    """
    Observation data for a buoy at a specific time.
    """

    station_id: str
    timestamp: str
    lat_deg: float
    lon_deg: float
    description: str = ""
    wind_speed_mps: float = math.nan
    wind_direction_deg: float = math.nan
    gust_speed_mps: float = math.nan
    wave_height_m: float = math.nan
    dominant_wave_period_s: float = math.nan
    average_wave_period_s: float = math.nan
    mean_wave_direction_deg: float = math.nan
    atmospheric_pressure_hpa: float = math.nan
    air_temperature_c: float = math.nan
    water_temperature_c: float = math.nan
    dewpoint_temperature_c: float = math.nan
    pressure_tendency_hpa: float = math.nan
    visibility_nmi: float = math.nan
    tide_m: float = math.nan
    image_direction_deg: float = math.nan

    def __str__(self):
        return (
            f"Observation(station_id={self.station_id} timestamp={self.timestamp}, description={self.description},"
            f" lat={self.lat_deg}, lon={self.lon_deg},wind_speed_mps={self.wind_speed_mps},"
            f" wind_direction_deg={self.wind_direction_deg}, gust_speed_mps={self.gust_speed_mps},"
            f" wave_height_m={self.wave_height_m}, dominant_wave_period_s={self.dominant_wave_period_s},"
            f" average_wave_period_s={self.average_wave_period_s},"
            f" mean_wave_direction_deg={self.mean_wave_direction_deg},"
            f" atmospheric_pressure_hpa={self.atmospheric_pressure_hpa}, air_temperature_c={self.air_temperature_c},"
            f" water_temperature_c={self.water_temperature_c}, dewpoint_temperature_c={self.dewpoint_temperature_c},"
            f" visibility_nmi={self.visibility_nmi}, pressure_tendency_hpa={self.pressure_tendency_hpa},"
            f" tide_m={self.tide_m}, image_direction_deg={self.image_direction_deg})"
        )

    def to_dict(self):
        """Convert the observation to a dictionary"""
        return asdict(self)


@dataclass
class ImageObservation:
    """A class to store an image's path and it's associated observation data"""

    image_path: str
    observation: Observation

    def base_filename(self):
        """return the base filename of the image"""
        return os.path.splitext(os.path.basename(self.image_path))[0]


def to_webdataset(image_observations: List[ImageObservation], output_dir: str):
    """Write the image observations to webdataset format"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        utils.clear_directory(output_dir)

    # write a webdataset recipe file
    with wds.ShardWriter(os.path.join(output_dir, "%06d.tar"), maxsize=1e9) as sink:
        for io in image_observations:
            image = utils.load_image(io.image_path)
            observation_dict = io.observation.to_dict()

            sink.write({"__key__": io.base_filename(), "jpg": image, "json": observation_dict})


def get_image_number(image_path: str) -> int:
    """Get the image number from the image path"""
    base_filename = os.path.basename(image_path)
    file_name, _ = os.path.splitext(base_filename)

    # image number is after the final underscore
    return int(file_name.split("_")[-1])


def get_all_image_observations(input_dir: str) -> List[ImageObservation]:
    """Get all the image observations from the input directory"""
    observation_file_paths = utils.get_all_files(input_dir, re.compile(r"observation.json", re.IGNORECASE))
    image_observations: List[ImageObservation] = []

    for obs_path in observation_file_paths:

        # Load the observation from the file
        observation_json = utils.load_json(obs_path)
        if observation_json is None:
            continue

        first_image_obs = Observation(**observation_json)
        image_paths = utils.get_all_files(os.path.dirname(obs_path), re.compile(r"\d+.jpg", re.IGNORECASE))

        first_image_angle = first_image_obs.image_direction_deg

        for image_path in image_paths:

            if first_image_angle is not None:
                # copy the observation and add the image direction
                obs = copy.copy(first_image_obs)
                image_number = get_image_number(image_path)
                assert image_number < 6, f"Image number {image_number} is greater than 5"
                obs.image_direction_deg = (first_image_angle + image_number * DEGREES_PER_IMAGE) % 360.0

            image_observations.append(ImageObservation(image_path, obs))

    return image_observations
