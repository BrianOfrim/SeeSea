import json
import os
from dataclasses import dataclass, asdict
from typing import List
import re

from seesea import utils
import webdataset as wds


@dataclass
class Observation:
    """
    Observation data for a buoy at a specific time.
    """

    station_id: str
    timestamp: str
    description: str
    lat_deg: float
    lon_deg: float
    wind_speed_mps: float
    wind_direction_deg: float
    gust_speed_mps: float
    wave_height_m: float
    dominant_wave_period_s: float
    average_wave_period_s: float
    mean_wave_direction_deg: float
    atmospheric_pressure_hpa: float
    air_temperature_c: float
    water_temperature_c: float
    dewpoint_temperature_c: float
    visibility_nmi: float
    pressure_tendency_hpa: float
    tide_m: float
    bearing_of_first_image_deg: int = None

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
            f" tide_m={self.tide_m}, bearing_of_first_image_deg={self.bearing_of_first_image_deg})"
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
        # return the base filename of the image
        return os.path.splitext(os.path.basename(self.image_path))[0]


def to_webdataset(image_observations: List[ImageObservation], output_dir: str):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write a webdataset recipe file
    with wds.ShardWriter(os.path.join(output_dir, "%06d.tar"), maxsize=1e9) as sink:
        for io in image_observations:
            image = utils.load_image(io.image_path)
            sink.write(
                {
                    "__key__": io.base_filename(),
                    "jpg": image,
                    "json": io.observation.to_dict(),
                }
            )


def get_all_image_observations(input_dir: str) -> List[ImageObservation]:
    observation_file_paths = utils.get_all_files(input_dir, re.compile(r"observation.json", re.IGNORECASE))
    image_observations: List[ImageObservation] = []

    for obs_path in observation_file_paths:

        # Load the observation from the file
        observation_json = utils.load_json(obs_path)
        if observation_json is None:
            continue

        observation = Observation(**observation_json)
        image_paths = utils.get_all_files(os.path.dirname(obs_path), re.compile(r"\d+.jpg", re.IGNORECASE))
        image_observations.extend([ImageObservation(image_path, observation) for image_path in image_paths])

    return image_observations
