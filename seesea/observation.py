import json
import os
from dataclasses import dataclass, asdict
from typing import List

from seesea import utils


@dataclass
class Observation:
    """
    Observation data for a buoy at a specific time.
    """

    id: str
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
            f"Observation(id={self.id} timestamp={self.timestamp}, description={self.description}, lat={self.lat_deg},"
            f" lon={self.lon_deg},wind_speed_mps={self.wind_speed_mps}, wind_direction_deg={self.wind_direction_deg},"
            f" gust_speed_mps={self.gust_speed_mps}, wave_height_m={self.wave_height_m},"
            f" dominant_wave_period_s={self.dominant_wave_period_s},"
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

    def to_dict(self):
        """Convert the image observation to a dictionary"""
        return asdict(self)


def to_huggingface_dataset(image_observations: List[ImageObservation], output_file: str):
    """Convert the image observations to a format that can be used by Hugging Face datasets. Output is jsonl format."""

    # must convert the image paths to be relative to the output file
    data = [
        {"file_path": os.path.relpath(io.image_path, os.path.dirname(output_file)), **io.observation.to_dict()}
        for io in image_observations
    ]

    # write the data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def from_huggingface_dataset(filepath: str) -> List[ImageObservation]:
    """Load ImageObservations from a Hugging Face dataset file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # convert the file paths to be absolute

    img_obs = []
    for d in data:
        abs_path = os.path.join(os.path.dirname(filepath), d["file_path"])
        # exclude the file_path key
        d.pop("file_path")

        obs = utils.from_dict(Observation, d)

        img_obs.append(ImageObservation(abs_path, obs))

    return img_obs
