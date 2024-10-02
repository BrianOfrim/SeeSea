import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from seesea import utils
from seesea.observation import ImageObservation


class SeeSeaDataset(Dataset):
    """SeeSea dataset."""

    def __init__(self, image_observation_file: str, observation_key: str, transform=None, max_length=None):
        """
        Args:
            image_observation_file (string): Path to the JSON file with image filenames and observation data.
            transform (callable, optional): Optional transforms to be applied to the images.
        """
        image_observations_json = utils.load_json(image_observation_file)
        if image_observations_json is None:
            # throw an exception if the json file is not loaded
            raise Exception(f"Failed to load image observation data from {image_observation_file}")
        if len(image_observations_json) == 0:
            # throw an exception if the json file is empty
            raise Exception(f"No image observation data found in {image_observation_file}")

        self.image_observations = [utils.from_dict(ImageObservation, obs) for obs in image_observations_json]
        self.observation_key = observation_key
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        size = len(self.image_observations)
        if self.max_length is not None:
            size = min(size, self.max_length)
        return size

    def __getitem__(self, idx):
        image_path = self.image_observations[idx].image_path
        image = Image.open(image_path).convert("RGB")
        observation = self.image_observations[idx].observation

        if self.transform:
            image = self.transform(image)

        value = getattr(observation, self.observation_key, None)
        if value is None:
            raise ValueError(
                f"Observation {observation.id}:{observation.timestamp} does not have a valid value for"
                f" {self.observation_key}"
            )

        value = np.array([value]).astype("float32")
        value = torch.from_numpy(value)

        return image, value
