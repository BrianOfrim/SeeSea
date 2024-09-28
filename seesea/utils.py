"""
General Utility functions
"""

import logging
import os
import re
import json
from io import BytesIO
from typing import Type, Any

import requests
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def mps_to_kts(mps: float) -> float:
    """Convert meters per second to knots"""
    return mps * 1.94384


def nmi_to_m(nmi: float) -> float:
    """Convert nautical miles to meters"""
    return nmi * 1852


def is_match(pattern: re.Pattern, text: str) -> bool:
    """Check if a pattern matches a string"""
    return bool(re.search(pattern, text))


def entry_exists(obj, key):
    """Check if a key exists in a dictionary and it's value is not None"""
    return key in obj and obj[key] is not None


def entries_exist(obj, keys):
    """Check if all keys exist in a dictionary and their values are not None"""
    return all(entry_exists(obj, key) for key in keys)


def attribute_exists(obj, key):
    """Check if a key exists in an object and it's value is not None"""
    return hasattr(obj, key) and getattr(obj, key) is not None


def attributes_exist(obj, keys):
    """Check if all keys exist in an object and their values are not None"""
    return all(attribute_exists(obj, key) for key in keys)


def fetch_json(url, timeout=None):
    """Get a json object from a url"""
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        LOGGER.debug("Failed to get json from %s due to %s", url, e)
        return None

    # verify that the request was successful
    if response.status_code != 200:
        LOGGER.warning("Failed to get json from %s due to status code %s", url, response.status_code)
        return None
    return response.json()


def load_json(file_path):
    """Load a json object from a file path"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        LOGGER.warning("Failed to load json from %s due to %s", file_path, e)
        return None


def fetch_image(url, timeout=None) -> Image:
    """Get an image file from a url"""
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        LOGGER.warning("Failed to get image file at %s due to %s", url, e)
        return None
    if response.status_code != 200:
        LOGGER.debug("Failed to get image file at %s, code: %d, reason: %s", url, response.status_code, response.reason)
        return None
    LOGGER.debug("Image file retrieved from %s ", url)
    return Image.open(BytesIO(response.content))


def load_image(image_path: str) -> Image:
    """Load an image from a file path"""
    try:
        img = Image.open(image_path)
    except Exception as e:
        LOGGER.warning("\tError loading image %s: %s", image_path, e)
        return None
    return img


def get_all_files(directory, reegex_pattern: re.Pattern = None):
    """Get all files under a directory that match a regex pattern"""
    found_files = []

    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file matches regex
            if reegex_pattern is None or is_match(reegex_pattern, file):
                # Get the full path of the file
                full_path = os.path.join(os.path.abspath(root), file)
                found_files.append(full_path)

    return found_files


def fraction_black(img: Image) -> float:
    """Get the fraction of black pixels in an image"""
    img_array = np.array(img.convert("L")).flatten()
    return np.count_nonzero(img_array == 0) / len(img_array)


def get_brightness(img: Image) -> float:
    """Get the mean pixel channel value of an image"""
    img_array = np.array(img, dtype=np.uint8)
    mean = np.mean(img_array)
    return mean


def from_dict(dataclass_type: Type[Any], data: dict) -> Any:
    """Recursive function to populate the dataclass and its nested members"""
    # Create a new dictionary for the attributes that will be passed to the dataclass
    fieldtypes = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}

    # Iterate over the fields and check if they are dataclass types themselves
    for field, fieldtype in fieldtypes.items():
        # If the field is a dataclass, recursively deserialize it
        if hasattr(fieldtype, "__dataclass_fields__"):
            data[field] = from_dict(fieldtype, data[field])
        # If the field is a List of dataclasses, handle it recursively
        elif hasattr(fieldtype, "__origin__") and fieldtype.__origin__ == list:
            if hasattr(fieldtype.__args__[0], "__dataclass_fields__"):
                data[field] = [from_dict(fieldtype.__args__[0], item) for item in data[field]]

    # Return an instance of the dataclass, passing the processed dictionary
    return dataclass_type(**data)