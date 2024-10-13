"""
General Utility functions
"""

import logging
import os
import re
import json
import math
from io import BytesIO
from typing import Type, Any, Tuple, Callable

import requests
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torch
import torchvision.models as models

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
    return hasattr(obj, key) and getattr(obj, key) is not None and getattr(obj, key) is not math.nan


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


def get_sharpness(img: Image) -> float:
    """Get the sharpness of an image"""

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(np.array(img.convert("L")), cv2.CV_64F)

    # Compute the variance of the Laplacian
    return laplacian.var()


def detect_water_droplets(img):
    """Detect water droplets in an image"""

    # Apply a Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(np.array(img.convert("L")), (9, 9), 2)

    # Use HoughCircles to detect circular shapes
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=10, maxRadius=100
    )

    if circles is None:
        return None

    return np.round(circles[0, :]).astype("int")


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


def continuous_single_output_model_factory(model_name: str, weights_path: str = None) -> Tuple[nn.Module, Callable]:
    """
    Create a model with one continuous output from the model name and weights path

    Args:
        model_name: The name of the model to create
        weights_path: The path to the weights file to load into the model. If None, the default weights are used

    Returns:
        The model and an image transform function
    """
    model_builder = models.get_model_builder(model_name)

    weight_enum = models.get_model_weights(model_builder)
    default_weights = weight_enum.DEFAULT
    transform = default_weights.transforms()
    model = model_builder(weights=default_weights)

    # Modify the fully connected layer to output a single value
    # For ResNet models, the fully connected layer is model.fc
    # For convnext models, the fully connected layer is model.classifier[2]
    if model_name.startswith("resnet") or model_name.startswith("wide_resnet") or model_name.startswith("regnet"):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)
    elif model_name.startswith("convnext"):
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=1)
    else:
        raise ValueError(f"Model {model_name} not supported")

    LOGGER.info("Using model %s", model_name)

    if weights_path is not None:
        # verify that the weights file exists
        if not os.path.exists(weights_path):
            raise ValueError(f"Model weights file {weights_path} does not exist")
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        LOGGER.debug("Loaded weights from %s", weights_path)
    else:
        LOGGER.debug("Using default weights %s", default_weights)

    LOGGER.debug("Using image transforms: %s", transform)

    return model, transform
