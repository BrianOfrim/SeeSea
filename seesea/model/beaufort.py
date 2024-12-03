"""Common functions for the Beaufort models"""

from typing import Callable


def mps_to_beaufort(wind_speed):
    """
    Convert wind speed from meters per second to Beaufort scale

    Args:
        wind_speed (float): Wind speed in meters per second

    Returns:
        int: Beaufort scale number (0-12)
    """
    if wind_speed < 0.5:
        return 0  # Calm
    elif wind_speed < 1.5:
        return 1  # Light air
    elif wind_speed < 3.3:
        return 2  # Light breeze
    elif wind_speed < 5.5:
        return 3  # Gentle breeze
    elif wind_speed < 7.9:
        return 4  # Moderate breeze
    elif wind_speed < 10.7:
        return 5  # Fresh breeze
    elif wind_speed < 13.8:
        return 6  # Strong breeze
    elif wind_speed < 17.1:
        return 7  # Near gale
    elif wind_speed < 20.7:
        return 8  # Gale
    elif wind_speed < 24.4:
        return 9  # Strong gale
    elif wind_speed < 28.4:
        return 10  # Storm
    elif wind_speed < 32.6:
        return 11  # Violent storm
    else:
        return 12  # Hurricane force


id2label_beaufort = {
    0: "Calm",
    1: "Light air",
    2: "Light breeze",
    3: "Gentle breeze",
    4: "Moderate breeze",
    5: "Fresh breeze",
    6: "Strong breeze",
    7: "Near gale",
    8: "Gale",
    9: "Strong gale",
    10: "Storm",
    11: "Violent storm",
    12: "Hurricane force",
}
label2id_beaufort = {v: k for k, v in id2label_beaufort.items()}


def preprocess_batch_beaufort(transform: Callable, samples):
    """Preprocess a batch of samples"""
    samples["pixel_values"] = transform(samples["jpg"])["pixel_values"]
    samples["labels"] = [mps_to_beaufort(obj["wind_speed_mps"]) for obj in samples["json"]]
    return samples
