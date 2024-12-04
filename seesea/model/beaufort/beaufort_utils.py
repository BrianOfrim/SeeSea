"""Common functions for the Beaufort models"""

from typing import Callable


BeaufortRanges = [
    (0, 0.5),  # Calm
    (0.5, 1.5),  # Light air
    (1.5, 3.3),  # Light breeze
    (3.3, 5.5),  # Gentle breeze
    (5.5, 7.9),  # Moderate breeze
    (7.9, 10.7),  # Fresh breeze
    (10.7, 13.8),  # Strong breeze
    (13.8, 17.1),  # Near gale
    (17.1, 20.7),  # Gale
    (20.7, 24.4),  # Strong gale
    (24.4, 28.4),  # Storm
    (28.4, 32.6),  # Violent storm
    (32.6, float("inf")),  # Hurricane force
]


def mps_to_beaufort(wind_speed):
    """
    Convert wind speed from meters per second to Beaufort scale

    Args:
        wind_speed (float): Wind speed in meters per second

    Returns:
        int: Beaufort scale number (0-12)
    """
    for i, (lower, upper) in enumerate(BeaufortRanges):
        if lower <= wind_speed < upper:
            return i
    assert False, "Wind speed out of range somehow"
    return len(BeaufortRanges) - 1


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
