"""Retrieve, process and save buoycam images and buoy observation datat from the NOAA"""

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
import logging
import os
import threading
import time
from typing import Dict, List
import requests

# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import easyocr

BUOYCAM_LIST_URL = "https://www.ndbc.noaa.gov/buoycams.php"
BUOYCAM_IMAGE_FILE_URL_BASE = "https://www.ndbc.noaa.gov/images/buoycam"
OBSERVATION_URL = "https://www.ndbc.noaa.gov/data/realtime2"
BUOYCAM_IMAGE_ROW_LENGTH = 6
IMAGE_WIDTH = 2880
IMAGE_HEIGHT = 300
SUB_IMAGE_WIDTH = 480
SUB_IMAGE_HEIGHT = 270

FRACTION_BLACK_THRESHOLD = 0.85

MISSING_DATA_INDICATOR = "MM"


# Rate limiting lock to be nice to the NOAA buoycam website
request_rate_limit_lock = threading.Lock()
# Max requests per second
MAX_REQUESTS_PER_SECOND = 20

REQUEST_TIMEOUT_SECONDS = 15

# Number of worker threads
NUM_WORKER_THREADS = 100

LOGGER = logging.getLogger(__name__)

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


class BuoyPosition:
    """
    BuoyPosition is a class that encapsulates the position of a buoy in degrees latitude and longitude.
    """

    def __init__(self, lat_deg, lon_deg):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg

    def __str__(self):
        return f"BuoyPosition(lat={self.lat_deg}, lon={self.lon_deg})"


class BuoyInfo:
    """
    BuoyInfo is a class that encapsulates the information for a buoy.
    """

    def __init__(
        self,
        identifier: str,
        tag: str,
        description: str,
        position: BuoyPosition,
        date: datetime.datetime,
    ):
        self.id = identifier
        self.tag = tag
        self.description = description
        self.position = position
        self.date = date

    def __str__(self):
        return (
            f"BuoyInfo(id={self.id}, tag={self.tag}, description={self.description}, position={self.position},"
            f" date={self.date})"
        )

    def date_string(self):
        return self.date.strftime("%Y_%m_%d_%H%M")

    def image_name(self):
        return f"{self.tag}_{self.date_string()}"

    def image_url(self):
        return f"{BUOYCAM_IMAGE_FILE_URL_BASE}/{self.image_name()}.jpg"

    def observation_url(self):
        return f"{OBSERVATION_URL}/{self.id}.txt"

    def save_directory(self, root_dir: str = "images"):
        return f"{root_dir}/{self.id}/{self.date_string()}"

    def image_full_path(self, root_dir: str = "images", image_suffix: str = None):
        return f"{self.save_directory(root_dir)}/{self.image_name()}{'_' + image_suffix if image_suffix else ""}.jpg"


class Observation:
    """
    Observation data for a buoy at a specific time.
    """

    def __init__(
        self,
        station_id: str,
        timestamp: str,
        description: str,
        lat_deg: float,
        lon_deg: float,
        wind_speed_kts: float,
        wind_direction_deg: float,
        gust_speed_kts: float,
        wave_height_m: float,
        dominant_wave_period_s: float,
        average_wave_period_s: float,
        mean_wave_direction_deg: float,
        atmospheric_pressure_hpa: float,
        air_temperature_c: float,
        water_temperature_c: float,
        dewpoint_temperature_c: float,
        visibility_m: float,
        pressure_tendency_hpa: float,
        tide_m: float,
        bearing_of_first_image: int = None,
    ):
        self.id: str = station_id
        self.timestamp: str = timestamp
        self.description: str = description
        self.lat_deg: float = lat_deg
        self.lon_deg: float = lon_deg
        self.wind_speed_kts: float = wind_speed_kts
        self.wind_direction_deg: float = wind_direction_deg
        self.gust_speed_kts: float = gust_speed_kts
        self.wave_height_m: float = wave_height_m
        self.dominant_wave_period_s: float = dominant_wave_period_s
        self.average_wave_period_s: float = average_wave_period_s
        self.mean_wave_direction_deg: float = mean_wave_direction_deg
        self.atmospheric_pressure_hpa: float = atmospheric_pressure_hpa
        self.air_temperature_c: float = air_temperature_c
        self.water_temperature_c: float = water_temperature_c
        self.dewpoint_temperature_c: float = dewpoint_temperature_c
        self.visibility_m: float = visibility_m
        self.pressure_tendency_hpa: float = pressure_tendency_hpa
        self.tide_m: float = tide_m
        self.bearing_of_first_image: int = bearing_of_first_image

    def __str__(self):
        return (
            f"Observation(id={self.id} timestamp={self.timestamp}, description={self.description}, lat={self.lat_deg},"
            f" lon={self.lon_deg},wind_speed_kts={self.wind_speed_kts}, wind_direction_deg={self.wind_direction_deg},"
            f" gust_speed_kts={self.gust_speed_kts}, wave_height_m={self.wave_height_m},"
            f" dominant_wave_period_s={self.dominant_wave_period_s},"
            f" average_wave_period_s={self.average_wave_period_s},"
            f" mean_wave_direction_deg={self.mean_wave_direction_deg},"
            f" atmospheric_pressure_hpa={self.atmospheric_pressure_hpa}, air_temperature_c={self.air_temperature_c},"
            f" water_temperature_c={self.water_temperature_c}, dewpoint_temperature_c={self.dewpoint_temperature_c},"
            f" visibility_m={self.visibility_m}, pressure_tendency_hpa={self.pressure_tendency_hpa},"
            f" tide_m={self.tide_m})"
        )

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class BuoyData:
    """
    A collection of observations for a buoy.
    """

    def __init__(self, station_id: str):
        self.id = station_id
        # key is the date string, value is the observation
        self.observations: Dict[str, Observation] = {}

    def add_observation(self, observation: Observation):
        self.observations[observation.timestamp] = observation

    def has_observation(self, timestamp):
        return timestamp in self.observations

    def get_observation(self, timestamp):
        if self.has_observation(timestamp):
            return self.observations[timestamp]
        return None


class OCR:
    """
    Extracts text from images using OCR
    """

    def __init__(self):
        self.reader = easyocr.Reader(["en"])

    def get_all_text_from_image(self, image):
        # Get all text from the image
        results = self.reader.readtext(np.array(image))
        if len(results) == 0:
            return None

        # Concatenate all the text results
        text_result = ""
        for result in results:
            text_result += result[1]

        return text_result

    def get_angle_from_image(self, image):
        # Get angle in degrees from the image
        results = self.reader.readtext(np.array(image))

        if len(results) == 0:
            return None

        # There should be at least 1 digit and 1 degree symbol
        text_result = results[0][1]
        if len(text_result) < 2:
            return None

        # Remove the last character (degree symbol) from the result
        first_result = text_result[:-1]
        # verify that the result is a number
        if not first_result.isnumeric():
            return None

        angle = int(first_result)

        if angle < 0 or angle > 360:
            return None

        return angle


def entry_exists(obj, key):
    return key in obj and obj[key] is not None


def get_json(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        LOGGER.debug("Failed to get json from %s due to %s", url, e)
        return None

    # verify that the request was successful
    if response.status_code != 200:
        LOGGER.warning("Failed to get json from %s due to status code %s", url, response.status_code)
        return None
    return response.json()


def get_latest_buoy_info() -> List[BuoyInfo]:
    buoy_json = get_json(BUOYCAM_LIST_URL)
    if buoy_json is None:
        LOGGER.error("Failed to get the buoy cam list from %s", BUOYCAM_LIST_URL)
        return None
    buoy_info_list = []
    for buoy in buoy_json:
        if (
            entry_exists(buoy, "id")
            and entry_exists(buoy, "img")
            and entry_exists(buoy, "name")
            and entry_exists(buoy, "lat")
            and entry_exists(buoy, "lng")
        ):

            tag = buoy["img"].split("_")[0]
            try:
                # Date is everything after the first underscore and is in the format YYYY_MM_DD_HHMM"
                date_str = buoy["img"].split("_", 1)[1].replace(".jpg", "")
                date = datetime.datetime.strptime(date_str, "%Y_%m_%d_%H%M")
            except ValueError:
                LOGGER.warning("Failed to parse date for buoy %s: %s", buoy["id"], buoy["img"])
                continue
            buoy_info_list.append(
                BuoyInfo(
                    buoy["id"],
                    tag,
                    buoy["name"],
                    BuoyPosition(buoy["lat"], buoy["lng"]),
                    date,
                )
            )
        else:
            LOGGER.warning("Failed to parse buoy info for buoy %s", buoy["id"])
    return buoy_info_list


def split_image(full_image: Image):
    sub_images = []
    for i in range(BUOYCAM_IMAGE_ROW_LENGTH):
        left = i * SUB_IMAGE_WIDTH
        sub_images.append(full_image.crop((left, 0, left + SUB_IMAGE_WIDTH, SUB_IMAGE_HEIGHT)))
    return sub_images


def extract_table_data(url: str) -> list:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        LOGGER.debug("Failed to get table data from %s due to %s", url, e)
        return None

    # verify that the request was successful
    if response.status_code != 200:
        LOGGER.debug(
            "Failed to get table data from %s due to status code %s",
            url,
            response.status_code,
        )
        return None

    data = response.text
    lines = data.split("\n")

    # Verify that the file has at least 3 lines (header, units, and data)
    if len(lines) < 3:
        raise ValueError("File does not have enough lines")

    # Extract the header and the first row.
    # Skip the first character in the first row as it is just to denote it's a non data row
    header = lines[0][1:].split()
    # Skip the second line which is the units
    rows = lines[2:]

    result = []
    for row in rows:
        values = row.split()
        if len(values) == 0:
            continue
        entry = {}
        for i, h in enumerate(header):
            entry[h] = values[i]
        result.append(entry)

    return result


def get_float(row: dict, key: str, convert_func=None) -> float:
    if key not in row:
        return None
    if row[key] == MISSING_DATA_INDICATOR:
        return None

    if convert_func is None:
        return float(row[key])

    return convert_func(float(row[key]))


def mps_to_kts(mps: float) -> float:
    # Convert meters per second to knots
    return mps * 1.94384


def nmi_to_m(nmi: float) -> float:
    # Convert nautical miles to meters
    return nmi * 1852


def table_row_to_observation(row, info: BuoyInfo) -> Observation:
    # Must have a timestamp
    if "YY" not in row or "MM" not in row or "DD" not in row or "hh" not in row or "mm" not in row:
        return None
    return Observation(
        station_id=info.id,
        timestamp=f'{row["YY"]}_{row["MM"]}_{row["DD"]}_{row["hh"]}{row["mm"]}',
        description=info.description,
        lat_deg=info.position.lat_deg,
        lon_deg=info.position.lon_deg,
        wind_speed_kts=get_float(row, "WSPD", mps_to_kts),
        wind_direction_deg=get_float(row, "WDIR"),
        gust_speed_kts=get_float(row, "GST", mps_to_kts),
        wave_height_m=get_float(row, "WVHT"),
        dominant_wave_period_s=get_float(row, "DPD"),
        average_wave_period_s=get_float(row, "APD"),
        mean_wave_direction_deg=get_float(row, "MWD"),
        atmospheric_pressure_hpa=get_float(row, "PRES"),
        air_temperature_c=get_float(row, "ATMP"),
        water_temperature_c=get_float(row, "WTMP"),
        dewpoint_temperature_c=get_float(row, "DEWP"),
        visibility_m=get_float(row, "VIS", nmi_to_m),
        pressure_tendency_hpa=get_float(row, "PTDY"),
        tide_m=get_float(row, "TIDE"),
    )


def get_observation_data(buoy_info: BuoyInfo) -> BuoyData:

    with request_rate_limit_lock:
        # Simple rate limiting mechanism to be nice to the NOAA website
        time.sleep(1 / MAX_REQUESTS_PER_SECOND)

    observation_data = extract_table_data(buoy_info.observation_url())
    if observation_data is None:
        LOGGER.warning("Failed to get buoy data for buoy %s", buoy_info.id)
        return None
    data = BuoyData(buoy_info.id)
    for row in observation_data:
        observation = table_row_to_observation(row, buoy_info)
        if observation is not None:
            data.add_observation(observation)
        else:
            LOGGER.warning("Failed to parse observation data for buoy %s", buoy_info.id)
    return data


def get_image_file(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        LOGGER.debug("Failed to get image file at %s due to %s", url, e)
        return None
    if response.status_code != 200:
        LOGGER.debug("Failed to get image file at %s", url)
        return None
    LOGGER.debug("Image file retrieved from %s ", url)
    return Image.open(BytesIO(response.content))


# the fraction of the image that is black
def fraction_black(img):
    img_array = np.array(img.convert("L")).flatten()
    return np.count_nonzero(img_array == 0) / len(img_array)


def get_angle_from_image(img: Image, ocr_reader: OCR):
    assert img.width == IMAGE_WIDTH and img.height == IMAGE_HEIGHT
    angle_crop = img.crop((150, img.height - 30, 250, img.height))
    # Extract the angle from the image
    return ocr_reader.get_angle_from_image(angle_crop)


def fetch_image(request: BuoyInfo) -> Image:

    # Simple rate limiting mechanism to be nice to the NOAA buoycam website
    with request_rate_limit_lock:
        time.sleep(1 / MAX_REQUESTS_PER_SECOND)

    # Get the image file
    img = get_image_file(request.image_url())
    if img is None:
        LOGGER.debug("\t%s failed", request)
        return None

    # Verify the image size
    if img.width != IMAGE_WIDTH or img.height != IMAGE_HEIGHT:
        LOGGER.warning("\t%d: image has invalid dimensions %dx%d", request, img.width, img.height)
        return None

    return img


def extend_to_past(latest_list: List[BuoyInfo], hours_in_past) -> list[BuoyInfo]:
    extended_list = []
    for latest in latest_list:
        # Added
        extended_list.extend(
            [
                BuoyInfo(
                    latest.id,
                    latest.tag,
                    latest.description,
                    latest.position,
                    latest.date - datetime.timedelta(minutes=10 * i),
                )
                for i in range(0, hours_in_past * 6)
            ]
        )

    return extended_list


def save_observation_data(observation: Observation, info: BuoyInfo, output_dir: str):
    # make the folder if it doesn't exist
    if not os.path.exists(info.save_directory(output_dir)):
        os.makedirs(info.save_directory(output_dir))

    observation_path = f"{info.save_directory(output_dir)}/observation.json"
    with open(observation_path, "w", encoding="utf-8") as file:
        file.write(observation.to_json())

    LOGGER.debug(
        "\tObservation for buoy %s at %s saved at %s",
        info.id,
        info.date_string(),
        observation_path,
    )


def save_image(img: Image, info: BuoyInfo, output_dir: str):
    # make the folder if it doesn't exist
    if not os.path.exists(info.save_directory(output_dir)):
        os.makedirs(info.save_directory(output_dir))

    # Save the full image
    full_image_path = info.image_full_path(output_dir, "full")
    img.save(full_image_path)
    LOGGER.debug("\tFull image saved at %s", full_image_path)

    # Save the sub images
    for i, sub_img in enumerate(split_image(img)):
        # Check if the sub image is mostly black
        fraction_black_value = fraction_black(sub_img)
        if fraction_black_value > FRACTION_BLACK_THRESHOLD:
            LOGGER.debug(
                "\t\tSub image %s is %.2f%% black, skipping",
                i,
                fraction_black_value * 100,
            )
            continue
        LOGGER.debug("\t\tSub image %i is %.2f%% black", i, fraction_black_value * 100)

        sub_img_path = info.image_full_path(output_dir, str(i))
        sub_img.save(sub_img_path)
        LOGGER.debug("\t\tSub image %d saved at %s", i, sub_img_path)


def image_pipeline(info: BuoyInfo, observation: Observation, output_dir: str, ocr_reader: OCR = None) -> bool:

    image = fetch_image(info)
    if image is None:
        return False

    if ocr_reader is not None:
        observation.bearing_of_first_image = get_angle_from_image(image, ocr_reader)

    save_image(image, info, output_dir)
    save_observation_data(observation, info, output_dir)
    return True


def main(args):

    LOGGER.info("Starting buoycam fetcher, getting images for the last %d hours", args.hours)

    latest_info = get_latest_buoy_info()

    if latest_info is None:
        LOGGER.fatal("Failed to get the buoy list")
        return

    LOGGER.info("Found %d buoycams", len(latest_info))

    # Observation look up object
    buoy_data_lookup: Dict[str, BuoyData] = {}

    # Get observation data for all buoycams
    with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as executor:
        futures_to_input = {executor.submit(get_observation_data, info): info for info in latest_info}

        for future in as_completed(futures_to_input):
            result = future.result()
            buoy = futures_to_input[future]
            LOGGER.info(
                "Completed observation request for %s, success: %s",
                buoy.id,
                result is not None,
            )
            if result is not None:
                buoy_data_lookup[result.id] = result

    LOGGER.info("Retrieved observation data for %s buoycams", len(buoy_data_lookup))

    # OCR reader
    ocr_reader = OCR()

    # Generate image requests for all buoycams that have observation data
    image_requests = extend_to_past([info for info in latest_info], args.hours)

    # Only request images that we have observation data for
    filtered_requests = []
    for ir in image_requests:
        if ir.id not in buoy_data_lookup:
            LOGGER.warning("Buoy %s does not have observation data", ir.id)
        if not buoy_data_lookup[ir.id].has_observation(ir.date_string()):
            LOGGER.warning("Buoy %s does not have observation data for %s", ir.id, ir.date_string())
        filtered_requests.append(ir)

    LOGGER.info("Generated %s image requests", len(filtered_requests))

    # Fetch, process and save the images
    with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as executor:

        futures_to_request = {}
        for request in filtered_requests:
            obs = buoy_data_lookup[request.id].get_observation(request.date_string())
            future = executor.submit(image_pipeline, request, obs, args.output, ocr_reader)
            futures_to_request[future] = request

        for future in as_completed(futures_to_request):
            result = future.result()
            request = futures_to_request[future]
            LOGGER.info("Completed %s, success: %s", request, future.result())


if __name__ == "__main__":
    execution_start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        description="A script to fetch, process, and save buoycam images and observation data from the NOAA website"
    )

    arg_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for images and observation data",
        default="images",
    )
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
    arg_parser.add_argument(
        "--hours",
        type=int,
        help="Number of hours in the past to retrieve images for",
        default=24,
    )

    input_args = arg_parser.parse_args()

    # setup the loggers
    LOGGER.setLevel(input_args.log)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    if input_args.log_file is not None:
        file_logging_handler = logging.FileHandler(input_args.log_file)
        file_logging_handler.setFormatter(log_formatter)
        LOGGER.addHandler(file_logging_handler)

    main(input_args)

    LOGGER.info("Total runtime %s", execution_start_time - time.time())
