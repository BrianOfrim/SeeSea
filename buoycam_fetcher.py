import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageStat
from io import BytesIO
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import easyocr
import logging
import threading
import time
from typing import Dict, List

# Retrieve, process and save buoycam images from the NOAA buoycam website

BUOYCAM_LIST_URL = "https://www.ndbc.noaa.gov/buoycams.php"
BUOYCAM_IMAGE_FILE_URL_BASE = "https://www.ndbc.noaa.gov/images/buoycam/"
OBSERVATION_URL = "https://www.ndbc.noaa.gov/data/realtime2"
BUOYCAM_IMAGE_ROW_LENGTH = 6
IMAGE_WIDTH = 2880
IMAGE_HEIGHT = 300
SUB_IMAGE_WIDTH = 480
SUB_IMAGE_HEIGHT = 270

FRACTION_BLACK_THRESHOLD = 0.85

MISSING_DATA_INDICATOR = "MM"

NUMBER_OF_HOURS_TO_GET_IMAGES_FOR = 24

# Rate limiting lock to be nice to the NOAA buoycam website
request_rate_limit_lock = threading.Lock()
# Max requests per second
MAX_REQUESTS_PER_SECOND = 10


LOGGER = logging.getLogger(__name__)


def get_json(url):
    response = requests.get(url)
    return response.json()


class OCR:
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


class BuoyImageRequest:
    def __init__(self, id, tag, date):
        self.id: str = id
        self.tag: str = tag
        self.date: datetime.datetime = date

    def date_string(self):
        return self.date.strftime("%Y_%m_%d_%H%M")

    def image_name(self):
        return f"{self.tag}_{self.date_string()}"

    def image_filename(self):
        return f"{self.image_name()}.jpg"

    def url(self):
        return f"{BUOYCAM_IMAGE_FILE_URL_BASE}{self.image_filename()}"

    def __str__(self):
        return f"BuoyImageRequest(id={self.id}, tag={self.tag}, date={self.date})"


def extract_table_data(url: str) -> list:
    response = requests.get(url)
    # verify that the request was successful
    if response.status_code != 200:
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
        for i in range(len(header)):
            entry[header[i]] = values[i]
        result.append(entry)

    return result


def get_float(dict: dict, key: str, convert_func=None) -> float:
    if key not in dict:
        return None
    if dict[key] == MISSING_DATA_INDICATOR:
        return None

    if convert_func is None:
        return float(dict[key])

    return convert_func(float(dict[key]))


def table_row_to_db_entry(row, station_id):
    return {
        "station_id": station_id,
        "timestamp": f'{row["YY"]}_{row["MM"]}_{row["DD"]}_{row["hh"]}{row["mm"]}',
        "wind_speed_kts": get_float(row, "WSPD", mps_to_kts),
        "wind_direction_deg": get_float(row, "WDIR"),
        "gust_speed_kts": get_float(row, "GST", mps_to_kts),
        "wave_height_m": get_float(row, "WVHT"),
        "dominant_wave_period_s": get_float(row, "DPD"),
        "average_wave_period_s": get_float(row, "APD"),
        "mean_wave_direction_deg": get_float(row, "MWD"),
        "atmospheric_pressure_hpa": get_float(row, "PRES"),
        "air_temperature_c": get_float(row, "ATMP"),
        "water_temperature_c": get_float(row, "WTMP"),
        "dewpoint_temperature_c": get_float(row, "DEWP"),
        "visibility_m": get_float(row, "VIS", nmi_to_m),
        "pressure_tendency_hpa": get_float(row, "PTDY"),
        "tide_m": get_float(row, "TIDE"),
    }


def mps_to_kts(mps: float) -> float:
    # Convert meters per second to knots
    return mps * 1.94384


def nmi_to_m(nmi: float) -> float:
    # Convert nautical miles to meters
    return nmi * 1852


class Observation:
    def __init__(
        self,
        timestamp,
        wind_speed_kts,
        wind_direction_deg,
        gust_speed_kts,
        wave_height_m,
        dominant_wave_period_s,
        average_wave_period_s,
        mean_wave_direction_deg,
        atmospheric_pressure_hpa,
        air_temperature_c,
        water_temperature_c,
        dewpoint_temperature_c,
        visibility_m,
        pressure_tendency_hpa,
        tide_m,
    ):
        self.timestamp = timestamp
        self.wind_speed_kts = wind_speed_kts
        self.wind_direction_deg = wind_direction_deg
        self.gust_speed_kts = gust_speed_kts
        self.wave_height_m = wave_height_m
        self.dominant_wave_period_s = dominant_wave_period_s
        self.average_wave_period_s = average_wave_period_s
        self.mean_wave_direction_deg = mean_wave_direction_deg
        self.atmospheric_pressure_hpa = atmospheric_pressure_hpa
        self.air_temperature_c = air_temperature_c
        self.water_temperature_c = water_temperature_c
        self.dewpoint_temperature_c = dewpoint_temperature_c
        self.visibility_m = visibility_m
        self.pressure_tendency_hpa = pressure_tendency_hpa
        self.tide_m = tide_m

    def __str__(self):
        return f"Observation(timestamp={self.timestamp}, wind_speed_kts={self.wind_speed_kts}, wind_direction_deg={self.wind_direction_deg}, gust_speed_kts={self.gust_speed_kts}, wave_height_m={self.wave_height_m}, dominant_wave_period_s={self.dominant_wave_period_s}, average_wave_period_s={self.average_wave_period_s}, mean_wave_direction_deg={self.mean_wave_direction_deg}, atmospheric_pressure_hpa={self.atmospheric_pressure_hpa}, air_temperature_c={self.air_temperature_c}, water_temperature_c={self.water_temperature_c}, dewpoint_temperature_c={self.dewpoint_temperature_c}, visibility_m={self.visibility_m}, pressure_tendency_hpa={self.pressure_tendency_hpa}, tide_m={self.tide_m})"


def get_observation_str_for_file(id: str, observation: Observation):
    str = f"id {id}\n"
    str += f"timestamp {observation.timestamp}\n"
    str += f"wind_speed_kts {observation.wind_speed_kts}\n"
    str += f"wind_direction_deg {observation.wind_direction_deg}\n"
    str += f"gust_speed_kts {observation.gust_speed_kts}\n"
    str += f"wave_height_m: {observation.wave_height_m}\n"
    str += f"dominant_wave_period_s {observation.dominant_wave_period_s}\n"
    str += f"average_wave_period_s {observation.average_wave_period_s}\n"
    str += f"mean_wave_direction_deg {observation.mean_wave_direction_deg}\n"
    str += f"atmospheric_pressure_hpa {observation.atmospheric_pressure_hpa}\n"
    str += f"air_temperature_c {observation.air_temperature_c}\n"
    str += f"water_temperature_c {observation.water_temperature_c}\n"
    str += f"dewpoint_temperature_c {observation.dewpoint_temperature_c}\n"
    str += f"visibility_m {observation.visibility_m}\n"
    str += f"pressure_tendency_hpa {observation.pressure_tendency_hpa}\n"
    str += f"tide_m {observation.tide_m}\n"
    return str


class BuoyData:
    def __init__(self, id):
        self.id = id
        # key is the date string, value is the observation
        self.observations: Dict[str, Observation] = {}

    def add_observation(self, observation: Observation):
        self.observations[observation.timestamp] = observation

    def has_observation(self, timestamp):
        return timestamp in self.observations

    def get_observation(self, timestamp):
        return self.observations[timestamp]


def table_row_to_observation(id, row):
    # Must have a timestamp
    if "YY" not in row or "MM" not in row or "DD" not in row or "hh" not in row or "mm" not in row:
        return None
    return Observation(
        timestamp=f'{row["YY"]}_{row["MM"]}_{row["DD"]}_{row["hh"]}{row["mm"]}',
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


def get_observation_data(id) -> BuoyData:

    with request_rate_limit_lock:
        # Simple rate limiting mechanism to be nice to the NOAA website
        time.sleep(1 / MAX_REQUESTS_PER_SECOND)

    url = f"{OBSERVATION_URL}/{id}.txt"
    observation_data = extract_table_data(url)
    if observation_data is None:
        LOGGER.warning(f"Failed to get buoy data for buoy {id}")
        return None
    data = BuoyData(id)
    for row in observation_data:
        observation = table_row_to_observation(id, row)
        if observation is not None:
            data.add_observation(observation)
    return data


def get_observations(ids: List[str]) -> Dict[str, BuoyData]:
    data = {}
    for id in ids:
        observation_data = get_observation_data(id)
        if observation_data is not None:
            data[id] = observation_data
    return data


def extract_date_string(filename):
    # Find the position of the first underscore
    start_index = filename.find("_") + 1
    if start_index == -1:
        return None
    # The end index is where ".jpg" starts
    end_index = filename.find(".jpg")
    if end_index == -1:
        return None
    # Extract the substring from start_index to end_index
    return filename[start_index:end_index]


def date_string_to_date(date_string):
    # Extract the year, month, and day from the date string. All times are in UTC.
    # Expected date format "YYYY_MM_DD_HHMM"
    if len(date_string) != 15:
        return None

    year = int(date_string[:4])
    month = int(date_string[5:7])
    day = int(date_string[8:10])
    hour = int(date_string[11:13])
    minute = int(date_string[13:15])
    return datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc)


def get_image_file(url):
    response = requests.get(url)
    if response.status_code != 200:
        LOGGER.debug(f"Failed to get image file at {url}")
        return None
    LOGGER.debug(f"Image file retrieved from {url}")
    return Image.open(BytesIO(response.content))


def date_to_date_string(date):
    return date.strftime("%Y_%m_%d_%H%M")


def split_image(img):
    sub_images = []
    for i in range(BUOYCAM_IMAGE_ROW_LENGTH):
        left = i * SUB_IMAGE_WIDTH
        sub_img = img.crop((left, 0, left + SUB_IMAGE_WIDTH, SUB_IMAGE_HEIGHT))
        sub_images.append(sub_img)
    return sub_images


# the fraction of the image that is black
def fraction_black(img):
    img_array = np.array(img.convert("L")).flatten()
    return np.count_nonzero(img_array == 0) / len(img_array)


def fetch_and_process_image(
    request: BuoyImageRequest, observation_data: Dict[str, BuoyData], ocr_reader: OCR = None
) -> bool:

    # Simple rate limiting mechanism to be nice to the NOAA buoycam website
    with request_rate_limit_lock:
        time.sleep(1 / MAX_REQUESTS_PER_SECOND)

    img_datetime_string = request.date_string()

    img_name = request.image_name()
    img_filename = request.image_filename()
    LOGGER.debug(f"\tImage filename: {img_filename}")

    # Get the image file
    img = get_image_file(request.url())
    if img is None:
        LOGGER.debug(f"\t{request} failed")
        return False

    # Verify the image size
    if img.width != IMAGE_WIDTH or img.height != IMAGE_HEIGHT:
        LOGGER.warning(f"\t{request}: image has invalid dimensions {img.width}x{img.height}")
        return False

    # Extract the sub images from the full image
    sub_images = split_image(img)

    # make the folder if it doesn't exist
    dir = f"images/{img_datetime_string}/{request.id}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save the full image
    full_image_path = f"{dir}/{img_name}_full.jpg"
    img.save(full_image_path)
    LOGGER.debug(f"\tFull image saved at {full_image_path}")

    # Save the sub images
    for i, sub_img in enumerate(sub_images):

        # Check if the sub image is mostly black
        fraction_black_value = fraction_black(sub_img)
        if fraction_black_value > FRACTION_BLACK_THRESHOLD:
            LOGGER.debug(f"\t\tSub image {i} is {fraction_black_value * 100:.2f}% black, skipping")
            continue

        LOGGER.debug(f"\t\tSub image {i} is {fraction_black_value * 100:.2f}% black")

        sub_img_path = f"{dir}/{img_name}_{i}.jpg"
        sub_img.save(sub_img_path)
        LOGGER.debug(f"\t\tSub image {i} saved at {sub_img_path}")

    # Look up the observation data
    observation_data = buoy_data.get(request.id)
    if observation_data is not None and observation_data.has_observation(img_datetime_string):
        observation = observation_data.get_observation(img_datetime_string)
        LOGGER.debug(f"\tObservation for buoy {request.id} at {img_datetime_string}: {observation}")
        # Save the observation data
        observation_path = f"{dir}/observation.txt"
        with open(observation_path, "w") as file:
            file.write(get_observation_str_for_file(request.id, observation))
    else:
        LOGGER.warning(f"\tNo observation data for buoy {request.id} at {img_datetime_string}")

    if ocr_reader is not None:
        angle_crop = img.crop((150, img.height - 30, 250, img.height))
        # plt.imshow(angle_crop)
        # plt.axis("off")
        # plt.show()

        # Extract the angle from the image
        angle = ocr_reader.get_angle_from_image(angle_crop)
        if angle is None:
            LOGGER.warning(f"\tFailed to extract angle from buoycam {station_id}: {station_name}")
        else:
            LOGGER.debug(f"\tExtracted angle: {angle}")
            # Save the angle
            angle_path = f"{dir}/angle.txt"
            with open(angle_path, "w") as file:
                file.write(str(angle))

    return True


def generate_requests(buoy_cam_list) -> list[BuoyImageRequest]:
    image_requests = []
    for buoy_cam in buoy_cam_list:
        if "id" not in buoy_cam or "name" not in buoy_cam or "img" not in buoy_cam:
            LOGGER.error(f"Buoycam does not have all required fields. Invalid dictionary: {buoy_cam}")
            continue

        station_id = buoy_cam["id"]
        if station_id is None:
            LOGGER.error(f"Buoycam does not have an id. Invalid dictionary: {buoy_cam}")
            continue

        img_filename = buoy_cam["img"]
        if img_filename is None:
            LOGGER.warning(f"Buoycam {station_id}: does not have an image")
            continue

        # before the first '_"
        station_tag = img_filename.split("_")[0]

        img_datetime_string = extract_date_string(img_filename)

        if img_datetime_string is None:
            LOGGER.error(f"Buoycam {station_id}: has an invalid image filename: {img_filename}")
            continue

        latest_img_date = date_string_to_date(img_datetime_string)

        if latest_img_date is None:
            LOGGER.error(f"Buoycam {station_id}: has an invalid image date: {img_datetime_string}")
            continue

        image_dates = [
            latest_img_date - datetime.timedelta(minutes=10 * i) for i in range(NUMBER_OF_HOURS_TO_GET_IMAGES_FOR * 6)
        ]

        image_requests.extend([BuoyImageRequest(station_id, station_tag, date) for date in image_dates])
    return image_requests


def nowutc():
    return datetime.datetime.now(datetime.UTC)


if __name__ == "__main__":
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.info("Starting buoycam fetcher")

    timer = nowutc()

    buoy_cam_list = get_json(BUOYCAM_LIST_URL)

    LOGGER.info(f"Found {len(buoy_cam_list)} buoycams")

    # Observation look up object
    # buoy_data = get_observations([buoy["id"] for buoy in buoy_cam_list if "id" in buoy])
    buoy_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_to_id = {
            executor.submit(get_observation_data, buoy["id"]): buoy for buoy in buoy_cam_list if "id" in buoy
        }

        for future in as_completed(futures_to_id):
            data = future.result()
            id = futures_to_id[future]
            LOGGER.info(f"Completed observation request for {id}, success: {data is not None}")

            if data is not None:
                buoy_data[data.id] = data

    LOGGER.debug(f"Completed all observation requests in {nowutc() - timer}")

    LOGGER.info(f"Retrieved observation data for {len(buoy_data)} buoycams")

    ocr_reader = None  # OCR()
    oldest_image_date = None
    success_count = 0
    fault_count = 0

    # Generate image requests for all buoycams that have observation data
    image_requests = generate_requests([b for b in buoy_cam_list if b["id"] in buoy_data])

    LOGGER.info(f"Generated {len(image_requests)} image requests")

    timer = nowutc()

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_to_request = {
            executor.submit(fetch_and_process_image, request, ocr_reader): request for request in image_requests
        }

        for future in as_completed(futures_to_request):
            request = futures_to_request[future]
            results.append((request, future.result()))
            LOGGER.info(f"Completed {request}, success: {future.result()}")

    LOGGER.info(f"Completed all image requests in {nowutc() - timer}")
