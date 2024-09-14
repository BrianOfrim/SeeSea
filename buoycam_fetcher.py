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
from dataclasses import dataclass

# Retrieve, process and save buoycam images from the NOAA buoycam website

# util functions
BUOYCAM_LIST_URL = "https://www.ndbc.noaa.gov/buoycams.php"
BUOYCAM_IMAGE_FILE_URL_BASE = "https://www.ndbc.noaa.gov/images/buoycam/"
BUOYCAM_IMAGE_ROW_LENGTH = 6
IMAGE_WIDTH = 2880
IMAGE_HEIGHT = 300
SUB_IMAGE_WIDTH = 480
SUB_IMAGE_HEIGHT = 270

FRACTION_BLACK_THRESHOLD = 0.85


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


def fetch_and_process_image(request: BuoyImageRequest, ocr_reader: None):

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

        image_dates = [latest_img_date - datetime.timedelta(minutes=10 * i) for i in range(73 * 6)]

        image_requests.extend([BuoyImageRequest(station_id, station_tag, date) for date in image_dates])
    return image_requests


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.info("Starting buoycam fetcher")

    timer = datetime.datetime.now()

    buoy_cam_list = get_json(BUOYCAM_LIST_URL)

    LOGGER.info(f"Found {len(buoy_cam_list)} buoycams")

    ocr_reader = None  # OCR()
    now = datetime.datetime.now(datetime.timezone.utc)

    oldest_image_date = None
    success_count = 0
    fault_count = 0

    image_requests = generate_requests(buoy_cam_list)

    LOGGER.info(f"Generated {len(image_requests)} image requests")

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_to_request = {
            executor.submit(fetch_and_process_image, request, ocr_reader): request for request in image_requests
        }

        for future in as_completed(futures_to_request):
            request = futures_to_request[future]
            results.append((request, future.result()))
            LOGGER.info(f"Completed {request}, success: {future.result()}")

    LOGGER.info(f"Completed all requests in {datetime.datetime.now() - timer}")
