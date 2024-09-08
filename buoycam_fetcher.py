
import requests
from PIL import Image, ImageStat
from io import BytesIO
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import easyocr
import logging

# Retrieve, process and save buoycam images from the NOAA buoycam website

# util functions
BUOYCAM_LIST_URL = "https://www.ndbc.noaa.gov/buoycams.php"
BUOYCAM_IMAGE_FILE_URL_BASE = "https://www.ndbc.noaa.gov/images/buoycam/"
BUOYCAM_IMAGE_ROW_LENGTH = 6
IMAGE_WIDTH = 2880
IMAGE_HEIGHT = 300
SUB_IMAGE_WIDTH = 480
SUB_IMAGE_HEIGHT = 270

LOGGER = logging.getLogger(__name__)

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


def get_json(url):
    response = requests.get(url)
    return response.json()


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


def get_buoycam_image_file(image_filename):
    url = BUOYCAM_IMAGE_FILE_URL_BASE + image_filename
    response = requests.get(url)
    if response.status_code != 200:
        LOGGER.warn(f"Failed to get image file {image_filename}")
        return None
    img = Image.open(BytesIO(response.content))
    return img


if __name__ == '__main__':
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.info("Starting buoycam fetcher")

    buoy_cam_list = get_json(BUOYCAM_LIST_URL)

    LOGGER.info(f"Found {len(buoy_cam_list)} buoycams")

    ocr_reader = OCR()

    image_count = 0


    for buoy_cam in buoy_cam_list:
        if "id" not in buoy_cam or "name" not in buoy_cam or "img" not in buoy_cam:
            LOGGER.error(f"Buoycam does not have all required fields. Invalid dictionary: {buoy_cam}")
            continue

        station_id = buoy_cam["id"]
        if station_id is None:
            LOGGER.error(f"Buoycam does not have an id. Invalid dictionary: {buoy_cam}")
            continue

        name = buoy_cam["name"]
        if name is None:
            LOGGER.error(f"Buoycam {station_id} does not have a name. Invalid dictionary: {buoy_cam}")
            continue
        img_filename = buoy_cam["img"]

        if img_filename is None:
            LOGGER.warn(f"Buoycam {station_id}: {name} does not have an image")
            continue

        img_name = img_filename.split(".")[0]
        img_datetime_string = extract_date_string(img_filename)
        
        if img_datetime_string is None:
            LOGGER.error(f"Buoycam {station_id}: {name} has an invalid image filename: {img_filename}")
            continue

        img_date = date_string_to_date(img_datetime_string)

        LOGGER.info(f"Processing buoycam {station_id}: {name} at {img_date}")

        # Get the image file
        img = get_buoycam_image_file(img_filename)
        if img is None:
            LOGGER.error(f"Buoycam {station_id}: {name} failed to get image file")
            continue

        image_count += 1

        # Display the full image
        # plt.title(f"{station_id}")
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

        # Verify the image size
        if img.width != IMAGE_WIDTH or img.height != IMAGE_HEIGHT:
            LOGGER.error(f"Buoycam {station_id}: {name} image has invalid dimensions {img.width}x{img.height}")
            continue

        # Extract the sub images from the full image
        sub_images = []
        for i in range(BUOYCAM_IMAGE_ROW_LENGTH):
            left = i * SUB_IMAGE_WIDTH
            sub_img = img.crop((left, 0, left + SUB_IMAGE_WIDTH, SUB_IMAGE_HEIGHT))
            sub_images.append(sub_img)

        # make the folder if it doesn't exist
        dir = f"images/{img_datetime_string}/{station_id}"
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Save the full image
        full_image_path = f"{dir}/{img_name}_full.jpg"
        img.save(full_image_path)
        LOGGER.debug(f"Full image saved at {full_image_path}")

        # Save the sub images
        for i, sub_img in enumerate(sub_images):
            mean = ImageStat.Stat(sub_img.convert("L")).mean
            LOGGER.debug(f"Sub image {i} mean gray value: {mean}")

            # Run OCR on the sub image
            image_text = ocr_reader.get_all_text_from_image(sub_img)

            # Check if the OCR results contain the text "no image", case insensitive
            if image_text is not None and "no image" in image_text.lower():
                LOGGER.warn(f"Sub image {i} for buoy {station_id} contains no image")
                continue

            sub_img_path = f"{dir}/{img_name}_{i}.jpg"
            sub_img.save(sub_img_path)
            LOGGER.debug(f"Sub image {i} saved at {sub_img_path}")

        angle_crop = img.crop((150, img.height - 30, 250, img.height))
        # plt.imshow(angle_crop)
        # plt.axis("off")
        # plt.show()

        # Extract the angle from the image
        angle = ocr_reader.get_angle_from_image(angle_crop)
        if angle is None:
            LOGGER.warn(f"Failed to extract angle from buoycam {station_id}: {name}")
        else:
            LOGGER.debug(f"Extracted angle: {angle}")
            # Save the angle
            angle_path = f"{dir}/angle.txt"
            with open(angle_path, "w") as file:
                file.write(str(angle))

    LOGGER.info(f"Processed buoycam images for {image_count} buoys")