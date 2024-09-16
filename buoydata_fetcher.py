import requests
import logging
import sqlite3
import json
import os

# Retrieve and process buoy data from the NOAA buoy website
# Save the data in a sqlite database
BUOYCAM_LIST_URL = "https://www.ndbc.noaa.gov/buoycams.php"
OBSERVATION_URL = "https://www.ndbc.noaa.gov/data/realtime2"

MISSING_DATA_INDICATOR = "MM"

CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS buoydata (
            station_id TEXT,
            timestamp TEXT,
            wind_speed_kts REAL,
            wind_direction_deg REAL,
            gust_speed_kts REAL,
            wave_height_m REAL,
            dominant_wave_period_s REAL,
            average_wave_period_s REAL,
            mean_wave_direction_deg REAL,
            atmospheric_pressure_hpa REAL,
            air_temperature_c REAL,
            water_temperature_c REAL,
            dewpoint_temperature_c REAL,
            visibility_m REAL,
            pressure_tendency_hpa REAL,
            tide_m REAL,
            PRIMARY KEY (station_id, timestamp)
        )
    """


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


def get_json(url: str) -> dict:
    response = requests.get(url)
    # verify that the request was successful
    if response.status_code != 200:
        return None
    return response.json()


def mps_to_kts(mps: float) -> float:
    # Convert meters per second to knots
    return mps * 1.94384


def nmi_to_m(nmi: float) -> float:
    # Convert nautical miles to meters
    return nmi * 1852


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


if __name__ == "__main__":

    LOGGER = logging.getLogger(__name__)

    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler())

    LOGGER.info("Starting buoy data fetcher")

    buoycam_list = get_json(BUOYCAM_LIST_URL)
    if buoycam_list is None:
        LOGGER.critical("Failed to get buoycam list")
        exit()

    # Make a data directory if it does not exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Create a connection to the database
    conn = sqlite3.connect("data/buoydata.db")
    cursor = conn.cursor()

    # Create the table if it does not exist
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()

    LOGGER.info(f"Found {len(buoycam_list)} buoycams")

    for buoy in buoycam_list:
        id = buoy["id"]
        buoy_data = extract_table_data(f"{OBSERVATION_URL}/{id}.txt")
        if buoy_data is None:
            LOGGER.warning(f"Failed to get buoy data for buoy {id}")
            continue

        db_entries = [table_row_to_db_entry(row, id) for row in buoy_data]

        LOGGER.info(f"Buoy: {id}, inserting {len(db_entries)} entries into the database")

        # Insert the data into the database
        for entry in db_entries:
            cursor.execute(
                """
                INSERT OR REPLACE INTO buoydata
                (station_id, timestamp, wind_speed_kts, wind_direction_deg, gust_speed_kts, wave_height_m,
                dominant_wave_period_s, average_wave_period_s, mean_wave_direction_deg, atmospheric_pressure_hpa,
                air_temperature_c, water_temperature_c, dewpoint_temperature_c, visibility_m, pressure_tendency_hpa,
                tide_m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["station_id"],
                    entry["timestamp"],
                    entry["wind_speed_kts"],
                    entry["wind_direction_deg"],
                    entry["gust_speed_kts"],
                    entry["wave_height_m"],
                    entry["dominant_wave_period_s"],
                    entry["average_wave_period_s"],
                    entry["mean_wave_direction_deg"],
                    entry["atmospheric_pressure_hpa"],
                    entry["air_temperature_c"],
                    entry["water_temperature_c"],
                    entry["dewpoint_temperature_c"],
                    entry["visibility_m"],
                    entry["pressure_tendency_hpa"],
                    entry["tide_m"],
                ),
            )

    conn.commit()
    conn.close()
