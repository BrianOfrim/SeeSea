import json


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

    @staticmethod
    def from_json(json_str: str) -> "Observation":
        return Observation(**json.loads(json_str))
