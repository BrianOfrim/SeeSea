import unittest
from datetime import datetime, timedelta
from seesea.buoycam_fetcher import extend_to_past, BuoyInfo, BuoyPosition


class TestExtendToPast(unittest.TestCase):

    class MockBuoyInfo:
        def __init__(self, id, date):
            self.id = id
            self.date = date
            self.tag = "tag"
            self.description = "description"
            self.position = BuoyPosition(1.0, 2.0)

        def save_directory(self, output_dir):
            return f"{output_dir}/{self.id}"

        def image_full_path(self, output_dir, suffix):
            return f"{self.save_directory(output_dir)}/{suffix}.jpg"

    def test_extend_to_past_no_hours_full_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 0, [0, 10, 20, 30, 40, 50])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].date, datetime(2023, 10, 1, 12, 30))

    def test_extend_to_past_no_hours_partial_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 0, [0, 50])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].date, datetime(2023, 10, 1, 12, 30))

    def test_extend_to_past_one_hour_full_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 1, [0, 30, 20, 10, 40, 50])
        expected_dates = [
            datetime(2023, 10, 1, 12, 30),
            datetime(2023, 10, 1, 12, 20),
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 50),
            datetime(2023, 10, 1, 11, 40),
            datetime(2023, 10, 1, 11, 30),
        ]

        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_one_hour_partial_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 1, [0, 10, 40, 50])
        expected_dates = [
            datetime(2023, 10, 1, 12, 30),
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 50),
            datetime(2023, 10, 1, 11, 40),
        ]

        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_two_hours_full_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 2, [30, 40, 10])
        expected_dates = [
            datetime(2023, 10, 1, 12, 30),
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 11, 40),
            datetime(2023, 10, 1, 11, 30),
            datetime(2023, 10, 1, 11, 10),
            datetime(2023, 10, 1, 10, 40),
            datetime(2023, 10, 1, 10, 30),
        ]
        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_three_hours_full_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 3, [0, 10, 20, 30, 40, 50])
        expected_dates = [
            datetime(2023, 10, 1, 12, 30),
            datetime(2023, 10, 1, 12, 20),
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 50),
            datetime(2023, 10, 1, 11, 40),
            datetime(2023, 10, 1, 11, 30),
            datetime(2023, 10, 1, 11, 20),
            datetime(2023, 10, 1, 11, 10),
            datetime(2023, 10, 1, 11, 0),
            datetime(2023, 10, 1, 10, 50),
            datetime(2023, 10, 1, 10, 40),
            datetime(2023, 10, 1, 10, 30),
            datetime(2023, 10, 1, 10, 20),
            datetime(2023, 10, 1, 10, 10),
            datetime(2023, 10, 1, 10, 0),
            datetime(2023, 10, 1, 9, 50),
            datetime(2023, 10, 1, 9, 40),
            datetime(2023, 10, 1, 9, 30),
        ]
        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_three_hours_partial_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 30))]
        result = extend_to_past(latest_list, 3, [0, 10])
        expected_dates = [
            datetime(2023, 10, 1, 12, 30),
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 10),
            datetime(2023, 10, 1, 11, 0),
            datetime(2023, 10, 1, 10, 10),
            datetime(2023, 10, 1, 10, 0),
        ]
        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_five_hours_full_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 10))]
        result = extend_to_past(latest_list, 5, [0, 20, 10, 50, 30, 40])
        expected_dates = [
            datetime(2023, 10, 1, 12, 10),
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 50),
            datetime(2023, 10, 1, 11, 40),
            datetime(2023, 10, 1, 11, 30),
            datetime(2023, 10, 1, 11, 20),
            datetime(2023, 10, 1, 11, 10),
            datetime(2023, 10, 1, 11, 0),
            datetime(2023, 10, 1, 10, 50),
            datetime(2023, 10, 1, 10, 40),
            datetime(2023, 10, 1, 10, 30),
            datetime(2023, 10, 1, 10, 20),
            datetime(2023, 10, 1, 10, 10),
            datetime(2023, 10, 1, 10, 0),
            datetime(2023, 10, 1, 9, 50),
            datetime(2023, 10, 1, 9, 40),
            datetime(2023, 10, 1, 9, 30),
            datetime(2023, 10, 1, 9, 20),
            datetime(2023, 10, 1, 9, 10),
            datetime(2023, 10, 1, 9, 0),
            datetime(2023, 10, 1, 8, 50),
            datetime(2023, 10, 1, 8, 40),
            datetime(2023, 10, 1, 8, 30),
            datetime(2023, 10, 1, 8, 20),
            datetime(2023, 10, 1, 8, 10),
            datetime(2023, 10, 1, 8, 0),
            datetime(2023, 10, 1, 7, 50),
            datetime(2023, 10, 1, 7, 40),
            datetime(2023, 10, 1, 7, 30),
            datetime(2023, 10, 1, 7, 20),
            datetime(2023, 10, 1, 7, 10),
        ]
        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)

    def test_extend_to_past_five_hours_partial_minute_list(self):
        latest_list = [self.MockBuoyInfo("buoy1", datetime(2023, 10, 1, 12, 00))]
        result = extend_to_past(latest_list, 5, [0, 10])
        expected_dates = [
            datetime(2023, 10, 1, 12, 0),
            datetime(2023, 10, 1, 11, 10),
            datetime(2023, 10, 1, 11, 0),
            datetime(2023, 10, 1, 10, 10),
            datetime(2023, 10, 1, 10, 0),
            datetime(2023, 10, 1, 9, 10),
            datetime(2023, 10, 1, 9, 0),
            datetime(2023, 10, 1, 8, 10),
            datetime(2023, 10, 1, 8, 0),
            datetime(2023, 10, 1, 7, 10),
            datetime(2023, 10, 1, 7, 0),
        ]
        self.assertEqual(len(result), len(expected_dates))
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result[i].date, expected_date)


if __name__ == "__main__":
    unittest.main()
