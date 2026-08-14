"""tests/test_validator.py — Tests for grounding and math validation logic."""

import unittest
from app import load_data, validate_itinerary


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_valid_itinerary_passes(self):
        valid = {
            "status": "fulfilled",
            "destination": "Munnar",
            "days": [
                {
                    "day": 1,
                    "title": "Day 1",
                    "hotel": {"id": "HOT-004", "name": "Munnar Hikers Hostel"},
                    "activities": [{"id": "ACT-002", "name": "Munnar Tea Estate Walk & Tasting"}],
                }
            ],
            "quote": {
                "line_items": [
                    {
                        "catalog_id": "HOT-004",
                        "name": "Munnar Hikers Hostel",
                        "unit_price_inr": 1500.0,
                        "quantity": 1,
                        "line_total_inr": 1500.0,
                    },
                    {
                        "catalog_id": "ACT-002",
                        "name": "Munnar Tea Estate Walk & Tasting",
                        "unit_price_inr": 900.0,
                        "quantity": 2,
                        "line_total_inr": 1800.0,
                    },
                ],
                "total": 3300.0,
            },
        }
        errors = validate_itinerary(valid, self.data)
        self.assertEqual(errors, [])

    def test_fake_id_detected(self):
        broken = {
            "days": [],
            "quote": {
                "line_items": [
                    {
                        "catalog_id": "FAKE-999",
                        "name": "Fake Hotel",
                        "unit_price_inr": 1000.0,
                        "quantity": 1,
                        "line_total_inr": 1000.0,
                    }
                ],
                "total": 1000.0,
            },
        }
        errors = validate_itinerary(broken, self.data)
        self.assertTrue(any("FAKE-999" in e and "does not exist" in e for e in errors))

    def test_price_mismatch_detected(self):
        broken = {
            "days": [],
            "quote": {
                "line_items": [
                    {
                        "catalog_id": "HOT-001",
                        "name": "Backwater Breeze Homestay",
                        "unit_price_inr": 9999.0,  # Real price is 3200
                        "quantity": 1,
                        "line_total_inr": 9999.0,
                    }
                ],
                "total": 9999.0,
            },
        }
        errors = validate_itinerary(broken, self.data)
        self.assertTrue(any("unit_price_inr mismatch" in e for e in errors))

    def test_total_mismatch_detected(self):
        broken = {
            "days": [],
            "quote": {
                "line_items": [
                    {
                        "catalog_id": "HOT-001",
                        "name": "Backwater Breeze Homestay",
                        "unit_price_inr": 3200.0,
                        "quantity": 2,
                        "line_total_inr": 6400.0,
                    }
                ],
                "total": 10000.0,  # Should be 6400
            },
        }
        errors = validate_itinerary(broken, self.data)
        self.assertTrue(any("total_inr mismatch" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
