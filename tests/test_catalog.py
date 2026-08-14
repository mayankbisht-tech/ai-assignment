"""tests/test_catalog.py — Tests for dynamic catalog extraction and matching."""

import unittest
from app import (
    detect_out_of_catalog_destination,
    get_catalog_locations,
    get_catalog_regions,
    load_data,
    resolve_destination,
)


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_dynamic_locations_extracted(self):
        locs = get_catalog_locations(self.data)
        self.assertIn("Alleppey", locs)
        self.assertIn("Kochi", locs)
        self.assertIn("Munnar", locs)

    def test_destination_resolution_and_aliases(self):
        # Direct
        self.assertEqual(resolve_destination("I want to visit Munnar", self.data), "Munnar")
        self.assertEqual(resolve_destination("Kochi trip", self.data), "Kochi")

        # Alias
        self.assertEqual(resolve_destination("Trip to Cochin", self.data), "Kochi")
        self.assertEqual(resolve_destination("Alappuzha backwaters", self.data), "Alleppey")

        # Region
        self.assertEqual(resolve_destination("Kerala holiday", self.data), "Kerala")

    def test_out_of_catalog_detection(self):
        self.assertEqual(detect_out_of_catalog_destination("Plan trip to Goa", self.data), "Goa")
        self.assertEqual(detect_out_of_catalog_destination("Visit Paris next week", self.data), "Paris")
        self.assertIsNone(detect_out_of_catalog_destination("Visit Munnar", self.data))


if __name__ == "__main__":
    unittest.main()
