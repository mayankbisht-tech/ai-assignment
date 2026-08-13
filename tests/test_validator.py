import json
import os
import pytest

from app import load_data, validate_node


def make_state(catalog, itinerary, destination_known=True):
    state = {
        "itinerary": itinerary,
        "catalog": catalog,
        "all_ids": {item["id"] for item in catalog},
        "destination_known": destination_known,
    }
    return state


def test_unfulfillable_when_destination_unknown():
    data = load_data()
    catalog = data["suppliers"]
    itinerary = {"status": "ok", "days": [], "quote": {"line_items": [], "total": 0}, "cited_ids": []}
    state = make_state(catalog, itinerary, destination_known=False)
    out = validate_node(state)
    assert out["validation_errors"], "Should flag error when destination unknown but status ok"


def test_price_arithmetic_and_grounding_valid():
    data = load_data()
    catalog = data["suppliers"]
    # create a simple valid itinerary referencing HOT-001 and ACT-001
    line_items = [
        {"id": "HOT-001", "name": "Backwater Breeze Homestay", "unit_price": 3200, "quantity": 2, "subtotal": 6400},
        {"id": "ACT-001", "name": "Alleppey Houseboat Day Cruise", "unit_price": 2200, "quantity": 4, "subtotal": 8800},
    ]
    itinerary = {
        "status": "ok",
        "destination": "Alleppey",
        "days": [],
        "quote": {"line_items": line_items, "total": 15200},
        "cited_ids": ["HOT-001", "ACT-001"]
    }
    state = make_state(catalog, itinerary, destination_known=True)
    out = validate_node(state)
    assert not out["validation_errors"], f"Validation failed: {out['validation_errors']}"


def test_subtotal_mismatch_detected():
    data = load_data()
    catalog = data["suppliers"]
    line_items = [
        {"id": "HOT-001", "name": "Backwater Breeze Homestay", "unit_price": 3200, "quantity": 2, "subtotal": 6300},
    ]
    itinerary = {"status": "ok", "destination": "Alleppey", "days": [], "quote": {"line_items": line_items, "total": 6300}, "cited_ids": ["HOT-001"]}
    state = make_state(catalog, itinerary, destination_known=True)
    out = validate_node(state)
    assert out["validation_errors"], "Should detect subtotal mismatch"
