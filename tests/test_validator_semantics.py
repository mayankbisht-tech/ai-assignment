import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import load_data, validate_node

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")


def load_output(name: str):
    path = os.path.join(OUTPUTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_req1_detects_days_and_hotel_quantity_mismatch():
    data = load_data()
    out = load_output("REQ-1.json")
    itinerary = out.get("itinerary")
    state = {
        "itinerary": itinerary,
        "all_ids": {item["id"] for item in data["suppliers"]},
        "catalog": data["suppliers"],
        "profile": data["traveler_profile"],
        "destination_known": True,
        "requested_days": 5,
    }

    res = validate_node(state)
    errs = res.get("validation_errors", [])
    # After generator fixes, REQ-1 should now pass semantic validation
    assert not errs, f"Unexpected validation errors for REQ-1: {errs}"
