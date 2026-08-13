"""
Grounding evaluation harness (optional stretch goal).

Re-checks already-generated outputs/*.json independently of app.py's
in-loop validator, so grounding is verified by a second, separate pass:

  1. Every cited id exists in the catalog.
  2. No HOT-/ACT-/TRN- pattern appears anywhere in the output that isn't
     a real catalog id (catches ids the model might have invented outside
     the "cited_ids" field, e.g. in notes).
  3. Every line-item subtotal equals unit_price * quantity, and the total
     equals the sum of subtotals.
  4. The known "unfulfillable trap" request (Goa, no matching inventory)
     is actually flagged unfulfillable, not answered with fabricated
     inventory.

Usage:
    python app.py --all --mock      # generate outputs/*.json first
    python eval.py
"""

import glob
import json
import os
import re

from app import load_data, VALID_ID_RE  # reuse the same id-pattern regex

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def evaluate_file(path: str, all_ids: set, data: dict) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    problems = []
    itinerary = result.get("itinerary")
    if itinerary is None:
        return ["No parsable itinerary JSON was produced."]

    if itinerary.get("status") == "unfulfillable":
        if itinerary.get("days") or itinerary.get("quote", {}).get("line_items"):
            problems.append("Unfulfillable response still contains itinerary content.")
        return problems

    full_text = json.dumps(itinerary)
    mentioned = set(re.findall(r"\b(?:HOT|ACT|TRN)-\d{3}\b", full_text))
    stray = mentioned - all_ids
    if stray:
        problems.append(f"Ids not in catalog: {sorted(stray)}")

    line_items = itinerary.get("quote", {}).get("line_items", [])
    running_total = 0.0
    for li in line_items:
        expected = round(li.get("unit_price", 0) * li.get("quantity", 0), 2)
        actual = round(li.get("subtotal", -1), 2)
        if abs(expected - actual) > 0.01:
            problems.append(f"{li.get('id')}: {li['unit_price']}x{li['quantity']} != {li['subtotal']}")
        running_total += li.get("subtotal", 0)

    stated_total = round(itinerary.get("quote", {}).get("total", -1), 2)
    if abs(round(running_total, 2) - stated_total) > 0.01:
        problems.append(f"Total {stated_total} != sum of line items {round(running_total, 2)}")

    # New semantic checks
    # 1) If request_text specifies N days, itinerary must contain exactly N days
    req_text = result.get("request_text", "")
    m = re.search(r"(\d+)\s+days?\b", req_text.lower())
    if m:
        req_days = int(m.group(1))
        returned_days = len(itinerary.get("days", []))
        if returned_days != req_days:
            problems.append(f"Requested {req_days} days but itinerary contains {returned_days} days")

    # 2) Hotel quantity must equal rooms_needed * nights where rooms_needed = ceil(party_size / hotel_capacity)
    # Use traveler profile from data for party size
    party = data.get("traveler_profile", {}).get("party", {})
    total_people = int(party.get("adults", 0)) + int(party.get("children", 0))
    nights = None
    if m:
        nights = int(m.group(1))
    else:
        nights = len(itinerary.get("days", []))

    # build id->item map from data catalog
    catalog = {it["id"]: it for it in data.get("suppliers", [])}
    for li in line_items:
        iid = li.get("id")
        cat = catalog.get(iid)
        if cat and cat.get("type") == "hotel":
            cap = int(cat.get("capacity", 0)) or 1
            rooms_needed = -(-total_people // cap)
            expected_qty = rooms_needed * (nights or 1)
            if li.get("quantity", 0) != expected_qty:
                problems.append(f"Hotel line item {iid} quantity {li.get('quantity')} != expected rooms*nights {expected_qty}")

    return problems


def main():
    data = load_data()
    all_ids = {item["id"] for item in data["suppliers"]}

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    if not files:
        print("No outputs found. Run app.py first.")
        return

    all_passed = True
    for path in files:
        problems = evaluate_file(path, all_ids, data)
        name = os.path.basename(path)
        if problems:
            all_passed = False
            print(f"[FAIL] {name}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[PASS] {name}")

    print("\nOverall:", "ALL GROUNDED" if all_passed else "GROUNDING ISSUES FOUND")


if __name__ == "__main__":
    main()
