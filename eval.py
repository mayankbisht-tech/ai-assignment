from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from helpers import load_env
load_env()
from app import load_data, validate_itinerary

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"

# Deliberately broken itinerary for self-test.
# Errors baked in: (a) GOA-999 non-existent, (b) wrong name for HOT-001,
# (c) wrong unit_price_inr, (d) wrong line_total_inr, (e) total mismatch,
# (f) ACT-099 cited in days but no line item.
_BROKEN = {
    "days": [{"day": 1, "title": "Fabricated Goa beach day.",
               "catalog_ids": ["GOA-999", "ACT-099"]}],
    "quote": {
        "line_items": [
            {"catalog_id": "GOA-999", "name": "Goa Beach Paradise",
             "unit_price_inr": 9999, "quantity": 3, "line_total_inr": 29000},
            {"catalog_id": "HOT-001", "name": "Wrong Name Hotel",
             "unit_price_inr": 5000, "quantity": 2, "line_total_inr": 10000},
        ],
        "total": 50000,
    },
}

_CHECKS = [
    ("non-existent catalog_id GOA-999",  lambda e: "GOA-999" in e and "does not exist" in e),
    ("name mismatch for HOT-001",        lambda e: "name mismatch" in e and "HOT-001" in e),
    ("unit_price mismatch for HOT-001",  lambda e: "unit_price_inr mismatch" in e and "HOT-001" in e),
    ("line_total mismatch for HOT-001",  lambda e: "line_total_inr mismatch" in e and "HOT-001" in e),
    ("grand total mismatch",             lambda e: "total_inr mismatch" in e),
    ("ACT-099 missing from line_items",  lambda e: "ACT-099" in e),
]

SEP = "=" * 60


def validate_output_file(path: Path, data: dict) -> tuple[bool, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    itin = payload.get("itinerary")
    if itin is None:
        status = payload.get("status", "")
        if status in ("declined_no_inventory", "unfulfillable"):
            return True, []
        return False, ["itinerary is null but status is not unfulfillable/declined"]
    errors = validate_itinerary(itin, data)
    return len(errors) == 0, errors


def main() -> None:
    data = load_data()
    overall_pass = True

    print(f"\n{SEP}\nREAL OUTPUT VALIDATION\n{SEP}")
    files = sorted(OUTPUTS_DIR.glob("*.json"))
    if not files:
        print("⚠  No files in outputs/.")
    for fpath in files:
        passed, errors = validate_output_file(fpath, data)
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'}  {fpath.name}")
        if errors:
            for e in errors: print(f"     • {e}")
            overall_pass = False
        else:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
            itin = payload.get("itinerary") or {}
            quote = itin.get("quote") or {}
            tot = quote.get("total", itin.get("total_inr", "n/a"))
            status = payload.get("status") or itin.get("status", "ok")
            print(f"     status={status}  total=₹{tot if tot != 'n/a' else 'n/a'}")

    print(f"\n{SEP}\nBROKEN ITINERARY SELF-TEST (must catch all error types)\n{SEP}")
    detected = validate_itinerary(_BROKEN, data)
    self_test_pass = True
    for label, pred in _CHECKS:
        caught = any(pred(e) for e in detected)
        print(f"  {'✅' if caught else '❌ MISSED'}  {label}")
        if not caught: self_test_pass = False
    print("\n  All detected errors:")
    for e in detected: print(f"    • {e}")
    if not self_test_pass:
        overall_pass = False

    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    print(f"  Real outputs    : {'ALL PASS ✅' if overall_pass else 'SOME FAILED ❌'}")
    print(f"  Broken self-test: {'ALL ERRORS CAUGHT ✅' if self_test_pass else 'SOME MISSED ❌'}")
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
