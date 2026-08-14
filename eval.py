"""eval.py — Evaluation Harness: Re-validate output files + broken itinerary self-test.

Usage:
    python eval.py

Checks:
  1. Validates generated output files in outputs/ against catalog rules & prices.
  2. Runs a hand-crafted broken itinerary exercising ALL error-detection paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure Unicode output works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from helpers import load_env
load_env()

from app import load_data, validate_itinerary

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"

# ── Deliberately broken itinerary ─────────────────────────────────────────
# Errors baked in:
#   (a) GOA-999 — non-existent catalog_id
#   (b) Wrong name for HOT-001
#   (c) Wrong unit_price_inr for HOT-001 (should be 3200)
#   (d) Wrong line_total_inr for HOT-001 (should be 3200 × 2 = 6400)
#   (e) total_inr doesn't match sum of line totals
#   (f) ACT-099 cited in days but has no line item and doesn't exist

_BROKEN = {
    "days": [
        {
            "day": 1,
            "title": "Fabricated Goa beach day.",
            "catalog_ids": ["GOA-999", "ACT-099"],   # (a) fake id, (f) no line item
        }
    ],
    "quote": {
        "line_items": [
            {
                "catalog_id": "GOA-999",            # (a) non-existent
                "name": "Goa Beach Paradise",        # fabricated
                "unit_price_inr": 9999,
                "quantity": 3,
                "line_total_inr": 29000,             # (d) wrong (9999×3=29997)
            },
            {
                "catalog_id": "HOT-001",
                "name": "Wrong Name Hotel",          # (b) name mismatch
                "unit_price_inr": 5000,              # (c) should be 3200
                "quantity": 2,
                "line_total_inr": 10000,             # (d) should be 6400
            },
        ],
        "total": 50000,                              # (e) doesn't match sum
    },
    "notes": "Fabricated itinerary for harness self-test.",
}

_EXPECTED_ERROR_CHECKS = [
    ("non-existent catalog_id GOA-999",     lambda e: "GOA-999" in e and "does not exist" in e),
    ("name mismatch for HOT-001",           lambda e: "name mismatch" in e and "HOT-001" in e),
    ("unit_price mismatch for HOT-001",     lambda e: "unit_price_inr mismatch" in e and "HOT-001" in e),
    ("line_total mismatch for HOT-001",     lambda e: "line_total_inr mismatch" in e and "HOT-001" in e),
    ("grand total mismatch",                lambda e: "total_inr mismatch" in e),
    ("ACT-099 missing from line_items",     lambda e: "ACT-099" in e),
]


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

    # ── 1. Real output files ──────────────────────────────────────────────
    files = sorted(OUTPUTS_DIR.glob("*.json"))
    print(f"\n{'=' * 60}")
    print("REAL OUTPUT VALIDATION")
    print(f"{'=' * 60}")

    if not files:
        print("⚠  No files in outputs/.")
    else:
        for fpath in files:
            passed, errors = validate_output_file(fpath, data)
            icon = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{icon}  {fpath.name}")
            if errors:
                for e in errors:
                    print(f"     • {e}")
                overall_pass = False
            else:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
                itin = payload.get("itinerary") or {}
                quote = itin.get("quote") or {}
                tot = quote.get("total", itin.get("total_inr", "n/a"))
                status = payload.get("status") or itin.get("status", "ok")
                print(f"     status={status}  total=₹{tot if tot != 'n/a' else 'n/a'}")

    # ── 2. Broken itinerary self-test ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("BROKEN ITINERARY SELF-TEST (must catch all error types)")
    print(f"{'=' * 60}")

    detected = validate_itinerary(_BROKEN, data)
    self_test_pass = True

    for label, pred in _EXPECTED_ERROR_CHECKS:
        caught = any(pred(e) for e in detected)
        icon = "✅" if caught else "❌ MISSED"
        print(f"  {icon}  {label}")
        if not caught:
            self_test_pass = False

    print("\n  All detected errors:")
    for e in detected:
        print(f"    • {e}")

    if not self_test_pass:
        overall_pass = False

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Real outputs    : {'ALL PASS ✅' if overall_pass else 'SOME FAILED ❌'}")
    print(f"  Broken self-test: {'ALL ERRORS CAUGHT ✅' if self_test_pass else 'SOME MISSED ❌'}")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
