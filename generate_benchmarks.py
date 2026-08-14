"""generate_benchmarks.py — Run all benchmark test requests and save REQ-1.json, REQ-2.json, REQ-3.json.

Usage:
    python generate_benchmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure Unicode output works cleanly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app import build_graph, create_initial_state, load_data, process_turn

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def generate_all_benchmarks() -> list[Path]:
    """Execute all benchmark test requests in sample_data.json and save structured outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    app = build_graph()
    test_requests = data.get("test_requests", [])

    created_files: list[Path] = []
    print("=" * 60)
    print(f"GENERATING BENCHMARK REQUEST OUTPUTS ({len(test_requests)} requests)")
    print("=" * 60)

    for req in test_requests:
        req_id = req["request_id"]
        req_text = req["text"]

        state = create_initial_state()
        state["is_benchmark"] = True
        state, response = process_turn(app, state, req_text)


        itin = state.get("itinerary")
        val_errors = state.get("validation_errors", [])

        if itin and itin.get("status") == "fulfilled":
            status_val = "ok_pending_human_review"
        else:
            status_val = "declined_no_inventory"

        output_payload = {
            "request_id": req_id,
            "request_text": req_text,
            "note_for_reviewer": req.get("note_for_reviewer", ""),
            "status": status_val,
            "destination": state.get("destination"),
            "days": state.get("days"),
            "adults": state.get("adults"),
            "children": state.get("children", 0),
            "budget": state.get("budget"),
            "preferences": state.get("preferences", []),
            "itinerary": itin,
            "grounded_and_valid": len(val_errors) == 0,
            "validation_errors": val_errors,
        }

        out_path = OUTPUTS_DIR / f"{req_id}.json"
        out_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        created_files.append(out_path)

        total_price = itin.get("quote", {}).get("total", "n/a") if itin else "n/a"
        price_str = f"₹{total_price:,.0f}" if isinstance(total_price, (int, float)) else total_price
        print(f"✅ Generated outputs/{out_path.name:<12} | Status: {status_val:<25} | Total: {price_str}")

    print("=" * 60)
    print("ALL BENCHMARK FILES GENERATED SUCCESSFULLY")
    print("=" * 60)
    return created_files


if __name__ == "__main__":
    generate_all_benchmarks()
