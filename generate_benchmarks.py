from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app import build_graph, create_initial_state, load_data, process_turn

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def generate_all_benchmarks() -> list[Path]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    app = build_graph()
    test_requests = data.get("test_requests", [])
    created: list[Path] = []

    print(f"{'=' * 60}\nGENERATING BENCHMARKS ({len(test_requests)} requests)\n{'=' * 60}")

    for req in test_requests:
        req_id, req_text = req["request_id"], req["text"]
        state = create_initial_state()
        state["is_benchmark"] = True
        state, _ = process_turn(app, state, req_text)

        itin = state.get("itinerary")
        val_errors = state.get("validation_errors", [])
        status_val = "ok_pending_human_review" if itin and itin.get("status") == "fulfilled" else "declined_no_inventory"

        payload = {
            "request_id": req_id, "request_text": req_text,
            "note_for_reviewer": req.get("note_for_reviewer", ""),
            "status": status_val,
            "destination": state.get("destination"), "days": state.get("days"),
            "adults": state.get("adults"), "children": state.get("children", 0),
            "budget": state.get("budget"), "preferences": state.get("preferences", []),
            "itinerary": itin,
            "grounded_and_valid": len(val_errors) == 0,
            "validation_errors": val_errors,
        }

        out_path = OUTPUTS_DIR / f"{req_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        created.append(out_path)

        total = itin.get("quote", {}).get("total", "n/a") if itin else "n/a"
        price_str = f"₹{total:,.0f}" if isinstance(total, (int, float)) else total
        print(f"✅ {out_path.name:<12} | {status_val:<25} | {price_str}")

    print(f"{'=' * 60}\nDONE\n{'=' * 60}")
    return created


if __name__ == "__main__":
    generate_all_benchmarks()
