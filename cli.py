"""cli.py — Optional CLI utility commands (run, sample, eval, chat).

Note: The primary entry point for normal conversational use is `python main.py`.
This module provides supplementary utilities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from helpers import load_env
load_env()

from app import (
    build_graph,
    create_initial_state,
    load_data,
    process_turn,
    validate_itinerary,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def cmd_sample() -> None:
    """Print all sample test requests from sample_data.json."""
    data = load_data()
    for req in data.get("test_requests", []):
        print(f"{req['request_id']}: {req['text']}")


def cmd_eval(request_id: str | None, all_outputs: bool) -> None:
    """Validate generated output JSON files against the catalog."""
    data = load_data()
    if all_outputs:
        files = sorted(OUTPUTS_DIR.glob("*.json"))
    elif request_id:
        files = [OUTPUTS_DIR / f"{request_id}.json"]
    else:
        print("Provide a request_id or --all for eval.", file=sys.stderr)
        sys.exit(1)

    for fpath in files:
        if not fpath.exists():
            print(f"Missing file: {fpath}")
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        itin = payload.get("itinerary")
        if itin:
            errors = validate_itinerary(itin, data)
            status = "OK" if len(errors) == 0 else "FAILED"
            print(f"{fpath.name}: {status}")
            for err in errors:
                print(f"  • {err}")
        else:
            print(f"{fpath.name}: {payload.get('status', 'NO_ITINERARY')}")


def cmd_chat(prompt: str | None, save: bool) -> None:
    """Run a single-turn or piped prompt through the planner."""
    text = prompt or sys.stdin.read().strip()
    if not text:
        print("No prompt provided.", file=sys.stderr)
        sys.exit(1)

    app = build_graph()
    state = create_initial_state()
    state, response = process_turn(app, state, text)

    print(response)
    if save and state.get("output_saved"):
        print("\nSaved output to outputs/CHAT.json")


def cmd_benchmark() -> None:
    """Generate benchmark output files (REQ-1.json, REQ-2.json, REQ-3.json)."""
    from generate_benchmarks import generate_all_benchmarks
    generate_all_benchmarks()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Travel Planner CLI Utilities")
    sub = parser.add_subparsers(dest="command")

    p_sample = sub.add_parser("sample", help="List test requests")
    p_bench = sub.add_parser("benchmark", help="Generate REQ-1.json, REQ-2.json, REQ-3.json")
    p_eval = sub.add_parser("eval", help="Validate output files")
    p_eval.add_argument("request_id", nargs="?")
    p_eval.add_argument("--all", action="store_true")

    p_chat = sub.add_parser("chat", help="Run a prompt non-interactively")
    p_chat.add_argument("prompt", nargs="?")
    p_chat.add_argument("--save", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "sample":
        cmd_sample()
    elif args.command == "benchmark":
        cmd_benchmark()
    elif args.command == "eval":
        cmd_eval(args.request_id, args.all)
    elif args.command == "chat":
        cmd_chat(args.prompt, args.save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

