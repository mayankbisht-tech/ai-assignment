"""Lightweight CLI wrapper for the grounded planner (no UI).

Provides three commands:
  - run   : execute one or all test requests (writes outputs/*.json)
  - sample: list available test request IDs and texts
  - eval  : inspect an existing outputs/<REQUEST_ID>.json file

This reuses the same pipeline in `app.py` and supports `--mock` for
offline deterministic runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from app import load_data, build_graph, run_request


# Load a local `.env` file if present so users who keep keys there will work
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def cmd_sample(data: dict) -> None:
    for r in data.get("test_requests", []):
        print(f"{r['request_id']}: {r['text']}")


def cmd_run(request_id: str | None, all_flag: bool) -> None:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Set the environment variable or put it in .env.", file=sys.stderr)
        sys.exit(1)

    data = load_data()
    app = build_graph(use_mock=False)

    if all_flag:
        requests = data["test_requests"]
    elif request_id:
        requests = [r for r in data["test_requests"] if r["request_id"] == request_id]
        if not requests:
            print(f"Unknown request_id {request_id}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide a request id or --all.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    for req in requests:
        result = run_request(app, data, req)
        out_path = os.path.join("outputs", f"{req['request_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        status = "OK" if result["grounded_and_valid"] else "FAILED VALIDATION"
        print(f"{req['request_id']}: {status} (attempts={result['attempts']}) -> {out_path}")


def cmd_eval(request_id: str | None, all_flag: bool) -> None:
    if all_flag:
        files = [f for f in os.listdir("outputs") if f.endswith(".json")]
    elif request_id:
        files = [f"{request_id}.json"]
    else:
        print("Provide a request id or --all for eval.", file=sys.stderr)
        sys.exit(1)

    for fname in files:
        path = os.path.join("outputs", fname)
        if not os.path.exists(path):
            print(f"Missing output file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rid = obj.get("request_id") or fname.replace(".json", "")
        ok = obj.get("grounded_and_valid")
        errs = obj.get("validation_errors", [])
        print(f"{rid}: {'OK' if ok else 'FAILED'}")
        if errs:
            print("  Validation errors:")
            for e in errs:
                print(f"   - {e}")


def cmd_chat(prompt: str | None, write_output: bool) -> None:
    """Send a free-text request through the planner pipeline and print result.

    If `prompt` is None, read from stdin. If `write_output` is True, write
    the final `final` object to `outputs/CHAT.json`.
    """
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Set the environment variable or put it in .env.", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        sys.exit(1)

    data = load_data()
    app = build_graph(use_mock=False)
    req = {"request_id": "CHAT", "text": prompt}
    result = run_request(app, data, req)
    print(json.dumps(result, indent=2))
    if write_output:
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "CHAT.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CLI-only runner for the grounded planner")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Execute request(s)")
    p_run.add_argument("request_id", nargs="?", help="e.g. REQ-1")
    p_run.add_argument("--all", action="store_true", dest="all", help="run all test requests")
    p_chat = sub.add_parser("chat", help="Send free-text request through the planner")
    p_chat.add_argument("prompt", nargs="?", help="Free-text request prompt. If omitted, reads stdin")
    p_chat.add_argument("--save", action="store_true", dest="save", help="Save output to outputs/CHAT.json")

    p_sample = sub.add_parser("sample", help="List available test requests")

    p_eval = sub.add_parser("eval", help="Inspect outputs/<REQUEST_ID>.json")
    p_eval.add_argument("request_id", nargs="?", help="e.g. REQ-1")
    p_eval.add_argument("--all", action="store_true", dest="all", help="eval all outputs in outputs/")

    args = parser.parse_args(argv)

    data = load_data()
    if args.cmd == "sample":
        cmd_sample(data)
    elif args.cmd == "run":
        cmd_run(getattr(args, "request_id", None), getattr(args, "all", False))
    elif args.cmd == "chat":
        cmd_chat(getattr(args, "prompt", None), getattr(args, "save", False))
    elif args.cmd == "eval":
        cmd_eval(getattr(args, "request_id", None), getattr(args, "all", False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
