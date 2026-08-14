"""main.py — Interactive Conversational CLI for the Grounded Travel Planner.

Usage:
    python main.py

The user interacts naturally across multiple turns. State (destination, days, party,
budget, preferences) is preserved between turns. When all required information is
collected, a 100% catalog-grounded itinerary and quote are printed to the console
and saved to outputs/CHAT.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure Unicode output works cleanly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app import build_graph, create_initial_state, process_turn


def run_cli() -> None:
    """Launch the interactive conversational CLI loop."""
    print("=" * 60)
    print("Travel Planner")
    print("Type 'exit', 'quit', or 'bye' to quit.")
    print("=" * 60)

    app = build_graph()
    state = create_initial_state()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nPlanner: Goodbye! Happy travels!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print("Planner: Goodbye! Have a great trip!")
            break

        state, response = process_turn(app, state, user_input)

        if response.startswith("=" * 60):
            # Itinerary block
            print(f"\n{response}")
            saved_file = state.get("last_saved_file", "CHAT.json")
            print(f"\nSaved itinerary to outputs/{saved_file}")
        else:
            print(f"Planner: {response}")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
