"""Start a conversational guided REPL for the grounded planner.

Behavior:
- Press Enter to start guided flow.
- Short/vague prompts ("i want to visit", "help me") are routed to guided flow.
- One-line concrete prompts are analyzed and may be offered fixes before running.

This file only handles CLI UX; grounding, planning, and validation remain in `app.py`.
"""
from __future__ import annotations

import os
import sys
import json
import re
from typing import Optional

from app import load_data, build_graph, run_request
from helpers import max_capacities


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
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


def _choose_from_suggestions(prompt: str, suggestions: list[str]) -> Optional[str]:
    if not suggestions:
        return None
    print(prompt)
    for i, s in enumerate(suggestions, start=1):
        print(f"  {i}. {s}")
    choice = input("Choose number, type a destination name, or press Enter to skip: ").strip()
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(suggestions):
            return suggestions[idx]
        print("Number out of range.")
        return None
    # free-text fuzzy match
    import difflib

    norm = choice.lower()
    candidates = [s.lower() for s in suggestions]
    close = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.6)
    if close:
        matched = suggestions[candidates.index(close[0])]
        yn = input(f"Did you mean '{matched}'? [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            return matched
        print("Okay — not selecting that.\n")
        return None
    print("I couldn't match that to a catalog destination. Please choose one of the available destinations.")
    return None


def _extract_destination_from_text(text: str) -> Optional[str]:
    m = re.search(r"(?:visit|visiting|go to|going to|in)\s+([A-Za-z ]{2,40})", text, re.I)
    if m:
        return m.group(1).strip().rstrip('.,?')
    return None


def _extract_party_from_text(text: str) -> Optional[tuple[int, int]]:
    m = re.search(r"party of\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1)), 0
    ma = re.search(r"(\d+)\s*adults?", text, re.I)
    mc = re.search(r"(\d+)\s*children?", text, re.I)
    if ma or mc:
        adults = int(ma.group(1)) if ma else 0
        children = int(mc.group(1)) if mc else 0
        return adults, children
    return None


def analyze_and_offer_fix(text: str, data: dict) -> str:
    # Keep simple: offer destination suggestions or party edit if obvious
    catalog = data.get("suppliers", [])
    locs = sorted({item.get("location") for item in catalog if item.get("location")})
    dest_guess = _extract_destination_from_text(text)
    if dest_guess and dest_guess.title() not in locs:
        import difflib

        close = difflib.get_close_matches(dest_guess.lower(), [l.lower() for l in locs], n=3, cutoff=0.6)
        close = [c.title() for c in sorted(set(close))]
        if close:
            print(f"Note: we don't have inventory for '{dest_guess}'. I can suggest: {', '.join(close)}.")
            pick = input("Replace destination with a suggested one? Enter number or press Enter to keep original: ").strip()
            if pick.isdigit():
                idx = int(pick) - 1
                if 0 <= idx < len(close):
                    new_dest = close[idx]
                    return re.sub(re.escape(dest_guess), new_dest, text, flags=re.I)
            print("Keeping original destination.")
    party = _extract_party_from_text(text)
    if party and party[1] == 0 and party[0] > 0:
        total = party[0]
        max_hotel = max((h.get("capacity", 0) for h in catalog if h.get("type") == "hotel"), default=0)
        if max_hotel and total > max_hotel:
            print(f"Note: party of {total} may need multiple rooms (max per room {max_hotel}).")
            adj = input("Edit party now? (y/N): ").strip().lower()
            if adj.startswith('y'):
                new = input("Enter 'adults,children' (e.g. 2,4): ").strip()
                if new:
                    return re.sub(r"party of\s*\d+", f"party of {new}", text, flags=re.I)
    return text


def _is_vague_prompt(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 30 and any(k in t for k in ("want to", "help me", "plan a", "i want", "vacation", "holiday", "help", "plan", "trip")):
        return True
    return False


def _is_catalog_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    patterns = [
        r"what (cities|destinations|places|options)",
        r"which (places|destinations|cities|options)",
        r"where can i (go|travel)",
        r"show destinations",
        r"what options do i have",
        r"what can i visit",
        r"what cities",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    # also simple help requests
    if t.strip() in ("help", "?"):
        return True
    return False


def guided_flow(data: dict, app, counter: int, initial_dest: Optional[str] = None, initial_prompt: Optional[str] = None):
    """Stateful, conversational guided flow.

    Returns (result, saved_path_or_none).
    """
    catalog = data.get("suppliers", [])
    locs = sorted({item.get("location") for item in catalog if item.get("location")})
    normalized = [l.lower() for l in locs]

    def is_help_query(text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        # cheap deterministic checks
        return bool(re.search(r"\b(help|\?|what|which|options|available|format|how many|maximum)\b", t))

    def handle_help(field: str):
        if field == "destination":
            print("Available destinations:")
            for i, l in enumerate(locs, start=1):
                print(f"  {i}. {l}")
            return
        if field == "days":
            print("Trip duration must be between 1 and 14 days. Default: 3 days.")
            return
        if field == "party":
            print("Enter the party as:\n  adults,children\nExamples:\n  2,0\n  2,1\n  2,2\nAt least one adult required. Max demo party size: 20.")
            return
        if field == "budget":
            supported_budgets = ["budget", "mid-range", "luxury"]
            print("Available budget levels:")
            for i, b in enumerate(supported_budgets, start=1):
                print(f"  {i}. {b}")
            return
        if field == "preferences":
            available_tags = sorted({tag for item in catalog for tag in item.get('tags', [])})
            print("Available preferences:")
            for t in available_tags:
                print(f"  - {t}")
            print("You can enter multiple values separated by commas.")
            return

    # state holds validated values only
    state: dict = {
        "initial_prompt": None,
        "destination": None,
        "days": None,
        "adults": None,
        "children": None,
        "budget": None,
        "preferences": None,
    }

    # apply initial dest if provided
    if initial_dest:
        init = initial_dest.strip()
        if init and init.lower() in normalized:
            state["destination"] = locs[normalized.index(init.lower())]
        elif init:
            print(f"I don't have inventory for '{init}'.")
            pick = _choose_from_suggestions("Available catalog locations:", locs)
            if pick:
                state["destination"] = pick
    # preserve high-level intent signals (solo/family) but do not infer party
    if initial_prompt:
        lp = initial_prompt.lower()
        if "solo" in lp:
            state["trip_type"] = "solo"
            print("Sure — I can help plan a solo trip. I'll still ask for the party (adults,children).")
        if "family" in lp:
            state["trip_type"] = "family"
            print("Sure — I can help plan a family trip. I'll still ask for the party (adults,children).")

    # ordered fields
    fields = ["destination", "days", "party", "budget", "preferences"]

    # defaults
    profile_party = data.get("traveler_profile", {}).get("party", {})
    default_adults = profile_party.get('adults', 2)
    default_children = profile_party.get('children', 0)
    default_party = (default_adults, default_children)
    default_budget = data.get("traveler_profile", {}).get("typical_budget_level", "mid-range")
    default_prefs = data.get("traveler_profile", {}).get("preferences", [])

    # track continue_anyway decisions to avoid repeating warnings
    continue_anyway = set()

    current_idx = 0
    while current_idx < len(fields):
        field = fields[current_idx]

        # skip if already validated
        if field == "party":
            if state.get("adults") is not None and state.get("children") is not None:
                current_idx += 1
                continue
        else:
            if state.get(field) is not None:
                current_idx += 1
                continue

        # prompt and parse per-field
        if field == "destination":
            prompt = "Destination (city or state): "
            if state.get("destination"):
                current_idx += 1
                continue
            while True:
                dest_in = input(prompt).strip()
                if not dest_in:
                    print("No destination entered. Aborting guided flow.")
                    return None, None
                if is_help_query(dest_in):
                    handle_help("destination")
                    continue
                if dest_in.lower() in normalized:
                    state["destination"] = locs[normalized.index(dest_in.lower())]
                    break
                import difflib

                close = difflib.get_close_matches(dest_in.lower(), normalized, n=1, cutoff=0.6)
                if close:
                    suggestion = locs[normalized.index(close[0])]
                    yn = input(f"Did you mean '{suggestion}'? [Y/n]: ").strip().lower()
                    if yn in ("", "y", "yes"):
                        state["destination"] = suggestion
                        break
                    else:
                        print("Okay — let's pick from available catalog locations.")
                print(f"We don't have inventory for '{dest_in}'.")
                pick = _choose_from_suggestions("Available catalog locations:", locs)
                if pick:
                    state["destination"] = pick
                    break
                retry = input("Type 'r' to re-enter destination, or 'q' to cancel guided flow: ").strip().lower()
                if retry == 'r':
                    continue
                print("Aborting guided flow.")
                return None, None

        elif field == "days":
            prompt = f"Number of days [3]: "
            while True:
                days_in = input(prompt).strip()
                if is_help_query(days_in):
                    handle_help("days")
                    continue
                if days_in == "":
                    state["days"] = 3
                    break
                if days_in.isdigit():
                    days = int(days_in)
                    if 1 <= days <= 14:
                        state["days"] = days
                        break
                    print("Please enter a reasonable number of days (1-14).")
                    continue
                print("Please enter a whole number for days (e.g. 3).")

        elif field == "party":
            prompt = f"Party (adults,children) [{default_adults} adults, {default_children} children]: "
            while True:
                party_in = input(prompt).strip()
                if is_help_query(party_in):
                    handle_help("party")
                    continue
                if not party_in:
                    adults = default_adults
                    children = default_children
                    state["adults"] = adults
                    state["children"] = children
                    break
                m = re.search(r"(\d+)\s*adults?", party_in, re.I)
                mc = re.search(r"(\d+)\s*children?", party_in, re.I)
                parts = [p.strip() for p in party_in.split(',') if p.strip().isdigit()]
                if m:
                    adults = int(m.group(1))
                elif parts:
                    adults = int(parts[0])
                else:
                    print("Please enter party as '2 adults, 1 child' or '2,1'.")
                    continue
                children = int(mc.group(1)) if mc else (int(parts[1]) if len(parts) > 1 else 0)
                total = adults + children
                if adults < 1:
                    print("At least one adult is required.")
                    continue
                if total > 20:
                    print("Party too large for this demo (max 20). Please split into smaller groups or contact an agent.")
                    continue
                # validated party
                state["adults"] = adults
                state["children"] = children
                break

            # incremental: accommodation capacity check for this field only
            dest = state.get("destination")
            if dest:
                dest_hotels = [h for h in catalog if h.get('type') == 'hotel' and (h.get('location') == dest or (dest == 'Kerala' and h.get('location') == 'Kerala'))]
                if dest_hotels:
                    max_hotel_cap = max(int(h.get('capacity', 0)) for h in dest_hotels)
                    total_people = state["adults"] + state["children"]
                    if max_hotel_cap < total_people and "party" not in continue_anyway:
                        print(f"\n⚠️  This party size may not be fulfillable with the available {dest} inventory.")
                        print(f"The largest available accommodation supports {max_hotel_cap} guests, while your party has {total_people} people.")
                        choice = input("What would you like to do?\n  1. Change party size\n  2. Continue anyway\n  3. Cancel\nChoose 1/2/3: ").strip()
                        if choice == '1':
                            # clear party and stay on party
                            state["adults"] = None
                            state["children"] = None
                            continue
                        if choice == '3':
                            print("Aborting guided flow.")
                            return None, None
                        # continue anyway
                        continue_anyway.add("party")

        elif field == "budget":
            supported_budgets = {"budget", "mid-range", "luxury"}
            prompt = f"Budget level [default: {default_budget}]: "
            while True:
                budget_in = input(prompt).strip()
                if is_help_query(budget_in):
                    handle_help("budget")
                    continue
                if not budget_in:
                    state["budget"] = default_budget
                    break
                bnorm_raw = budget_in.lower().strip()
                no_pref = {"any", "anything", "no preference", "no preference for budget", "doesn't matter", "doesnt matter", "whatever", "flexible", "any budget", "no-budget"}
                if bnorm_raw in no_pref:
                    state["budget"] = None
                    print("Budget preference: No preference.")
                    break
                bnorm = bnorm_raw.replace(' ', '-')
                if bnorm in supported_budgets:
                    state["budget"] = bnorm
                    break
                print(f"Unknown budget level '{budget_in}'. Supported: {', '.join(sorted(supported_budgets))}.")

            # incremental: budget feasibility
            budget = state.get("budget")
            dest = state.get("destination")
            # If user has no budget preference (None), skip exact-tag budget checks
            if budget is not None:
                budget_items = [it for it in catalog if budget in it.get('tags', []) and (it.get('location') == dest or (dest == 'Kerala' and it.get('location') == 'Kerala'))]
            else:
                budget_items = None
            if budget is not None and not budget_items and "budget" not in continue_anyway:
                # Check whether there is any inventory at this destination at all
                dest_items = [it for it in catalog if it.get('location') == dest or (dest == 'Kerala' and it.get('location') == 'Kerala')]
                if not dest_items:
                    # No inventory at all -> treat as destination-level problem
                    print(f"\n⚠️  I couldn't find any inventory for {dest} in the catalog.")
                    choice = input("What would you like to do?\n  1. Change destination\n  2. Cancel\nChoose 1/2: ").strip()
                    if choice == '1':
                        # clear destination and return to destination field
                        state["destination"] = None
                        # also clear downstream fields
                        state["budget"] = None
                        state["preferences"] = None
                        current_idx = 0
                        continue
                    print("Aborting guided flow.")
                    return None, None

                # Inventory exists but no exact budget tag matches; show sample pricing and offer to continue
                print(f"\n⚠️  I couldn't find an exact budget-category match in the {dest} catalog, but other inventory exists.")
                print("Here are some available items and their prices (catalog values):")
                def unit_price(item):
                    t = item.get('type')
                    if t == 'hotel':
                        return item.get('price_per_night')
                    if t == 'activity':
                        return item.get('price_per_person')
                    if t == 'transport':
                        return item.get('price_per_day', item.get('price_flat'))
                    return None

                sample = dest_items[:8]
                for it in sample:
                    up = unit_price(it)
                    price_str = f"{up}" if up is not None else "N/A"
                    print(f"  - {it.get('id')}: {it.get('name')} — {price_str}")

                choice = input("What would you like to do?\n  1. Change budget\n  2. Continue anyway (use available items and show actual prices)\n  3. Cancel\nChoose 1/2/3: ").strip()
                if choice == '1':
                    state["budget"] = None
                    continue
                if choice == '3':
                    print("Aborting guided flow.")
                    return None, None
                # continue anyway with available inventory (do not assign artificial budget tags)
                continue_anyway.add("budget")

        elif field == "preferences":
            prompt = f"Preferences (comma separated) [default: {', '.join(default_prefs)}]: "
            while True:
                prefs_in = input(prompt).strip()
                if is_help_query(prefs_in):
                    handle_help("preferences")
                    continue

                # determine destination-specific tags
                dest = state.get("destination")
                dest_activity_tags = sorted({
                    t
                    for item in catalog
                    if item.get('type') == 'activity' and (item.get('location') == dest or (dest == 'Kerala' and item.get('location') == 'Kerala'))
                    for t in item.get('tags', [])
                })

                # If user pressed Enter, consider defaults but filter to available tags
                if not prefs_in:
                    filtered = [p for p in default_prefs if p in dest_activity_tags]
                    missing = [p for p in default_prefs if p not in filtered]
                    if missing and dest_activity_tags:
                        print(f"\nNote: these default preferences have no matches in {dest}: {', '.join(missing)}")
                        print("Available activity preferences for this destination:")
                        for i, t in enumerate(dest_activity_tags, start=1):
                            print(f"  {i}. {t}")
                        choice = input("Continue with available defaults, choose new preferences, or change destination?\n  1. Continue with available defaults\n  2. Choose from available preferences\n  3. Change destination\nChoose 1/2/3: ").strip()
                        if choice == '1':
                            state["preferences"] = filtered
                            break
                        if choice == '2':
                            sel = input("Enter the number of the preference to choose: ").strip()
                            if sel.isdigit():
                                idx = int(sel) - 1
                                if 0 <= idx < len(dest_activity_tags):
                                    state["preferences"] = [dest_activity_tags[idx]]
                                    break
                            print("Invalid selection; let's re-enter preferences.")
                            continue
                        if choice == '3':
                            state["destination"] = None
                            state["preferences"] = None
                            current_idx = 0
                            break
                        # otherwise loop to re-enter
                    else:
                        state["preferences"] = filtered
                        if not dest_activity_tags and not filtered:
                            print(f"\n⚠️ The catalog does not contain activities for {dest}.")
                            choice = input("What would you like to do?\n  1. Continue without an activity preference\n  2. Change destination\n  3. Cancel\nChoose 1/2/3: ").strip()
                            if choice == '1':
                                state["preferences"] = []
                                break
                            if choice == '2':
                                state["destination"] = None
                                state["preferences"] = None
                                current_idx = 0
                                break
                            print("Aborting guided flow.")
                            return None, None
                        break

                # user provided explicit preferences
                prefs = [p.strip() for p in prefs_in.split(',') if p.strip()]
                # find any prefs that have no matching activity
                bad = [p for p in prefs if p not in dest_activity_tags]
                if bad:
                    if dest_activity_tags:
                        print(f"\n⚠️ No catalog activities in {dest} match: {', '.join(bad)}")
                        print("Available activity preferences for this destination:")
                        for i, t in enumerate(dest_activity_tags, start=1):
                            print(f"  {i}. {t}")
                        choice = input("What would you like to do?\n  1. Choose an available preference\n  2. Continue with your preference(s) anyway\n  3. Cancel\nChoose 1/2/3: ").strip()
                        if choice == '1':
                            sel = input("Enter number of preference to choose: ").strip()
                            if sel.isdigit():
                                idx = int(sel) - 1
                                if 0 <= idx < len(dest_activity_tags):
                                    state["preferences"] = [dest_activity_tags[idx]]
                                    break
                            print("Invalid selection; re-enter preferences.")
                            continue
                        if choice == '3':
                            print("Aborting guided flow.")
                            return None, None
                        # continue anyway
                        state["preferences"] = prefs
                        continue_anyway.add("preferences")
                        break
                    else:
                        print(f"\n⚠️ The catalog does not contain activities for {dest}.")
                        choice = input("What would you like to do?\n  1. Continue without an activity preference\n  2. Change destination\n  3. Cancel\nChoose 1/2/3: ").strip()
                        if choice == '1':
                            state["preferences"] = []
                            break
                        if choice == '2':
                            state["destination"] = None
                            state["preferences"] = None
                            current_idx = 0
                            break
                        print("Aborting guided flow.")
                        return None, None

                # all provided prefs are available
                state["preferences"] = prefs
                break

        # move to next field
        current_idx += 1

    # all fields collected — build preview
    days = state["days"]
    dest = state["destination"]
    adults = state["adults"]
    children = state["children"]
    budget = state["budget"]
    prefs = state["preferences"] or []

    req_text = f"{days} days in {dest} for a party of {adults} adults and {children} children. Budget: {budget}. Preferences: {', '.join(prefs)}."
    print("\nRequest preview:")
    print(f"  {days} days in {dest} for {adults} adults and {children} children.")
    print(f"  Budget: {budget}")
    print(f"  Preferences: {', '.join(prefs)}")
    ok = input("Run planner for this request? (y/N): ").strip().lower()
    if not ok.startswith('y'):
        print("Aborted guided flow.")
        return None, None

    req_id = f"GUIDED-{counter}"
    req = {"request_id": req_id, "text": req_text}

    # run planner with retries handled in app.py
    result = run_request(app, data, req)

    return result, None


def main():
    _load_env()
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Put it in the environment or .env.")
        sys.exit(1)

    print("Planner chatbot — guided mode by default.\nPress Enter to start guided flow, type a one-line prompt to run immediately, or ':quit' to exit.")
    data = load_data()
    app = build_graph(use_mock=False)
    counter = 1
    last_result = None
    while True:
        try:
            raw = input(f"(guided) press Enter for guided flow or enter prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw in (":quit", ":exit"):
            break

        if raw == "":
            res, saved_path = guided_flow(data, app, counter)
            if res is not None:
                last_result = res
                counter += 1
            if saved_path:
                print(f"Saved: {saved_path}")
            continue

        text = raw
        save = False
        if raw.endswith(" --save"):
            save = True
            text = raw[: -len(" --save")].rstrip()

        # Pre-checks: catalog/help questions or natural region prompts should not call planner
        locs = sorted({item.get("location") for item in data.get("suppliers", []) if item.get("location")})
        normalized = [l.lower() for l in locs]

        if _is_catalog_question(text):
            print("Available destinations in the catalog:")
            for i, l in enumerate(locs, start=1):
                print(f"  {i}. {l}")
            print("Which destination would you like to visit?")
            # hand over to the guided flow to let user pick
            res, saved_path = guided_flow(data, app, counter)
            if res is not None:
                last_result = res
                counter += 1
            if saved_path:
                print(f"Saved: {saved_path}")
            continue

        # If user mentioned a destination-like phrase, prefer guided flow instead of calling planner directly
        dest_guess = _extract_destination_from_text(text)
        if dest_guess:
            if dest_guess.lower() in normalized:
                # known catalog destination: start guided flow with initial dest
                res, saved_path = guided_flow(data, app, counter, initial_dest=dest_guess)
                if res is not None:
                    last_result = res
                    counter += 1
                if saved_path:
                    print(f"Saved: {saved_path}")
                continue
            else:
                # region or non-specific place mentioned: show catalog and ask user to pick
                print(f"I can help with that. The catalog doesn't include '{dest_guess}'.")
                print("Available destinations in the catalog:")
                for i, l in enumerate(locs, start=1):
                    print(f"  {i}. {l}")
                print("Which destination would you like to visit?")
                res, saved_path = guided_flow(data, app, counter)
                if res is not None:
                    last_result = res
                    counter += 1
                if saved_path:
                    print(f"Saved: {saved_path}")
                continue

        if _is_vague_prompt(text):
            dest_guess = _extract_destination_from_text(text)
            res, saved_path = guided_flow(data, app, counter, initial_dest=dest_guess)
            if res is not None:
                last_result = res
                counter += 1
            if saved_path:
                print(f"Saved: {saved_path}")
            continue

        text = analyze_and_offer_fix(text, data)
        req = {"request_id": f"CHAT-{counter}", "text": text}
        result = run_request(app, data, req)
        last_result = result

        it = result.get("itinerary") or {}
        status = it.get("status")
        if status == "ok":
            dest = it.get("destination")
            days = it.get("days", [])
            total = it.get("quote", {}).get("total", 0)
            cited = it.get("cited_ids", [])
            print(f"Destination: {dest or 'Unknown'} — Days: {len(days)} — Total: {total} {data.get('currency','')}")
            if cited:
                print(f"Cited IDs: {', '.join(cited[:5])}")
            notes = it.get("personalization_notes")
            if notes:
                print(f"Notes: {notes}")
        else:
            reason = it.get("reason_if_unfulfillable") or "Unfulfillable or no itinerary returned."
            print(f"Cannot fulfill request: {reason}")

        if save:
            os.makedirs("outputs", exist_ok=True)
            fname = f"CHAT-{counter}.json"
            path = os.path.join("outputs", fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {path}")
        else:
            if isinstance(result, dict) and result.get("grounded_and_valid"):
                print("Result is grounded and valid (not saved). Use '--save' to save this run.")

        counter += 1


if __name__ == "__main__":
    main()
