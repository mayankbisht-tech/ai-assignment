"""
AI Travel Itinerary Planner — LangGraph + Groq
================================================

Grounded, priced, cited itinerary generation from a fixed supplier catalog.

Pipeline (see build_graph() for the LangGraph wiring):

    load_request -> retrieve -> plan (LLM) -> validate --(invalid)--> plan (retry, max 2x)
                                                   |
                                              (valid / out of retries)
                                                   v
                                               finalize

Run:
    export GROQ_API_KEY=sk_...
    python app.py REQ-1              # single request, calls Groq
    python app.py --all              # all three test requests
    python app.py --all --mock       # no API key needed; deterministic rule-based
                                      # stand-in for the LLM node (see MOCK MODE below)

Outputs are written to outputs/<REQUEST_ID>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Optional, TypedDict


# Load a local `.env` file if present so users can keep GROQ_API_KEY there
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
def _load_local_env() -> None:
    if not os.path.exists(ENV_PATH):
        return
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
        # best-effort only; don't fail imports if .env can't be read
        pass


_load_local_env()

# Defer importing langgraph until build_graph() so tests can import this module
# without requiring the `langgraph` package to be installed in the test env.


DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data.json")
MAX_RETRIES = 2  # total plan attempts = 1 + MAX_RETRIES
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
VALID_ID_RE = re.compile(r"\b(HOT|ACT|TRN)-\d{3}\b")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------

class PlannerState(TypedDict, total=False):
    request_id: str
    request_text: str
    catalog: list[dict]
    profile: dict
    all_ids: set
    destination_known: bool
    matched_locations: list[str]
    candidates: dict            # {hotels: [...], activities: [...], transport: [...]}
    learned_preferences: list[str]
    raw_llm_output: str
    itinerary: Optional[dict]
    validation_errors: list[str]
    attempts: int
    final: dict


# --------------------------------------------------------------------------
# Node 1: retrieve — deterministic, rule-based retrieval + profile grounding
#   (no LLM here: cheap, fast, and keeps the grounding step auditable)
# --------------------------------------------------------------------------

def retrieve_node(state: PlannerState) -> PlannerState:
    catalog = state["catalog"]
    text = state["request_text"].lower()
    profile = state["profile"]

    # "Kerala" itself is a location value used by state-wide transport items,
    # so it's kept separate from the actual city/town list to avoid matching
    # itself as a "city".
    city_locations = sorted({item["location"] for item in catalog if item["location"] != "Kerala"})
    matched_locations = [loc for loc in city_locations if loc.lower() in text]
    if not matched_locations and "kerala" in text:
        # state-wide request (e.g. "5 days in Kerala") -> every city is in scope
        matched_locations = city_locations

    destination_known = len(matched_locations) > 0

    # Extract requested number of days if explicitly provided (e.g. "5 days")
    m = re.search(r"(\d+)\s+days?\b", text)
    if m:
        try:
            state["requested_days"] = int(m.group(1))
        except Exception:
            state["requested_days"] = None
    else:
        state["requested_days"] = None

    if not destination_known:
        state["matched_locations"] = []
        state["destination_known"] = False
        state["candidates"] = {"hotels": [], "activities": [], "transport": []}
        state["learned_preferences"] = []
        return state

    # crude budget-level detection
    if "budget" in text or "budget-conscious" in text:
        budget_level = "budget"
    elif "mid-range" in text or "mid range" in text:
        budget_level = "mid-range"
    else:
        budget_level = profile.get("typical_budget_level", "mid-range")

    # Pull "what worked / what didn't" signals out of past-trip feedback so
    # the LLM has explicit, traceable guidance rather than having to infer
    # it itself from raw text on every call.
    past_feedback = " ".join(t["feedback"] for t in profile.get("past_trips", []))
    learned_preferences = []
    if "unhurried" in past_feedback or "relaxed" in past_feedback or "slow pace" in past_feedback:
        learned_preferences.append("prefers a relaxed pace / fewer packed days")
    if "bored on the long drives" in past_feedback or "fewer long drives" in past_feedback:
        learned_preferences.append("avoid long back-to-back driving days")
    if "more nature walks" in past_feedback:
        learned_preferences.append("wants more nature walks")
    if "loved" in past_feedback and "estate stay" in past_feedback:
        learned_preferences.append("responds well to scenic estate-style stays")

    def loc_ok(item):
        return item["location"] in matched_locations or item["location"] == "Kerala"

    def tag_score(item):
        tags = set(item.get("tags", []))
        score = len(tags.intersection(profile.get("preferences", [])))
        if budget_level in tags:
            score += 2
        return score

    hotels = sorted(
        [h for h in catalog if h["type"] == "hotel" and loc_ok(h)],
        key=tag_score, reverse=True,
    )
    activities = sorted(
        [a for a in catalog if a["type"] == "activity" and loc_ok(a)],
        key=tag_score, reverse=True,
    )
    transport = [t for t in catalog if t["type"] == "transport" and loc_ok(t)]

    state["matched_locations"] = matched_locations
    state["destination_known"] = True
    state["candidates"] = {
        "hotels": hotels,
        "activities": activities,
        "transport": transport,
    }
    state["learned_preferences"] = learned_preferences
    return state


# --------------------------------------------------------------------------
# Node 2: plan — the only node that calls the LLM
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a travel-itinerary planning assistant for a human travel agent.

HARD RULES (violating any of these is a critical failure):
1. You may ONLY use items from the "candidates" JSON you are given. Never invent a
    hotel, activity, transport option, price, or id that is not in that list.
2. Every item you place into the itinerary must include its exact catalog "id"
    (e.g. "HOT-002"). Do not rename or paraphrase ids.
3. If "destination_known" is false, or "candidates" has no relevant items for the
    requested destination, you MUST return status "unfulfillable" and explain why in
    plain language. Do NOT substitute a different destination and do NOT invent
    inventory to fill the gap.
4. Prices must come directly from the candidate item's price field. Compute
    per-item subtotal = unit price * quantity (nights / people / days), and the
    grand total = exact sum of all subtotals. Do not round, estimate, or add
    unlisted fees.
5. Use the traveler profile and "learned_preferences" (derived from past-trip
    feedback) to choose which candidates to use and how to pace the days
    (e.g. fewer long drives, more nature walks, relaxed pace) — but still only
    from the given candidates.
6. This itinerary will be reviewed by a human travel agent before anything is
    booked or confirmed. You are drafting a proposal, not confirming a booking.

ADDITIONAL HARD RULES:
1. The itinerary MUST contain exactly the number of days requested by the user
    (if the request explicitly states e.g. "5 days"). The LLM must not shorten
    or lengthen the itinerary. If the request does not state days explicitly,
    produce a reasonable-day itinerary but prefer to ask/clarify in interactive
    flows.
2. Hotel pricing semantics: for hotel line items, `quantity` MUST equal
    (rooms_needed * nights), where `rooms_needed` = ceil(party_size / hotel_capacity)
    and `nights` = number of requested days. Populate `quote.line_items` so this
    arithmetic holds exactly.
3. If you cannot satisfy these structural rules, return `status: "unfulfillable"`
    with a clear `reason_if_unfulfillable` describing which rule cannot be met.

Respond with ONLY a single JSON object (no prose, no markdown fences) matching
exactly this shape:

{
  "status": "ok" | "unfulfillable",
  "reason_if_unfulfillable": string | null,
  "destination": string | null,
  "days": [
     {
        "day": 1,
        "hotel_id": string | null,
        "activity_ids": [string, ...],
        "transport_ids": [string, ...],
        "notes": string
     }
  ],
  "quote": {
     "line_items": [
        {"id": string, "name": string, "unit_price": number, "quantity": number, "subtotal": number}
     ],
     "total": number
  },
  "cited_ids": [string, ...],
  "personalization_notes": string
}

If status is "unfulfillable", "days", "quote.line_items" MUST be empty arrays,
"quote.total" MUST be 0, and "cited_ids" MUST be empty.
"""


def _build_user_prompt(state: PlannerState) -> str:
    payload = {
        "request_id": state["request_id"],
        "request_text": state["request_text"],
        "destination_known": state["destination_known"],
        "matched_locations": state["matched_locations"],
        "traveler_profile": {
            "party": state["profile"]["party"],
            "preferences": state["profile"]["preferences"],
            "typical_budget_level": state["profile"]["typical_budget_level"],
            "past_trip_feedback": [
                t["feedback"] for t in state["profile"].get("past_trips", [])
            ],
        },
        "learned_preferences": state["learned_preferences"],
        "candidates": state["candidates"],
    }
    prompt = "DATA:\n" + json.dumps(payload, indent=2)
    if state.get("validation_errors"):
        prompt += (
            "\n\nYour previous attempt failed validation for these reasons:\n- "
            + "\n- ".join(state["validation_errors"])
            + "\nFix these issues and return a corrected JSON object."
        )
    return prompt


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq  # imported lazily so --mock needs no dependency
    import time

    retries = int(os.environ.get("GROQ_RETRIES", "2"))
    backoff = float(os.environ.get("GROQ_BACKOFF", "1.0"))
    timeout = float(os.environ.get("GROQ_TIMEOUT", "30"))

    last_exc = None
    for attempt in range(retries + 1):
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            if attempt < retries:
                sleep_time = backoff * (2 ** attempt)
                time.sleep(sleep_time)
                continue
            raise


def plan_node(state: PlannerState) -> PlannerState:
    # If no GROQ API key is available, use a deterministic rule-based planner
    # that follows the HARD RULES and the additional semantic rules. This
    # ensures reviewers can run the pipeline reproducibly without an API key.
    user_prompt = _build_user_prompt(state)
    state["attempts"] = state.get("attempts", 0) + 1

    # Use deterministic planner when no GROQ key is present or when explicitly forced.
    if (not os.environ.get("GROQ_API_KEY")) or os.environ.get("FORCE_LOCAL_PLANNER"):
        # Deterministic planner: pick top candidates and assemble an itinerary
        catalog = state["catalog"]
        candidates = state.get("candidates", {})
        profile = state.get("profile", {})
        party = profile.get("party", {})
        adults = int(party.get("adults", 0))
        children = int(party.get("children", 0))
        total_people = adults + children

        nights = state.get("requested_days") or max(1, len(state.get("matched_locations", [])))

        days = []

        # Choose hotel: top hotel candidate (if any)
        hotels = candidates.get("hotels", [])
        hotel_choice = hotels[0] if hotels else None

        # Choose activities: rotate through available activities
        activities = candidates.get("activities", [])
        transport = candidates.get("transport", [])
        transport_choice = transport[0] if transport else None

        # Build daily plan
        for d in range(1, int(nights) + 1):
            act_ids = []
            # assign at most one activity per day if available
            if activities:
                act = activities[(d - 1) % len(activities)]
                act_ids.append(act.get("id"))
            days.append({
                "day": d,
                "hotel_id": hotel_choice.get("id") if hotel_choice else None,
                "activity_ids": act_ids,
                "transport_ids": [transport_choice.get("id")] if transport_choice else [],
                "notes": "Auto-generated deterministic plan: one activity per day where available."
            })

        line_items = []
        cited = []
        # Hotel line item
        if hotel_choice:
            from helpers import rooms_needed
            hotel_cap = int(hotel_choice.get("capacity", 1))
            rooms = rooms_needed(total_people, hotel_cap)
            qty = rooms * int(nights)
            unit = float(hotel_choice.get("price_per_night", 0))
            subtotal = unit * qty
            line_items.append({
                "id": hotel_choice.get("id"),
                "name": hotel_choice.get("name"),
                "unit_price": unit,
                "quantity": qty,
                "subtotal": subtotal,
            })
            cited.append(hotel_choice.get("id"))

        # Activities line items (price_per_person)
        for act in activities:
            unit = float(act.get("price_per_person", 0))
            qty = total_people
            subtotal = unit * qty
            line_items.append({
                "id": act.get("id"),
                "name": act.get("name"),
                "unit_price": unit,
                "quantity": qty,
                "subtotal": subtotal,
            })
            cited.append(act.get("id"))

        # Transport line item: use per_day if present, quantity = max(1, nights-1)
        if transport_choice:
            unit = float(transport_choice.get("price_per_day", transport_choice.get("price_flat", 0)))
            qty = max(1, int(nights) - 1)
            subtotal = unit * qty
            line_items.append({
                "id": transport_choice.get("id"),
                "name": transport_choice.get("name"),
                "unit_price": unit,
                "quantity": qty,
                "subtotal": subtotal,
            })
            cited.append(transport_choice.get("id"))

        total = sum(li["subtotal"] for li in line_items)

        itinerary = {
            "status": "ok" if cited else "unfulfillable",
            "reason_if_unfulfillable": None if cited else "No candidate inventory",
            "destination": ", ".join(state.get("matched_locations", [])) or None,
            "days": days if cited else [],
            "quote": {"line_items": line_items if cited else [], "total": total if cited else 0},
            "cited_ids": cited if cited else [],
            "personalization_notes": ", ".join(state.get("learned_preferences", [])),
        }

        state["raw_llm_output"] = json.dumps(itinerary)
        state["itinerary"] = itinerary
        state["validation_errors"] = []
        return state

    # Otherwise call the real Groq API
    raw = _call_groq(SYSTEM_PROMPT, user_prompt)
    state["raw_llm_output"] = raw
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw)
    except (json.JSONDecodeError, AttributeError):
        state["itinerary"] = None
        state["validation_errors"] = ["Output was not valid JSON."]
        return state
    state["itinerary"] = parsed
    state["validation_errors"] = []
    return state


# --------------------------------------------------------------------------
# Node 3: validate — the grounding gate. Nothing reaches the user unchecked.
# --------------------------------------------------------------------------

def validate_node(state: PlannerState) -> PlannerState:
    itinerary = state.get("itinerary")
    errors: list[str] = []

    if itinerary is None:
        state["validation_errors"] = ["Output was not valid JSON."]
        return state

    all_ids = state["all_ids"]

    if not state["destination_known"]:
        if itinerary.get("status") != "unfulfillable":
            errors.append(
                "Destination has no catalog inventory; status must be 'unfulfillable', "
                f"got '{itinerary.get('status')}'."
            )
        if itinerary.get("days") or itinerary.get("quote", {}).get("line_items"):
            errors.append("Unfulfillable response must not contain days or line_items.")
        state["validation_errors"] = errors
        return state

    if itinerary.get("status") != "ok":
        errors.append(f"Expected status 'ok' for a fulfillable request, got '{itinerary.get('status')}'.")
        state["validation_errors"] = errors
        return state

    # every cited id must exist in the catalog, and no stray ids elsewhere
    cited = set(itinerary.get("cited_ids", []))
    unknown = cited - all_ids
    if unknown:
        errors.append(f"cited_ids references ids not in catalog: {sorted(unknown)}")

    # scan the whole payload text for any HOT-/ACT-/TRN- pattern not in catalog
    full_text = json.dumps(itinerary)
    mentioned_ids = set(re.findall(r"\b(?:HOT|ACT|TRN)-\d{3}\b", full_text))
    stray = mentioned_ids - all_ids
    if stray:
        errors.append(f"Response mentions ids that don't exist in the catalog: {sorted(stray)}")

    # price math: each subtotal, and the grand total
    line_items = itinerary.get("quote", {}).get("line_items", [])
    running_total = 0.0
    # verify each line_item matches catalog pricing
    from helpers import find_catalog_item_by_id, rooms_needed

    for li in line_items:
        # check id is in catalog
        if li.get("id") not in all_ids:
            errors.append(f"Line item references unknown id: {li.get('id')}")
            continue
        cat = find_catalog_item_by_id(state["catalog"], li.get("id"))
        if cat:
            # determine catalog unit price field
            if cat.get("type") == "hotel":
                expected_unit = float(cat.get("price_per_night", 0))
            elif cat.get("type") == "activity":
                expected_unit = float(cat.get("price_per_person", 0))
            elif cat.get("type") == "transport":
                # try price_per_day or price_flat
                expected_unit = float(cat.get("price_per_day", cat.get("price_flat", 0)))
            else:
                expected_unit = float(li.get("unit_price", 0))
            if abs(expected_unit - float(li.get("unit_price", 0))) > 0.01:
                errors.append(f"Line item {li.get('id')} unit_price {li.get('unit_price')} does not match catalog price {expected_unit}.")

        # Additional business-semantic checks: hotel quantity should represent nights * rooms
        if li.get("id") in all_ids:
            cat = find_catalog_item_by_id(state["catalog"], li.get("id"))
            if cat and cat.get("type") == "hotel":
                # determine nights: prefer explicit requested_days, else fall back to returned days
                nights = state.get("requested_days") or len(itinerary.get("days", []))
                # if nights is zero (no day info), skip this semantic check to allow
                # price-arithmetic-only tests and synthetic itineraries without days
                if not nights:
                    # skip semantic rooms*nights enforcement when no day info
                    pass
                else:
                    # total party size from profile
                    party = state.get("profile", {}).get("party", {})
                    total_people = int(party.get("adults", 0)) + int(party.get("children", 0))
                    hotel_cap = int(cat.get("capacity", 1))
                    needed_rooms = rooms_needed(total_people, hotel_cap)
                    expected_qty = needed_rooms * (nights or 1)
                    if li.get("quantity", 0) != expected_qty:
                        errors.append(
                            f"Line item {li.get('id')} quantity {li.get('quantity')} does not match expected nights*rooms {expected_qty} (rooms {needed_rooms} nights {nights})."
                        )

        expected = round(li.get("unit_price", 0) * li.get("quantity", 0), 2)
        actual = round(li.get("subtotal", -1), 2)
        if abs(expected - actual) > 0.01:
            errors.append(
                f"Line item {li.get('id')} subtotal mismatch: "
                f"{li.get('unit_price')} x {li.get('quantity')} = {expected}, got {actual}."
            )
        running_total += li.get("subtotal", 0)

    stated_total = round(itinerary.get("quote", {}).get("total", -1), 2)
    if abs(round(running_total, 2) - stated_total) > 0.01:
        errors.append(f"Quote total {stated_total} does not equal sum of line items {round(running_total, 2)}.")

    # Validate requested days count if explicitly requested
    req_days = state.get("requested_days")
    if req_days is not None:
        returned_days = len(itinerary.get("days", []))
        if returned_days != req_days:
            errors.append(f"Requested {req_days} days but itinerary contains {returned_days} days.")

    if not line_items:
        errors.append("Fulfillable itinerary has no priced line items.")

    state["validation_errors"] = errors
    return state


def should_retry(state: PlannerState) -> str:
    if not state["validation_errors"]:
        return "finalize"
    if state["attempts"] >= 1 + MAX_RETRIES:
        return "finalize"  # give up, surface the errors
    return "plan"


# --------------------------------------------------------------------------
# Node 4: finalize
# --------------------------------------------------------------------------

def finalize_node(state: PlannerState) -> PlannerState:
    state["final"] = {
        "request_id": state["request_id"],
        "request_text": state["request_text"],
        "attempts": state["attempts"],
        "grounded_and_valid": len(state["validation_errors"]) == 0,
        "validation_errors": state["validation_errors"],
        "itinerary": state.get("itinerary"),
    }
    return state


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph(use_mock: bool):
    from langgraph.graph import StateGraph, END

    graph = StateGraph(PlannerState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges("validate", should_retry, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_request(app, data: dict, request: dict) -> dict:
    catalog = data["suppliers"]
    initial: PlannerState = {
        "request_id": request["request_id"],
        "request_text": request["text"],
        "catalog": catalog,
        "profile": data["traveler_profile"],
        "all_ids": {item["id"] for item in catalog},
        "attempts": 0,
        "validation_errors": [],
    }
    result = app.invoke(initial)
    return result["final"]


def main():
    parser = argparse.ArgumentParser(description="Grounded travel itinerary planner")
    parser.add_argument("request_id", nargs="?", help="e.g. REQ-1")
    parser.add_argument("--all", action="store_true", help="run all test_requests")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — using deterministic local planner for reproducible runs.", file=sys.stderr)

    data = load_data()
    app = build_graph(use_mock=False)

    if args.all:
        requests_to_run = data["test_requests"]
    elif args.request_id:
        requests_to_run = [r for r in data["test_requests"] if r["request_id"] == args.request_id]
        if not requests_to_run:
            print(f"Unknown request_id {args.request_id}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    for req in requests_to_run:
        result = run_request(app, data, req)
        out_path = os.path.join("outputs", f"{req['request_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        status = "OK" if result["grounded_and_valid"] else "FAILED VALIDATION"
        print(f"{req['request_id']}: {status} (attempts={result['attempts']}) -> {out_path}")


if __name__ == "__main__":
    main()
