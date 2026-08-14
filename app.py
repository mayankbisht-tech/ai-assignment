from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import END, StateGraph
from helpers import load_env
load_env()

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"

# ── Catalog helpers ──────────────────────────────────────────────────────────

def load_data() -> dict[str, Any]:
    return json.loads((ROOT / "sample_data.json").read_text(encoding="utf-8"))

def get_catalog(data: dict) -> list[dict]:
    return data.get("suppliers", [])

def catalog_by_id(data: dict) -> dict[str, dict]:
    return {i["id"]: i for i in get_catalog(data)}

def get_catalog_locations(data: dict) -> list[str]:
    return sorted({i["location"] for i in get_catalog(data) if i.get("type") in ("hotel", "activity") and i.get("location")})

def get_catalog_regions(data: dict) -> list[str]:
    all_locs = {i["location"] for i in get_catalog(data) if i.get("location")}
    return sorted(all_locs - set(get_catalog_locations(data))) or ["Kerala"]

def get_traveler_profile(data: dict) -> dict:
    return data.get("traveler_profile", {})

def unit_price(item: dict) -> float:
    t = item.get("type")
    if t == "hotel":   return float(item.get("price_per_night", 0))
    if t == "activity": return float(item.get("price_per_person", 0))
    if t == "transport":
        return float(item.get("price_per_day", item.get("price_flat", 0)))
    return 0.0

_ALIASES = {"cochin": "Kochi", "fort kochi": "Kochi", "alappuzha": "Alleppey",
            "alappuzha beach": "Alleppey", "munnar hills": "Munnar"}
_ACRONYMS = {"uk": "UK", "usa": "USA", "uae": "UAE"}
_KNOWN_OUT = {
    "goa","mumbai","delhi","jaipur","bangalore","bengaluru","chennai","coorg","wayanad",
    "ooty","mysore","hyderabad","kolkata","shimla","manali","agra","varanasi","udaipur",
    "pondicherry","andaman","lakshadweep","kashmir","ladakh","bali","thailand","dubai",
    "paris","uk","usa","london","assam","rajasthan","sikkim"
}
_STOP = {"a","an","the","my","our","some","weekend","few","short","long","family",
         "vacation","hotel","hotels","activity","activities","price","budget","days","people","adults"}

def resolve_destination(text: str, data: dict) -> str | None:
    low = text.strip().lower()
    for loc in get_catalog_locations(data):
        if re.search(r'\b' + re.escape(loc.lower()) + r'\b', low):
            return loc
    for alias, canon in _ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', low):
            return canon
    for reg in get_catalog_regions(data):
        if re.search(r'\b' + re.escape(reg.lower()) + r'\b', low):
            return reg
    return None

def detect_out_of_catalog_destination(text: str, data: dict) -> str | None:
    low = text.strip().lower()
    for place in _KNOWN_OUT:
        if re.search(r'\b' + re.escape(place) + r'\b', low):
            return _ACRONYMS.get(place, place.capitalize())
    m = re.search(r'\b(?:go\s+to|visit|trip\s+to|in|for|about)\s+([a-zA-Z]+)\b', low)
    if m:
        word = m.group(1).lower()
        candidate = _ACRONYMS.get(word, word.capitalize())
        valid = {v.lower() for v in get_catalog_locations(data) + get_catalog_regions(data)}
        if candidate.lower() not in valid and candidate.lower() not in _STOP:
            return candidate
    return None

# ── State ────────────────────────────────────────────────────────────────────

class PlannerState(TypedDict, total=False):
    messages: list[dict]
    latest_user_message: str
    destination: str | None
    days: int | None
    adults: int | None
    children: int | None
    budget: float | None
    budget_level: str | None
    preferences: list[str]
    pending_question: str | None
    preferences_asked: bool
    budget_asked: bool
    out_of_catalog_destination: str | None
    inquiry_response: str | None
    turn_intent: str | None
    itinerary: dict | None
    status: Literal["in_progress", "fulfilled", "unfulfillable"]
    last_response: str
    validation_errors: list[str]
    output_saved: bool
    last_saved_file: str | None
    is_benchmark: bool

def create_initial_state() -> PlannerState:
    return {"messages": [], "latest_user_message": "", "destination": None, "days": None,
            "adults": None, "children": 0, "budget": None, "budget_level": None,
            "preferences": [], "pending_question": None, "preferences_asked": False,
            "budget_asked": False, "out_of_catalog_destination": None, "inquiry_response": None,
            "turn_intent": None, "itinerary": None, "status": "in_progress",
            "last_response": "", "validation_errors": [], "output_saved": False,
            "last_saved_file": None, "is_benchmark": False}

def llm_understand_turn(text: str, state: PlannerState, data: dict) -> dict:
    """
    Single LLM call that understands the full conversation context and returns:
    - reply: a natural response if the user asked a question (None if purely slot-filling)
    - slots: any trip parameters extracted from the message
    Falls back to empty dict if LLM is unavailable.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return {}
    try:
        import groq
        client = groq.Groq(api_key=key)
        catalog = get_catalog(data)
        valid_locs = get_catalog_locations(data)
        catalog_summary = "\n".join(
            f"- [{i['id']}] {i['name']} | {i['location']} | {i.get('type')} | "
            f"₹{unit_price(i):,.0f} | tags: {', '.join(i.get('tags', []))}"
            for i in catalog
        )
        # Current state summary
        known = (
            f"destination={state.get('destination')}, days={state.get('days')}, "
            f"adults={state.get('adults')}, children={state.get('children')}, "
            f"budget={state.get('budget')}, budget_level={state.get('budget_level')}, "
            f"preferences={state.get('preferences')}"
        )
        # Recent conversation history (last 6 turns)
        history = state.get("messages", [])[-6:]
        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)

        system = f"""You are a friendly AI travel planner assistant for a Kerala travel agency.

CATALOG (use ONLY these items — never invent anything):
{catalog_summary}

VALID DESTINATIONS: {', '.join(valid_locs)} (also Kerala as a region)

Your job per turn:
1. If the user asks a question or makes a comment that needs a real answer, write a SHORT, 
   helpful reply (1-2 sentences max, grounded strictly in the catalog above). Set "reply" to that.
2. Extract any trip parameters mentioned. Set null for anything not mentioned.
3. If the user just provides a slot value (number, destination, etc.) with no question, set "reply" to null.

Return ONLY valid JSON with these keys:
{{
  "reply": "<your response or null>",
  "destination": "<Kochi|Munnar|Alleppey or null>",
  "days": <integer or null>,
  "adults": <integer or null>,
  "children": <integer or null>,
  "budget": <number or null>,
  "budget_level": "<budget|mid-range|luxury or null>",
  "preferences": ["<tag>", ...] or null,
  "out_of_catalog_destination": "<name if user asked for somewhere NOT in catalog, else null>"
}}"""

        user_msg = f"""CONVERSATION SO FAR:
{history_text}

CURRENT BOOKING STATE: {known}

User just said: "{text}"

Respond with JSON only."""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {k: v for k, v in parsed.items() if v is not None}
    except Exception:
        return {}


def extract_slots_deterministic(text: str, state: PlannerState, data: dict) -> dict:
    low = text.strip().lower()
    updates: dict = {}
    pending = state.get("pending_question")

    dest = resolve_destination(text, data)
    if dest:
        updates.update(destination=dest, out_of_catalog_destination=None, pending_question=None)
    else:
        out = detect_out_of_catalog_destination(text, data)
        if out:
            updates["out_of_catalog_destination"] = out

    # Days
    m = re.search(r'\b(\d+)\s*(?:nights?|days?)\b', low)
    if m:
        updates.update(days=int(m.group(1)), pending_question=None)
    elif "weekend" in low:
        updates.update(days=2, pending_question=None)
    elif re.match(r'^\s*(\d+)\s*$', low) and state.get("days") is None and state.get("destination"):
        updates.update(days=int(low.strip()), pending_question=None)
    elif re.search(r'\b(?:for|make it|change to)\s+(\d+)\b', low) and "adult" not in low:
        m2 = re.search(r'\b(?:for|make it|change to)\s+(\d+)\b', low)
        if m2: updates.update(days=int(m2.group(1)), pending_question=None)

    # Party
    pair = re.match(r'^\s*(\d+)\s*,\s*(\d+)\s*$', low)
    if pair:
        updates.update(adults=int(pair.group(1)), children=int(pair.group(2)), pending_question=None)
    else:
        ma = re.search(r'\b(\d+)\s*adults?\b', low)
        if ma: updates.update(adults=int(ma.group(1)), pending_question=None)
        mk = re.search(r'\b(\d+)\s*(?:children|child|kids?)\b', low)
        if mk: updates.update(children=int(mk.group(1)), pending_question=None)
        elif re.search(r'\bkids?\s+aged?\s+([\d\s,and]+)', low):
            ages = re.findall(r'\b\d+\b', re.search(r'\bkids?\s+aged?\s+([\d\s,and]+)', low).group(1))
            updates.update(children=len(ages), pending_question=None)
        if "couple" in low or "two of us" in low or "me and my wife" in low or "me and my partner" in low:
            updates.update(adults=2, pending_question=None)
            if "children" not in updates and state.get("children") is None: updates["children"] = 0
        elif "solo" in low or "just me" in low or "alone" in low or "1 person" in low:
            updates.update(adults=1, children=0, pending_question=None)
        elif re.search(r'\bfamily\s+of\s+(\d+)\b', low):
            total = int(re.search(r'\bfamily\s+of\s+(\d+)\b', low).group(1))
            updates.update(adults=updates.get("adults", 2), children=max(0, total - updates.get("adults", 2)), pending_question=None)
        elif re.search(r'\b(\d+)\s*(?:people|persons|travellers?|travelers?)\b', low):
            num = int(re.search(r'\b(\d+)\s*(?:people|persons|travellers?|travelers?)\b', low).group(1))
            if "adults" not in updates and state.get("adults") is None:
                updates.update(adults=num, children=updates.get("children", 0), pending_question=None)
        elif re.match(r'^\s*(\d+)\s*$', low) and state.get("days") is not None and state.get("adults") is None:
            updates.update(adults=int(low.strip()), children=0, pending_question=None)

    # Preferences
    known_tags = {"nature","hiking","tea-estate","local-food","relaxed","culture","family-friendly",
                  "wildlife","adventure","backwaters","heritage","views","food","trek","cruise","dance","ayurveda"}
    found = set(state.get("preferences", []))
    for tag in known_tags:
        if tag.replace("-", " ") in low or tag in low:
            found.add(tag)
    if found:
        updates.update(preferences=sorted(found), preferences_asked=True)
        if pending == "preferences": updates["pending_question"] = None

    # Budget
    mk2 = re.search(r'\b(?:budget\s*(?:of|is|around|under)?\s*)?(\d+)\s*k\b', low)
    if mk2:
        updates.update(budget=float(mk2.group(1)) * 1000, budget_asked=True)
        if pending == "budget": updates["pending_question"] = None
    else:
        for m3 in re.finditer(r'[\d,]+', text.replace("₹", "").replace(",", "")):
            try:
                val = float(m3.group(0))
                if 1000 <= val <= 1000000 and val not in (updates.get("days"), state.get("days")):
                    updates.update(budget=val, budget_asked=True)
                    if pending == "budget": updates["pending_question"] = None
                    break
            except Exception: pass

    for kw, blvl in [("budget","budget"),("budget-conscious","budget"),("cheap","budget"),
                     ("affordable","budget"),("hostel","budget"),("backpacker","budget"),
                     ("mid-range","mid-range"),("midrange","mid-range"),("moderate","mid-range"),
                     ("luxury","luxury"),("5-star","luxury"),("premium","luxury")]:
        if kw in low:
            updates.update(budget_level=blvl, budget_asked=True)
            if pending == "budget": updates["pending_question"] = None
            break

    # Skip / flexible
    if any(w in low for w in ("flexible","skip","none","no preference","anything","standard","no budget","not sure","any")):
        if pending == "preferences" or (not state.get("preferences_asked") and not updates.get("preferences")):
            updates.update(preferences_asked=True, pending_question=None)
        elif pending == "budget" or (not state.get("budget_asked") and not updates.get("budget")):
            updates.update(budget_asked=True, pending_question=None)

    return updates


# ── Itinerary Builder ────────────────────────────────────────────────────────

def build_grounded_itinerary(destination: str, days: int, adults: int, children: int,
                              budget: float | None, preferences: list[str],
                              budget_level: str | None, data: dict) -> dict:
    catalog = get_catalog(data)
    people = max(1, adults + (children or 0))
    active_prefs = set(preferences) | set(get_traveler_profile(data).get("preferences", []))
    is_regional = destination.lower() in ("kerala", "all")
    allowed = {i["location"] for i in catalog} if is_regional else {destination, "Kerala"}

    hotels    = [i for i in catalog if i["type"] == "hotel"     and i.get("location") in allowed]
    activities = [i for i in catalog if i["type"] == "activity"  and i.get("location") in allowed]
    transport  = [i for i in catalog if i["type"] == "transport" and i.get("location") in allowed]

    if not hotels or not activities:
        return {"status": "unfulfillable",
                "reason_if_unfulfillable": f"No catalog inventory for '{destination}'.",
                "destination": destination, "days": [], "quote": {"line_items": [], "total": 0.0}}

    def score_hotel(h: dict) -> float:
        s = float(h.get("rating", 3.0)) + len(set(h.get("tags", [])) & active_prefs) * 1.5
        if budget_level == "budget"    and "budget"    in h.get("tags", []): s += 3.0
        if budget_level == "mid-range" and "mid-range" in h.get("tags", []): s += 2.0
        if h.get("capacity", 2) >= people: s += 2.0
        return s

    def score_activity(a: dict) -> float:
        return float(a.get("rating", 3.0)) + len(set(a.get("tags", [])) & active_prefs) * 2.0

    chosen_hotel = sorted(hotels, key=score_hotel, reverse=True)[0]
    sorted_acts  = sorted(activities, key=score_activity, reverse=True)
    rooms = max(1, math.ceil(people / chosen_hotel.get("capacity", 4)))

    suitable_trans = [t for t in transport if t.get("capacity", 4) >= people and "price_per_day" in t]
    chosen_trans = suitable_trans[0] if suitable_trans else (transport[0] if transport else None)

    itinerary_days, used_acts = [], set()
    for d in range(1, days + 1):
        act = next((a for a in sorted_acts if a["id"] not in used_acts), None)
        if act: used_acts.add(act["id"])

        def make_act_entry(a):
            return [{"id": a["id"], "name": a["name"],
                     "duration_hours": a.get("duration_hours", 3), "unit_price": unit_price(a)}]

        if d == 1:
            title  = f"Arrival & {act['name']}" if act else f"Arrival & Check-in at {chosen_hotel['name']}"
            notes  = (f"Check in to {chosen_hotel['name']}. Enjoy a relaxed start with {act['name']} ({act.get('duration_hours',3)} hrs)."
                      if act else f"Check in to {chosen_hotel['name']} and enjoy a relaxing evening.")
            day_acts = make_act_entry(act) if act else []
        elif d == days:
            title  = f"{act['name']} & Departure" if act else "Leisure Morning & Departure"
            notes  = (f"Morning {act['name']} ({act.get('duration_hours',3)} hrs) then checkout and departure."
                      if act else f"Relaxed morning in {destination}, checkout, and onward departure.")
            day_acts = make_act_entry(act) if act else []
        else:
            title  = f"Discover {destination} — {act['name']}" if act else f"Scenic Leisure in {destination}"
            notes  = (f"Full day exploring {act['name']} ({act.get('duration_hours',3)} hrs). Leisure evening."
                      if act else f"Self-guided exploration of {destination}'s scenic spots and cafes.")
            day_acts = make_act_entry(act) if act else []

        day_obj: dict = {"day": d, "title": title,
                         "hotel": {"id": chosen_hotel["id"], "name": chosen_hotel["name"],
                                   "price_per_night": unit_price(chosen_hotel)},
                         "activities": day_acts, "notes": notes}
        if chosen_trans:
            day_obj["transport"] = {"id": chosen_trans["id"], "name": chosen_trans["name"],
                                    "price_per_day": unit_price(chosen_trans)}
        itinerary_days.append(day_obj)

    # Quote
    h_price = unit_price(chosen_hotel)
    h_qty   = days * rooms
    line_items = [{"catalog_id": chosen_hotel["id"], "name": chosen_hotel["name"],
                   "unit_price_inr": h_price, "quantity": h_qty, "rooms": rooms,
                   "line_total_inr": round(h_price * h_qty, 2)}]
    for act in sorted_acts:
        if act["id"] in used_acts:
            ap = unit_price(act)
            line_items.append({"catalog_id": act["id"], "name": act["name"],
                                "unit_price_inr": ap, "quantity": people,
                                "line_total_inr": round(ap * people, 2)})
    if chosen_trans:
        tp = unit_price(chosen_trans)
        tq = 1 if "price_flat" in chosen_trans else days
        line_items.append({"catalog_id": chosen_trans["id"], "name": chosen_trans["name"],
                           "unit_price_inr": tp, "quantity": tq, "line_total_inr": round(tp * tq, 2)})

    return {"status": "fulfilled", "destination": destination, "days": itinerary_days,
            "quote": {"line_items": line_items, "total": round(sum(li["line_total_inr"] for li in line_items), 2)}}

# ── Validator ────────────────────────────────────────────────────────────────

def validate_itinerary(itinerary: dict, data: dict) -> list[str]:
    if not itinerary: return ["GROUNDING: itinerary is empty or None."]
    if itinerary.get("status") == "unfulfillable": return []
    index = catalog_by_id(data)
    quote = itinerary.get("quote", {})
    line_items = itinerary.get("line_items") or quote.get("line_items", [])
    if not line_items: return ["GROUNDING: line_items is empty or missing."]

    errors, computed, li_ids = [], 0.0, set()
    for li in line_items:
        cid = li.get("catalog_id") or li.get("id", "")
        li_ids.add(cid)
        if cid not in index:
            errors.append(f"GROUNDING: catalog_id '{cid}' does not exist in catalog."); continue
        cat = index[cid]
        if li.get("name", "").strip() != cat["name"].strip():
            errors.append(f"GROUNDING: name mismatch for {cid}: claimed='{li.get('name')}' expected='{cat['name']}'.")
        ep = unit_price(cat)
        cp = float(li.get("unit_price_inr", li.get("unit_price", -999)))
        if abs(cp - ep) > 0.01:
            errors.append(f"GROUNDING: unit_price_inr mismatch for {cid}: claimed={cp} expected={ep}.")
        qty = int(li.get("quantity", 0))
        elt = round(ep * qty, 2)
        clt = round(float(li.get("line_total_inr", li.get("subtotal", -999))), 2)
        if abs(clt - elt) > 0.01:
            errors.append(f"GROUNDING: line_total_inr mismatch for {cid}: claimed={clt} expected={elt}.")
        computed += ep * qty

    claimed_total = float(itinerary.get("total_inr", quote.get("total", itinerary.get("total", -999))))
    if abs(claimed_total - round(computed, 2)) > 0.01:
        errors.append(f"GROUNDING: total_inr mismatch: claimed={claimed_total} recomputed={round(computed,2)}.")

    day_cited: set[str] = set()
    for day in itinerary.get("days", []):
        day_cited.update(day.get("catalog_ids", []))
        for key in ("hotel", "transport"):
            if day.get(key, {}).get("id"): day_cited.add(day[key]["id"])
        for act in day.get("activities", []):
            if act.get("id"): day_cited.add(act["id"])

    for cid in day_cited:
        if cid not in index:
            errors.append(f"GROUNDING: catalog_id '{cid}' cited in days does not exist in catalog.")
        elif cid not in li_ids:
            errors.append(f"GROUNDING: catalog_id '{cid}' cited in days but missing from line_items.")
    return errors

# ── CLI Formatting & Save ────────────────────────────────────────────────────

def format_itinerary_cli(state: PlannerState) -> str:
    itin = state.get("itinerary")
    if not itin or itin.get("status") != "fulfilled":
        return state.get("last_response", "Unable to fulfill itinerary.")
    dest, days = state.get("destination", "Destination"), state.get("days", 0)
    adults, kids, budget = state.get("adults", 1), state.get("children", 0), state.get("budget")
    travellers = f"{adults} adults" + (f", {kids} children" if kids else "")
    budget_str = f"₹{budget:,.0f}" if budget else "Flexible / Not specified"
    sep = "=" * 60
    lines = [sep, "FINAL ITINERARY", sep, f"Destination: {dest}", f"Duration:    {days} days",
             f"Travellers:  {travellers}", f"Budget:      {budget_str}", ""]
    for d in itin.get("days", []):
        lines.append(f"Day {d['day']} — {d.get('title', '')}")
        if d.get("hotel"):
            lines.append(f"- Hotel:    {d['hotel']['name']} ({d['hotel']['id']})")
        for act in d.get("activities", []):
            lines.append(f"- Activity: {act['name']} ({act['id']}) — {act.get('duration_hours','?')} hrs")
        if not d.get("activities"):
            lines.append("- Activity: Leisure & self-guided exploration")
        if d.get("notes"): lines.append(f"- Notes:    {d['notes']}")
        lines.append("")
    quote = itin.get("quote", {})
    lines += [sep, "QUOTE", sep]
    for li in quote.get("line_items", []):
        rooms_note = f" ({li['rooms']} rooms)" if li.get("rooms", 1) > 1 else ""
        lines.append(f"• {li['name']} ({li['catalog_id']}): ₹{li['unit_price_inr']:,.0f} × {li['quantity']}{rooms_note} = ₹{li['line_total_inr']:,.0f}")
    total = quote.get("total", 0.0)
    lines.append("-" * 60)
    lines.append(f"Total: ₹{total:,.0f}")
    if budget and total > budget:
        lines.append(f"Note: Total is ₹{total - budget:,.0f} above requested budget (using lowest catalog rates).")
    lines += [sep, "TRIP COMPLETE", sep]
    return "\n".join(lines)

def save_chat_output(state: PlannerState) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CHAT_{ts}.json"
    payload = {
        "request_id": f"CHAT_{ts}", "timestamp": ts,
        "request_text": state.get("latest_user_message", ""),
        "destination": state.get("destination"), "days": state.get("days"),
        "adults": state.get("adults"), "children": state.get("children", 0),
        "budget": state.get("budget"), "preferences": state.get("preferences", []),
        "itinerary": state.get("itinerary"),
        "grounded_and_valid": len(state.get("validation_errors", [])) == 0,
        "validation_errors": state.get("validation_errors", []),
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    path = OUTPUTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    (OUTPUTS_DIR / "CHAT.json").write_text(content, encoding="utf-8")
    state["last_saved_file"] = filename
    return path

# ── LangGraph Nodes ──────────────────────────────────────────────────────────

def node_understand_and_update(state: PlannerState) -> PlannerState:
    text = state.get("latest_user_message", "").strip()
    data = load_data()
    state["inquiry_response"] = None

    # One unified AI call: understands context, extracts slots, generates reply when needed
    ai = llm_understand_turn(text, state, data)
    reply = ai.pop("reply", None)
    if reply:
        state["inquiry_response"] = reply

    # Deterministic extraction runs as grounding override
    det = extract_slots_deterministic(text, state, data)
    for k, v in {**ai, **det}.items():
        if v is not None: state[k] = v  # type: ignore
    return state


def _simple_node(response_key: str, response_val: str | None = None):
    def node(state: PlannerState) -> PlannerState:
        state["last_response"] = response_val or state.get(response_key, "How can I help?")
        state["status"] = "in_progress"
        return state
    return node

def node_answer_inquiry(state: PlannerState) -> PlannerState:
    """Answer the inquiry, then append the next missing-slot question so the flow never stalls."""
    inquiry = state.get("inquiry_response", "How can I help?")
    # Determine next missing slot and append its question
    if not state.get("destination"):
        follow_up = f" Which destination would you prefer? ({', '.join(get_catalog_locations(load_data()))})"
    elif not state.get("days"):
        follow_up = f" How many days would you like to spend in {state.get('destination', 'your destination')}?"
    elif not state.get("adults"):
        follow_up = " How many people are travelling (adults and children)?"
    elif not state.get("preferences_asked"):
        follow_up = " What kind of experience do you prefer (e.g. nature, culture, adventure — or 'flexible')?"
    elif not state.get("budget_asked"):
        follow_up = " Do you have a target budget in mind (or type 'flexible')?"
    else:
        follow_up = ""
    state["last_response"] = inquiry.rstrip() + follow_up
    state["status"] = "in_progress"
    return state

def node_ask_destination(state: PlannerState) -> PlannerState:
    locs_str = ", ".join(get_catalog_locations(load_data()))
    state["pending_question"] = "destination"
    state["last_response"] = f"Where would you like to travel? We have inventory in: {locs_str} (Kerala)."
    state["status"] = "in_progress"
    return state

def node_ask_days(state: PlannerState) -> PlannerState:
    state["pending_question"] = "days"
    state["last_response"] = f"How many days would you like to spend in {state.get('destination', 'your destination')}?"
    state["status"] = "in_progress"
    return state

def node_ask_party(state: PlannerState) -> PlannerState:
    state["pending_question"] = "party"
    state["last_response"] = "How many people are travelling (adults and children)?"
    state["status"] = "in_progress"
    return state

def node_ask_preferences(state: PlannerState) -> PlannerState:
    state["pending_question"] = "preferences"
    state["preferences_asked"] = True
    dest = state.get("destination", "your destination")
    catalog = get_catalog(load_data())
    tags = sorted({t for i in catalog if i.get("location") in (dest, "Kerala") for t in i.get("tags", [])
                   if t not in ("city", "flexible", "transfer", "group")})
    state["last_response"] = f"What kind of experience do you prefer? (e.g. {', '.join(tags[:6])} — or type 'flexible')"
    state["status"] = "in_progress"
    return state

def node_ask_budget(state: PlannerState) -> PlannerState:
    state["pending_question"] = "budget"
    state["budget_asked"] = True
    state["last_response"] = "Do you have a target budget (e.g. ₹50,000, budget-conscious / mid-range / luxury)? (Or type 'flexible')"
    state["status"] = "in_progress"
    return state

def node_recover_invalid_dest(state: PlannerState) -> PlannerState:
    out = state.get("out_of_catalog_destination", "that location")
    locs_str = ", ".join(get_catalog_locations(load_data()))
    state["last_response"] = f"I don't have inventory for {out}. Available: {locs_str}. Which would you prefer?"
    state["out_of_catalog_destination"] = None
    state["destination"] = None
    state["status"] = "in_progress"
    return state

def node_plan_itinerary(state: PlannerState) -> PlannerState:
    data = load_data()
    itin = build_grounded_itinerary(
        destination=state.get("destination", "Munnar"), days=state.get("days", 3),
        adults=state.get("adults", 2), children=state.get("children", 0),
        budget=state.get("budget"), preferences=state.get("preferences", []),
        budget_level=state.get("budget_level"), data=data,
    )
    errors = validate_itinerary(itin, data)
    state.update(itinerary=itin, validation_errors=errors,
                 status="fulfilled" if itin.get("status") == "fulfilled" else "unfulfillable",
                 last_response=format_itinerary_cli({**state, "itinerary": itin}))  # type: ignore
    if not state.get("is_benchmark"):
        saved = save_chat_output(state)
        state.update(last_saved_file=saved.name, output_saved=True)
    return state

def route_next_step(state: PlannerState) -> str:
    if state.get("out_of_catalog_destination"): return "recover_invalid_dest"
    if state.get("inquiry_response"):
        if not all([state.get("destination"), state.get("days"), state.get("adults"),
                    state.get("preferences_asked"), state.get("budget_asked")]):
            return "answer_inquiry"
    if not state.get("destination"): return "ask_destination"
    if not state.get("days"):        return "ask_days"
    if not state.get("adults"):      return "ask_party"
    if not state.get("preferences_asked") and not state.get("preferences"): return "ask_preferences"
    if not state.get("budget_asked") and not state.get("budget") and not state.get("budget_level"): return "ask_budget"
    return "plan_itinerary"

def build_graph():
    g = StateGraph(PlannerState)
    nodes = ["understand", "answer_inquiry", "ask_destination", "ask_days", "ask_party",
             "ask_preferences", "ask_budget", "recover_invalid_dest", "plan_itinerary"]
    fns = [node_understand_and_update, node_answer_inquiry, node_ask_destination,
           node_ask_days, node_ask_party, node_ask_preferences, node_ask_budget,
           node_recover_invalid_dest, node_plan_itinerary]
    for name, fn in zip(nodes, fns):
        g.add_node(name, fn)
    g.set_entry_point("understand")
    g.add_conditional_edges("understand", route_next_step, {n: n for n in nodes[1:]})
    for n in nodes[1:]:
        g.add_edge(n, END)
    return g.compile()

def process_turn(app, state: PlannerState, user_input: str) -> tuple[PlannerState, str]:
    state["latest_user_message"] = user_input
    state["messages"].append({"role": "user", "content": user_input})
    result = app.invoke(state)
    response = result.get("last_response", "")
    result["messages"].append({"role": "planner", "content": response})
    return result, response
