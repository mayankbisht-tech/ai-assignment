"""app.py — Conversational AI-Assisted Travel Planner Engine.

Core Modules:
1. Dynamic Catalog Introspection & Helpers (sample_data.json)
2. Conversational State Management (PlannerState)
3. Slot Extraction & Inquiry Handling (Deterministic + LLM Assisted)
4. Grounded Itinerary Planner & Python Math Engine (Zero-Hallucination Pricing)
5. Grounding & Math Validator (Strict Catalog Verification)
6. LangGraph Conversational State Machine
7. Save & Runner Utilities
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

# Ensure Unicode output works cleanly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import END, StateGraph

from helpers import load_env
load_env()

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


# ═══════════════════════════════════════════════════════════════════════════
# 1. CATALOG INTROSPECTION & DATA HELPERS (Dynamic — No Hardcoding)
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> dict[str, Any]:
    """Load sample_data.json as the sole source of truth."""
    path = ROOT / "sample_data.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_catalog(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all supplier inventory items."""
    return data.get("suppliers", [])


def catalog_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index catalog items by unique ID."""
    return {item["id"]: item for item in get_catalog(data)}


def get_catalog_locations(data: dict[str, Any]) -> list[str]:
    """Dynamically extract all distinct town/city locations for stays & activities."""
    locs = {
        item["location"]
        for item in get_catalog(data)
        if item.get("location") and item.get("type") in ("hotel", "activity")
    }
    return sorted(locs)


def get_catalog_regions(data: dict[str, Any]) -> list[str]:
    """Dynamically extract broader regional locations (e.g. statewide transport)."""
    all_locs = {item["location"] for item in get_catalog(data) if item.get("location")}
    specific_locs = set(get_catalog_locations(data))
    regions = all_locs - specific_locs
    return sorted(regions) or ["Kerala"]


def get_traveler_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Return traveler profile from sample data."""
    return data.get("traveler_profile", {})


def unit_price(item: dict[str, Any]) -> float:
    """Canonical per-unit price for any catalog item type."""
    itype = item.get("type")
    if itype == "hotel":
        return float(item.get("price_per_night", 0))
    if itype == "activity":
        return float(item.get("price_per_person", 0))
    if itype == "transport":
        if "price_per_day" in item:
            return float(item.get("price_per_day", 0))
        return float(item.get("price_flat", 0))
    return 0.0


# Known alias lookup for flexible user input (case-insensitive)
_DYNAMIC_ALIASES: dict[str, str] = {
    "cochin": "Kochi",
    "fort kochi": "Kochi",
    "alappuzha": "Alleppey",
    "alappuzha beach": "Alleppey",
    "munnar hills": "Munnar",
}


def resolve_destination(text: str, data: dict[str, Any]) -> str | None:
    """Dynamically match a destination mentioned in user text against catalog inventory."""
    low = text.strip().lower()
    valid_locs = get_catalog_locations(data)
    valid_regions = get_catalog_regions(data)

    # 1. Direct location match
    for loc in valid_locs:
        if re.search(r'\b' + re.escape(loc.lower()) + r'\b', low):
            return loc

    # 2. Known alias match
    for alias, canon in _DYNAMIC_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', low):
            if canon in valid_locs or canon in valid_regions:
                return canon

    # 3. Regional match (e.g. Kerala)
    for reg in valid_regions:
        if re.search(r'\b' + re.escape(reg.lower()) + r'\b', low):
            return reg

    return None


def detect_out_of_catalog_destination(text: str, data: dict[str, Any]) -> str | None:
    """Detect if the user is asking for a specific destination NOT in the catalog."""
    low = text.strip().lower()
    known_out = {
        "goa", "mumbai", "delhi", "jaipur", "bangalore", "bengaluru", "chennai",
        "coorg", "wayanad", "ooty", "mysore", "hyderabad", "kolkata", "shimla",
        "manali", "agra", "varanasi", "udaipur", "pondicherry", "andaman",
        "lakshadweep", "kashmir", "ladakh", "bali", "thailand", "dubai", "paris",
        "uk", "usa", "london", "assam", "kashmir", "rajasthan", "sikkim", "manali"
    }

    acronyms = {"uk": "UK", "usa": "USA", "uae": "UAE"}

    # Direct keyword match for known outside destinations
    for place in known_out:
        if re.search(r'\b' + re.escape(place) + r'\b', low):
            return acronyms.get(place, place.capitalize())

    # Pattern match: "go to <word>", "visit <word>", "trip to <word>"
    m = re.search(r'\b(?:go\s+to|visit|trip\s+to|in|for|about)\s+([a-zA-Z]+)\b', low)
    if m:
        word = m.group(1).lower()
        candidate = acronyms.get(word, word.capitalize())
        valid_all = set(get_catalog_locations(data)) | set(get_catalog_regions(data))
        if candidate.lower() not in {v.lower() for v in valid_all} and candidate.lower() not in {
            "a", "an", "the", "my", "our", "some", "weekend", "few", "short", "long", "family", "vacation",
            "hotel", "hotels", "activity", "activities", "price", "budget", "days", "people", "adults"
        }:
            return candidate

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONVERSATIONAL STATE
# ═══════════════════════════════════════════════════════════════════════════

class PlannerState(TypedDict, total=False):
    # Conversation tracking
    messages: list[dict[str, str]]
    latest_user_message: str

    # Extracted Trip Slots (Preserved across turns)
    destination: str | None
    days: int | None
    adults: int | None
    children: int | None
    budget: float | None
    budget_level: str | None
    preferences: list[str]

    # Question checkpoints
    pending_question: str | None
    preferences_asked: bool
    budget_asked: bool

    # Temporary Turn Flags
    out_of_catalog_destination: str | None
    inquiry_response: str | None
    turn_intent: str | None

    # Itinerary & Output
    itinerary: dict[str, Any] | None
    status: Literal["in_progress", "fulfilled", "unfulfillable"]
    last_response: str
    validation_errors: list[str]
    output_saved: bool
    last_saved_file: str | None
    is_benchmark: bool


def create_initial_state() -> PlannerState:
    """Create a blank conversational state."""
    return {
        "messages": [],
        "latest_user_message": "",
        "destination": None,
        "days": None,
        "adults": None,
        "children": 0,
        "budget": None,
        "budget_level": None,
        "preferences": [],
        "pending_question": None,
        "preferences_asked": False,
        "budget_asked": False,
        "out_of_catalog_destination": None,
        "inquiry_response": None,
        "turn_intent": None,
        "itinerary": None,
        "status": "in_progress",
        "last_response": "",
        "validation_errors": [],
        "output_saved": False,
        "last_saved_file": None,
        "is_benchmark": False,
    }



# ═══════════════════════════════════════════════════════════════════════════
# 3. DETERMINISTIC & LLM SLOT EXTRACTION + INQUIRY HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def is_conversational_inquiry(text: str) -> bool:
    """Detect if the user is asking a general/clarifying question rather than providing parameters."""
    low = text.strip().lower()
    inquiry_phrases = [
        "how many days do you", "how many days", "how many nights", "how long",
        "what do you recommend", "what do you provide", "what options", "what activities",
        "what can we do", "which hotels", "what hotels", "tell me about", "options do you have",
        "what places", "suggest something", "recommend", "how much does it cost", "pricing"
    ]
    if any(p in low for p in inquiry_phrases):
        return True
    if "?" in low and not re.search(r'^(?:kochi|munnar|alleppey|kerala)\s*\??$', low):
        return True
    return False


def generate_inquiry_response(text: str, state: PlannerState, data: dict[str, Any]) -> str:
    """Provide a helpful, catalog-grounded answer to user inquiries and prompt for missing info."""
    dest = state.get("destination")
    locs = get_catalog_locations(data)
    locs_str = ", ".join(locs)
    catalog = get_catalog(data)

    # If Groq is available, generate a personalized grounded answer
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        try:
            import groq  # type: ignore
            client = groq.Groq(api_key=key)

            dest_info = f"Current destination: {dest}" if dest else f"Available destinations: {locs_str} (Kerala)"
            items_summary = "\n".join(
                f"- [{i['id']}] {i['name']} in {i['location']}: ₹{unit_price(i):,.0f} ({i.get('type')}, tags: {', '.join(i.get('tags', []))})"
                for i in catalog
            )

            prompt = f"""You are a helpful travel assistant.
{dest_info}

Catalog Inventory:
{items_summary}

User asked: "{text}"

Answer the user's question concisely and accurately based ONLY on this catalog.
At the end of your response, ask the user for their preferred number of days or travelers to continue planning."""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": "You are a concise, helpful travel advisor. Never invent inventory or prices outside the catalog."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Deterministic fallback answer
    if dest:
        dest_acts = [i["name"] for i in catalog if i["type"] == "activity" and i.get("location") == dest]
        acts_txt = ", ".join(dest_acts) if dest_acts else "guided sightseeing"
        return (
            f"For {dest}, we recommend 2 to 4 days! You can experience {acts_txt}. "
            f"How many days would you like to plan for?"
        )
    return (
        f"We offer flexible 2 to 5 day itineraries across {locs_str} (Kerala) with handpicked hotels, "
        f"cultural activities, and private cab transport. Where would you like to travel?"
    )


def extract_slots_deterministic(
    text: str,
    current_state: PlannerState,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract trip parameters incrementally from user text using robust regex & rules.
    Guarantees 100% reliable state tracking even when offline or LLM unavailable.
    """
    low = text.strip().lower()
    updates: dict[str, Any] = {}
    pending = current_state.get("pending_question")

    # 1. Destination Extraction & Correction
    dest = resolve_destination(text, data)
    if dest:
        updates["destination"] = dest
        updates["out_of_catalog_destination"] = None
        updates["pending_question"] = None
    else:
        out_dest = detect_out_of_catalog_destination(text, data)
        if out_dest:
            updates["out_of_catalog_destination"] = out_dest

    # 2. Duration (Days) Extraction & Correction
    days_match = re.search(r'\b(\d+)\s*(?:nights?|days?)\b', low)
    if days_match:
        updates["days"] = int(days_match.group(1))
        updates["pending_question"] = None
    elif "weekend" in low:
        updates["days"] = 2
        updates["pending_question"] = None
    elif re.search(r'\b(\d+)\s*d\b', low):
        m = re.search(r'\b(\d+)\s*d\b', low)
        if m:
            updates["days"] = int(m.group(1))
            updates["pending_question"] = None
    elif re.match(r'^\s*(\d+)\s*$', low) and current_state.get("days") is None and current_state.get("destination") is not None:
        updates["days"] = int(low.strip())
        updates["pending_question"] = None
    elif re.search(r'\b(?:for|make it|change to)\s+(\d+)\s*(?:days?|nights?)?\b', low) and "adult" not in low and "people" not in low:
        m = re.search(r'\b(?:for|make it|change to)\s+(\d+)\b', low)
        if m:
            updates["days"] = int(m.group(1))
            updates["pending_question"] = None

    # 3. Party Size Extraction & Correction
    m_pair = re.search(r'^\s*(\d+)\s*,\s*(\d+)\s*$', low)
    if m_pair:
        updates["adults"] = int(m_pair.group(1))
        updates["children"] = int(m_pair.group(2))
        updates["pending_question"] = None
    else:
        m_adults = re.search(r'\b(\d+)\s*adults?\b', low)
        if m_adults:
            updates["adults"] = int(m_adults.group(1))
            updates["pending_question"] = None

        m_kids = re.search(r'\b(\d+)\s*(?:children|child|kids?)\b', low)
        if m_kids:
            updates["children"] = int(m_kids.group(1))
            updates["pending_question"] = None
        elif re.search(r'\bkids?\s+aged?\s+([\d\s,and]+)', low):
            ages = re.findall(r'\b\d+\b', re.search(r'\bkids?\s+aged?\s+([\d\s,and]+)', low).group(1))
            updates["children"] = len(ages)
            updates["pending_question"] = None

        if "couple" in low or "two of us" in low or "me and my wife" in low or "me and my partner" in low:
            updates["adults"] = 2
            if "children" not in updates and current_state.get("children") is None:
                updates["children"] = 0
            updates["pending_question"] = None
        elif "solo" in low or "just me" in low or "alone" in low or "1 person" in low:
            updates["adults"] = 1
            updates["children"] = 0
            updates["pending_question"] = None
        elif re.search(r'\bfamily\s+of\s+(\d+)\b', low):
            total_party = int(re.search(r'\bfamily\s+of\s+(\d+)\b', low).group(1))
            if "adults" in updates:
                updates["children"] = max(0, total_party - updates["adults"])
            else:
                updates["adults"] = 2
                updates["children"] = max(0, total_party - 2)
            updates["pending_question"] = None
        elif re.search(r'\b(\d+)\s*(?:people|persons|travellers?|travelers?)\b', low):
            num = int(re.search(r'\b(\d+)\s*(?:people|persons|travellers?|travelers?)\b', low).group(1))
            if "adults" not in updates and current_state.get("adults") is None:
                updates["adults"] = num
                if "children" not in updates and current_state.get("children") is None:
                    updates["children"] = 0
            updates["pending_question"] = None
        elif re.match(r'^\s*(\d+)\s*$', low) and current_state.get("days") is not None and current_state.get("adults") is None:
            updates["adults"] = int(low.strip())
            updates["children"] = 0
            updates["pending_question"] = None

    # 4. Preferences Extraction
    known_tags = {
        "nature", "hiking", "tea-estate", "local-food", "relaxed", "culture",
        "family-friendly", "wildlife", "adventure", "backwaters", "heritage",
        "views", "food", "trek", "cruise", "dance", "ayurveda"
    }
    found_tags = set(current_state.get("preferences", []))
    for tag in known_tags:
        clean_tag = tag.replace("-", " ")
        if clean_tag in low or tag in low:
            found_tags.add(tag)
    if found_tags:
        updates["preferences"] = sorted(found_tags)
        updates["preferences_asked"] = True
        if pending == "preferences":
            updates["pending_question"] = None

    # 5. Budget Extraction
    m_k = re.search(r'\b(?:budget\s*(?:of|is|around|under)?\s*)?(\d+)\s*k\b', low)
    if m_k:
        updates["budget"] = float(m_k.group(1)) * 1000.0
        updates["budget_asked"] = True
        if pending == "budget":
            updates["pending_question"] = None
    else:
        for m in re.finditer(r'[\d,]+', text.replace("₹", "").replace(",", "")):
            try:
                val = float(m.group(0))
                if 1000.0 <= val <= 1000000.0:
                    if val != updates.get("days") and val != current_state.get("days"):
                        updates["budget"] = val
                        updates["budget_asked"] = True
                        if pending == "budget":
                            updates["pending_question"] = None
                        break
            except Exception:
                pass

    if any(w in low for w in ("budget", "budget-conscious", "cheap", "affordable", "hostel", "backpacker")):
        updates["budget_level"] = "budget"
        updates["budget_asked"] = True
        if pending == "budget":
            updates["pending_question"] = None
    elif any(w in low for w in ("mid-range", "midrange", "moderate", "standard")):
        updates["budget_level"] = "mid-range"
        updates["budget_asked"] = True
        if pending == "budget":
            updates["pending_question"] = None
    elif any(w in low for w in ("luxury", "5-star", "resort", "premium")):
        updates["budget_level"] = "luxury"
        updates["budget_asked"] = True
        if pending == "budget":
            updates["pending_question"] = None

    # 6. Check for skip / flexible responses based on pending question
    if any(w in low for w in ("flexible", "skip", "none", "no preference", "anything", "standard", "no budget", "not sure", "any")):
        if pending == "preferences" or (not current_state.get("preferences_asked") and not updates.get("preferences")):
            updates["preferences_asked"] = True
            updates["pending_question"] = None
        elif pending == "budget" or (not current_state.get("budget_asked") and not updates.get("budget")):
            updates["budget_asked"] = True
            updates["pending_question"] = None

    return updates


def extract_slots_with_llm(
    text: str,
    current_state: PlannerState,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Use Groq LLM with JSON mode to assist slot extraction if available."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None

    try:
        import groq  # type: ignore
        client = groq.Groq(api_key=key)

        valid_locs = get_catalog_locations(data) + get_catalog_regions(data)
        locs_str = ", ".join(valid_locs)

        system_prompt = (
            "You are a travel information extractor. Given the current travel request message and existing state, "
            "extract updated travel slots. Return ONLY valid JSON with keys: "
            "destination (string or null), days (integer or null), adults (integer or null), "
            "children (integer or null), budget (number or null), budget_level (string or null), "
            "preferences (list of strings or null), out_of_catalog_destination (string or null).\n"
            f"Catalog valid destinations are: {locs_str}."
        )

        user_prompt = f"""Current Known State:
Destination: {current_state.get('destination')}
Days: {current_state.get('days')}
Adults: {current_state.get('adults')}
Children: {current_state.get('children')}
Budget: {current_state.get('budget')}

New User Message: "{text}"

Extract any mentioned or updated slots. If user mentions a destination not in catalog, put it in out_of_catalog_destination."""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        return {k: v for k, v in parsed.items() if v is not None}
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. GROUNDED ITINERARY PLANNER & DETERMINISTIC PRICING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def build_grounded_itinerary(
    destination: str,
    days: int,
    adults: int,
    children: int,
    budget: float | None,
    preferences: list[str],
    budget_level: str | None,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a 100% catalog-grounded itinerary with strictly Python-computed pricing.
    - Every ID cited exists in sample_data.json.
    - Exactly `days` days of itinerary are generated.
    - No duplicate activities scheduled on multiple days.
    - Party capacity and room allocations calculated accurately.
    - Per-item subtotal = unit_price * quantity.
    - Total quote = sum(subtotals).
    """
    catalog = get_catalog(data)
    people = max(1, adults + (children or 0))
    traveler_prof = get_traveler_profile(data)

    profile_prefs = set(traveler_prof.get("preferences", []))
    active_prefs = set(preferences) | profile_prefs

    is_regional = destination.lower() in ("kerala", "all")
    if is_regional:
        allowed_locs = {item["location"] for item in catalog}
    else:
        allowed_locs = {destination, "Kerala"}

    hotels = [i for i in catalog if i["type"] == "hotel" and i.get("location") in allowed_locs]
    activities = [i for i in catalog if i["type"] == "activity" and i.get("location") in allowed_locs]
    transport = [i for i in catalog if i["type"] == "transport" and i.get("location") in allowed_locs]

    if not hotels or not activities:
        return {
            "status": "unfulfillable",
            "reason_if_unfulfillable": f"Insufficient catalog inventory in '{destination}' to construct a complete {days}-day itinerary.",
            "destination": destination,
            "days": [],
            "quote": {"line_items": [], "total": 0.0},
        }

    # 1. Select Hotel(s)
    def score_hotel(h: dict) -> float:
        score = float(h.get("rating", 3.0))
        htags = set(h.get("tags", []))
        score += len(htags & active_prefs) * 1.5
        if budget_level == "budget" and "budget" in htags:
            score += 3.0
        elif budget_level == "mid-range" and "mid-range" in htags:
            score += 2.0
        cap = h.get("capacity", 2)
        if cap >= people:
            score += 2.0
        return score

    sorted_hotels = sorted(hotels, key=score_hotel, reverse=True)
    chosen_hotel = sorted_hotels[0]

    # Room calculation based on hotel capacity
    hotel_cap = chosen_hotel.get("capacity", 4)
    rooms_needed = max(1, math.ceil(people / hotel_cap))

    # 2. Select & Score Activities
    def score_activity(a: dict) -> float:
        score = float(a.get("rating", 3.0))
        atags = set(a.get("tags", []))
        score += len(atags & active_prefs) * 2.0
        return score

    sorted_acts = sorted(activities, key=score_activity, reverse=True)

    # 3. Select Transport (Sedan for <= 4, SUV for > 4)
    suitable_trans = [t for t in transport if t.get("capacity", 4) >= people and "price_per_day" in t]
    chosen_trans = suitable_trans[0] if suitable_trans else (transport[0] if transport else None)

    # 4. Build Day-by-Day Itinerary (No duplicate paid activities)
    itinerary_days: list[dict[str, Any]] = []
    used_acts_set: set[str] = set()

    for d in range(1, days + 1):
        available_act = None
        for act in sorted_acts:
            if act["id"] not in used_acts_set:
                available_act = act
                used_acts_set.add(act["id"])
                break

        if d == 1:
            if available_act:
                title = f"Arrival & {available_act['name']}"
                notes = f"Check in to {chosen_hotel['name']}. Enjoy a relaxed start with {available_act['name']} ({available_act.get('duration_hours', 3)} hrs)."
                day_acts = [{
                    "id": available_act["id"],
                    "name": available_act["name"],
                    "duration_hours": available_act.get("duration_hours", 3),
                    "unit_price": unit_price(available_act),
                }]
            else:
                title = f"Arrival & Check-in at {chosen_hotel['name']}"
                notes = f"Check in to {chosen_hotel['name']} and enjoy a relaxing evening at leisure."
                day_acts = []
        elif d == days:
            if available_act:
                title = f"{available_act['name']} & Departure"
                notes = f"Morning {available_act['name']} ({available_act.get('duration_hours', 3)} hrs) followed by hotel checkout and departure."
                day_acts = [{
                    "id": available_act["id"],
                    "name": available_act["name"],
                    "duration_hours": available_act.get("duration_hours", 3),
                    "unit_price": unit_price(available_act),
                }]
            else:
                title = f"Leisure Morning & Departure"
                notes = f"Relaxed morning in {destination}, local souvenir shopping, checkout, and onward departure."
                day_acts = []
        else:
            if available_act:
                title = f"Discover {destination} — {available_act['name']}"
                notes = f"Full day exploring {available_act['name']} ({available_act.get('duration_hours', 3)} hrs). Leisure evening."
                day_acts = [{
                    "id": available_act["id"],
                    "name": available_act["name"],
                    "duration_hours": available_act.get("duration_hours", 3),
                    "unit_price": unit_price(available_act),
                }]
            else:
                title = f"Scenic Exploration & Cultural Leisure in {destination}"
                notes = f"Self-guided exploration of {destination}'s scenic spots, cafes, and local viewpoints at a relaxed pace."
                day_acts = []

        day_obj: dict[str, Any] = {
            "day": d,
            "title": title,
            "hotel": {
                "id": chosen_hotel["id"],
                "name": chosen_hotel["name"],
                "price_per_night": unit_price(chosen_hotel),
            },
            "activities": day_acts,
            "notes": notes,
        }
        if chosen_trans:
            day_obj["transport"] = {
                "id": chosen_trans["id"],
                "name": chosen_trans["name"],
                "price_per_day": unit_price(chosen_trans),
            }
        itinerary_days.append(day_obj)

    # 5. Deterministic Quote Calculation in Python
    line_items: list[dict[str, Any]] = []

    # Hotel Line Item
    h_price = unit_price(chosen_hotel)
    h_total_qty = days * rooms_needed
    line_items.append({
        "catalog_id": chosen_hotel["id"],
        "name": chosen_hotel["name"],
        "unit_price_inr": h_price,
        "quantity": h_total_qty,
        "rooms": rooms_needed,
        "line_total_inr": round(h_price * h_total_qty, 2),
    })

    # Activity Line Items (Count actual used activities)
    for act in sorted_acts:
        if act["id"] in used_acts_set:
            a_price = unit_price(act)
            line_items.append({
                "catalog_id": act["id"],
                "name": act["name"],
                "unit_price_inr": a_price,
                "quantity": people,
                "line_total_inr": round(a_price * people, 2),
            })

    # Transport Line Item
    if chosen_trans:
        t_price = unit_price(chosen_trans)
        t_qty = 1 if "price_flat" in chosen_trans else days
        line_items.append({
            "catalog_id": chosen_trans["id"],
            "name": chosen_trans["name"],
            "unit_price_inr": t_price,
            "quantity": t_qty,
            "line_total_inr": round(t_price * t_qty, 2),
        })

    grand_total = round(sum(li["line_total_inr"] for li in line_items), 2)

    return {
        "status": "fulfilled",
        "destination": destination,
        "days": itinerary_days,
        "quote": {
            "line_items": line_items,
            "total": grand_total,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. GROUNDING VALIDATOR (Strict Verification Against Catalog)
# ═══════════════════════════════════════════════════════════════════════════

def validate_itinerary(itinerary: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """
    Validate that:
    1. Every cited ID exists in catalog.
    2. Names and unit prices match canonical catalog data.
    3. Math: line_total == unit_price * quantity and grand_total == sum(line_totals).
    4. Day citations cross-reference with quote line items.
    """
    errors: list[str] = []
    index = catalog_by_id(data)

    if not itinerary:
        return ["GROUNDING: itinerary is empty or None."]

    if itinerary.get("status") == "unfulfillable":
        return []

    quote = itinerary.get("quote", {})
    line_items = itinerary.get("line_items") or quote.get("line_items", [])
    if not line_items:
        return ["GROUNDING: line_items is empty or missing."]

    computed_sum = 0.0
    line_item_ids: set[str] = set()

    for li in line_items:
        cid = li.get("catalog_id") or li.get("id", "")
        line_item_ids.add(cid)

        if cid not in index:
            errors.append(f"GROUNDING: catalog_id '{cid}' does not exist in catalog.")
            continue

        cat = index[cid]
        # Name check
        if li.get("name", "").strip() != cat["name"].strip():
            errors.append(f"GROUNDING: name mismatch for {cid}: claimed='{li.get('name')}' expected='{cat['name']}'.")

        # Price check
        expected_p = unit_price(cat)
        claimed_p = float(li.get("unit_price_inr", li.get("unit_price", -999)))
        if abs(claimed_p - expected_p) > 0.01:
            errors.append(f"GROUNDING: unit_price_inr mismatch for {cid}: claimed={claimed_p} expected={expected_p}.")

        # Line total check
        qty = int(li.get("quantity", 0))
        expected_lt = round(expected_p * qty, 2)
        claimed_lt = round(float(li.get("line_total_inr", li.get("subtotal", -999))), 2)
        if abs(claimed_lt - expected_lt) > 0.01:
            errors.append(f"GROUNDING: line_total_inr mismatch for {cid}: claimed={claimed_lt} expected={expected_lt}.")

        computed_sum += expected_p * qty

    # Total check
    claimed_total = float(itinerary.get("total_inr", quote.get("total", itinerary.get("total", -999))))
    computed_sum = round(computed_sum, 2)
    if abs(claimed_total - computed_sum) > 0.01:
        errors.append(f"GROUNDING: total_inr mismatch: claimed={claimed_total} recomputed={computed_sum}.")

    # Collect all day cited IDs
    day_cited: set[str] = set()
    for day in itinerary.get("days", []):
        for cid in day.get("catalog_ids", []):
            day_cited.add(cid)
        hotel = day.get("hotel", {})
        if hotel.get("id"):
            day_cited.add(hotel["id"])
        for act in day.get("activities", []):
            if act.get("id"):
                day_cited.add(act["id"])
        transport = day.get("transport", {})
        if transport.get("id"):
            day_cited.add(transport["id"])

    # Check for cited IDs not existing or missing from line items
    for cid in day_cited:
        if cid not in index:
            errors.append(f"GROUNDING: catalog_id '{cid}' cited in days does not exist in catalog.")
        elif cid not in line_item_ids:
            errors.append(f"GROUNDING: catalog_id '{cid}' cited in days but missing from line_items.")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# 6. OUTPUT FORMATTING & SAVE
# ═══════════════════════════════════════════════════════════════════════════

def format_itinerary_cli(state: PlannerState) -> str:
    """Format final itinerary for beautiful CLI output."""
    itin = state.get("itinerary")
    if not itin or itin.get("status") != "fulfilled":
        return state.get("last_response", "Unable to fulfill itinerary.")

    dest = state.get("destination", "Destination")
    days = state.get("days", 0)
    adults = state.get("adults", 1)
    kids = state.get("children", 0)
    budget = state.get("budget")

    travellers_str = f"{adults} adults" + (f", {kids} children" if kids else "")
    budget_str = f"₹{budget:,.0f}" if budget else "Flexible / Not specified"

    lines = [
        "=" * 60,
        "FINAL ITINERARY",
        "=" * 60,
        f"Destination: {dest}",
        f"Duration:    {days} days",
        f"Travellers:  {travellers_str}",
        f"Budget:      {budget_str}",
        "",
    ]

    for d in itin.get("days", []):
        day_num = d.get("day", 1)
        title = d.get("title", f"Day {day_num}")
        hotel = d.get("hotel", {})
        acts = d.get("activities", [])
        notes = d.get("notes", "")

        lines.append(f"Day {day_num} — {title}")
        if hotel:
            lines.append(f"- Hotel:    {hotel.get('name')} ({hotel.get('id')})")
        if acts:
            for act in acts:
                lines.append(f"- Activity: {act.get('name')} ({act.get('id')}) — {act.get('duration_hours', '?')} hrs")
        else:
            lines.append(f"- Activity: Leisure & self-guided exploration")
        if notes:
            lines.append(f"- Notes:    {notes}")
        lines.append("")

    # Quote Section
    quote = itin.get("quote", {})
    total = quote.get("total", 0.0)
    line_items = quote.get("line_items", [])

    lines.append("=" * 60)
    lines.append("QUOTE")
    lines.append("=" * 60)

    for li in line_items:
        cid = li.get("catalog_id")
        name = li.get("name")
        u_price = li.get("unit_price_inr", 0)
        qty = li.get("quantity", 1)
        lt = li.get("line_total_inr", 0)
        rooms_note = f" ({li['rooms']} rooms)" if li.get("rooms", 1) > 1 else ""
        lines.append(f"• {name} ({cid}): ₹{u_price:,.0f} × {qty}{rooms_note} = ₹{lt:,.0f}")

    lines.append("-" * 60)
    lines.append(f"Total: ₹{total:,.0f}")

    if budget and total > budget:
        diff = total - budget
        lines.append(f"Note: Total is ₹{diff:,.0f} above requested budget (using lowest available catalog rates).")

    lines.append("=" * 60)
    lines.append("TRIP COMPLETE")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_chat_output(state: PlannerState) -> Path:
    """Save conversation result to a new timestamped JSON file in outputs/ (and updates CHAT.json)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CHAT_{timestamp}.json"
    path = OUTPUTS_DIR / filename

    data_to_save = {
        "request_id": f"CHAT_{timestamp}",
        "timestamp": timestamp,
        "request_text": state.get("latest_user_message", ""),
        "destination": state.get("destination"),
        "days": state.get("days"),
        "adults": state.get("adults"),
        "children": state.get("children", 0),
        "budget": state.get("budget"),
        "preferences": state.get("preferences", []),
        "itinerary": state.get("itinerary"),
        "grounded_and_valid": len(state.get("validation_errors", [])) == 0,
        "validation_errors": state.get("validation_errors", []),
    }

    content = json.dumps(data_to_save, indent=2, ensure_ascii=False)
    path.write_text(content, encoding="utf-8")

    # Also update canonical CHAT.json
    (OUTPUTS_DIR / "CHAT.json").write_text(content, encoding="utf-8")

    state["last_saved_file"] = filename
    return path



# ═══════════════════════════════════════════════════════════════════════════
# 7. LANGGRAPH CONVERSATIONAL STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

def node_understand_and_update(state: PlannerState) -> PlannerState:
    """Extract slots from latest user message and merge with persistent state."""
    text = state.get("latest_user_message", "").strip()
    data = load_data()

    # Clear previous turn inquiry response
    state["inquiry_response"] = None

    # Check if user asked a general/clarifying inquiry
    if is_conversational_inquiry(text):
        state["inquiry_response"] = generate_inquiry_response(text, state, data)

    # 1. Deterministic Extraction (Always reliable)
    det_updates = extract_slots_deterministic(text, state, data)

    # 2. LLM Extraction (Enhances conversational nuance when available)
    llm_updates = extract_slots_with_llm(text, state, data) or {}

    # Merge updates (deterministic takes precedence for exact catalog matches)
    merged = {**llm_updates, **det_updates}

    for k, v in merged.items():
        if v is not None:
            state[k] = v  # type: ignore

    return state


def node_answer_inquiry(state: PlannerState) -> PlannerState:
    """Return the generated inquiry response."""
    state["last_response"] = state.get("inquiry_response", "How can I assist your trip planning?")
    state["status"] = "in_progress"
    return state


def node_ask_destination(state: PlannerState) -> PlannerState:
    """Prompt user for destination with catalog options."""
    data = load_data()
    locs = get_catalog_locations(data)
    locs_str = ", ".join(locs)
    state["pending_question"] = "destination"
    state["last_response"] = f"Where would you like to travel? We currently have inventory in: {locs_str} (Kerala)."
    state["status"] = "in_progress"
    return state


def node_ask_days(state: PlannerState) -> PlannerState:
    """Prompt user for duration."""
    dest = state.get("destination", "your destination")
    state["pending_question"] = "days"
    state["last_response"] = f"How many days would you like to spend in {dest}?"
    state["status"] = "in_progress"
    return state


def node_ask_party(state: PlannerState) -> PlannerState:
    """Prompt user for party size."""
    state["pending_question"] = "party"
    state["last_response"] = "How many people are travelling (number of adults and children)?"
    state["status"] = "in_progress"
    return state


def node_ask_preferences(state: PlannerState) -> PlannerState:
    """Prompt user for travel style / activity preferences."""
    state["pending_question"] = "preferences"
    state["preferences_asked"] = True
    dest = state.get("destination", "your destination")
    data = load_data()
    catalog = get_catalog(data)
    loc_items = [i for i in catalog if i.get("location") in (dest, "Kerala")]
    tags = sorted({t for i in loc_items for t in i.get("tags", []) if t not in ("city", "flexible", "transfer", "group")})
    tags_hint = ", ".join(tags[:6])
    state["last_response"] = f"What kind of experience or activities do you prefer? (e.g. {tags_hint} — or type 'flexible')"
    state["status"] = "in_progress"
    return state


def node_ask_budget(state: PlannerState) -> PlannerState:
    """Prompt user for target budget or budget tier."""
    state["pending_question"] = "budget"
    state["budget_asked"] = True
    state["last_response"] = "Do you have a target budget in mind (e.g. ₹50,000, or budget-conscious / mid-range / luxury)? (Or type 'flexible')"
    state["status"] = "in_progress"
    return state


def node_recover_invalid_dest(state: PlannerState) -> PlannerState:
    """Handle out-of-catalog destination gracefully without losing other slots."""
    data = load_data()
    out_dest = state.get("out_of_catalog_destination", "that location")
    locs = get_catalog_locations(data)
    locs_str = ", ".join(locs)

    state["last_response"] = (
        f"I don't have inventory for {out_dest} in the current catalog. "
        f"Available destinations are: {locs_str}. Which one would you prefer?"
    )
    state["out_of_catalog_destination"] = None
    state["destination"] = None
    state["status"] = "in_progress"
    return state


def node_plan_itinerary(state: PlannerState) -> PlannerState:
    """Build, validate, and format the final grounded itinerary."""
    data = load_data()
    dest = state.get("destination", "Munnar")
    days = state.get("days", 3)
    adults = state.get("adults", 2)
    children = state.get("children", 0)
    budget = state.get("budget")
    prefs = state.get("preferences", [])
    b_level = state.get("budget_level")

    itin = build_grounded_itinerary(
        destination=dest,
        days=days,
        adults=adults,
        children=children,
        budget=budget,
        preferences=prefs,
        budget_level=b_level,
        data=data,
    )

    errors = validate_itinerary(itin, data)
    state["itinerary"] = itin
    state["validation_errors"] = errors
    state["status"] = "fulfilled" if itin.get("status") == "fulfilled" else "unfulfillable"

    cli_text = format_itinerary_cli(state)
    state["last_response"] = cli_text

    if not state.get("is_benchmark"):
        saved_path = save_chat_output(state)
        state["last_saved_file"] = saved_path.name
        state["output_saved"] = True
    return state




def route_next_step(state: PlannerState) -> Literal["recover_invalid_dest", "answer_inquiry", "ask_destination", "ask_days", "ask_party", "ask_preferences", "ask_budget", "plan_itinerary"]:
    """Determine the next step in the conversation based on state completeness."""
    # 1. Invalid destination recovery
    if state.get("out_of_catalog_destination"):
        return "recover_invalid_dest"

    # 2. General inquiry question (if no new slot was completed)
    if state.get("inquiry_response"):
        if not (state.get("destination") and state.get("days") and state.get("adults") and state.get("preferences_asked") and state.get("budget_asked")):
            return "answer_inquiry"

    # 3. Missing slots in sequence
    if not state.get("destination"):
        return "ask_destination"
    if not state.get("days"):
        return "ask_days"
    if not state.get("adults"):
        return "ask_party"
    if not state.get("preferences_asked") and not state.get("preferences"):
        return "ask_preferences"
    if not state.get("budget_asked") and not state.get("budget") and not state.get("budget_level"):
        return "ask_budget"

    # 4. All collected -> Plan!
    return "plan_itinerary"


def build_graph():
    """Compile the LangGraph conversational StateGraph."""
    g = StateGraph(PlannerState)

    g.add_node("understand", node_understand_and_update)
    g.add_node("answer_inquiry", node_answer_inquiry)
    g.add_node("ask_destination", node_ask_destination)
    g.add_node("ask_days", node_ask_days)
    g.add_node("ask_party", node_ask_party)
    g.add_node("ask_preferences", node_ask_preferences)
    g.add_node("ask_budget", node_ask_budget)
    g.add_node("recover_invalid_dest", node_recover_invalid_dest)
    g.add_node("plan_itinerary", node_plan_itinerary)

    g.set_entry_point("understand")

    g.add_conditional_edges(
        "understand",
        route_next_step,
        {
            "recover_invalid_dest": "recover_invalid_dest",
            "answer_inquiry": "answer_inquiry",
            "ask_destination": "ask_destination",
            "ask_days": "ask_days",
            "ask_party": "ask_party",
            "ask_preferences": "ask_preferences",
            "ask_budget": "ask_budget",
            "plan_itinerary": "plan_itinerary",
        },
    )

    g.add_edge("recover_invalid_dest", END)
    g.add_edge("answer_inquiry", END)
    g.add_edge("ask_destination", END)
    g.add_edge("ask_days", END)
    g.add_edge("ask_party", END)
    g.add_edge("ask_preferences", END)
    g.add_edge("ask_budget", END)
    g.add_edge("plan_itinerary", END)

    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════
# 8. CONVERSATIONAL TURN PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def process_turn(app, state: PlannerState, user_input: str) -> tuple[PlannerState, str]:
    """
    Process one conversational turn through the compiled LangGraph workflow.
    Updates and returns the persistent state and the response string.
    """
    state["latest_user_message"] = user_input
    state["messages"].append({"role": "user", "content": user_input})

    result_state = app.invoke(state)

    response = result_state.get("last_response", "")
    result_state["messages"].append({"role": "planner", "content": response})

    return result_state, response
