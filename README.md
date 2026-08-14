# Conversational AI Travel Planner

A conversational, 100% catalog-grounded travel planning assistant built with **LangGraph**. Every hotel, activity, transport item, and price in the output is traced directly to `sample_data.json` — the system **never fabricates inventory or prices**.

---

## Quick Start

You only need to run:

```bash
python main.py
```

---

## Conversational Interaction Example

When you run `python main.py`, the interactive CLI launches:

```text
============================================================
Travel Planner
Type 'exit', 'quit', or 'bye' to quit.
============================================================

You: I want to visit Munnar
Planner: How many days would you like to spend in Munnar?

You: 4 days
Planner: How many people are travelling (number of adults and children)?

You: 2 adults
Planner: What kind of experience or activities do you prefer? (e.g. adventure, backpacker, budget, couples, family-friendly, hiking — or type 'flexible')

You: hiking and nature
Planner: Do you have a target budget in mind (e.g. ₹50,000, or budget-conscious / mid-range / luxury)? (Or type 'flexible')

You: budget-conscious

============================================================
FINAL ITINERARY
============================================================
Destination: Munnar
Duration:    4 days
Travellers:  2 adults
Budget:      Flexible / Not specified

Day 1 — Arrival & Munnar Tea Estate Walk & Tasting
- Hotel:    Munnar Hikers Hostel (HOT-004)
- Activity: Munnar Tea Estate Walk & Tasting (ACT-002) — 3 hrs
- Notes:    Check in to Munnar Hikers Hostel. Enjoy a relaxed start with Munnar Tea Estate Walk & Tasting (3 hrs).

Day 2 — Discover Munnar — Eravikulam National Park Trek
- Hotel:    Munnar Hikers Hostel (HOT-004)
- Activity: Eravikulam National Park Trek (ACT-003) — 5 hrs
- Notes:    Full day exploring Eravikulam National Park Trek (5 hrs). Leisure evening.

Day 3 — Scenic Exploration & Cultural Leisure in Munnar
- Hotel:    Munnar Hikers Hostel (HOT-004)
- Activity: Leisure & self-guided exploration
- Notes:    Self-guided exploration of Munnar's scenic spots, cafes, and local viewpoints at a relaxed pace.

Day 4 — Leisure Morning & Departure
- Hotel:    Munnar Hikers Hostel (HOT-004)
- Activity: Leisure & self-guided exploration
- Notes:    Relaxed morning in Munnar, local souvenir shopping, checkout, and onward departure.

============================================================
QUOTE
============================================================
• Munnar Hikers Hostel (HOT-004): ₹1,500 × 4 = ₹6,000
• Munnar Tea Estate Walk & Tasting (ACT-002): ₹900 × 2 = ₹1,800
• Eravikulam National Park Trek (ACT-003): ₹1,400 × 2 = ₹2,800
• Private Cab (Sedan, per day) (TRN-001): ₹3,000 × 4 = ₹12,000
------------------------------------------------------------
Total: ₹22,600
============================================================
TRIP COMPLETE
============================================================

Saved itinerary to outputs/CHAT.json
```

### Natural Corrections & State Preservation
The conversation preserves state between turns:
- **"Actually make it 3 days"** $\to$ updates only duration to 3 days while keeping Munnar and 2 adults.
- **"Actually Kochi"** $\to$ updates only destination to Kochi while keeping 3 days and 2 adults.
- **"I want to go to Goa"** $\to$ recognizes Goa is not in the catalog, suggests available catalog destinations (`Alleppey, Kochi, Munnar`), and preserves other collected parameters.

---

## File Layout

```
travel-planner/
├── sample_data.json      # Supplier inventory + traveler profile (source of truth)
├── app.py                # LangGraph state machine, slot extraction, grounded planner & pricer
├── main.py               # Conversational CLI entry point (`python main.py`)
├── eval.py               # Evaluation harness & broken-itinerary self-test
├── cli.py                # Optional supplementary CLI tools
├── helpers.py            # Environment loader
├── outputs/              # Saved output JSON files (e.g. CHAT.json)
├── tests/                # Unit test suite (13 tests)
├── writeup.md            # Part B Architecture & Design Write-up
└── requirements.txt      # Dependencies (langgraph, groq, python-dotenv)
```

---


Run evaluation harness:
```bash
python eval.py
```
