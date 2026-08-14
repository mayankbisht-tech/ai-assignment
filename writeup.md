# Part B — Architecture & Reasoning Write-Up

AI-Assisted Grounded Travel Planner
====================================

## 1. Architecture

The system is designed as an interactive, stateful conversational CLI built on **LangGraph**. Unlike a naive chatbot that passes an unbounded chat history to an LLM every turn, this architecture uses a structured **`PlannerState`** as the single source of truth across conversational turns.

```
                  ┌────────────────────────────────────────┐
                  │          START (User Input)            │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    node_understand_and_update          │
                  │ (Deterministic + LLM Slot Extraction)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                              [route_next_step]
                       /             |              \
     (out_of_catalog) /      (missing slots)         \ (all required slots present)
                     ▼               ▼                ▼
       ┌──────────────────────┐ ┌───────────────┐ ┌──────────────────────┐
       │recover_invalid_dest  │ │ask_destination│ │  plan_itinerary      │
       │ (lists catalog locs) │ │   ask_days    │ │(Catalog Match & Pure │
       └─────────────┬────────┘ │   ask_party   │ │  Python Pricing)     │
                     │          └───────┬───────┘ └──────────┬───────────┘
                     │                  │                    │
                     └──────────────────┼────────────────────┘
                                        ▼
                                      [ END ]
                    (Outputs response & saves CHAT.json)
```

### Why this architecture?
- **Separation of Extraction vs Execution**: The LLM assists with intent understanding and slot extraction (when online), but the actual candidate retrieval, inventory matching, and quote calculations are strictly handled by deterministic Python logic.
- **Incremental State Preservation**: Each turn updates only the slots mentioned, preserving previously collected information (e.g. asking for "4 days", then "2 adults", then "Actually Kochi" switches only destination without losing days or party size).
- **Dynamic Catalog Grounding**: Available destinations and items are extracted dynamically from `sample_data.json` on initialization. No city names or IDs are hardcoded in planning branches.

---

## 2. Grounding (Zero Hallucinations)

Grounding is enforced at four distinct levels:

1. **Dynamic Catalog Introspection**:
   The system queries `sample_data.json` to derive the set of valid destinations (`Alleppey`, `Kochi`, `Munnar`) and regional scopes (`Kerala`). Any destination outside this inventory (e.g., `Goa`, `Paris`, `Mumbai`) is caught before planning begins and gracefully declined with an explanation of available destinations.

2. **Catalog ID Preservation**:
   Every hotel (`HOT-xxx`), activity (`ACT-xxx`), and transport (`TRN-xxx`) option in the generated itinerary references a verified item in `suppliers`.

3. **Deterministic Python Pricing (Never Computed by LLM)**:
   LLMs frequently make arithmetic errors (e.g. multiplying prices or summing 8 line items incorrectly). In our system:
   $$\text{Line Total} = \text{Unit Price (from catalog)} \times \text{Quantity}$$
   $$\text{Grand Total} = \sum \text{Line Totals}$$
   All calculations occur exclusively in Python code reading canonical fields directly from `sample_data.json`.

4. **Independent Post-Validation (`validate_itinerary`)**:
   Before an itinerary is presented or saved, `validate_itinerary()` independently audits the payload against `sample_data.json`:
   - Checks that all referenced IDs exist in the catalog.
   - Checks that item names and unit prices match canonical catalog data verbatim.
   - Recomputes all line totals and the grand total to guarantee exact mathematical consistency.

---

## 3. Cost & Latency Optimization

In a production deployment, keeping latency sub-second and token costs minimal is critical:

- **Model Choice**:
  `llama-3.3-70b-versatile` via Groq is used as the primary LLM with native JSON schema formatting. Groq LPU inference achieves ~500–800 tokens/sec with sub-second time-to-first-token (TTFT).
- **Token Budget**:
  Prompts only include current state slots and the latest message (~150–300 tokens input). We never dump entire catalog databases into the prompt context.
- **Deterministic Offline Fallback**:
  If the LLM API is unavailable, rate-limited, or offline, the system seamlessly uses its built-in deterministic slot extractor (`extract_slots_deterministic`) without crashing.
- **Prompt Caching & Streaming**:
  In a production UI, system prompts and catalog metadata are cached at the inference gateway. For real-time chat, response tokens can be streamed directly to the frontend.

---

## 4. Failure Handling

| Scenario | System Response |
|---|---|
| **Invalid Destination (e.g. Goa, REQ-3)** | Catches out-of-catalog location immediately; responds informing user that Goa is not in the catalog and lists available destinations (`Alleppey, Kochi, Munnar`); preserves existing slots. |
| **Missing API Key / Offline Mode** | Automatically falls back to deterministic regex & rule-based slot extraction; zero downtime and 100% functional conversational CLI. |
| **API Timeout / Rate Limit (429/500)** | Caught gracefully in `extract_slots_with_llm`, falling back to deterministic extraction for the current turn. |
| **Budget Exceeded** | If requested budget is below catalog rates, the system builds the most affordable valid combination and explicitly notes the budget delta rather than hallucinating fake discounts. |

---

## 5. Evaluation Strategy & Metrics

To ensure product quality and verify zero hallucinations in production, we implement automated metrics:

### Primary Grounding Metrics
- **Catalog ID Precision**: % of cited item IDs that exist in `sample_data.json` (Target: 100%).
- **Price Accuracy**: % of items whose unit price exactly matches the catalog canonical price (Target: 100%).
- **Mathematical Integrity**: Binary check that $\text{Total} == \sum \text{Line Totals}$ (Target: 100%).
- **Decline Recall**: % of unfulfillable requests (e.g. Goa) correctly flagged and declined without fabricated inventory (Target: 100%).

### Business & Quality Metrics
- **Budget Fit Rate**: Adherence of total quote to requested budget limits.
- **Preference Alignment**: Tag overlap between traveler interests and selected hotel/activities.
- **Turn Efficiency**: Average number of conversational turns needed to produce a confirmed itinerary.

### Evaluation Harness (`eval.py`)
`eval.py` provides automated verification:
1. Re-validates every output JSON file in `outputs/` against raw catalog data.
2. Runs a self-test with a deliberately corrupted itinerary testing all error detectors (fake IDs, altered prices, subtotal discrepancies, grand total errors).

---

## 6. Human-in-the-Loop Design

Our core philosophy is **AI assists, human confirms**.

- The AI generates a structured itinerary draft with exact supplier IDs and computed pricing.
- The output structure explicitly designates status as `"fulfilled"` or `"ok_pending_human_review"` and requires human travel agent confirmation before any booking or payment processing occurs.
- The system produces a reviewable document, not an irreversible booking action.

---

## 7. With More Time: Top 3 Enhancements

1. **Multi-City Route & Inter-City Travel Optimization**:
   For region-wide trips (e.g., 5 days across Kerala), implement a routing module (graph shortest path) that sequences hubs (Kochi $\to$ Alleppey $\to$ Munnar) to minimize transfer time (directly honoring feedback like *"kids got bored on long drives"*).
2. **Pydantic Structured Output Enforcement**:
   Use Pydantic v2 schemas with `instructor` or LangChain structured outputs to guarantee schema compliance at compile-time.
3. **OpenTelemetry Observability & Slot Drift Monitoring**:
   Instrument every LangGraph node with latency tracing, token tracking, and slot-update telemetry to monitor conversation health and detect extraction anomalies in production.
