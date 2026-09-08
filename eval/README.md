# Evaluation harness

Offline evaluation for the medical multi-agent assistant, replaying a fixed test set through the full LangGraph pipeline (planner → rag → evaluator → reviser/escalate → summarizer) and producing aggregate and per-question metrics.

```
eval/
├── testset.jsonl   # 23 fixed questions (input)
├── run_eval.py     # the harness
├── results.json    # full detail: every metric + the actual answer per item
└── results.md      # aggregate + per-question tables
```

---

## Running it

Run it directly with the project's Python environment — no Docker.

Prerequisites:

- Dependencies installed: `pip install -r requirements.txt`.
- `config/properties.env` populated and a valid `OPENROUTER_API_KEY` exported.
- A Qdrant instance reachable at `QDRANT_URL` (default `http://localhost:6333`).
  Either a standalone Qdrant on the pre-built index, or Qdrant Cloud via
  `QDRANT_URL` / `QDRANT_API_KEY`.

```bash
# from the project root
python eval/run_eval.py
```

On startup the harness calls `ensure_ingested(MEDICAL_PDF_PATH)`, so it reuses
the pre-built Qdrant index if it is already in place and only ingests when the
collection is empty. It then runs the 23 items **as one continuous
conversation** — each turn's question and answer are appended to
`chat_history`, and the Summarizer's running `summary` is carried into the next
turn. This is deliberate: the two `meta` items ("summarize what we discussed")
need real history to work on.

---

## Test set (`testset.jsonl`)

One JSON object per line:

| Field | Meaning |
|-------|---------|
| `id` | 1-based item number, also the conversation order. |
| `question` | The user turn. English and Indonesian are mixed on purpose. |
| `type` | Gold intent label: `medical` (18), `meta` (2), `small_talk` (3). |
| `must_include` | List of **concepts** expected in a correct answer / in the retrieved context. Each concept is itself a list of accepted lowercase surface forms (English + Indonesian + synonyms); a concept counts as covered when **any** form appears. A bare string is still accepted as a one-form concept. Empty for non-medical items. |

`must_include` is a substring check, not a semantic one, but the per-concept
synonym lists remove the old English-only bias — a correct Indonesian answer
that says `"gula darah"` now matches the `"blood sugar"` concept. It is still a
cheap proxy for "did the pipeline surface the right facts", not a rubric: keep
reading `concept_coverage` alongside `factuality`, not instead of it. To make it
fully semantic you would swap the substring match for an LLM-graded rubric.

---

## Metrics

Computed in `run_eval.py`; medical-only metrics skip `meta` / `small_talk`
items (they show as `None`).

| Metric | How it is computed | Notes |
|--------|--------------------|-------|
| `routing_accuracy` | Share of items where the Planner's `intent` equals `type`. | Also broken out `by_type`. |
| `retrieval_hit_rate` | For each medical item, re-run `retrieve_context(question)` and check that any surface form of any `must_include` concept appears in the returned context. | Isolates retrieval from generation. |
| `avg_factuality` | Mean of the Evaluator's `factuality` score (LLM-as-judge, 0–1), medical only. | Draft vs. retrieved context. |
| `avg_tone` | Mean of the Evaluator's `tone` score (0–1), medical only. | |
| `avg_concept_coverage` | Mean fraction of `must_include` concepts covered in the **final answer** (any accepted EN/ID surface form counts), medical only. | Substring proxy for fact coverage, not a quality rubric — read with `factuality`. |
| `revision_rate` | Share of medical answers the Reviser rewrote (`revised == True`). | |
| `escalation_rate` | Share of medical answers escalated after `MAX_RETRIES` (`escalated == True`). | |
| `avg_rag_attempts` | Mean RAG generations per medical question. `> 1` means the evaluator forced a re-query. | Re-query cost. |
| `avg_latency_s` | Wall-clock seconds per `graph.invoke`, all items. | Dominated by LLM calls; absolute value depends on the model/endpoint. |

`per_item` in `results.json` additionally carries `summary_used`,
`summary_after`, and the full `answer` string for manual inspection.

---

## Current results

From the committed `results.json` / `results.md` (23 items, 18 medical).

| Metric | Value | Reading |
|--------|-------|---------|
| Routing accuracy | **1.0** | 23/23. medical 1.0, meta 1.0, small_talk 1.0. |
| &nbsp;&nbsp;small_talk | 1.0 | Off-topic items (weather, thanks, coding help) route to `small_talk` and get a fixed redirect instead of the RAG path. |
| Retrieval hit-rate | **0.944** | 17/18. Only item 8 *"function of the liver"* misses — the source has no matching passage, and the answer says so rather than guessing. |
| Avg factuality | **0.978** | 17 of 18 medical items score 1.0; item 15 (*"virus vs bacterium"*) scores 0.6 on a formatting artefact in the generated answer, not a grounding error. |
| Avg tone | **0.961** | Most items score 0.9–1.0. No answer required a revision. |
| Avg concept coverage | **0.75** | The three 0.0s are all cases where the answer declines rather than assert an unsupported fact (items 3, 8, 16). The 0.5s (items 1, 6, 11) are correct answers that phrase the concept without the exact keyword. Substring proxy — read with factuality. |
| Revision rate | **0** | The Reviser never fired; drafts cleared both thresholds. |
| Escalation rate | **0** | No item exhausted `MAX_RETRIES`. |
| Avg RAG attempts | **1.0** | No item needed a re-query this run. |
| Avg latency | **17.3 s** | Range ~1.2 s (small_talk, no RAG) to ~34.9 s (item 4, a long asthma-treatment answer). |

### Takeaways

- **Routing and retrieval are solid.** Intent routing is correct across the
  set, and the single retrieval miss is a genuine gap in the source that the
  answer handles by declining rather than guessing.
- **Grounded generation is strong** — factuality and tone stay high across the
  medical set, with zero revisions and zero escalations.
- **`concept_coverage` is a coverage proxy, not a quality score.** Per-concept
  EN/ID synonym lists mean it no longer penalises correct Indonesian answers,
  but it is still a substring match — a low value can simply be an answer that
  correctly declines when the source has nothing. Read it next to `factuality`.
- **Latency is LLM-bound.** The reasoning model's thinking budget is the main
  lever (`*_REASONING_MAX_TOKENS`).
