# Evaluation harness

Offline evaluation for the medical multi-agent assistant. It replays a fixed
test set through the **full LangGraph pipeline** (planner → rag → evaluator →
reviser/escalate → summarizer) and writes aggregate + per-question metrics for
the presentation's "evaluation metrics and results" section.

```
eval/
├── testset.jsonl   # 23 fixed questions (input)
├── run_eval.py     # the harness
├── results.json    # full detail: every metric + the actual answer per item
└── results.md      # a table to paste into the slides
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
need real history to work on. One bad item is caught, recorded with an `error`
field, and the run continues.

Both output files are overwritten on every run.

---

## Test set (`testset.jsonl`)

One JSON object per line:

| Field | Meaning |
|-------|---------|
| `id` | 1-based item number, also the conversation order. |
| `question` | The user turn. English and Indonesian are mixed on purpose. |
| `type` | Gold intent label: `medical` (18), `meta` (2), `small_talk` (3). |
| `must_include` | Lowercase substrings expected in a correct answer / in the retrieved context. Empty for non-medical items. |

`must_include` is a coarse, English-centric keyword list. It is a cheap proxy
for "did the pipeline surface the right facts", **not** a semantic check — an
answer can be fully correct in Indonesian and still miss `"blood sugar"` or
`"pale"`. Read `keyword_recall` with that in mind (see below).

---

## Metrics

Computed in `run_eval.py`; medical-only metrics skip `meta` / `small_talk`
items (they show as `None`).

| Metric | How it is computed | Notes |
|--------|--------------------|-------|
| `routing_accuracy` | Share of items where the Planner's `intent` equals `type`. | Also broken out `by_type`. |
| `retrieval_hit_rate` | For each medical item, re-run `retrieve_context(question)` and check any `must_include` keyword appears in the returned context. | Isolates retrieval from generation. |
| `avg_factuality` | Mean of the Evaluator's `factuality` score (LLM-as-judge, 0–1), medical only. | Draft vs. retrieved context. |
| `avg_tone` | Mean of the Evaluator's `tone` score (0–1), medical only. | |
| `avg_keyword_recall` | Mean fraction of `must_include` keywords found as substrings in the **final answer**, medical only. | Low by design — see the caveat above. |
| `revision_rate` | Share of medical answers the Reviser rewrote (`revised == True`). | |
| `escalation_rate` | Share of medical answers escalated after `MAX_RETRIES` (`escalated == True`). | |
| `avg_rag_attempts` | Mean RAG generations per medical question. `> 1` means the evaluator forced a re-query. | Re-query cost. |
| `avg_latency_s` | Wall-clock seconds per `graph.invoke`, all items. | Dominated by LLM calls; absolute value depends on the model/endpoint. |

`per_item` in `results.json` additionally carries `summary_used`,
`summary_after`, and the full `answer` string for manual inspection.

---

## Current results

From the committed `results.json` / `results.md` (23 items, 18 medical):

| Metric | Value | Reading |
|--------|-------|---------|
| Routing accuracy | **0.957** | 22/23. medical 1.0, meta 1.0, small_talk 0.667. |
| &nbsp;&nbsp;small_talk | 0.667 | Item 21 *"What's the weather like today?"* is routed `medical`; the RAG agent still declines it correctly ("not found in the provided source"), so the user-visible behaviour is fine even though the label is wrong. |
| Retrieval hit-rate | **0.944** | 17/18. Only item 8 *"function of the liver"* misses — the book has no `"bile"` passage the retriever surfaces, and the answer correctly says so. |
| Avg factuality | **0.979** | Lowest is item 5 (0.8), *"Apa itu pneumonia?"* — the drafted answer leans on opportunistic-infection context rather than a general definition. |
| Avg tone | **0.994** | One dip: item 4 (0.9). |
| Avg keyword recall | **0.361** | Expected to be low: strict English substring match against answers that are often in Indonesian or use synonyms. Not a quality signal on its own — cross-check with factuality. |
| Revision rate | **0** | The Reviser never fired; drafts cleared both thresholds. |
| Escalation rate | **0** | No item exhausted `MAX_RETRIES`. |
| Avg RAG attempts | **1.111** | Items 3 and 15 needed a second retrieval pass. |
| Avg latency | **17.2 s** | Range ~1.8 s (small_talk, no RAG) to ~36.9 s (item 3, two RAG passes). |

### Takeaways

- **Routing and retrieval are solid.** The only routing miss is a harmless
  small-talk → medical slip; the only retrieval miss is a genuine gap in the
  source book, handled gracefully.
- **Grounded generation is strong** (factuality 0.98, tone 0.99, zero
  revisions/escalations).
- **`keyword_recall` is the weakest-looking number but the least trustworthy
  one** — it penalises correct Indonesian answers and synonyms. Treat it as a
  loose retrieval sanity check, not an answer-quality score. A better version
  would use per-language keyword sets or an LLM-graded rubric.
- **Latency is LLM-bound.** Re-query passes roughly double it; the reasoning
  model's thinking budget is the main lever (`*_REASONING_MAX_TOKENS`).

---

## Extending the eval

- **Add items:** append lines to `testset.jsonl`. Keep `id` sequential — it is
  also the conversation order. Put the most discriminating keywords in
  `must_include` (lowercase).
- **Add a metric:** compute it per item in `run_item()`, aggregate it in
  `aggregate()`, and add a row in `write_markdown()`.
- **Break the single-conversation replay** (score each item cold) by resetting
  `chat_history` and `summary` inside the loop in `main()` — but then the
  `meta` items lose their history.
