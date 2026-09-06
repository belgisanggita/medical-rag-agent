"""
Offline evaluation harness for the medical multi-agent assistant.

Runs a fixed test set (eval/testset.jsonl) through the full LangGraph pipeline
and reports aggregate metrics for the presentation's "evaluation metrics and
results" section:

  * routing_accuracy      - did the Planner classify intent correctly
                            (medical / meta / small_talk)?
  * retrieval_hit_rate    - did Qdrant return context containing the expected
                            keywords for medical questions?
  * avg_factuality        - Evaluator factuality score (LLM-as-judge), medical only
  * avg_tone              - Evaluator tone score, medical only
  * avg_keyword_recall    - fraction of expected keywords present in the answer
  * revision_rate         - share of medical answers the Reviser rewrote
  * escalation_rate       - share of medical answers flagged low-confidence
  * avg_rag_attempts      - mean RAG generations per medical question (re-query cost)
  * avg_latency_s         - wall-clock seconds per question

Usage (from the project root, with Qdrant running and config/properties.env set):

    python eval/run_eval.py

Writes eval/results.json (full detail) and eval/results.md (a table to paste
into the slides).
"""

import json
import os
import statistics as stats
import sys
import time

# make `import app...` work when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.graph import build_graph
from app.agent.tools.rag_agent import retrieve_context
from app.config import properties_setup as settings
from ingest import ensure_ingested

HERE = os.path.dirname(os.path.abspath(__file__))
TESTSET_PATH = os.path.join(HERE, "testset.jsonl")
RESULTS_JSON = os.path.join(HERE, "results.json")
RESULTS_MD = os.path.join(HERE, "results.md")

EXPECTED_INTENT = {"medical": "medical", "meta": "meta", "small_talk": "small_talk"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(stats.mean(xs), 3) if xs else None


def keyword_recall(text: str, keywords):
    if not keywords:
        return None
    t = (text or "").lower()
    return round(sum(k.lower() in t for k in keywords) / len(keywords), 3)


def retrieval_hit(question: str, keywords):
    if not keywords:
        return None
    ctx = retrieve_context(question).lower()
    return any(k.lower() in ctx for k in keywords)


def run_item(graph, item: dict) -> dict:
    t0 = time.time()
    out = graph.invoke({"question": item["question"], "chat_history": [], "summary": ""})
    latency = round(time.time() - t0, 2)

    answer = out.get("answer", "")
    intent = out.get("intent")
    is_medical = item["type"] == "medical"

    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "intent_predicted": intent,
        "routing_correct": intent == EXPECTED_INTENT[item["type"]],
        "factuality": out.get("factuality") if is_medical else None,
        "tone": out.get("tone") if is_medical else None,
        "keyword_recall": keyword_recall(answer, item.get("must_include")) if is_medical else None,
        "retrieval_hit": retrieval_hit(item["question"], item.get("must_include")) if is_medical else None,
        "rag_attempts": out.get("rag_attempts", 0),
        "revised": bool(out.get("revised")),
        "escalated": bool(out.get("escalated")),
        "latency_s": latency,
        "answer": answer,
    }


def aggregate(rows):
    med = [r for r in rows if r["type"] == "medical"]
    agg = {
        "n_total": len(rows),
        "n_medical": len(med),
        "routing_accuracy": _mean([r["routing_correct"] for r in rows]),
        "routing_accuracy_by_type": {
            t: _mean([r["routing_correct"] for r in rows if r["type"] == t])
            for t in ("medical", "meta", "small_talk")
        },
        "retrieval_hit_rate": _mean([r["retrieval_hit"] for r in med]),
        "avg_factuality": _mean([r["factuality"] for r in med]),
        "avg_tone": _mean([r["tone"] for r in med]),
        "avg_keyword_recall": _mean([r["keyword_recall"] for r in med]),
        "revision_rate": _mean([r["revised"] for r in med]),
        "escalation_rate": _mean([r["escalated"] for r in med]),
        "avg_rag_attempts": _mean([r["rag_attempts"] for r in med]),
        "avg_latency_s": _mean([r["latency_s"] for r in rows]),
    }
    return agg


def write_markdown(agg, rows):
    lines = [
        "# Evaluation results",
        "",
        f"Test set: `eval/testset.jsonl` ({agg['n_total']} questions, "
        f"{agg['n_medical']} medical).",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Routing accuracy (Planner intent) | {agg['routing_accuracy']} |",
        f"| &nbsp;&nbsp;medical | {agg['routing_accuracy_by_type']['medical']} |",
        f"| &nbsp;&nbsp;meta | {agg['routing_accuracy_by_type']['meta']} |",
        f"| &nbsp;&nbsp;small_talk | {agg['routing_accuracy_by_type']['small_talk']} |",
        f"| Retrieval hit-rate | {agg['retrieval_hit_rate']} |",
        f"| Avg factuality (Evaluator) | {agg['avg_factuality']} |",
        f"| Avg tone (Evaluator) | {agg['avg_tone']} |",
        f"| Avg keyword recall (answer) | {agg['avg_keyword_recall']} |",
        f"| Revision rate | {agg['revision_rate']} |",
        f"| Escalation rate | {agg['escalation_rate']} |",
        f"| Avg RAG attempts / question | {agg['avg_rag_attempts']} |",
        f"| Avg latency (s) | {agg['avg_latency_s']} |",
        "",
        "## Per-question",
        "",
        "| id | type | routed ok | factuality | tone | kw recall | retr hit | attempts | revised | escalated | latency s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['type']} | {r['routing_correct']} | {r['factuality']} | "
            f"{r['tone']} | {r['keyword_recall']} | {r['retrieval_hit']} | {r['rag_attempts']} | "
            f"{r['revised']} | {r['escalated']} | {r['latency_s']} |"
        )
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if settings.MEDICAL_PDF_PATH:
        print(f"Ensuring '{settings.MEDICAL_PDF_PATH}' is ingested...")
        ensure_ingested(settings.MEDICAL_PDF_PATH)

    with open(TESTSET_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    graph = build_graph()
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] ({item['type']}) {item['question'][:70]}")
        try:
            rows.append(run_item(graph, item))
        except Exception as e:  # keep going so one bad item doesn't lose the run
            print(f"    ERROR: {e}")
            rows.append({
                "id": item["id"], "type": item["type"], "question": item["question"],
                "error": str(e), "routing_correct": None, "factuality": None, "tone": None,
                "keyword_recall": None, "retrieval_hit": None, "rag_attempts": 0,
                "revised": False, "escalated": False, "latency_s": None,
            })

    agg = aggregate(rows)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_item": rows}, f, indent=2, ensure_ascii=False)
    write_markdown(agg, rows)

    print("\n=== Aggregate ===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_JSON}\nWrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
