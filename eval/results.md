# Evaluation results

Test set: `eval/testset.jsonl` (23 questions, 18 medical).

## Aggregate metrics

| Metric | Value |
| --- | --- |
| Routing accuracy (Planner intent) | 1 |
| &nbsp;&nbsp;medical | 1 |
| &nbsp;&nbsp;meta | 1 |
| &nbsp;&nbsp;small_talk | 1 |
| Retrieval hit-rate | 0.944 |
| Avg factuality (Evaluator) | 0.998 |
| Avg tone (Evaluator) | 0.989 |
| Avg keyword recall (answer) | 0.444 |
| Revision rate | 0 |
| Escalation rate | 0 |
| Avg RAG attempts / question | 1.056 |
| Avg latency (s) | 8.428 |

## Per-question

| id | type | routed ok | factuality | tone | kw recall | retr hit | attempts | revised | escalated | latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medical | True | 1.0 | 1.0 | 0.5 | True | 2 | False | False | 32.1 |
| 2 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 8.78 |
| 3 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 5.21 |
| 4 | medical | True | 0.96 | 1.0 | 1.0 | True | 1 | False | False | 22.03 |
| 5 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 6.09 |
| 6 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 7.4 |
| 7 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 13.12 |
| 8 | medical | True | 1.0 | 1.0 | 0.0 | False | 1 | False | False | 6.55 |
| 9 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 6.25 |
| 10 | medical | True | 1.0 | 0.8 | 1.0 | True | 1 | False | False | 18.0 |
| 11 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 6.15 |
| 12 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 16.38 |
| 13 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 6.08 |
| 14 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 4.21 |
| 15 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 4.26 |
| 16 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 7.01 |
| 17 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 6.91 |
| 18 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 7.19 |
| 19 | meta | True | None | None | None | None | 0 | False | False | 1.44 |
| 20 | meta | True | None | None | None | None | 0 | False | False | 1.73 |
| 21 | small_talk | True | None | None | None | None | 0 | False | False | 2.55 |
| 22 | small_talk | True | None | None | None | None | 0 | False | False | 1.91 |
| 23 | small_talk | True | None | None | None | None | 0 | False | False | 2.5 |
