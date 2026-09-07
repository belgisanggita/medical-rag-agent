# Evaluation results

Test set: `eval/testset.jsonl` (23 questions, 18 medical).

## Aggregate metrics

| Metric | Value |
| --- | --- |
| Routing accuracy (Planner intent) | 0.957 |
| &nbsp;&nbsp;medical | 1 |
| &nbsp;&nbsp;meta | 1 |
| &nbsp;&nbsp;small_talk | 0.667 |
| Retrieval hit-rate | 0.944 |
| Avg factuality (Evaluator) | 0.979 |
| Avg tone (Evaluator) | 0.994 |
| Avg keyword recall (answer) | 0.361 |
| Revision rate | 0 |
| Escalation rate | 0 |
| Avg RAG attempts / question | 1.111 |
| Avg latency (s) | 17.199 |

## Per-question

| id | type | routed ok | factuality | tone | kw recall | retr hit | attempts | revised | escalated | latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 27.17 |
| 2 | medical | True | 0.95 | 1.0 | 0.0 | True | 1 | False | False | 20.67 |
| 3 | medical | True | 1.0 | 1.0 | 1.0 | True | 2 | False | False | 36.86 |
| 4 | medical | True | 0.98 | 0.9 | 1.0 | True | 1 | False | False | 19.73 |
| 5 | medical | True | 0.8 | 1.0 | 0.0 | True | 1 | False | False | 17.73 |
| 6 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 10.48 |
| 7 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 10.03 |
| 8 | medical | True | 1.0 | 1.0 | 0.0 | False | 1 | False | False | 13.97 |
| 9 | medical | True | 0.9 | 1.0 | 0.0 | True | 1 | False | False | 18.87 |
| 10 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 21.69 |
| 11 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 24.75 |
| 12 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 23.73 |
| 13 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 9.51 |
| 14 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 14.4 |
| 15 | medical | True | 1.0 | 1.0 | 0.0 | True | 2 | False | False | 20.68 |
| 16 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 20.47 |
| 17 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 13.98 |
| 18 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 11.05 |
| 19 | meta | True | None | None | None | None | 0 | False | False | 8.76 |
| 20 | meta | True | None | None | None | None | 0 | False | False | 28.05 |
| 21 | small_talk | False | None | None | None | None | 1 | False | False | 12.11 |
| 22 | small_talk | True | None | None | None | None | 0 | False | False | 9.11 |
| 23 | small_talk | True | None | None | None | None | 0 | False | False | 1.77 |
