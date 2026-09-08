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
| Avg factuality (Evaluator) | 0.978 |
| Avg tone (Evaluator) | 0.961 |
| Avg concept coverage (answer) | 0.75 |
| Revision rate | 0 |
| Escalation rate | 0 |
| Avg RAG attempts / question | 1 |
| Avg latency (s) | 17.261 |

## Per-question

| id | type | routed ok | factuality | tone | concept cov | retr hit | attempts | revised | escalated | latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 25.33 |
| 2 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 14.76 |
| 3 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 16.73 |
| 4 | medical | True | 1.0 | 0.9 | 1.0 | True | 1 | False | False | 34.94 |
| 5 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 30.73 |
| 6 | medical | True | 1.0 | 1.0 | 0.5 | True | 1 | False | False | 17.42 |
| 7 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 6.96 |
| 8 | medical | True | 1.0 | 1.0 | 0.0 | False | 1 | False | False | 23.4 |
| 9 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 13.64 |
| 10 | medical | True | 1.0 | 0.9 | 1.0 | True | 1 | False | False | 11.29 |
| 11 | medical | True | 1.0 | 0.95 | 0.5 | True | 1 | False | False | 22.79 |
| 12 | medical | True | 1.0 | 0.9 | 1.0 | True | 1 | False | False | 26.1 |
| 13 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 10.19 |
| 14 | medical | True | 1.0 | 0.9 | 1.0 | True | 1 | False | False | 23.8 |
| 15 | medical | True | 0.6 | 0.8 | 1.0 | True | 1 | False | False | 24.44 |
| 16 | medical | True | 1.0 | 1.0 | 0.0 | True | 1 | False | False | 12.28 |
| 17 | medical | True | 1.0 | 0.95 | 1.0 | True | 1 | False | False | 17.07 |
| 18 | medical | True | 1.0 | 1.0 | 1.0 | True | 1 | False | False | 19.49 |
| 19 | meta | True | None | None | None | None | 0 | False | False | 18.37 |
| 20 | meta | True | None | None | None | None | 0 | False | False | 16.99 |
| 21 | small_talk | True | None | None | None | None | 0 | False | False | 4.64 |
| 22 | small_talk | True | None | None | None | None | 0 | False | False | 1.48 |
| 23 | small_talk | True | None | None | None | None | 0 | False | False | 4.16 |
