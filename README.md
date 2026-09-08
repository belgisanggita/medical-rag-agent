# Medical RAG Agent

A multi-agent **Retrieval-Augmented Generation** assistant over the *Gale
Encyclopedia of Medicine* (`docs/Medical_Book.pdf`). Ask medical questions in
Indonesian or English and get answers grounded in the source book, with an
automatic fact-check / tone-check / self-revision loop.

---

## Table of contents

- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Pre-built Qdrant index](#pre-built-qdrant-index)
- [How it works](#how-it-works)
- [Services, volumes & network](#services-volumes--network)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)
- [Local development (without Docker)](#local-development-without-docker)
- [Evaluation harness](#evaluation-harness)
- [Room for future work](#room-for-future-work)
- [Project layout](#project-layout)

---

## Architecture

A single [LangGraph](https://langchain-ai.github.io/langgraph/) state machine
orchestrates five specialised agents. The **planner** owns every routing
decision — both *which* agents to run and, after evaluation, whether to accept,
re-query, revise, or escalate.

```
planner ─┬─ small_talk ─────────────────────────────► END   (off-topic redirect)
         ├─ meta ──────────► meta ──────────────────► END   (summarise the chat)
         └─ medical ──────► rag ──► evaluator ─┬─ accept ──► summarizer ► END
                              ▲                ├─ revise ──► reviser ───► summarizer ► END
                              └── retry ───────┤
                                               └─ escalate ► escalate ──► summarizer ► END
```

| Agent | Responsibility |
|-------|----------------|
| **Planner** | Classifies intent (`medical` / `meta` / `small_talk`), resolves follow-up context, picks the post-evaluation action. |
| **RAG** | Embeds the question, retrieves top-`k` chunks from Qdrant, drafts a grounded answer. |
| **Evaluator** | Scores the draft for `factuality` and `tone` against the retrieved context. |
| **Reviser** | Rewrites the answer to fix issues the evaluator flagged. |
| **Escalate** | Prepends a "could not verify with confidence" banner when retries are exhausted. |
| **Summarizer** | Maintains a rolling conversation summary used for follow-ups. |

```
┌──────────────┐        ┌──────────────┐        ┌───────────────────┐
│  Browser     │ ─────► │  app         │ ─────► │  qdrant           │
│  :8501       │        │  Streamlit + │        │  vector store     │
│              │ ◄───── │  LangGraph   │ ◄───── │  :6333 / :6334    │
└──────────────┘        └──────┬───────┘        └───────────────────┘
                               │ HTTPS
                               ▼
                        OpenRouter (LLM API)
```

---

## Tech stack

- **UI:** Streamlit
- **Orchestration:** LangGraph + langchain-core
- **LLM:** any OpenAI-compatible endpoint via [OpenRouter](https://openrouter.ai/) (default `openai/gpt-oss-120b`)
- **Embeddings:** `intfloat/multilingual-e5-base` via `sentence-transformers` (runs on CPU, downloaded on first use)
- **Vector DB:** Qdrant
- **PDF parsing:** pdfplumber
- **Packaging:** Docker + Docker Compose

---

## Prerequisites

- **Docker Engine 24+** and the **Compose v2** plugin (`docker compose`, not `docker-compose`).
- An **OpenRouter API key** — <https://openrouter.ai/keys>.
- ~3 GB free disk for the image + embedding-model cache.
- Outbound internet from the `app` container (HuggingFace model download + OpenRouter calls).

---

## Getting started

### 1. Provide your OpenRouter key

Tuning knobs live in `config/properties.env`, which is **committed and holds
defaults only** — no secrets. Secrets are injected from the environment, where
they override anything in that file:

```bash
export OPENROUTER_API_KEY=sk-or-...your key...
```

Prefer a file? Put it in a root `.env` (git-ignored) — Compose reads it
automatically:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...your key...' > .env
```

### 2. Bring the stack up

**Self-hosted Qdrant — the default.** Pulls and runs the Qdrant container
alongside the app:

```bash
docker compose up -d --build
```

**Qdrant Cloud** — skip the local Qdrant container with `--scale qdrant=0` and
point the app at your cluster:

```bash
export QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
export QDRANT_API_KEY=<cluster-key>
docker compose up -d --scale qdrant=0
```

### 3. First run

In self-hosted mode Qdrant boots on the **pre-built index** that ships with the
repo (`docs/prebuilt_qdrant_data`), so `docs/Medical_Book.pdf` is already
embedded and indexed — ingestion is skipped and the app is usable right away.

The only one-time cost left is downloading the embedding model (~1 GB). The app
does this **at startup**, not lazily on your first question, and says so in the
UI — `⏬ Embedding Model Downloading... (This is only for first time)` — so a
slow first boot is visible instead of looking like a hang.

### 4. Shut down

```bash
docker compose down          # stop & remove containers, keep data
docker compose down -v       # also delete model cache + logs (named volumes)
```

---

## Configuration

Defaults live in `config/properties.env`; **every one of them can be overridden
by a real environment variable** of the same name (Compose `environment:`,
`export`, a platform secret store). Only the LLM key is required, and it should
always come from the environment.

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENROUTER_API_KEY` | — | **Required.** Your OpenRouter key — pass it as an env var, don't commit it. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | Any model your key can access. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | sentence-transformers model id. |
| `QDRANT_URL` | `http://localhost:6333` | Compose defaults it to `http://qdrant:6333`; set it to your cluster URL for Qdrant Cloud. |
| `QDRANT_API_KEY` | *(unset)* | Required for Qdrant Cloud / any secured Qdrant. |
| `QDRANT_COLLECTION_NAME` | `rag-documents` | Collection to store chunks in. Must stay `rag-documents` to use the pre-built index. |
| `QDRANT_STORAGE` | `./docs/prebuilt_qdrant_data` | Compose-only. Host path or named volume backing `/qdrant/storage`; change it to [start from a clean index](#starting-from-a-clean-index-re-ingest-from-the-pdf). |
| `MEDICAL_PDF_PATH` | `docs/Medical_Book.pdf` | Source PDF; Compose pins it to the copy inside the image. |
| `APP_NAME` | `Medical AI` | Shown in the UI title. |
| `RAG_TOP_K` | `4` | Chunks retrieved per query. |
| `RAG_RECENT_TURNS` | `4` | Prior turns passed to the RAG agent. |
| `RAG_TEMPERATURE` / `RAG_MAX_TOKENS` | `0.3` / `1024` | RAG generation. |
| `CONFIDENCE_THRESHOLD` | `0.6` | Min factuality score to accept an answer. |
| `TONE_THRESHOLD` | `0.6` | Min tone score to accept an answer. |
| `MAX_RETRIES` | `2` | RAG re-query attempts before escalating. |
| `EVALUATOR_*` / `REVISER_*` / `SUMMARIZER_*` / `PLANNER_*` | see file | Per-agent temperature / token budgets. |

> **Reasoning-model note:** the default model always spends part of its token
> budget "thinking". `*_REASONING_MAX_TOKENS` caps that internal budget so short
> answers (e.g. the Evaluator's numeric score) don't come back empty.

After changing `config/properties.env`, restart the app container:

```bash
docker compose restart app
```

---

## Pre-built Qdrant index

`docs/prebuilt_qdrant_data/` is a **ready-made Qdrant storage directory**: the
Medical Book is already chunked, embedded and indexed in the `rag-documents`
collection, with the matching marker in `rag-documents_ingestion_state`. It is
committed to the repo and bind-mounted straight into the `qdrant` service:

```yaml
volumes:
  - ${QDRANT_STORAGE:-./docs/prebuilt_qdrant_data}:/qdrant/storage
```

Why: embedding 243 pages on CPU takes several minutes on every fresh
environment. Shipping the index removes that wait entirely.

### Starting from a clean index (re-ingest from the PDF)

Point `QDRANT_STORAGE` at a different location and Qdrant boots empty, so the
app re-chunks and re-embeds `docs/Medical_Book.pdf` on startup. The pre-built
directory is left untouched.

**Option A — a fresh directory:**

```bash
mkdir -p qdrant-storage                       # already in .dockerignore
export QDRANT_STORAGE=./qdrant-storage
docker compose up -d
docker compose logs -f app                    # watch the ingestion run
```

**Option B — a path outside the repo** (keeps the data out of your working
tree entirely, so `git status` stays clean):

```bash
export QDRANT_STORAGE=$HOME/medical-rag-qdrant-fresh
docker compose up -d
```

`QDRANT_STORAGE` must be a **filesystem path** (absolute, or relative starting
with `./` or `../`) — Compose rejects a bare name, reading it as an undeclared
named volume. Docker creates the directory if it does not exist.

---

## How it works

- **Config** — `app/config/properties_setup.py` reads two layers via its
  `_env()` helper: the process environment first (Compose `environment:`,
  `export`, a platform secret store), then `config/properties.env` for
  defaults. That is how the OpenRouter key stays out of the repo and how
  `QDRANT_URL` / `MEDICAL_PDF_PATH` get their container-correct values. Each
  LLM setting also answers to its historical dotted name
  (`open_router.api_key` → `OPENROUTER_API_KEY`), so old config files keep
  working. Empty values count as unset, so an unused `${VAR:-}` override
  cannot blank out a default.
- **Startup** — `app.py` warms the embedding model before anything else
  (`warm_up_model()`), using `is_model_cached()` — an offline HF-cache probe —
  to decide whether to show a download notice or a quick "loading" spinner.
  Both run once per server process via `st.cache_resource`.
- **Ingestion** — `ingest.py::ensure_ingested()` extracts text per page
  (`pdfplumber`), chunks it, embeds each chunk with the e5 model
  (`passage: ` prefix), and upserts into Qdrant. A marker point in a
  `<collection>_ingestion_state` collection is written **last**, so a crash
  mid-ingest is retried cleanly on the next start instead of leaving a partial
  index.
- **Retrieval** — questions are embedded with the `query: ` prefix (e5's
  asymmetric convention) and matched by cosine similarity.
- **Serving** — `app.py` is a pure Streamlit UI. It builds the LangGraph once
  per session and calls `graph.invoke({...})` per turn — no HTTP layer between
  UI and agents.
- **State** — chat history and the rolling summary live in
  `st.session_state` (per browser session); the vector index lives in Qdrant
  (shared, persistent).

---

## Services, volumes & network

### Services

| Service | Image | Ports | Healthcheck |
|---------|-------|-------|-------------|
| `qdrant` | `qdrant/qdrant:v1.12.4` | `6333` (REST/dashboard), `6334` (gRPC) | TCP probe on `6333` |
| `app` | built from `./Dockerfile` | `8501` (Streamlit) | `GET /_stcore/health` |

`app` starts only after `qdrant` reports **healthy**
(`depends_on: condition: service_healthy`). Both restart automatically
(`restart: unless-stopped`).

### Volumes (named, persistent)

| Volume | Mounted at | Purpose |
|--------|-----------|---------|
| `hf-cache` | `/home/appuser/.cache/huggingface` | Downloaded embedding model. |
| `app-logs` | `/app/logs` | `medical_generative.log` debug log. |

### Bind mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./config` | `/app/config` (`ro`) | Holds `properties.env` — non-secret defaults & tuning, read at app start. |
| `./docs/prebuilt_qdrant_data` | `/qdrant/storage` | [Pre-built index](#pre-built-qdrant-index) — override with `QDRANT_STORAGE` to start clean. |


---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `app` stuck in `Created`, never starts | `qdrant` not healthy yet (self-hosted). `docker compose logs qdrant`; give it a few more seconds on first boot. |
| UI error: *"Gagal menyiapkan index dokumen"* | Ingestion failed. Check `docker compose logs -f app` (full traceback also in the `app-logs` volume). Usually a bad/empty `OPENROUTER_API_KEY` or no internet for the model download. |
| Long wait on the very first startup | Expected once — the embedding model download (~1 GB), shown in the UI as *Embedding Model Downloading...*. Cached in `hf-cache` afterwards. |
| App re-ingests despite the pre-built index | The index is keyed to this PDF, `EMBEDDING_MODEL` and `QDRANT_COLLECTION_NAME` — check none of the three changed, and that `QDRANT_STORAGE` still points at `docs/prebuilt_qdrant_data`. |
| `git status` dirty under `docs/prebuilt_qdrant_data` | Qdrant writes to its storage dir while running. `git checkout -- docs/prebuilt_qdrant_data` to restore, or run with `QDRANT_STORAGE` pointing elsewhere. |
| `401` / `Reasoning is mandatory` from the LLM | Check `OPENROUTER_API_KEY` and that `OPENROUTER_MODEL` is available to your account. |
| Image build pulls huge CUDA packages | `requirements.txt` pins `torchvision==0.29.0+cpu`; the Dockerfile adds PyTorch's CPU wheel index. If `torch` still resolves to a CUDA build, pin it explicitly to the matching `+cpu` version in `requirements.txt`. |
| Changed `config/properties.env`, no effect | `docker compose restart app` — the file is read at process start. |
| Port already in use | Edit the `ports:` mapping in `docker-compose.yaml` (e.g. `"8502:8501"`). |

---

## Local development (without Docker)

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Qdrant on its own, on the pre-built index (swap the -v source for an
# empty directory to ingest the PDF from scratch instead)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$PWD/docs/prebuilt_qdrant_data:/qdrant/storage" qdrant/qdrant

# config/properties.env already defaults to QDRANT_URL=http://localhost:6333
# and MEDICAL_PDF_PATH=docs/Medical_Book.pdf; only the key is missing:
export OPENROUTER_API_KEY=sk-or-...your key...

streamlit run app.py
```

---

## Evaluation harness

A small offline eval lives in `eval/` (`testset.jsonl` → `run_eval.py` →
`results.md` / `results.json`). See [`eval/README.md`](eval/README.md). To run
it against the containerised stack:

```bash
docker compose exec app python eval/run_eval.py
```

---

## Room for future work

This project is far from perfect and has plenty of gaps — it was built in just a
few days. What follows is an honest note on where it could go next.

### Evaluation method

The harness in `eval/` uses no named eval library (RAGAS, DeepEval, promptfoo),
but it does follow well-known practices: a **fixed golden test set run through
the full pipeline**, with **component-level** metrics (routing accuracy,
retrieval hit-rate) and **end-to-end** ones (faithfulness / groundedness and
tone via *LLM-as-a-judge*), in the shape of the **RAG evaluation triad**. It is
a directed check on a single run — not a large benchmark with significance
testing.

Next steps: adopt a named framework so numbers compare to published baselines;
grow the test set and run each item several times for mean ± variance; make
`concept_coverage` semantic with an LLM rubric instead of a substring match; gate
CI on the key metrics; and use a different model family for the judge.

### The project overall

- **Retrieval:** hybrid (dense + sparse) search with a re-ranker,
  heading-aware chunking, and incremental multi-document ingestion.
- **Agents:** token streaming to the UI, per-agent model selection, and
  showing the retrieved passages + evaluator scores.
- **Serving:** split the LangGraph into a FastAPI service behind the UI,
  add tracing (Langfuse), auth and rate limiting, and a
  full CI pipeline.
- **Product & safety:** inline source citations, an explicit medical
  disclaimer and emergency refusal path, user feedback capture, and
  wider language coverage.