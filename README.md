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
- [How it works](#how-it-works)
- [Services, volumes & network](#services-volumes--network)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)
- [Local development (without Docker)](#local-development-without-docker)
- [Evaluation harness](#evaluation-harness)
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

### 1. Create your config file

The app reads secrets and tuning knobs from `config/properties.env`
(git-ignored). Copy the template and fill in your OpenRouter key:

```bash
cp config/example.properties.env config/properties.env
```

Then edit `config/properties.env` and set at least:

```properties
open_router.api_key = sk-or-...your key...
```

You can leave everything else at its defaults. `QDRANT_URL` and
`MEDICAL_PDF_PATH` are **overridden automatically** by Compose, so their values
in this file do not matter when running under Docker.

### 2. Bring the stack up

```bash
docker compose up -d
```

This builds the app image, starts Qdrant, waits until Qdrant is healthy, then
starts the app. Open:

- **App:** <http://localhost:8501>
- **Qdrant dashboard:** <http://localhost:6333/dashboard>

### 3. First run

On the **first question you ask**, the app will:

1. Download the embedding model (~1 GB) — cached in the `hf-cache` volume afterwards.
2. Ingest `docs/Medical_Book.pdf` into Qdrant (243 pages → embedded in batches).

This one-time step can take several minutes on CPU. Watch progress with:

```bash
docker compose logs -f app
```

Ingestion is **idempotent** (keyed on a content hash of the PDF), so restarts
and redeploys skip straight past it.

### 4. Shut down

```bash
docker compose down          # stop & remove containers, keep data
docker compose down -v       # also delete Qdrant data + model cache + logs
```

---

## Configuration

All variables live in `config/properties.env`. Only the LLM key is required.

| Variable | Default | Notes |
|----------|---------|-------|
| `open_router.api_key` | — | **Required.** Your OpenRouter key. |
| `open_router.url` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `open_router.model` | `openai/gpt-oss-120b` | Any model your key can access. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | sentence-transformers model id. |
| `QDRANT_URL` | `http://localhost:6333` | **Overridden by Compose** to `http://qdrant:6333`. |
| `QDRANT_API_KEY` | *(unset)* | Only needed for a secured Qdrant. |
| `QDRANT_COLLECTION_NAME` | `rag-documents` | Collection to store chunks in. |
| `MEDICAL_PDF_PATH` | — | **Overridden by Compose** to `docs/Medical_Book.pdf`. |
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

## How it works

- **Config** — `app/config/properties_setup.py` loads `config/properties.env`
  with `python-dotenv`. Real environment variables (set by Compose) take
  precedence over the file, which is how `QDRANT_URL` / `MEDICAL_PDF_PATH` get
  their container-correct values.
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
| `qdrant-data` | `/qdrant/storage` | Vector collections + payloads. |
| `hf-cache` | `/home/appuser/.cache/huggingface` | Downloaded embedding model. |
| `app-logs` | `/app/logs` | `medical_generative.log` debug log. |

### Bind mount

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./config` | `/app/config` (`ro`) | Holds `properties.env` — secrets & tuning, read at app start. |

### Network

A dedicated bridge network `medical-net`. The app reaches Qdrant by service
name at `http://qdrant:6333`. Only `8501`, `6333` and `6334` are published to
the host.

---

## Operations

```bash
# Status / health
docker compose ps

# Follow logs
docker compose logs -f app
docker compose logs -f qdrant

# Rebuild after code or dependency changes
docker compose up -d --build

# Restart just the app (e.g. after editing properties.env)
docker compose restart app

# Open a shell in the app container
docker compose exec app bash

# Re-ingest from scratch (drops all vector data)
docker compose down
docker volume rm medical-rag-agent_qdrant-data
docker compose up -d

# Full teardown incl. model cache + logs
docker compose down -v
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `app` stuck in `Created`, never starts | `qdrant` not healthy yet. `docker compose logs qdrant`; give it a few more seconds on first boot. |
| UI error: *"Gagal menyiapkan index dokumen"* | Ingestion failed. Check `docker compose logs -f app` (full traceback also in the `app-logs` volume). Usually a bad/empty `open_router.api_key` or no internet for the model download. |
| First question hangs for minutes | Expected once — embedding model download + PDF ingestion. Subsequent runs are fast (cached + idempotent). |
| `401` / `Reasoning is mandatory` from the LLM | Check `open_router.api_key` and that `open_router.model` is available to your account. |
| Image build pulls huge CUDA packages | `requirements.txt` pins `torchvision==0.29.0+cpu`; the Dockerfile adds PyTorch's CPU wheel index. If `torch` still resolves to a CUDA build, pin it explicitly to the matching `+cpu` version in `requirements.txt`. |
| Changed `config/properties.env`, no effect | `docker compose restart app` — the file is read at process start. |
| Port already in use | Edit the `ports:` mapping in `docker-compose.yaml` (e.g. `"8502:8501"`). |

---

## Local development (without Docker)

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Qdrant on its own
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant-data:/qdrant/storage qdrant/qdrant

cp config/example.properties.env config/properties.env
# edit: open_router.api_key, QDRANT_URL=http://localhost:6333,
#       MEDICAL_PDF_PATH=docs/Medical_Book.pdf

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

## Project layout

```
app.py                     Streamlit entrypoint (UI only)
ingest.py                  Idempotent PDF -> Qdrant ingestion
requirements.txt
Dockerfile                 App image (Python 3.12, CPU-only torch)
docker-compose.yaml        app + qdrant + volumes + network + healthchecks
config/
  example.properties.env   Template — copy to properties.env
app/
  agent/graph.py           LangGraph wiring
  agent/tools/             planner / rag / evaluator / reviser / summarizer
  prompts/                 One prompt module per agent
  index/qdrant_index.py    Collections, embedding, search, ingest-state
  infra/qdrant_infra.py    Qdrant client singleton
  llm/openai_llm.py        OpenRouter ChatOpenAI factory + logging
  config/properties_setup.py
  utils/                   PDF extraction, logger
docs/Medical_Book.pdf      Source corpus (auto-ingested)
eval/                      Offline evaluation harness
```
