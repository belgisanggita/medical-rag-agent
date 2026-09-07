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
docker compose up -d
```

**Qdrant Cloud** — skip the local Qdrant container with `--scale qdrant=0` and
point the app at your cluster:

```bash
export QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
export QDRANT_API_KEY=<cluster-key>
docker compose up -d --scale qdrant=0
```

(`--scale qdrant=0` works because the app's `depends_on` entry for `qdrant` is
marked `required: false`.)

Either way this builds the app image; in self-hosted mode it also waits until
Qdrant reports healthy before starting the app. Open:

- **App:** <http://localhost:8501>
- **Qdrant dashboard** (self-hosted only): <http://localhost:6333/dashboard>

### 3. First run

In self-hosted mode Qdrant boots on the **pre-built index** that ships with the
repo (`docs/prebuilt_qdrant_data`), so `docs/Medical_Book.pdf` is already
embedded and indexed — ingestion is skipped and the app is usable right away.
See [Pre-built Qdrant index](#pre-built-qdrant-index) for how it works and how
to rebuild it from scratch.

The only one-time cost left is downloading the embedding model (~1 GB). The app
does this **at startup**, not lazily on your first question, and says so in the
UI — `⏬ Embedding Model Downloading... (This is only for first time)` — so a
slow first boot is visible instead of looking like a hang. The weights land in
the `hf-cache` volume and every later start just loads them from disk. Watch
progress with:

```bash
docker compose logs -f app
```

A line like `'docs/Medical_Book.pdf' already ingested (doc_id=...), skipping.`
confirms the pre-built index was picked up. On **Qdrant Cloud** there is no
pre-built data, so the first startup ingests the PDF (243 pages, several
minutes on CPU) — after which it is stored in your cluster for good.

Ingestion is **idempotent**, keyed on a content hash of the PDF, so restarts
and redeploys skip straight past it.

### 4. Shut down

```bash
docker compose down          # stop & remove containers, keep data
docker compose down -v       # also delete model cache + logs (named volumes)
```

`down -v` removes the *named* volumes only. The Qdrant index lives in the
bind-mounted directory `docs/prebuilt_qdrant_data`, so it survives both
commands — deleting it is always an explicit `rm -rf`.

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
environment. Shipping the index removes that wait entirely — all that is left
on first use is the one-off embedding-model download needed to embed your
*questions*.

A few properties worth knowing:

- **It replaces the old `qdrant-data` named volume.** The index is a normal
  directory in your working tree, so `docker compose down -v` no longer wipes
  it. Qdrant also *writes* to it while running (WAL, segment merges), so
  `git status` may show changes there after a session — that is expected.
- **It is tied to this exact PDF.** `ensure_ingested()` keys on a SHA-256 of
  `docs/Medical_Book.pdf`; swap the PDF and the hash changes, so the app
  ingests the new edition alongside the pre-built one instead of trusting it.
- **It is tied to the embedding model and collection name.** The collection is
  768-dimensional (`intfloat/multilingual-e5-base`) and named `rag-documents`.
  Changing `EMBEDDING_MODEL` or `QDRANT_COLLECTION_NAME` makes the pre-built
  data unusable — start fresh instead (below).
- **It is committed, with guardrails.** `.gitattributes` marks the tree
  `binary` so line-ending normalisation can never corrupt the SST / mmap / WAL
  files, and `.gitignore` re-includes it explicitly — the global `*.log*` rule
  would otherwise drop RocksDB's write-ahead logs and leave a corrupt index on
  clone.
- **Sizes.** The files are sparse: 54 MB on disk here, ~437 MB apparent. Git
  packs the mostly-zero bytes down to roughly 27 MB in history, but a fresh
  clone materialises the full ~437 MB (Git does not write sparse files). The
  same expansion applies to a Docker build context, which is why
  `.dockerignore` excludes the directory from the app image.
- **Qdrant Cloud ignores all of this.** A managed cluster has its own storage;
  the first startup against it ingests the PDF normally.

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

Expect a few minutes of CPU work; the log ends with `Indexing N chunks into
Qdrant...` followed by `Done.`. To go back to the shipped index, unset the
variable (`unset QDRANT_STORAGE`) and bring the stack up again.

To **regenerate the shipped index itself**, run one of the above, stop the
stack cleanly so Qdrant flushes (`docker compose down`),
then replace the directory:

```bash
docker compose down
rm -rf docs/prebuilt_qdrant_data
cp -a qdrant-storage docs/prebuilt_qdrant_data
git add docs/prebuilt_qdrant_data
```

Stopping the stack first matters: copying a live storage directory can capture
a half-written WAL.

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

### Network

A dedicated bridge network `medical-net`. Self-hosted, the app reaches Qdrant
by service name at `http://qdrant:6333`, and `8501`, `6333`, `6334` are
published to the host. Against Qdrant Cloud (`--scale qdrant=0`) only `8501`
is published — the `qdrant` container is never created.

---

## Operations

```bash
# Status / health
docker compose ps

# Follow logs
docker compose logs -f app
docker compose logs -f qdrant   # self-hosted only

# Rebuild after code or dependency changes
docker compose up -d --build

# Restart just the app (e.g. after editing properties.env)
docker compose restart app

# Open a shell in the app container
docker compose exec app bash

# Re-ingest from scratch, leaving the shipped index alone (self-hosted only)
docker compose down
QDRANT_STORAGE=./qdrant-storage docker compose up -d

# Full teardown incl. model cache + logs (the bind-mounted index survives)
docker compose down -v
```

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

## Project layout

```
app.py                     Streamlit entrypoint (UI only)
ingest.py                  Idempotent PDF -> Qdrant ingestion
requirements.txt
Dockerfile                 App image (Python 3.12, CPU-only torch)
docker-compose.yaml        app + qdrant (skip it with --scale qdrant=0) + volumes
config/
  properties.env           Committed defaults — secrets come from env vars
docs/
  Medical_Book.pdf         Source corpus
  prebuilt_qdrant_data/    Ready-made Qdrant index (bind-mounted, skips ingest)
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