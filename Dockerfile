# syntax=docker/dockerfile:1
#
# Medical RAG Agent - Streamlit app image.
#
# The app auto-downloads the sentence-transformers embedding model on first
# use (~1 GB) and writes it under $HF_HOME, which docker-compose maps to a
# named volume so it is downloaded once and reused across restarts.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # torch / torchvision "+cpu" wheels live on PyTorch's own index
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    # keep the big embedding-model download on a mounted volume
    HF_HOME=/home/appuser/.cache/huggingface

# curl -> container HEALTHCHECK; libgl1/libglib2.0-0 -> runtime libs pulled
# in by Pillow / torchvision image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not bust the (slow) pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser . .

# Writable paths: app logs + model cache (also become volume mount points).
RUN mkdir -p /app/logs "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8501

# Streamlit exposes a lightweight readiness endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
