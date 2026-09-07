import os
from dotenv import dotenv_values, load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(current_dir, "..", "..", "config", "properties.env")

_FILE = dotenv_values(dotenv_path=env_path)
load_dotenv(dotenv_path=env_path)


def _env(*names, default=None):
    """First non-empty value for `names`, from the environment then the file.

    Accepting several names is what lets each setting have a plain
    UPPER_SNAKE_CASE spelling on top of the historical dotted one
    (`open_router.api_key`), which many secret stores - Docker `-e`,
    Hugging Face Spaces, systemd - cannot express. Canonical name first,
    legacy aliases after; both keep working.
    """
    for source in (os.environ, _FILE):
        for name in names:
            value = source.get(name)
            if value is not None and value.strip():
                return value.strip()
    return default


# App
APP_NAME = _env("APP_NAME", default="CV Screening AI")
APP_VERSION = _env("APP_VERSION", default="0.1.0")
APP_PORT = int(_env("APP_PORT", default=8000))
DEBUG = _env("DEBUG", default="true").lower() == "true"
API_PREFIX = _env("API_PREFIX", default="/api/v1")

# LLM (OpenRouter, OpenAI-compatible)
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "open_router.api_key")
OPENROUTER_BASE_URL = _env(
    "OPENROUTER_BASE_URL", "open_router.url", default="https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "open_router.model")

#Embedding
EMBEDDING_MODEL = _env("EMBEDDING_MODEL")

# Qdrant - QDRANT_URL points at either a self-hosted instance (the Compose
# `qdrant` service, or localhost outside Docker) or a managed cluster;
# QDRANT_API_KEY is only needed by the latter.
QDRANT_URL = _env("QDRANT_URL", default="http://localhost:6333")
QDRANT_API_KEY = _env("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = _env("QDRANT_COLLECTION_NAME", default="rag-documents")

# Ingestion
MEDICAL_PDF_PATH = _env("MEDICAL_PDF_PATH")

# RAG Agent
RAG_TOP_K = int(_env("RAG_TOP_K", default=4))
RAG_RECENT_TURNS = int(_env("RAG_RECENT_TURNS", default=4))
RAG_TEMPERATURE = float(_env("RAG_TEMPERATURE", default=0.3))
RAG_MAX_TOKENS = int(_env("RAG_MAX_TOKENS", default=1024))
RAG_REASONING_MAX_TOKENS = int(_env("RAG_REASONING_MAX_TOKENS", default=300))

# Evaluator Agent
EVALUATOR_TEMPERATURE = float(_env("EVALUATOR_TEMPERATURE", default=0.0))
EVALUATOR_MAX_TOKENS = int(_env("EVALUATOR_MAX_TOKENS", default=150))
EVALUATOR_REASONING_MAX_TOKENS = int(_env("EVALUATOR_REASONING_MAX_TOKENS", default=100))
CONFIDENCE_THRESHOLD = float(_env("CONFIDENCE_THRESHOLD", default=0.6))
TONE_THRESHOLD = float(_env("TONE_THRESHOLD", default=0.6))
MAX_RETRIES = int(_env("MAX_RETRIES", default=2))

# Reviser Agent
REVISER_TEMPERATURE = float(_env("REVISER_TEMPERATURE", default=0.3))
REVISER_MAX_TOKENS = int(_env("REVISER_MAX_TOKENS", default=1024))
REVISER_REASONING_MAX_TOKENS = int(_env("REVISER_REASONING_MAX_TOKENS", default=300))

# Summarizer Agent
SUMMARIZER_TEMPERATURE = float(_env("SUMMARIZER_TEMPERATURE", default=0.3))
SUMMARIZER_MAX_TOKENS = int(_env("SUMMARIZER_MAX_TOKENS", default=600))
SUMMARIZER_REASONING_MAX_TOKENS = int(_env("SUMMARIZER_REASONING_MAX_TOKENS", default=200))

# Planner Agent
PLANNER_TEMPERATURE = float(_env("PLANNER_TEMPERATURE", default=0.0))
PLANNER_MAX_TOKENS = int(_env("PLANNER_MAX_TOKENS", default=150))
PLANNER_REASONING_MAX_TOKENS = int(_env("PLANNER_REASONING_MAX_TOKENS", default=100))