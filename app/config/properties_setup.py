import os
from dotenv import load_dotenv

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root and then to the config directory
env_path = os.path.join(current_dir, "..", "..", "config", "properties.env")

# Load environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# App
APP_NAME = os.getenv("APP_NAME", "CV Screening AI")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

# LLM (OpenRouter, OpenAI-compatible)
OPENROUTER_API_KEY = os.getenv("open_router.api_key")
OPENROUTER_BASE_URL = os.getenv("open_router.url", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("open_router.model")

#Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag-documents")

# Ingestion
MEDICAL_PDF_PATH = os.getenv("MEDICAL_PDF_PATH")

# RAG Agent
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 4))
RAG_RECENT_TURNS = int(os.getenv("RAG_RECENT_TURNS", 4))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", 0.3))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", 1024))
RAG_REASONING_MAX_TOKENS = int(os.getenv("RAG_REASONING_MAX_TOKENS", 300))

# Evaluator Agent
EVALUATOR_TEMPERATURE = float(os.getenv("EVALUATOR_TEMPERATURE", 0.0))
EVALUATOR_MAX_TOKENS = int(os.getenv("EVALUATOR_MAX_TOKENS", 150))
EVALUATOR_REASONING_MAX_TOKENS = int(os.getenv("EVALUATOR_REASONING_MAX_TOKENS", 100))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
TONE_THRESHOLD = float(os.getenv("TONE_THRESHOLD", 0.6))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))

# Reviser Agent
REVISER_TEMPERATURE = float(os.getenv("REVISER_TEMPERATURE", 0.3))
REVISER_MAX_TOKENS = int(os.getenv("REVISER_MAX_TOKENS", 1024))
REVISER_REASONING_MAX_TOKENS = int(os.getenv("REVISER_REASONING_MAX_TOKENS", 300))

# Summarizer Agent
SUMMARIZER_TEMPERATURE = float(os.getenv("SUMMARIZER_TEMPERATURE", 0.3))
SUMMARIZER_MAX_TOKENS = int(os.getenv("SUMMARIZER_MAX_TOKENS", 600))
SUMMARIZER_REASONING_MAX_TOKENS = int(os.getenv("SUMMARIZER_REASONING_MAX_TOKENS", 200))

# Planner Agent
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", 0.0))
PLANNER_MAX_TOKENS = int(os.getenv("PLANNER_MAX_TOKENS", 150))
PLANNER_REASONING_MAX_TOKENS = int(os.getenv("PLANNER_REASONING_MAX_TOKENS", 100))

