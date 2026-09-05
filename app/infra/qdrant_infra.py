"""
Qdrant connection only. Everything that operates ON the index (creating
collections, embedding, search, delete, ingestion-state tracking) lives in
app/index/qdrant_index.py - this module's only job is handing back a
connected client so that file (and anything else) doesn't each open its
own connection.
"""

from qdrant_client import QdrantClient

from app.config import properties_setup as settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _client
