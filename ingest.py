"""
Ingestion pipeline: Medical Book PDF -> chunks -> Qdrant.

ensure_ingested() is idempotent and keyed off a content hash of the PDF
(doc_id), so it's safe to call on every app startup: if this exact file
was already indexed, it's a fast Qdrant count check and nothing gets
re-embedded. If the source PDF is swapped for a different edition, its
hash changes and it gets ingested as a new document automatically.

app.py is the only entrypoint that calls this (on startup, via
@st.cache_resource) - this module has no CLI of its own.
"""

import hashlib

from app.utils.extract_pdf import extract_text_per_page, chunk_text
from app.index.qdrant_index import (
    ensure_collection,
    is_document_ingested,
    index_chunks,
    mark_document_ingested,
    delete_document,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _compute_doc_id(pdf_path: str) -> str:
    """Content hash of the PDF, not the file path - so the same file under a
    different path is recognized as already-ingested, and an edited/replaced
    file under the same path is correctly treated as a new document."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def ensure_ingested(pdf_path: str):
    ensure_collection()

    doc_id = _compute_doc_id(pdf_path)
    if is_document_ingested(doc_id):
        logger.info(f"'{pdf_path}' already ingested (doc_id={doc_id}), skipping.")
        return

    # a previous attempt for this doc_id may have crashed partway through
    # (no ingestion marker got written in that case) - clear any leftover
    # points first so this run always produces one clean, complete set.
    logger.info(f"Clearing any partial/stale points for doc_id={doc_id} before re-ingesting...")
    delete_document(doc_id)

    logger.info(f"New document detected (doc_id={doc_id}). Extracting text from {pdf_path}...")
    pages = extract_text_per_page(pdf_path)

    logger.info(f"Chunking {len(pages)} pages...")
    chunks = chunk_text(pages)
    for chunk in chunks:
        chunk["metadata"]["doc_id"] = doc_id

    logger.info(f"Indexing {len(chunks)} chunks into Qdrant...")
    index_chunks(chunks)

    mark_document_ingested(doc_id, len(chunks))
    logger.info("Done.")
