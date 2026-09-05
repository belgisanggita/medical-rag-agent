"""
Qdrant index operations for the medical RAG pipeline: creating/checking
collections, embedding text, writing chunks, searching, and tracking which
documents have been fully ingested.

Uses app.infra.qdrant_infra.get_client() for the connection - this module
owns what happens with that connection, not how it's obtained.
"""

import time
import uuid

from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.config import properties_setup as settings
from app.infra.qdrant_infra import get_client
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_model = None

# A doc is only "ingested" once every chunk made it in - tracked via a marker
# point in a tiny side collection, written last, after index_chunks()
# finishes without error. A run that crashes mid-way (e.g. a missing
# dependency partway through embedding) never writes this marker, so the
# next attempt correctly sees the document as NOT done instead of finding
# the partial content points and wrongly skipping the rest.
_MARKER_COLLECTION_SUFFIX = "_ingestion_state"


def _marker_collection_name() -> str:
    return f"{settings.QDRANT_COLLECTION_NAME}{_MARKER_COLLECTION_SUFFIX}"


def _marker_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL}' (first call downloads/loads weights)...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def ensure_collection():
    """Creates both the content collection and its ingestion-marker
    collection if missing. Each is checked independently - the content
    collection surviving from a previous run must never short-circuit
    creation of the marker collection (that gap caused a 404 the first
    time mark_document_ingested() ran against an old collection)."""
    client = get_client()

    if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        logger.info(f"Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' already exists")
    else:
        dim = _get_model().get_embedding_dimension()
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' (dim={dim})")

    marker_name = _marker_collection_name()
    if client.collection_exists(marker_name):
        logger.info(f"Qdrant ingestion-marker collection '{marker_name}' already exists")
    else:
        client.create_collection(
            collection_name=marker_name,
            vectors_config=qmodels.VectorParams(size=1, distance=qmodels.Distance.COSINE),
        )
        logger.info(f"Created Qdrant ingestion-marker collection '{marker_name}'")


def is_document_ingested(doc_id: str) -> bool:
    """True only if a previous run fully finished indexing this doc_id (see
    mark_document_ingested). A crash partway through leaves content points
    but no marker, so this correctly reports False and the caller re-runs."""
    client = get_client()
    marker_name = _marker_collection_name()
    if not client.collection_exists(marker_name):
        return False
    return len(client.retrieve(collection_name=marker_name, ids=[_marker_id(doc_id)])) > 0


def mark_document_ingested(doc_id: str, chunk_count: int):
    client = get_client()
    client.upsert(
        collection_name=_marker_collection_name(),
        points=[
            qmodels.PointStruct(
                id=_marker_id(doc_id),
                vector=[0.0],
                payload={"doc_id": doc_id, "chunk_count": chunk_count},
            )
        ],
    )


def delete_document(doc_id: str):
    """Remove any (partial or stale) points for this doc_id from the content
    collection - used before re-ingesting so a retry after a crash doesn't
    leave duplicate chunks alongside the fresh full set."""
    client = get_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        return
    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )
        ),
    )


def embed_query(text: str):
    """Embed a user question for retrieval. intfloat/multilingual-e5-* models
    are trained with an asymmetric "query: " / "passage: " prefix convention -
    using the right one on each side measurably improves retrieval quality."""
    return _get_model().encode(f"query: {text}").tolist()


def search(query_vector, top_k: int = 4):
    """Returns the top_k most similar points as a list of
    {"text": str, "score": float, **metadata} dicts."""
    client = get_client()
    result = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": p.score, **p.payload} for p in result.points]


def index_chunks(chunks, batch_size: int = 64):
    client = get_client()
    model = _get_model()

    total = len(chunks)
    total_batches = (total + batch_size - 1) // batch_size
    logger.info(f"Embedding + indexing {total} chunks into '{settings.QDRANT_COLLECTION_NAME}' in {total_batches} batches of {batch_size}...")

    start = time.monotonic()
    for batch_num, i in enumerate(range(0, total, batch_size), start=1):
        batch = chunks[i : i + batch_size]

        t0 = time.monotonic()
        # "passage: " prefix - the indexing-side half of e5's asymmetric
        # query/passage convention (see embed_query above).
        vectors = model.encode([f"passage: {c['text']}" for c in batch], show_progress_bar=False).tolist()
        embed_s = time.monotonic() - t0

        t0 = time.monotonic()
        points = [
            qmodels.PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
            for chunk, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)
        upsert_s = time.monotonic() - t0

        done = i + len(batch)
        elapsed = time.monotonic() - start
        logger.info(
            f"Batch {batch_num}/{total_batches}: embedded in {embed_s:.1f}s, upserted in {upsert_s:.1f}s "
            f"-> {done}/{total} chunks ({100 * done // total}%), elapsed {elapsed:.0f}s"
        )

    logger.info(f"Finished indexing {total} chunks in {time.monotonic() - start:.0f}s.")
