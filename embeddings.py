"""
Semantic search over LinkedIn connections using ChromaDB + sentence-transformers.

The embedding model (all-MiniLM-L6-v2, ~80MB) is downloaded on first use
and cached locally by sentence-transformers.
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import db

logger = logging.getLogger(__name__)

CHROMA_PATH = str(Path(__file__).parent / "data" / "chroma")
COLLECTION_NAME = "connections"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{MODEL_NAME}' (downloads ~80MB on first run)...")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded.")
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_index(progress_cb=None) -> int:
    """Embed all connections from SQLite into ChromaDB. Safe to re-run (upserts)."""
    connections = db.search_connections(limit=100_000)
    if not connections:
        logger.warning("No connections in DB to embed.")
        return 0

    model = get_model()
    collection = get_collection()

    ids, documents, metadatas = [], [], []
    for c in connections:
        text = " | ".join(filter(None, [c["name"], c["headline"], c["location"]]))
        ids.append(c["profile_id"])
        documents.append(text)
        metadatas.append({
            "name": c["name"] or "",
            "headline": c["headline"] or "",
            "location": c["location"] or "",
            "degree": int(c["degree"] or 3),
            "profile_url": c["profile_url"] or "",
        })

    BATCH = 256
    total = len(ids)
    for i in range(0, total, BATCH):
        batch_ids = ids[i : i + BATCH]
        batch_docs = documents[i : i + BATCH]
        batch_meta = metadatas[i : i + BATCH]
        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, embeddings=embeddings, metadatas=batch_meta)
        if progress_cb:
            progress_cb(f"🧠 Indexing... {min(i + BATCH, total)}/{total}")

    logger.info(f"Indexed {total} connections into ChromaDB")
    return total


def semantic_search(query: str, limit: int = 10, location: str = None) -> list:
    """Return connections semantically closest to the query string."""
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB index is empty — run /embed first")
        return []

    model = get_model()
    query_vec = model.encode(query).tolist()

    # Fetch extra results to allow post-filtering by location
    fetch = min(limit * 3, 100)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for rid, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if location and location.lower() not in (meta.get("location") or "").lower():
            continue
        output.append({
            "profile_id": rid,
            "name": meta.get("name", ""),
            "headline": meta.get("headline", ""),
            "location": meta.get("location", ""),
            "degree": meta.get("degree", 3),
            "profile_url": meta.get("profile_url", ""),
            "similarity": round(1 - dist, 3),
        })
        if len(output) >= limit:
            break

    return output


def index_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0
