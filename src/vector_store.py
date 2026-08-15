from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunking import Chunk

logger = logging.getLogger(__name__)

_OPEN_CLIENTS: List = []

COLLECTION_NAME = "alura_rag"
MANIFEST_FILENAME = "manifest.json"
MODEL_CACHE_DIR = "models"


class VectorStoreError(Exception):
    """Raised when the vector store is missing or inconsistent."""


def _embedding_function(model_name: str) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(Path(MODEL_CACHE_DIR).resolve()),
    )


def _compute_signature(chunks: List[Chunk]) -> str:
    """Hash of all chunk contents and metadata to detect base changes."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: (c.metadata.get("source", ""), c.metadata.get("page", 0), c.metadata.get("chunk_index", 0))):
        digest.update(chunk.text.encode("utf-8", errors="replace"))
        digest.update(json.dumps(chunk.metadata, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _write_manifest(persist_dir: Path, signature: str, n_chunks: int) -> None:
    (persist_dir / MANIFEST_FILENAME).write_text(
        json.dumps({"signature": signature, "n_chunks": n_chunks}, indent=2),
        encoding="utf-8",
    )


def _read_manifest(persist_dir: Path) -> Optional[dict]:
    manifest_path = persist_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _close_previous_clients() -> None:
    """Release Chroma clients so persisted files are not locked (Windows)."""
    while _OPEN_CLIENTS:
        client = _OPEN_CLIENTS.pop()
        try:
            client.close()
        except Exception as exc:
            logger.debug("falha ao fechar cliente anterior: %s", exc)


def _chunk_id(chunk: Chunk) -> str:
    meta = chunk.metadata
    return hashlib.sha1(
        f"{meta.get('source')}:{meta.get('page')}:{meta.get('chunk_index')}".encode()
    ).hexdigest()[:16]


def build_vector_store(
    chunks: List[Chunk],
    persist_dir: str | Path = "data/chromadb",
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Build the vector store from chunks, reusing it when unchanged.

    The base is only rebuilt when the documents actually change; otherwise
    the persisted store is opened as-is, so indexing is not repeated and the
    collection is not duplicated.
    """
    if not chunks:
        raise VectorStoreError("nenhum chunk para indexar")

    persist = Path(persist_dir)
    persist.mkdir(parents=True, exist_ok=True)

    signature = _compute_signature(chunks)
    manifest = _read_manifest(persist)

    if manifest and manifest.get("signature") == signature:
        store = _open_store(persist, collection_name, embedding_function)
        try:
            if store._collection.count() == manifest.get("n_chunks"):
                logger.info("base vetorial inalterada, reutilizando %d chunks", store._collection.count())
                return store
        except Exception as exc:
            logger.warning("colecao persistida inconsistente (%s); reindexando", exc)

    logger.info("documentos alterados; reindexando base vetorial")
    _close_previous_clients()
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)

    store = _open_store(persist, collection_name, embedding_function)
    documents = [
        Document(page_content=chunk.text, metadata=dict(chunk.metadata), id=_chunk_id(chunk))
        for chunk in chunks
    ]
    store.add_documents(documents, ids=[doc.id for doc in documents])
    _write_manifest(persist, signature, len(documents))
    logger.info("base vetorial indexada com %d chunks", len(documents))
    return store


def _open_store(
    persist_dir: Path,
    collection_name: str,
    embedding_function: Optional[Embeddings],
) -> Chroma:
    if embedding_function is None:
        from src import config

        cfg = config.Config.from_env()
        embedding_function = _embedding_function(cfg.embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_dir),
    )
    if hasattr(store, "_client"):
        _OPEN_CLIENTS.append(store._client)
    return store


def load_vector_store(
    persist_dir: str | Path = "data/chromadb",
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Open an existing vector store without reindexing."""
    persist = Path(persist_dir)
    if not (persist / MANIFEST_FILENAME).exists():
        raise VectorStoreError("base vetorial inexistente. Execute a indexacao antes.")
    return _open_store(persist, collection_name, embedding_function)