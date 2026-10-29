"""Local PDF retrieval over a small, explicitly allowlisted paddy corpus.

This module retrieves and cites source passages only. Similarity is not evidence
of factual correctness, and candidate documents are not approved authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
import pymupdf


ROOT = Path(__file__).resolve().parents[2]
RAW_PDF_DIR = ROOT / "data" / "raw"
INVENTORY_PATH = ROOT / "data" / "pdf_inventory.json"
INDEX_PATH = ROOT / "data" / "indexes" / "agriassist_chroma"
COLLECTION_NAME = "agriassist_paddy_knowledge_v1"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 140
EMBEDDING_BATCH_SIZE = 24

# Provisional, content-relevant sources only. Weather bulletins and the separate
# bacterial-blight experiment are deliberately excluded from this first index.
ALLOWED_PDF_FILENAMES = (
    "8.pdf",
    "04+-+Srujani+148717_Revised.pdf",
    "15+Emerging+minor+diseases+of+rice+in+Odisha.pdf",
    "11-14.pdf",
)


class KnowledgeError(RuntimeError):
    """Base error for corpus ingestion and retrieval failures."""


class KnowledgeIngestionError(KnowledgeError):
    """Raised when selected source documents cannot be safely indexed."""


class OllamaEmbeddingError(KnowledgeError):
    """Raised when local Ollama embeddings cannot be produced."""


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_filename: str
    page_number: int | None
    metadata: dict[str, Any]
    distance: float | None


class OllamaEmbedder:
    """Embedding adapter fixed to the local nomic-embed-text model."""

    def __init__(self, host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as error:  # pragma: no cover - depends on environment
            raise OllamaEmbeddingError(
                "The Ollama Python package is missing; install the project's local dependencies."
            ) from error

        self.model = EMBEDDING_MODEL
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}"])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = self.client.embed(model=self.model, input=inputs)
        except Exception as error:
            raise OllamaEmbeddingError(
                f"Could not create embeddings with Ollama model '{self.model}'. "
                "Confirm the Ollama service is running and the model is available "
                f"(ollama pull {self.model}). No alternate model was used. Details: {error}"
            ) from error

        vectors = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise OllamaEmbeddingError(
                f"Ollama model '{self.model}' returned an unexpected embedding response."
            )
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise OllamaEmbeddingError(
                f"Ollama model '{self.model}' returned an empty or invalid vector."
            )
        return vectors


def _clean_page_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def deterministic_chunk_id(
    source_filename: str, page_number: int, chunk_index: int, text: str
) -> str:
    """Create a repeatable ID from the original source location and chunk text."""

    payload = f"{source_filename}\0{page_number}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_page_text(
    text: str,
    *,
    source_filename: str,
    page_number: int,
    source_metadata: dict[str, str | int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split one page into overlapping chunks, retaining its source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")
    if page_number < 1:
        raise ValueError("page_number must be one-based and at least 1")

    cleaned = _clean_page_text(text)
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            lower_bound = start + chunk_size // 2
            boundary = max(cleaned.rfind("\n", lower_bound, end), cleaned.rfind(" ", lower_bound, end))
            if boundary > start:
                end = boundary

        chunk_text = cleaned[start:end].strip()
        if chunk_text:
            index = len(chunks)
            metadata: dict[str, str | int] = dict(source_metadata or {})
            metadata.update(
                {
                    "source_filename": source_filename,
                    "page_number": page_number,
                    "chunk_index": index,
                    "crop_scope": "paddy",
                }
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=deterministic_chunk_id(source_filename, page_number, index, chunk_text),
                    text=chunk_text,
                    metadata=metadata,
                )
            )
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _source_metadata(filename: str, inventory_entry: dict[str, Any]) -> dict[str, str | int]:
    if inventory_entry.get("extraction_status") != "ok":
        raise KnowledgeIngestionError(
            f"Inventory does not confirm successful text extraction for '{filename}'."
        )
    if inventory_entry.get("source_suitability") != "candidate_for_rag":
        raise KnowledgeIngestionError(
            f"Selected source '{filename}' is not marked candidate_for_rag in the inventory."
        )
    crop_relevance = inventory_entry.get("crop_relevance")
    if not isinstance(crop_relevance, str) or not re.search(r"\b(rice|paddy)\b", crop_relevance, re.I):
        raise KnowledgeIngestionError(
            f"Selected source '{filename}' has no paddy/rice relevance recorded in the inventory."
        )

    title = inventory_entry.get("extracted_title")
    publisher = inventory_entry.get("publisher")
    source_url = inventory_entry.get("source_url")
    metadata: dict[str, str | int] = {
        "topic_category": str(inventory_entry.get("topic_category") or "unknown"),
        "crop_relevance": crop_relevance,
        "geographic_applicability": str(inventory_entry.get("geographic_relevance") or "unknown"),
        "verification_status": "provisional; authority, applicability, and reuse permission unverified",
        "source_suitability": "candidate_for_rag",
        "source_title_status": "extracted_explicitly" if isinstance(title, str) and title else "unknown",
        "publisher_status": "extracted_explicitly" if isinstance(publisher, str) and publisher else "unknown",
        "source_url_status": "explicit_in_document" if isinstance(source_url, str) and source_url else "unknown",
        "confidence_or_uncertainty": str(inventory_entry.get("confidence_or_uncertainty") or "unknown"),
        "verification_notes": " | ".join(inventory_entry.get("verification_notes") or ["unknown"]),
    }
    if isinstance(title, str) and title:
        metadata["source_title"] = title
    if isinstance(publisher, str) and publisher:
        metadata["publisher"] = publisher
    if isinstance(source_url, str) and source_url:
        metadata["original_source_url"] = source_url
    return metadata


def extract_selected_chunks(
    *,
    data_directory: Path = RAW_PDF_DIR,
    inventory_path: Path = INVENTORY_PATH,
    selected_filenames: tuple[str, ...] = ALLOWED_PDF_FILENAMES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Extract only allowlisted PDFs and return non-empty, page-scoped chunks."""

    selected = tuple(selected_filenames)
    unsupported = sorted(set(selected) - set(ALLOWED_PDF_FILENAMES))
    if unsupported:
        raise KnowledgeIngestionError(
            "Only the explicit paddy PDF allowlist may be indexed; rejected: "
            + ", ".join(unsupported)
        )
    if not selected:
        raise KnowledgeIngestionError("No selected PDFs were requested for ingestion.")

    missing = [name for name in selected if not (data_directory / name).is_file()]
    if missing:
        raise KnowledgeIngestionError(
            "Required allowlisted PDF(s) are missing from "
            f"'{data_directory}': {', '.join(missing)}"
        )
    if not inventory_path.is_file():
        raise KnowledgeIngestionError(f"PDF inventory not found: {inventory_path}")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeIngestionError(f"Could not read PDF inventory '{inventory_path}': {error}") from error

    entries = {
        item.get("filename"): item
        for item in inventory.get("documents", [])
        if isinstance(item, dict) and isinstance(item.get("filename"), str)
    }
    chunks: list[DocumentChunk] = []
    for filename in selected:
        entry = entries.get(filename)
        if entry is None:
            raise KnowledgeIngestionError(
                f"No inventory metadata exists for selected PDF '{filename}'."
            )
        source_metadata = _source_metadata(filename, entry)
        pdf_path = data_directory / filename
        try:
            with pymupdf.open(pdf_path) as document:
                for page_number, page in enumerate(document, start=1):
                    chunks.extend(
                        chunk_page_text(
                            page.get_text(),
                            source_filename=filename,
                            page_number=page_number,
                            source_metadata=source_metadata,
                            chunk_size=chunk_size,
                            overlap=overlap,
                        )
                    )
        except Exception as error:
            raise KnowledgeIngestionError(
                f"Could not extract selected PDF '{filename}': {type(error).__name__}: {error}"
            ) from error
    if not chunks:
        raise KnowledgeIngestionError(
            "The selected PDFs produced no extractable text chunks; OCR may be required."
        )
    return chunks


def _open_collection(persist_directory: Path):
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    return client, collection


def _embed_document_chunks(embedder: Embedder, chunks: list[DocumentChunk]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = embedder.embed_documents([chunk.text for chunk in batch])
        if len(batch_vectors) != len(batch):
            raise OllamaEmbeddingError("Embedding count did not match the number of source chunks.")
        if any(not vector for vector in batch_vectors):
            raise OllamaEmbeddingError("Embedding service returned an empty vector.")
        vectors.extend(batch_vectors)
    return vectors


def build_index(
    *,
    data_directory: Path = RAW_PDF_DIR,
    inventory_path: Path = INVENTORY_PATH,
    persist_directory: Path = INDEX_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    rebuild: bool = False,
    embedder: Embedder | None = None,
    collection: Any | None = None,
) -> int:
    """Build/update the persistent Chroma index from the four selected PDFs."""

    chunks = extract_selected_chunks(
        data_directory=data_directory,
        inventory_path=inventory_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    embedding_client = embedder or OllamaEmbedder()
    embeddings = _embed_document_chunks(embedding_client, chunks)

    if collection is None:
        _, collection = _open_collection(persist_directory)

    if rebuild:
        existing_ids = collection.get().get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
    else:
        # Replace records for these sources before upserting, removing stale chunks
        # if a selected PDF has changed since the previous ingestion.
        for filename in ALLOWED_PDF_FILENAMES:
            collection.delete(where={"source_filename": filename})

    for start in range(0, len(chunks), 128):
        batch = chunks[start : start + 128]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings[start : start + len(batch)],
        )
    return len(chunks)


def _paddy_filter() -> dict[str, Any]:
    return {
        "$and": [
            {"source_filename": {"$in": list(ALLOWED_PDF_FILENAMES)}},
            {"crop_scope": "paddy"},
        ]
    }


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    crop: str = "paddy",
    persist_directory: Path = INDEX_PATH,
    embedder: Embedder | None = None,
    collection: Any | None = None,
) -> list[RetrievedChunk]:
    """Retrieve paddy-source passages and their provenance; make no advice claims.

    Chroma's cosine distance is returned as ``distance``. It is a ranking signal,
    not a measure of factual correctness or source authority.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 50:
        raise ValueError("top_k must be an integer between 1 and 50")
    if not isinstance(crop, str) or crop.strip().casefold() not in {"paddy", "rice"}:
        raise ValueError("AgriAssist retrieval currently supports paddy (rice) only")

    if collection is None:
        _, collection = _open_collection(persist_directory)
    if collection.count() == 0:
        return []

    embedding_client = embedder or OllamaEmbedder()
    query_vector = embedding_client.embed_query(query.strip())
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=_paddy_filter(),
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []

    retrieved: list[RetrievedChunk] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        filename = metadata.get("source_filename")
        if filename not in ALLOWED_PDF_FILENAMES or metadata.get("crop_scope") != "paddy":
            continue
        distance = distances[index] if index < len(distances) else None
        retrieved.append(
            RetrievedChunk(
                text=text,
                source_filename=filename,
                page_number=metadata.get("page_number"),
                metadata=metadata,
                distance=float(distance) if distance is not None else None,
            )
        )
    return retrieved


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build/query the local paddy PDF index.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Index the allowlisted PDFs.")
    build_parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild the collection.")
    query_parser = commands.add_parser("query", help="Search indexed passages.")
    query_parser.add_argument("query", help="A paddy-focused search query.")
    query_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "build":
        count = build_index(rebuild=args.rebuild)
        print(f"Indexed {count} chunks in {COLLECTION_NAME} under {INDEX_PATH.relative_to(ROOT)}")
    else:
        for result in retrieve(args.query, top_k=args.top_k):
            print(
                f"[{result.source_filename}, page {result.page_number}, "
                f"distance={result.distance}] {result.metadata.get('geographic_applicability')}\n"
                f"{result.text}\n"
            )
        print("Distances rank matches; they do not establish factual correctness.")


if __name__ == "__main__":
    _cli()
