import json
from pathlib import Path

import chromadb
import pytest
import pymupdf

from agriassist.knowledge import (
    ALLOWED_PDF_FILENAMES,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    KnowledgeIngestionError,
    OllamaEmbedder,
    OllamaEmbeddingError,
    _open_collection,
    build_index,
    chunk_page_text,
    deterministic_chunk_id,
    extract_selected_chunks,
    retrieve,
)


class FakeEmbedder:
    def __init__(self):
        self.document_batches = []
        self.queries = []

    @staticmethod
    def _vector(text):
        return [
            float(len(text) % 97 + 1),
            float(text.casefold().count("rice") + 1),
            float(sum(text.encode("utf-8")) % 89 + 1),
        ]

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)


def write_pdf(path: Path, pages: list[str]) -> None:
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)


@pytest.fixture
def test_corpus(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    titles = {
        "8.pdf": "Rice sheath blight article",
        "04+-+Srujani+148717_Revised.pdf": "Bacterial leaf blight research article",
        "15+Emerging+minor+diseases+of+rice+in+Odisha.pdf": "Minor rice diseases",
        "11-14.pdf": "Rice micronutrient management",
    }
    inventory_docs = []
    for index, filename in enumerate(ALLOWED_PDF_FILENAMES):
        write_pdf(raw_dir / filename, [f"Paddy rice source {index}. " + ("Useful passage. " * 20)])
        inventory_docs.append(
            {
                "filename": filename,
                "extraction_status": "ok",
                "source_suitability": "candidate_for_rag",
                "extracted_title": titles[filename],
                "publisher": "Verified Publisher" if filename.startswith("04+") else None,
                "topic_category": "rice_disease" if index != 3 else "nutrient_management",
                "crop_relevance": "Rice (paddy); explicit in the source.",
                "geographic_relevance": "Odisha context; applicability to coastal Karnataka is unverified.",
                "source_url": "https://example.invalid/source" if filename.startswith("04+") else None,
                "verification_notes": ["Provisional test source; not authority-approved."],
                "confidence_or_uncertainty": "Local applicability is uncertain.",
            }
        )
    inventory_path = tmp_path / "pdf_inventory.json"
    inventory_path.write_text(
        json.dumps({"documents": inventory_docs}), encoding="utf-8"
    )
    return raw_dir, inventory_path


def test_page_chunking_preserves_page_and_source_metadata():
    metadata = {"topic_category": "rice_disease", "publisher_status": "unknown"}
    chunks = chunk_page_text(
        "Rice symptoms. " * 20,
        source_filename="8.pdf",
        page_number=3,
        source_metadata=metadata,
        chunk_size=60,
        overlap=12,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source_filename"] == "8.pdf" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 3 for chunk in chunks)
    assert all(chunk.metadata["topic_category"] == "rice_disease" for chunk in chunks)
    assert all(len(chunk.text) <= 60 for chunk in chunks)
    assert set(chunks[0].text) & set(chunks[1].text)


def test_empty_page_does_not_create_a_chunk():
    assert chunk_page_text(" \n\t ", source_filename="8.pdf", page_number=2) == []


def test_chunk_ids_are_deterministic_and_content_sensitive():
    first = deterministic_chunk_id("8.pdf", 1, 0, "same passage")

    assert deterministic_chunk_id("8.pdf", 1, 0, "same passage") == first
    assert deterministic_chunk_id("8.pdf", 1, 0, "changed passage") != first
    assert deterministic_chunk_id("8.pdf", 1, 0, "same passage") != deterministic_chunk_id(
        "8.pdf", 2, 0, "same passage"
    )


def test_pdf_extraction_keeps_page_numbers_and_ignores_empty_pages(test_corpus):
    raw_dir, inventory_path = test_corpus
    write_pdf(raw_dir / "8.pdf", ["First paddy passage.", "", "Third paddy passage."])

    chunks = extract_selected_chunks(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        selected_filenames=("8.pdf",),
    )

    assert [chunk.metadata["page_number"] for chunk in chunks] == [1, 3]
    assert [chunk.text for chunk in chunks] == ["First paddy passage.", "Third paddy passage."]


def test_source_metadata_is_preserved_in_chroma(test_corpus, tmp_path):
    raw_dir, inventory_path = test_corpus
    collection_path = tmp_path / "chroma"
    source_chunks = extract_selected_chunks(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        selected_filenames=("04+-+Srujani+148717_Revised.pdf",),
    )
    fake = FakeEmbedder()
    # Build all four sources so the production allowlist path is exercised.
    build_index(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        persist_directory=collection_path,
        embedder=fake,
    )
    collection = chromadb.PersistentClient(path=str(collection_path)).get_collection(
        COLLECTION_NAME
    )
    records = collection.get(where={"source_filename": "04+-+Srujani+148717_Revised.pdf"})

    assert source_chunks
    metadata = records["metadatas"][0]
    assert metadata["source_title"] == "Bacterial leaf blight research article"
    assert metadata["publisher"] == "Verified Publisher"
    assert metadata["original_source_url"] == "https://example.invalid/source"
    assert metadata["page_number"] == 1
    assert metadata["topic_category"] == "rice_disease"
    assert "Odisha" in metadata["geographic_applicability"]
    assert metadata["verification_status"].startswith("provisional;")
    assert metadata["publisher_status"] == "extracted_explicitly"


def test_empty_index_returns_no_results_without_embedding(tmp_path):
    _, collection = _open_collection(tmp_path / "empty")

    class MustNotEmbed:
        def embed_query(self, text):
            raise AssertionError("empty index should not call Ollama")

    assert retrieve("rice leaf symptoms", collection=collection, embedder=MustNotEmbed()) == []


@pytest.mark.parametrize(
    "kwargs, query",
    [({}, "  "), ({"crop": "tomato"}, "spots"), ({"top_k": 0}, "spots"), ({"top_k": 51}, "spots")],
)
def test_invalid_queries_and_scope_are_rejected(kwargs, query, tmp_path):
    _, collection = _open_collection(tmp_path / "invalid")

    with pytest.raises(ValueError):
        retrieve(query, collection=collection, **kwargs)


def test_missing_allowlisted_pdf_fails_clearly(test_corpus):
    raw_dir, inventory_path = test_corpus
    (raw_dir / ALLOWED_PDF_FILENAMES[1]).unlink()

    with pytest.raises(KnowledgeIngestionError) as error:
        extract_selected_chunks(data_directory=raw_dir, inventory_path=inventory_path)
    assert ALLOWED_PDF_FILENAMES[1] in str(error.value)


def test_only_four_allowlisted_pdfs_can_be_selected(test_corpus):
    raw_dir, inventory_path = test_corpus
    write_pdf(raw_dir / "unlisted-tomato.pdf", ["Tomato article."])

    with pytest.raises(KnowledgeIngestionError, match="explicit paddy PDF allowlist"):
        extract_selected_chunks(
            data_directory=raw_dir,
            inventory_path=inventory_path,
            selected_filenames=("unlisted-tomato.pdf",),
        )


def test_extra_pdf_is_not_indexed_and_reingestion_does_not_duplicate(test_corpus, tmp_path):
    raw_dir, inventory_path = test_corpus
    write_pdf(raw_dir / "unlisted-tomato.pdf", ["Tomato article."])
    persist_directory = tmp_path / "allowlist-index"
    fake = FakeEmbedder()

    first_count = build_index(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        persist_directory=persist_directory,
        embedder=fake,
    )
    collection = chromadb.PersistentClient(path=str(persist_directory)).get_collection(
        COLLECTION_NAME
    )
    first_index_count = collection.count()
    second_count = build_index(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        persist_directory=persist_directory,
        embedder=fake,
    )

    records = collection.get()
    indexed_sources = {metadata["source_filename"] for metadata in records["metadatas"]}
    assert first_count == second_count
    assert collection.count() == first_index_count
    assert indexed_sources == set(ALLOWED_PDF_FILENAMES)
    assert "unlisted-tomato.pdf" not in indexed_sources


def test_retrieval_returns_text_provenance_metadata_and_distance(test_corpus, tmp_path):
    raw_dir, inventory_path = test_corpus
    persist_directory = tmp_path / "retrieval-index"
    fake = FakeEmbedder()
    build_index(
        data_directory=raw_dir,
        inventory_path=inventory_path,
        persist_directory=persist_directory,
        embedder=fake,
    )
    collection = chromadb.PersistentClient(path=str(persist_directory)).get_collection(
        COLLECTION_NAME
    )

    results = retrieve(
        "paddy rice disease passage",
        collection=collection,
        embedder=fake,
        top_k=3,
    )

    assert len(results) == 3
    assert all(result.text for result in results)
    assert all(result.source_filename in ALLOWED_PDF_FILENAMES for result in results)
    assert all(result.page_number == 1 for result in results)
    assert all(result.metadata["crop_scope"] == "paddy" for result in results)
    assert all(result.distance is not None for result in results)


def test_ollama_embedder_uses_same_fixed_model_for_docs_and_queries(monkeypatch):
    import ollama

    class StubClient:
        def __init__(self):
            self.calls = []

        def embed(self, *, model, input):
            self.calls.append((model, input))
            return {"embeddings": [[0.1, 0.2] for _ in input]}

    stub = StubClient()
    monkeypatch.setattr(ollama, "Client", lambda **kwargs: stub)
    embedder = OllamaEmbedder()

    docs = embedder.embed_documents(["one passage", "another passage"])
    query = embedder.embed_query("a paddy query")

    assert docs == [[0.1, 0.2], [0.1, 0.2]]
    assert query == [0.1, 0.2]
    assert stub.calls == [
        (EMBEDDING_MODEL, ["search_document: one passage", "search_document: another passage"]),
        (EMBEDDING_MODEL, ["search_query: a paddy query"]),
    ]


def test_ollama_failure_names_required_model_without_fallback(monkeypatch):
    import ollama

    class FailingClient:
        def embed(self, **kwargs):
            raise ConnectionError("service unavailable")

    monkeypatch.setattr(ollama, "Client", lambda **kwargs: FailingClient())

    with pytest.raises(OllamaEmbeddingError, match="nomic-embed-text"):
        OllamaEmbedder().embed_query("paddy")
