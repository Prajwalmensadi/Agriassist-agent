import json
from pathlib import Path

import pytest

from agriassist.evaluation import (
    EvaluationError,
    calculate_retrieval_metrics,
    citation_validation,
    evaluate_answer_checks,
    run_retrieval_evaluation,
    validate_dataset,
    write_json_report,
)


DATASET_PATH = Path(__file__).with_name("evaluation_cases.json")


def dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_has_unique_ids_and_verified_gold_only():
    cases = validate_dataset(dataset())
    assert len(cases) >= 20
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["gold_evidence"] for case in cases if case["gold_evidence_status"] == "verified")
    assert all(not case["gold_evidence"] for case in cases if case["gold_evidence_status"] == "manual_review")


def test_dataset_rejects_duplicate_case_ids():
    value = dataset()
    value["cases"].append(dict(value["cases"][0]))
    with pytest.raises(EvaluationError, match="Duplicate"):
        validate_dataset(value)


def test_retrieval_metrics_compute_page_recall_and_reciprocal_rank():
    gold = [{"source_filename": "rice.pdf", "page_numbers": [2, 4]}]
    retrieved = [
        {"source_filename": "other.pdf", "page_number": 1},
        {"source_filename": "rice.pdf", "page_number": 4},
        {"source_filename": "rice.pdf", "page_number": 2},
    ]
    metrics = calculate_retrieval_metrics(gold, retrieved, k=2)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["matched_gold_page_count"] == 1


def test_manual_review_case_has_no_retrieval_score():
    assert calculate_retrieval_metrics([], [], k=5) == {
        "scored": False,
        "recall_at_k": None,
        "mrr_at_k": None,
        "reason": "no_verified_gold_evidence",
    }


def test_retrieval_evaluation_calls_injected_retriever_and_reports_manual_cases(tmp_path):
    cases = dataset()
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").write_text("mock index marker")
    calls = []

    class Chunk:
        source_filename = "8.pdf"
        page_number = 2
        distance = 0.1
        metadata = {"geographic_applicability": "unknown", "verification_status": "unverified"}

    def fake_retrieve(query, **kwargs):
        calls.append((query, kwargs))
        return [Chunk()]

    report = run_retrieval_evaluation(cases, index_path=index, retriever=fake_retrieve)
    assert len(calls) == len(cases["cases"])
    assert calls[0][1]["crop"] == "paddy"
    assert report["summary"]["scored_cases"] == sum(
        case["gold_evidence_status"] == "verified" for case in cases["cases"]
    )
    assert report["cases"][0]["metrics"]["scored"] is False
    assert report["cases"][5]["metrics"]["recall_at_k"] == 1.0


def test_retrieval_requires_existing_index(tmp_path):
    with pytest.raises(EvaluationError, match="Chroma index is missing"):
        run_retrieval_evaluation(dataset(), index_path=tmp_path / "absent", retriever=lambda *a, **k: [])


def test_citation_validation_requires_source_and_page_in_retrieved_evidence():
    retrieved = [{"source_filename": "8.pdf", "page_number": 2}]
    assert citation_validation([{"source_filename": "8.pdf", "page_number": 2}], retrieved)["all_citations_valid"]
    invalid = citation_validation([{"source_filename": "8.pdf", "page_number": 3}], retrieved)
    assert invalid["all_citations_valid"] is False
    assert invalid["invalid_citations"][0]["reason"] == "not_in_retrieved_evidence"


def test_answer_checks_cover_abstention_scope_and_geographic_caution():
    case = {
        "citation_expected": False,
        "category": "unsupported_crop",
        "abstention_expected": True,
        "geographic_caution_required": True,
    }
    checked = evaluate_answer_checks(
        case,
        answer=("AgriAssist supports paddy only. The available documents do not establish applicability to "
                "coastal Karnataka; the source is from Odisha."),
        status="unsupported_crop",
        citations=[],
        retrieved=[],
    )
    assert checked["deterministic_checks_passed"] is True
    assert checked["semantic_factual_correctness"] == "requires_human_review"


def test_report_writer_outputs_readable_json(tmp_path):
    output = tmp_path / "nested" / "report.json"
    write_json_report({"ok": True}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
