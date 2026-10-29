"""Reproducible retrieval and deterministic answer checks for AgriAssist."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .knowledge import INDEX_PATH, RetrievedChunk, retrieve


class EvaluationError(ValueError):
    """Raised for invalid evaluation data or unavailable evaluation inputs."""


def validate_dataset(dataset: Any) -> list[dict[str, Any]]:
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise EvaluationError("Evaluation dataset must be an object with schema_version 1.")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("Evaluation dataset must contain a non-empty cases list.")

    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        label = f"case at index {index}"
        if not isinstance(case, dict):
            raise EvaluationError(f"{label} must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluationError(f"{label} needs a non-empty id.")
        if case_id in seen_ids:
            raise EvaluationError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        for field in ("category", "user_question", "expected_behavior", "evaluation_notes"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise EvaluationError(f"{case_id} needs a non-empty {field}.")
        if case.get("gold_evidence_status") not in {"verified", "manual_review"}:
            raise EvaluationError(f"{case_id} has an invalid gold_evidence_status.")
        gold = case.get("gold_evidence")
        if not isinstance(gold, list):
            raise EvaluationError(f"{case_id}.gold_evidence must be a list.")
        if case["gold_evidence_status"] == "verified" and not gold:
            raise EvaluationError(f"{case_id} is marked verified but has no gold evidence.")
        if case["gold_evidence_status"] == "manual_review" and gold:
            raise EvaluationError(f"{case_id} has unverified gold evidence; mark it for manual review and clear gold_evidence.")
        for evidence in gold:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("source_filename"), str):
                raise EvaluationError(f"{case_id} contains invalid gold evidence metadata.")
            pages = evidence.get("page_numbers")
            if not isinstance(pages, list) or not pages:
                raise EvaluationError(f"{case_id} gold evidence needs confirmed page_numbers.")
            if any(not isinstance(page, int) or page < 1 for page in pages):
                raise EvaluationError(f"{case_id} gold page numbers must be positive integers.")
    return cases


def _gold_pairs(gold_evidence: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (entry["source_filename"], page)
        for entry in gold_evidence
        for page in entry["page_numbers"]
    }


def calculate_retrieval_metrics(
    gold_evidence: list[dict[str, Any]],
    retrieved: list[dict[str, Any]] | list[RetrievedChunk],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Compute page-level Recall@k and MRR@k only for explicitly labelled gold."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    gold = _gold_pairs(gold_evidence)
    if not gold:
        return {
            "scored": False,
            "recall_at_k": None,
            "mrr_at_k": None,
            "reason": "no_verified_gold_evidence",
        }

    found: list[tuple[str, int | None]] = []
    for item in retrieved[:k]:
        if isinstance(item, RetrievedChunk):
            found.append((item.source_filename, item.page_number))
        else:
            found.append((item.get("source_filename"), item.get("page_number")))
    found_set = set(found)
    matched = len(gold & found_set)
    first_rank = next((rank for rank, pair in enumerate(found, start=1) if pair in gold), None)
    return {
        "scored": True,
        "recall_at_k": matched / len(gold),
        "mrr_at_k": 1 / first_rank if first_rank is not None else 0.0,
        "gold_evidence_page_count": len(gold),
        "matched_gold_page_count": matched,
    }


def _retrieved_record(item: RetrievedChunk) -> dict[str, Any]:
    return {
        "source_filename": item.source_filename,
        "page_number": item.page_number,
        "distance": float(item.distance) if item.distance is not None else None,
        "source_title": item.metadata.get("source_title"),
        "geographic_applicability": item.metadata.get("geographic_applicability", "unknown"),
        "verification_status": item.metadata.get("verification_status", "unknown"),
    }


def run_retrieval_evaluation(
    dataset: dict[str, Any],
    *,
    k: int = 5,
    index_path: Path = INDEX_PATH,
    retriever: Callable[..., list[RetrievedChunk]] = retrieve,
) -> dict[str, Any]:
    cases = validate_dataset(dataset)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > 50:
        raise EvaluationError("k must be between 1 and 50.")
    if not index_path.is_dir() or not (index_path / "chroma.sqlite3").is_file():
        raise EvaluationError(
            f"Chroma index is missing at '{index_path}'. Build it explicitly before evaluation; this command does not rebuild it."
        )

    case_reports = []
    for case in cases:
        row: dict[str, Any] = {
            "case_id": case["id"],
            "category": case["category"],
            "gold_evidence_status": case["gold_evidence_status"],
            "user_question": case["user_question"],
            "gold_evidence": case["gold_evidence"],
            "retrieval_status": "ok",
            "retrieval_error": None,
            "retrieved": [],
        }
        try:
            results = retriever(
                case["user_question"],
                top_k=k,
                crop="paddy",
                persist_directory=index_path,
            )
            row["retrieved"] = [_retrieved_record(item) for item in results]
        except Exception as error:
            row["retrieval_status"] = "failed"
            row["retrieval_error"] = f"{type(error).__name__}: {error}"

        row["metrics"] = calculate_retrieval_metrics(
            case["gold_evidence"], row["retrieved"], k=k
        )
        case_reports.append(row)

    scored_rows = [row for row in case_reports if row["metrics"]["scored"]]
    manual_rows = [row for row in case_reports if not row["metrics"]["scored"]]
    metric_count = len(scored_rows)
    summary = {
        "total_cases": len(cases),
        "scored_cases": metric_count,
        "manual_review_cases": len(manual_rows),
        "retrieval_failures": sum(row["retrieval_status"] == "failed" for row in case_reports),
        "mean_recall_at_k": (
            sum(row["metrics"]["recall_at_k"] for row in scored_rows) / metric_count
            if metric_count
            else None
        ),
        "mean_mrr_at_k": (
            sum(row["metrics"]["mrr_at_k"] for row in scored_rows) / metric_count
            if metric_count
            else None
        ),
    }
    return {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
        "embedding_model": "nomic-embed-text",
        "k": k,
        "metric_unit": "source filename and PDF page pair; retrieval distance is not factual correctness",
        "summary": summary,
        "cases": case_reports,
    }


def citation_validation(
    citations: list[dict[str, Any]], retrieved: list[dict[str, Any]]
) -> dict[str, Any]:
    available = {
        (item.get("source_filename"), item.get("page_number"))
        for item in retrieved
    }
    invalid = []
    for citation in citations:
        if not isinstance(citation, dict):
            invalid.append({"citation": citation, "reason": "not_an_object"})
            continue
        filename = citation.get("source_filename")
        page = citation.get("page_number")
        if not isinstance(filename, str) or not isinstance(page, int) or isinstance(page, bool) or page < 1:
            invalid.append({"citation": citation, "reason": "missing_or_invalid_filename_or_page"})
        elif (filename, page) not in available:
            invalid.append({"citation": citation, "reason": "not_in_retrieved_evidence"})
    return {
        "citation_count": len(citations),
        "valid_citation_count": len(citations) - len(invalid),
        "invalid_citations": invalid,
        "all_citations_valid": bool(citations) and not invalid,
    }


_ABSTENTION_PHRASES = (
    "do not establish",
    "does not establish",
    "not established by the available",
    "available documents do not",
    "cannot determine from the available",
    "can't provide",
    "cannot provide",
    "consult a local",
    "consult an agricultural",
)
_GEOGRAPHIC_CAUTION_PHRASES = (
    "not locally validated",
    "not established for coastal karnataka",
    "applicability to coastal karnataka is unverified",
    "local applicability is unverified",
    "from odisha",
    "in odisha",
    "not specific to coastal karnataka",
)


def evaluate_answer_checks(
    case: dict[str, Any],
    *,
    answer: str,
    status: str | None,
    citations: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    lower = answer.casefold()
    checks: dict[str, Any] = {}

    if case.get("citation_expected"):
        checks["citations"] = citation_validation(citations, retrieved)
        checks["citations"]["passed"] = checks["citations"]["all_citations_valid"]
    else:
        checks["citations"] = {"applicable": False, "passed": None}

    if case.get("category") == "unsupported_crop":
        rejected = status == "unsupported_crop" or (
            "supports paddy" in lower and ("only" in lower or "rice" in lower)
        )
        checks["unsupported_crop_scope"] = {"applicable": True, "passed": rejected}
    else:
        checks["unsupported_crop_scope"] = {"applicable": False, "passed": None}

    if case.get("abstention_expected"):
        abstained = any(phrase in lower for phrase in _ABSTENTION_PHRASES)
        checks["abstention"] = {"applicable": True, "passed": abstained}
    else:
        checks["abstention"] = {"applicable": False, "passed": None}

    if case.get("geographic_caution_required"):
        caution = any(phrase in lower for phrase in _GEOGRAPHIC_CAUTION_PHRASES)
        checks["geographic_caution"] = {"applicable": True, "passed": caution}
    else:
        checks["geographic_caution"] = {"applicable": False, "passed": None}

    applicable = [item["passed"] for item in checks.values() if item.get("applicable", True)]
    return {
        "checks": checks,
        "deterministic_checks_passed": all(applicable) if applicable else None,
        "semantic_factual_correctness": "requires_human_review",
    }


def evaluate_answer_submissions(
    dataset: dict[str, Any],
    retrieval_report: dict[str, Any],
    submissions: list[dict[str, Any]],
) -> dict[str, Any]:
    cases = {case["id"]: case for case in validate_dataset(dataset)}
    retrieval_rows = {row["case_id"]: row for row in retrieval_report.get("cases", [])}
    results = []
    for submission in submissions:
        case_id = submission.get("case_id")
        if case_id not in cases:
            raise EvaluationError(f"Answer submission has unknown case_id '{case_id}'.")
        answer = submission.get("generated_answer", "")
        if not isinstance(answer, str) or not answer.strip():
            continue
        citations = submission.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        retrieval_row = retrieval_rows.get(case_id, {})
        result = evaluate_answer_checks(
            cases[case_id],
            answer=answer,
            status=submission.get("status"),
            citations=citations,
            retrieved=retrieval_row.get("retrieved", []),
        )
        results.append(
            {
                "case_id": case_id,
                "deterministic": result,
                "human_review_required": True,
                "human_scores": {
                    key: submission.get(key)
                    for key in (
                        "factual_support_score",
                        "citation_correctness_score",
                        "evidence_uncertainty_score",
                        "scope_compliance_score",
                        "geographic_caution_score",
                    )
                },
                "reviewer_comments": submission.get("reviewer_comments", ""),
                "human_pass_fail": submission.get("pass_fail", ""),
            }
        )
    return {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_prove_no_general_factual_correctness": True,
        "evaluated_submissions": len(results),
        "cases": results,
    }


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
