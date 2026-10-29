"""Run deterministic checks on manually collected answers and citations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from agriassist.evaluation import (
    EvaluationError,
    evaluate_answer_submissions,
    write_json_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "evaluation_cases.json"
DEFAULT_RETRIEVAL_REPORT = ROOT / "data" / "processed" / "retrieval_evaluation.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "answer_evaluation.json"


def _read_submissions(path: Path) -> list[dict]:
    submissions = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                citations = json.loads(row.get("citations_json") or "[]")
            except json.JSONDecodeError as error:
                raise EvaluationError(f"Invalid citations_json on CSV line {line_number}: {error}") from error
            if not isinstance(citations, list):
                raise EvaluationError(f"citations_json on CSV line {line_number} must be a JSON list.")
            submissions.append(
                {
                    "case_id": row.get("case_id"),
                    "generated_answer": row.get("generated_answer", ""),
                    "status": row.get("status", ""),
                    "citations": citations,
                    "factual_support_score": row.get("factual_support_score", ""),
                    "citation_correctness_score": row.get("citation_correctness_score", ""),
                    "evidence_uncertainty_score": row.get("evidence_uncertainty_score", ""),
                    "scope_compliance_score": row.get("scope_compliance_score", ""),
                    "geographic_caution_score": row.get("geographic_caution_score", ""),
                    "reviewer_comments": row.get("reviewer_comments", ""),
                    "pass_fail": row.get("pass_fail", ""),
                }
            )
    return submissions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL_REPORT)
    parser.add_argument("--answers", type=Path, required=True, help="Completed review CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        retrieval_report = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
        submissions = _read_submissions(args.answers)
        report = evaluate_answer_submissions(dataset, retrieval_report, submissions)
    except (OSError, json.JSONDecodeError, EvaluationError) as error:
        print(f"Answer evaluation failed: {error}", file=sys.stderr)
        return 2

    write_json_report(report, args.output)
    print(f"Deterministically checked submissions: {report['evaluated_submissions']}")
    print("Semantic factual support still requires human review.")
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
