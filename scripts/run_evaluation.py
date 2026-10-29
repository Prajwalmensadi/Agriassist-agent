"""Run offline retrieval evaluation against the existing Chroma index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agriassist.evaluation import EvaluationError, run_retrieval_evaluation, write_json_report
from agriassist.knowledge import INDEX_PATH


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "evaluation_cases.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "retrieval_evaluation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff for Recall@k and MRR@k.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        report = run_retrieval_evaluation(
            dataset,
            k=args.k,
            index_path=args.index,
        )
    except (OSError, json.JSONDecodeError, EvaluationError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    write_json_report(report, args.output)
    summary = report["summary"]
    print(f"Cases: {summary['total_cases']}")
    print(f"Scored with verified gold evidence: {summary['scored_cases']}")
    print(f"Manual review / unscored: {summary['manual_review_cases']}")
    print(f"Retrieval failures: {summary['retrieval_failures']}")
    print(f"Mean Recall@{args.k}: {summary['mean_recall_at_k']}")
    print(f"Mean MRR@{args.k}: {summary['mean_mrr_at_k']}")
    print(f"JSON report: {args.output}")
    return 1 if summary["retrieval_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
