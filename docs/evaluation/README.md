# AgriAssist evaluation framework (Phase A)

This is a small, versioned evaluation set for the paddy-only MVP in coastal Karnataka. It checks retrieval against the current local document collection and records human review of generated answers. It does not prove that an answer is generally correct or suitable as farm advice.

## Dataset and evidence labels

The dataset is `tests/evaluation_cases.json`, schema version 1. It contains 21 cases: 7 with manually checked source/page gold evidence and 14 workflow or safety cases marked `manual_review`. Only verified gold evidence contributes to retrieval scores. A manual-review case is still sent to the retriever and included in the report, but has no Recall or MRR score. Do not promote a label to `verified` without checking the cited PDF page.

Verified page labels currently cover: iron-toxicity conditions and signs in `11-14.pdf` (pages 2–3), a sheath-blight field pattern in `8.pdf` (page 2), bacterial-leaf-blight symptoms in `04+-+Srujani+148717_Revised.pdf` (page 2), Bakanae symptoms in `15+Emerging+minor+diseases+of+rice+in+Odisha.pdf` (page 2), and the Odisha-specific framing in the Srujani document (page 1). The sources are provisional and mostly describe Odisha or general rice material. Inclusion does not certify authority or local applicability to coastal Karnataka.

## Retrieval evaluation

Run from the repository root:

```sh
PYTHONPATH=src ./.venv/bin/python scripts/run_evaluation.py --k 5
```

The command reads the existing Chroma index and writes `data/processed/retrieval_evaluation.json`. It does not build or modify the index. Each case records retrieved filename, PDF page, distance, available source metadata, errors, and labels. `Recall@k` is the fraction of verified gold filename/page pairs present in the top k; `MRR@k` is the reciprocal rank of the first such pair (zero if none). Means include only verified cases. Retrieval distance and these metrics do not measure factual correctness, citation entailment, diagnosis quality, or safe field applicability. If Ollama/embedding retrieval is unavailable, failures are recorded per case and the command exits nonzero.

## Answer review

Copy `data/evaluation/human_review_template.csv` to a working file, enter the generated answer and cited pages, and have a reviewer fill in the rubric fields. Use JSON in `citations_json`, for example `[ {"source_filename": "8.pdf", "page_number": 2} ]`. The rubric uses 1–5 scores (5 is strongest) for factual support, citation correctness, treatment of uncertainty, scope compliance, and geographic caution; record `pass_fail` as `pass`, `fail`, or `needs_review`. Review factual support against the actual source page and ensure evidence about unsupported crops is not applied to paddy.

```sh
cp data/evaluation/human_review_template.csv data/evaluation/human_review.csv
# Fill the working CSV, then:
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_answers.py \
  --answers data/evaluation/human_review.csv
```

The deterministic answer checks only flag narrow conditions such as whether a required citation points to retrieved evidence, an unsupported-crop case states the supported scope, and required abstention/geographic-caution wording appears. They do not determine whether a sentence is supported by a passage. Every answer still requires human review. The CSV can include notes and human scores; the output is `data/processed/answer_evaluation.json`.

Run tests with:

```sh
./.venv/bin/python -m pytest -q
```

## Current limitations

- The set is small and hand-curated; it is not a representative benchmark.
- Gold labels are page-level, not passage spans, and only cover the explicitly verified cases.
- Source authority and geographic transfer are not established by retrieval.
- Deterministic wording checks can be evaded by wording and can flag acceptable paraphrases; use them as workflow checks only.
- No score substitutes for a qualified local agricultural expert, field observation, or human review of citations and claims.
