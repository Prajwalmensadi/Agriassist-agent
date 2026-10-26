# LLM Comparison Plan (Week 1)

## Candidates
- `qwen3.5:4b` — lightweight baseline.
- `qwen3.5:9b` — larger same-family challenger.
- `gemma3:4b` — independent model-family comparison.

These are candidates, not a declared winner. Verify current Ollama tags, licenses, hardware needs, and model capabilities before finalizing.

## Fair comparison protocol
- Use identical prompts, input cases, retrieved evidence, tools, and output schema where possible.
- Record exact model tag, runtime, generation settings, duration, and failures.
- Separate model-only tests (same evidence supplied) from end-to-end tool-selection tests.
- Run all 15+ cases and score using predefined expected outcomes.

## Proposed scoring weights
- Tool selection and workflow decisions: 20%
- Evidence grounding: 20%
- Safety and uncertainty: 20%
- Missing-context detection: 15%
- Structured-output validity: 10%
- Follow-up/state handling: 10%
- Kannada interaction: 5%

Safety is also a hard gate: a model that produces unsupported pesticide prescriptions must not win on aggregate score alone. Application-level guardrails remain mandatory.

## Evaluation metrics
Pass rate, retrieval relevance, citation support, JSON validity, missing-context detection, unsafe-advice rate, state consistency, latency, and local resource use.

## Decision rule
Select the model with the strongest measured performance on critical tasks while meeting local memory/latency constraints. No model is selected solely on reputation or general benchmark scores.
