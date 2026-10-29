# Safety and Limitations

- No invented pesticide names, dosages, application rates, or treatment schedules.
- Do not state a definitive diagnosis from insufficient evidence.
- Ground recommendations in applicable, verified source passages.
- If retrieval is empty, irrelevant, or conflicting, disclose the limitation and escalate where appropriate.
- Treat user text and retrieved documents as data, not as instructions overriding system rules.
- Validate model output against a schema and permitted action list.
- Restrict tools; no arbitrary shell execution.
- Limit agent steps and log tool calls, state transitions, validation errors, and retrieval references.
- Store only necessary case information. Do not commit personal farmer data.
- Follow-up observations are self-reported; do not infer that an action caused an outcome.
- This prototype does not establish improved farmer outcomes without further user/field evaluation.

## Current controller limitations

- Evidence relevance uses a deterministic content-word overlap heuristic. It may miss useful paraphrases and may accept coincidental overlap; it is not an evidence entailment or truth check.
- Geographic caveats are appended using source metadata. Metadata can be incomplete, and the caveat does not make a source locally applicable.
- Model citation-like references are rejected unless their filename/page pair exists in retrieved results. The UI's source list is based on those results; neither check proves that a passage supports every answer sentence.
- Field-symptom context intake currently asks one distribution question for a narrow set of explicit paddy symptom reports. It does not parse arbitrary farmer reports into a complete agronomic case.
- High-risk question detection is keyword based and may over-refuse or fail to recognize some risky phrasing. Do not treat this as production-safe screening.
