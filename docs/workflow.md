# Advisory Workflow

1. Create a case from the farmer's report.
2. Collect relevant crop and symptom context.
3. Identify missing information and ask targeted questions.
4. Search the curated knowledge base.
5. Assess evidence relevance and uncertainty.
6. Produce a structured advisory, or request more information / expert review.
7. Record the farmer's chosen action and completion status.
8. Create a follow-up task.
9. Record later observations and summarize reported change without asserting causation.

Possible states: `NEW`, `COLLECTING_CONTEXT`, `RETRIEVING_EVIDENCE`, `ADVISORY_READY`, `AWAITING_FARMER_ACTION`, `FOLLOW_UP_DUE`, `FOLLOW_UP_RECORDED`, `AWAITING_EXPERT_REVIEW`, `INSUFFICIENT_EVIDENCE`, `CLOSED`.

The application, not the LLM, owns state transitions and validates permitted actions.
