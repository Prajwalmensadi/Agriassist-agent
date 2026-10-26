# AgriAssist

An agentic agricultural advisory and follow-up system for paddy-focused scenarios in coastal Karnataka.

## Project status
Week 1 — scope, design, knowledge-source verification, and LLM evaluation planning.

## Objective
AgriAssist helps turn verified agricultural guidance into contextualized action plans. It asks for missing context, retrieves approved source material, records farmer-reported actions, and supports follow-up. It is a decision-support prototype, not a substitute for qualified agricultural extension advice.

## Safety boundaries
- Do not invent diagnoses, pesticide names, dosages, or application rates.
- Recommendations must be grounded in applicable, verified sources.
- If evidence is insufficient, say so and escalate to a qualified agricultural expert.
- Record farmer-reported outcomes without claiming causation.
- Do not commit secrets, private farmer data, restricted documents, local databases, or generated indexes.

## Planned stack
- Python
- Ollama local inference (model choice pending evaluation)
- RAG with embeddings and FAISS or Chroma
- SQLite for case memory
- Streamlit UI
- pytest / evaluation scripts

## Five-week roadmap
1. Scope, architecture, model comparison, 15+ test cases, repository.
2. Core bounded agent loop, structured output, first tool, error handling.
3. Knowledge retrieval and persistent memory.
4. Planning, follow-up, guardrails, evaluation, failure analysis.
5. Streamlit UI, documentation, reproducible setup, demo.

## Quick start (scaffold only)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

The current scaffold contains planning artifacts and tests; the full agent is not implemented yet.
