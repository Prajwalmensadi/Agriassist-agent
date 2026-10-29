# AgriAssist

An agentic agricultural advisory and follow-up system for paddy (rice) farmers in coastal Karnataka, India.

The current MVP supports paddy (rice) only. Other crops are outside its scope and must not receive agricultural advice from AgriAssist. The architecture may be extended to other crops in a future version.

## Project status
Local MVP prototype: paddy context validation, PDF inventory, Chroma retrieval, Streamlit chat, and a bounded session-scoped controller are present. SQLite case memory and follow-up scheduling are not implemented. Sources remain provisional and are not established as locally approved guidance.

## Objective
AgriAssist helps turn verified agricultural guidance into contextualized action plans. It asks for missing context, retrieves approved source material, records farmer-reported actions, and supports follow-up. It is a decision-support prototype, not a substitute for qualified agricultural extension advice.

## Safety boundaries
- Do not invent diagnoses, pesticide names, dosages, or application rates.
- Recommendations must be grounded in applicable, verified sources.
- If evidence is insufficient, say so and escalate to a qualified agricultural expert.
- Record farmer-reported outcomes without claiming causation.
- Do not commit secrets, private farmer data, restricted documents, local databases, or generated indexes.

## Current and planned stack
- Python
- Ollama local inference (`qwen2.5:3b`) and `nomic-embed-text`
- Chroma retrieval over an explicit local PDF allowlist
- Streamlit chat UI with in-session conversation state
- SQLite case memory and follow-up workflow (planned)
- pytest / evaluation scripts

## Five-week roadmap
1. Scope, architecture, model comparison, 15+ test cases, repository.
2. Core bounded agent loop, structured output, first tool, error handling.
3. Knowledge retrieval and persistent memory.
4. Planning, follow-up, guardrails, evaluation, failure analysis.
5. Streamlit UI, documentation, reproducible setup, demo.

## Local launch
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Ensure Ollama is running and the configured local models are available. The app reads the existing Chroma index; it does not build or rebuild it automatically. See `docs/evaluation/README.md` for evaluation commands and limitations.
