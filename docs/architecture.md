# MVP Architecture

## Supported scope
The MVP supports paddy (rice) cases for farmers in coastal Karnataka, India. Other crops are outside the MVP scope; the system must not provide agricultural advice for them. The architecture can be extended to other crops in a future version.

```text
Farmer -> Streamlit UI -> Session-scoped Controller -> Scope / Output Policies
                                   |                 |
                                   |                 +--> Local Ollama generation
                                   +--> Existing Chroma retrieval -> Curated PDF passages

SQLite case memory, farmer-action recording, and follow-up scheduling remain planned.
```

## Design principles
1. Use a single bounded agent, not a multi-agent framework.
2. Deterministic Python rules own crop-scope clarification, retrieval control, and output checks.
3. Retrieval similarity is a ranking signal, not factual confidence. The current keyword-overlap check is a weak relevance heuristic, not a semantic judge.
4. The model receives retrieved passages as untrusted evidence and cannot invoke arbitrary tools.
5. The UI displays source filenames, PDF pages, and provisional geographic/verification metadata. Citations are checked against retrieved pages; this does not prove entailment.
6. Empty, irrelevant, high-risk, or failed retrieval does not produce a generated agricultural answer.
7. Session state is temporary and browser-session scoped. Persistent SQLite memory and follow-up workflows are not implemented.
