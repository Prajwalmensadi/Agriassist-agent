# Initial Architecture

```text
Farmer -> Streamlit UI -> Agent Controller
                         |      |
                         |      +--> Guardrails / Schema Validation
                         |
                         +--> Context & Case Tools
                         +--> RAG Retrieval -> Curated Agricultural Documents
                         +--> SQLite Case Memory
                         +--> Follow-up Scheduler / Tasks

Agent Controller -> Evidence-grounded advisory -> Save case/action -> Follow-up
```

## Design principles
1. Use a single bounded agent, not a multi-agent framework.
2. The LLM proposes permitted next actions; application code validates and executes them.
3. Retrieval and case memory are separate: documents provide domain knowledge; SQLite stores case history.
4. Every recommendation should be traceable to applicable evidence.
5. The agent must stop after a configured step limit and escalate when evidence is inadequate.
