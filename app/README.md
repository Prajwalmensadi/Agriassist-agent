Run the local chat UI from the repository root with:

```sh
PYTHONPATH=src ./.venv/bin/python -m streamlit run app/streamlit_app.py
```

The UI keeps conversation and pending clarification state in the current Streamlit session only. It uses the existing Chroma index and local Ollama models; it does not build the index or persist cases to SQLite.
