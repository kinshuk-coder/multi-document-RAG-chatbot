# Atlas — multi-document RAG chatbot

A small, local-first Retrieval-Augmented Generation app. Upload several documents, ask a question, and inspect the chunks that supported the answer.

**Live demo:** [multi-document-rag-chatbot-9375c02e.fastapicloud.dev](https://multi-document-rag-chatbot-9375c02e.fastapicloud.dev/)

## Run with uv

```powershell
uv sync
# Add your Gemini API key to .env (created locally and ignored by Git)
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

`uv sync` creates and maintains the project's `.venv` automatically. You never need to activate it: prefix commands with `uv run`, for example `uv run python -m compileall app`.

To enable Gemini-powered answers, set `GEMINI_API_KEY` before starting the server. Without it, Atlas still retrieves relevant passages and produces a transparent, extractive response.

## Dependencies

Add a runtime package with `uv add package-name`, or add a development-only package with `uv add --dev package-name`. After changing dependencies, commit both `pyproject.toml` and the generated `uv.lock` so every machine gets the same resolved versions.

## Notes

- Supported uploads: PDF, DOCX, TXT, Markdown.
- Documents and the in-memory index are deliberately ephemeral; restarting the server clears them.
- Retrieval uses a compact TF-IDF implementation to keep this starter local and dependency-light. Swap `Retriever` in `app/main.py` for a vector database/embedding provider when scaling.
