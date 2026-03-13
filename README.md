# Neostas

Streamlit chat app with optional document upload, RAG, and web search. Supports concise/detailed response modes and per-user conversation state.

## Run

```bash
streamlit run main.py
```

## Setup

1. **Environment**  
   Create a `.env` in the project root (or set in the shell):

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

2. **Dependencies**  
   `pip install -r requirements.txt`

## Codebase

- **Error handling**: Custom exceptions in `app/exceptions.py` (`DocumentProcessingError`, `RAGError`, `AgentError`, `ConfigurationError`, `EmbeddingError`). Use them for clearer control flow and user-facing messages.
- **Logging**: `app.utils.logger` — use `get_logger()` in modules and log with `logger.info/exception/warning` instead of `print`.
- **Documentation**: Modules and public functions have docstrings (Args, Returns, Raises where useful).
- **Quality**: Type hints on public APIs, constants for magic strings/numbers, and safe cleanup (e.g. temp files in document processing).
