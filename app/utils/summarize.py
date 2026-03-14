"""
Document summarization: load a file and return an LLM-generated summary.

Does not build or modify the vectorstore. Supports PDF and TXT.
"""
import os
import tempfile

from app.utils.process_doc.processing import load_document
from app.components.models.llm import load_groq_client
from app.utils.logger.logger import get_logger

logger = get_logger()

SUMMARIZE_EXTENSIONS = (".pdf", ".txt")


def summarize_document(file) -> str:
    """
    Load a document and return a summary using the configured LLM.

    Does not modify the vectorstore. Accepts the same file-like interface
    as process_document: .filename and .save(fp).

    Args:
        file: File-like object with .filename and .save(fp).

    Returns:
        Summary text string.

    Raises:
        ValueError: Unsupported file type (only .pdf and .txt).
        Exception: On load or LLM failure.
    """
    if not file:
        raise ValueError("No file provided")

    filename = getattr(file, "filename", "upload.txt")
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in SUMMARIZE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for summarization: {ext}. Supported: {', '.join(SUMMARIZE_EXTENSIONS)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        documents = load_document(tmp_path, ext)
        if not documents:
            return "The document produced no readable content."

        full_text = "\n\n".join(doc.page_content for doc in documents)
        if not full_text.strip():
            return "The document is empty."

        # Truncate if very long to stay within context limits
        max_chars = 120_000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[... document truncated for summarization ...]"

        client = load_groq_client()
        prompt = (
            "You are a concise summarizer. Provide a clear, structured summary. "
            "Use short paragraphs and bullet points where helpful.\n\n"
            f"Summarize the following document.\n\n---\n\n{full_text}"
        )
        response = client.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
        return summary.strip() if summary else "Could not generate summary."
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", tmp_path, e)
