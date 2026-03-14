"""
Document summarization: load a file and return an LLM-generated summary.

Processes page-by-page (or chunk-by-chunk). Each page is passed to the LLM
together with the current running summary; the summary is updated incrementally
and kept under MAX_SUMMARY_TOKENS so it does not grow with document length.

Does not build or modify the vectorstore. Supports PDF and TXT.
"""
import os
import tempfile

from app.utils.process_doc.processing import load_document
from app.components.models.llm import load_groq_client
from app.utils.logger.logger import get_logger

logger = get_logger()

SUMMARIZE_EXTENSIONS = (".pdf", ".txt")
MAX_SUMMARY_TOKENS = 800
# Rough chars per token for English; used to cap summary length
CHARS_PER_TOKEN = 4
MAX_SUMMARY_CHARS = MAX_SUMMARY_TOKENS * CHARS_PER_TOKEN  # ~3200


def _pages_from_documents(documents, ext: str):
    """
    Return a list of page/chunk strings. PDF → one item per page; TXT → split into chunks.
    """
    if not documents:
        return []
    if ext == ".pdf":
        return [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
    # .txt: single doc or few; split into ~page-sized chunks so we don't send huge blocks
    full = "\n\n".join(doc.page_content for doc in documents)
    if not full.strip():
        return []
    size = 4000  # chars per "page" for TXT
    pages = []
    for i in range(0, len(full), size):
        chunk = full[i : i + size].strip()
        if chunk:
            pages.append(chunk)
    return pages if pages else [full.strip()]


def _invoke_summary(client, system: str, user: str) -> str:
    """Call the LLM and return the summary text."""
    prompt = f"{system}\n\n{user}"
    response = client.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return (text or "").strip()


def summarize_document(file) -> str:
    """
    Load a document and return a summary using the configured LLM.

    Processes incrementally: first page → concise summary; then (summary + page 2) → updated
    summary; then (summary + page 3), etc. The running summary is always kept under
    MAX_SUMMARY_TOKENS (~800 tokens) so it does not grow with the number of pages.

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
        pages = _pages_from_documents(documents, ext)
        if not pages:
            return "The document produced no readable content."

        client = load_groq_client()
        summary = ""

        for i, page_text in enumerate(pages):
            if not page_text.strip():
                continue
            is_first = i == 0
            if is_first:
                system = (
                    "You are a concise summarizer. Summarize the following section of a document. "
                    f"Keep your summary under {MAX_SUMMARY_TOKENS} tokens (about {MAX_SUMMARY_CHARS} characters). "
                    "Use short paragraphs and bullet points where helpful. Output only the summary."
                )
                user = f"Section 1:\n\n{page_text}"
            else:
                system = (
                    "You are a concise summarizer. You have an existing summary and a new section. "
                    "Update the summary to incorporate the new section. "
                    f"Your output must be a single updated summary only, strictly under {MAX_SUMMARY_TOKENS} tokens "
                    f"(about {MAX_SUMMARY_CHARS} characters). Compress and merge; do not just append. "
                    "Output only the updated summary, nothing else."
                )
                user = (
                    f"Current summary (keep under {MAX_SUMMARY_TOKENS} tokens):\n\n{summary}\n\n"
                    f"New section to incorporate:\n\n{page_text}"
                )
            summary = _invoke_summary(client, system, user)
            if not summary:
                summary = "Could not generate summary."
            # Hard cap so we never exceed ~800 tokens even if the model ignores instructions
            if len(summary) > MAX_SUMMARY_CHARS:
                summary = summary[:MAX_SUMMARY_CHARS].rstrip() + "…"

        return summary
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", tmp_path, e)
