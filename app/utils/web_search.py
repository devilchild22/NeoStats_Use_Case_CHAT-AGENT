"""
Web search via DuckDuckGo for fallback or general queries.

"""
from ddgs import DDGS

from app.utils.logger.logger import get_logger

logger = get_logger()
MAX_RESULTS = 5


def web_search(query: str) -> str:
    """
    Search the web for the given query and return concatenated snippets.

    Args:
        query: Search query string.

    Returns:
        Newline-joined result bodies, or empty string on failure.
    """
    if not query or not str(query).strip():
        return ""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=MAX_RESULTS):
                body = r.get("body") or r.get("title", "")
                if body:
                    results.append(body)

                    logger.info(f"Web search results are : {body}")
        return "\n".join(results) if results else ""
    except Exception as e:
        logger.warning("Web search failed for %r: %s", query[:50], e)
        return ""