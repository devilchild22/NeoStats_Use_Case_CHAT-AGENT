"""
Embedding model for document and query vectors (HuggingFace sentence-transformers).
"""
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.logger.logger import get_logger
from app.exceptions import EmbeddingError

logger = get_logger()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embeddings() -> HuggingFaceEmbeddings | None:
    """
    Load the HuggingFace embedding model used for vectorising text.

    Returns:
        HuggingFaceEmbeddings instance, or None on failure (caller should handle).
    """
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.exception("Embedding model failed to load: %s", e)
        return None