"""
FAISS vectorstore creation from document chunks.

Used as an alternative path to process_doc.processing when you already
have chunks (e.g. from document_loader + chunking).
"""
from langchain_community.vectorstores import FAISS

from app.components.models.embedding import load_embeddings
from app.utils.logger.logger import get_logger

logger = get_logger()


def create_vector_store(chunks) -> FAISS | None:
    """
    Build a FAISS vectorstore from a list of document chunks.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        FAISS vectorstore, or None if embeddings fail or chunks are empty.
    """
    if not chunks:
        logger.warning("create_vector_store called with no chunks")
        return None
    try:
        embeddings = load_embeddings()
        if embeddings is None:
            logger.error("Embeddings not available for vector store")
            return None
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logger.exception("Vector store creation failed: %s", e)
        return None