"""
Split documents into chunks for embedding and retrieval.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger.logger import get_logger

logger = get_logger()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents) -> list:
    """
    Split a list of documents into overlapping chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunk documents, or [] on error.
    """
    if not documents:
        return []
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return splitter.split_documents(documents)
    except Exception as e:
        logger.exception("Chunking error: %s", e)
        return []