"""
Load documents from disk (PDF only in this module).

For PDF, TXT, DOCX, CSV use app.utils.process_doc.processing.load_document instead.
"""
import os
from langchain_community.document_loaders import PyPDFLoader

from app.utils.logger.logger import get_logger

logger = get_logger()


def load_financial_documents(file_path: str):
    """
    Load PDF documents from the given file path.

    Args:
        file_path: Path to a PDF file.

    Returns:
        List of LangChain Document objects, or [] on error.

    Raises:
        ValueError: If the file is not a PDF.
    """
    try:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension != ".pdf":
            raise ValueError("Only PDF financial reports supported")
        loader = PyPDFLoader(file_path)
        return loader.load()
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Document load error for %s: %s", file_path, e)
        return []