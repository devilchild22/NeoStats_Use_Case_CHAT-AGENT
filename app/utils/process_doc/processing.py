"""
Document processing: load supported file types and build a FAISS vectorstore.

Supports PDF, TXT, DOCX, and CSV. Uses temp files for uploads and ensures
cleanup on failure. Raises DocumentProcessingError or ValueError for
callers to handle.
"""
import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.components.models.embedding import load_embeddings
from app.exceptions import DocumentProcessingError
from app.utils.logger.logger import get_logger

logger = get_logger()

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx", ".csv")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_document(file_path: str, file_extension: str):
    """
    Load a document from disk using the appropriate LangChain loader.

    Args:
        file_path: Absolute path to the file.
        file_extension: Extension including dot (e.g. ".pdf").

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If the file type is not supported.
    """
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
    }
    loader_cls = loaders.get(file_extension.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    loader = loader_cls(file_path)
    return loader.load()


def process_document(file) -> FAISS | None:
    """
    Process an uploaded file into a FAISS vectorstore.

    Saves the file to a temp path, loads it, chunks it, embeds with the
    configured model, and returns a FAISS index. Temp file is always removed.

    Args:
        file: File-like object with .filename and .save(fp).

    Returns:
        FAISS vectorstore, or None if file is falsy.

    Raises:
        ValueError: Unsupported file type.
        DocumentProcessingError: Load, chunk, or embed failed.
    """
    if not file:
        return None

    filename = getattr(file, "filename", "upload.txt")
    ext = os.path.splitext(filename)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        documents = load_document(tmp_path, ext)
        if not documents:
            raise DocumentProcessingError("Document produced no content.", details=filename)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        embeddings = load_embeddings()
        if embeddings is None:
            raise DocumentProcessingError("Embedding model failed to load.", details=filename)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except (ValueError, DocumentProcessingError):
        raise
    except Exception as e:
        logger.exception("Document processing failed: %s", e)
        raise DocumentProcessingError(
            "Failed to process document.",
            details=str(e),
        ) from e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", tmp_path, e)     

