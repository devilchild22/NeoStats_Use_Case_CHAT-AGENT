"""
Custom exceptions for the Neostas application.

Centralising domain-specific errors improves error handling, logging,
and allows the UI to show user-friendly messages while preserving details in logs.
"""


class NeostasError(Exception):
    """Base exception for all Neostas application errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(NeostasError):
    """Raised when required configuration (e.g. API keys) is missing or invalid."""


class DocumentProcessingError(NeostasError):
    """Raised when document upload, loading, or parsing fails."""


class RAGError(NeostasError):
    """Raised when RAG retrieval or answer generation fails."""


class AgentError(NeostasError):
    """Raised when the agent invocation or response extraction fails."""


class EmbeddingError(NeostasError):
    """Raised when the embedding model fails to load or run."""
