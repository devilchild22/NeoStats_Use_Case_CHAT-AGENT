"""
LLM client for the agent (Groq Chat).
"""
from langchain_groq import ChatGroq
import os
from app.config.config import GROQ_API_KEY
from app.utils.logger.logger import get_logger
from app.exceptions import ConfigurationError

logger = get_logger()


MODEL_NAME = os.environ.get("GROQ_MODEL_NAME")
TEMPERATURE = 0.1


def load_groq_client() -> ChatGroq:
    """
    Create and return a Groq chat client.

    Returns:
        ChatGroq instance.

    Raises:
        ConfigurationError: If GROQ_API_KEY is missing or invalid.
    """
    if not GROQ_API_KEY or not str(GROQ_API_KEY).strip():
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )
    try:
        client = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        return client
    except Exception as e:
        logger.error("Error initializing Groq client: %s", e)
        raise ConfigurationError(
            "Failed to initialize LLM client.",
            details=str(e),
        ) from e


