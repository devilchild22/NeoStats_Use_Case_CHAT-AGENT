"""
Centralised logging for the application.

Configures format and level once; get_logger() returns the module logger
for consistent naming in log output.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Return the application logger for use in other modules."""
    return logger