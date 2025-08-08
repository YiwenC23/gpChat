"""
Vector Database integration for the application.

This package provides functionality for working with vector embeddings and similarity search.
"""

__version__ = "0.1.0"

# Set default app config for Django integration
default_app_config = 'vector_db.apps.VectorDbConfig'

# Import core functionality
from .core.vector_store import VectorStore  # noqa 