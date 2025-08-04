"""Utility functions for working with embeddings."""

from typing import List, Optional
import numpy as np


def get_embedding(
    text: str,
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None,
    **kwargs
) -> List[float]:
    """Get an embedding for the given text using the specified model.
    
    Args:
        text: The text to generate an embedding for
        model: The name of the embedding model to use
        api_key: Optional API key for the embedding service
        **kwargs: Additional arguments to pass to the embedding function
        
    Returns:
        A list of floats representing the text embedding
        
    Note:
        This is a placeholder implementation. In a real application, you would
        typically call an external embedding service like OpenAI's API.
    """
    # This is a mock implementation that returns a random embedding
    # In a real application, you would call an actual embedding API
    embedding_dim = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }.get(model, 1536)
    
    # Generate a random embedding for demonstration purposes
    # In production, replace this with an actual API call
    rng = np.random.RandomState(hash(text) % (2**32 - 1))
    return rng.randn(embedding_dim).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate the cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        The cosine similarity between the vectors, in the range [-1, 1]
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)
