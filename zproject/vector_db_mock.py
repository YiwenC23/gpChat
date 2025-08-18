"""
Enhanced mock implementation of the vector_db module for development.
Provides basic in-memory vector storage and search capabilities.
"""

import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced mock implementation of VectorStore with in-memory storage."""
    
    def __init__(self, embedding_dimension: int = 1536):
        # Initialize in-memory storage for collections
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self.embedding_dimension = embedding_dimension
        logger.info(f"Initialized VectorStore with embedding dimension {embedding_dimension}")
        
    def insert(self, collection: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert a document into the vector store"""
        if collection not in self.collections:
            self.collections[collection] = []
            
        # Ensure embedding is of correct dimension
        if len(embedding) != self.embedding_dimension:
            logger.warning(f"Expected embedding dimension {self.embedding_dimension}, got {len(embedding)}")
            # Pad or truncate embedding to match expected dimension
            if len(embedding) < self.embedding_dimension:
                embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
            else:
                embedding = embedding[:self.embedding_dimension]
        
        # Add timestamp if not already present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = str(time.time())
            
        # Add document_id for easier reference
        import uuid
        metadata['document_id'] = metadata.get('document_id', str(uuid.uuid4()))
        
        document = {
            'content': content,
            'embedding': embedding,
            'metadata': metadata
        }
        
        self.collections[collection].append(document)
        return True
        
    def add_document(self, collection: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Alias for insert"""
        return self.insert(collection, content, embedding, metadata)
        
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not a or not b:
            return 0.0
            
        try:
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
                
            return dot_product / (magnitude_a * magnitude_b)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def search_by_vector(
        self, 
        collection: str, 
        query_embedding: List[float], 
        limit: int = 5,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        if collection not in self.collections:
            return []
            
        results = []
        
        for doc in self.collections[collection]:
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            
            if similarity >= threshold:
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity_score': similarity
                })
                
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit results
        return results[:limit]
        
    def search_by_text(
        self,
        collection: str,
        query_text: str,
        limit: int = 5,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search using text - in a real implementation, this would generate an embedding first"""
        # In a mock version, we'll just do simple substring matching
        if collection not in self.collections:
            return []
            
        results = []
        query_lower = query_text.lower()
        
        for doc in self.collections[collection]:
            content_lower = doc['content'].lower()
            
            # Simple fuzzy matching based on substring
            if query_lower in content_lower:
                # Calculate a mock similarity score based on the relative position
                position = content_lower.find(query_lower)
                length_ratio = len(query_lower) / max(1, len(content_lower))
                position_factor = 1.0 - (position / max(1, len(content_lower)))
                similarity = 0.5 + (0.5 * position_factor * length_ratio)
                
                if similarity >= threshold:
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'similarity_score': similarity
                    })
                
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit results
        return results[:limit]
        
    def search(self, *args, **kwargs):
        """Legacy search method"""
        return self.search_by_vector(*args, **kwargs)
        
    def delete(self, collection: str, filter_func=None):
        """Delete documents from collection"""
        if collection not in self.collections:
            return False
            
        if filter_func is None:
            # Delete entire collection
            del self.collections[collection]
            return True
            
        # Filter documents to keep
        self.collections[collection] = [
            doc for doc in self.collections[collection]
            if not filter_func(doc)
        ]
        
        return True
