"""Document model for vector database."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class Document:
    """A document with its vector representation and metadata."""
    
    content: str
    """The text content of the document."""
    
    embedding: List[float]
    """The vector embedding of the document."""
    
    id: Optional[int] = None
    """Unique identifier for the document."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the document."""
    
    created_at: Optional[datetime] = None
    """Timestamp when the document was created."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary.
        
        Returns:
            A dictionary representation of the document.
        """
        return {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document from a dictionary.
        
        Args:
            data: Dictionary containing document data
            
        Returns:
            A new Document instance
        """
        return cls(
            id=data.get('id'),
            content=data['content'],
            embedding=data['embedding'],
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        ) 