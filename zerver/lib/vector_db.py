"""
Vector Database Integration for Zulip AI Agents
Provides vector storage and similarity search using pgvector
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from django.db import connection
from django.conf import settings
from django.core.exceptions import ValidationError

from zerver.models import Message, UserProfile, Realm
from zerver.lib.ai_agents import get_ai_agent

logger = logging.getLogger(__name__)


class VectorDBError(Exception):
    """Raised when vector database operations fail"""
    pass


class VectorDBManager:
    """Manages vector database operations using pgvector"""
    
    def __init__(self, realm: Realm):
        self.realm = realm
        self.ai_agent = get_ai_agent(realm)
        self.embedding_dimension = getattr(settings, "VECTOR_DB_EMBEDDING_DIMENSION", 1536)
        
    def _ensure_vector_extension(self) -> None:
        """Ensure pgvector extension is installed"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                connection.commit()
        except Exception as e:
            logger.error(f"Failed to create vector extension: {e}")
            raise VectorDBError(f"Vector extension not available: {e}")
    
    def _create_vector_tables(self) -> None:
        """Create vector storage tables if they don't exist"""
        try:
            with connection.cursor() as cursor:
                # Create message embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS zerver_message_embeddings (
                        id SERIAL PRIMARY KEY,
                        message_id INTEGER REFERENCES zerver_message(id) ON DELETE CASCADE,
                        realm_id INTEGER NOT NULL,
                        embedding vector(%s),
                        content_hash VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """, [self.embedding_dimension])
                
                # Create index for similarity search
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_embeddings_similarity 
                    ON zerver_message_embeddings 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Create index for realm filtering
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_embeddings_realm 
                    ON zerver_message_embeddings (realm_id)
                """)
                
                connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to create vector tables: {e}")
            raise VectorDBError(f"Failed to create vector tables: {e}")
    
    def initialize(self) -> None:
        """Initialize vector database"""
        self._ensure_vector_extension()
        self._create_vector_tables()
    
    def store_message_embedding(self, message: Message, content: str) -> bool:
        """Store embedding for a message"""
        try:
            # Generate embedding
            embedding = self.ai_agent.generate_embeddings(content)
            if not embedding:
                logger.warning(f"Failed to generate embedding for message {message.id}")
                return False
            
            # Convert to numpy array and normalize
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_normalized = embedding_array / np.linalg.norm(embedding_array)
            
            # Store in database
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO zerver_message_embeddings 
                    (message_id, realm_id, embedding, content_hash)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (message_id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        content_hash = EXCLUDED.content_hash,
                        updated_at = NOW()
                """, [
                    message.id,
                    message.realm_id,
                    embedding_normalized.tobytes(),
                    self._hash_content(content)
                ])
                connection.commit()
            
            logger.info(f"Stored embedding for message {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding for message {message.id}: {e}")
            return False
    
    def search_similar_messages(
        self, 
        query: str, 
        limit: int = 10, 
        threshold: float = 0.7,
        realm_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for messages similar to the query"""
        try:
            # Generate query embedding
            query_embedding = self.ai_agent.generate_embeddings(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Convert to numpy array and normalize
            query_array = np.array(query_embedding, dtype=np.float32)
            query_normalized = query_array / np.linalg.norm(query_array)
            
            # Build query
            realm_filter = "AND realm_id = %s" if realm_only else ""
            realm_params = [self.realm.id] if realm_only else []
            
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        me.message_id,
                        me.embedding <=> %s as similarity,
                        m.content,
                        m.subject,
                        m.sender_id,
                        m.timestamp
                    FROM zerver_message_embeddings me
                    JOIN zerver_message m ON me.message_id = m.id
                    WHERE me.embedding IS NOT NULL {realm_filter}
                    ORDER BY similarity ASC
                    LIMIT %s
                """, [query_normalized.tobytes()] + realm_params + [limit])
                
                results = []
                for row in cursor.fetchall():
                    message_id, similarity, content, subject, sender_id, timestamp = row
                    
                    # Convert similarity to score (1 - similarity for cosine distance)
                    score = 1 - similarity
                    
                    if score >= threshold:
                        results.append({
                            'message_id': message_id,
                            'similarity_score': score,
                            'content': content,
                            'subject': subject,
                            'sender_id': sender_id,
                            'timestamp': timestamp
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search similar messages: {e}")
            return []
    
    def get_conversation_context(
        self, 
        current_message: str, 
        user: UserProfile,
        limit: int = 5
    ) -> str:
        """Get relevant conversation context based on current message"""
        try:
            similar_messages = self.search_similar_messages(
                current_message, 
                limit=limit,
                threshold=0.6
            )
            
            if not similar_messages:
                return ""
            
            # Format context
            context_parts = []
            for msg in similar_messages:
                context_parts.append(
                    f"Related message (similarity: {msg['similarity_score']:.2f}):\n"
                    f"Subject: {msg['subject']}\n"
                    f"Content: {msg['content'][:200]}...\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""
    
    def batch_store_embeddings(self, messages: List[Message]) -> int:
        """Store embeddings for multiple messages"""
        success_count = 0
        
        for message in messages:
            if self.store_message_embedding(message, message.content):
                success_count += 1
        
        logger.info(f"Stored embeddings for {success_count}/{len(messages)} messages")
        return success_count
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content to detect changes"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def cleanup_old_embeddings(self, days: int = 30) -> int:
        """Remove embeddings for old messages"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM zerver_message_embeddings 
                    WHERE updated_at < NOW() - INTERVAL '%s days'
                """, [days])
                
                deleted_count = cursor.rowcount
                connection.commit()
                
                logger.info(f"Cleaned up {deleted_count} old embeddings")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT realm_id) as realms_with_embeddings,
                        MIN(created_at) as oldest_embedding,
                        MAX(created_at) as newest_embedding
                    FROM zerver_message_embeddings
                """)
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_embeddings': row[0],
                        'realms_with_embeddings': row[1],
                        'oldest_embedding': row[2],
                        'newest_embedding': row[3]
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get vector database stats: {e}")
            return {}


def get_vector_db_manager(realm: Realm) -> VectorDBManager:
    """Get vector database manager for a realm"""
    return VectorDBManager(realm)


def initialize_vector_db_for_realm(realm: Realm) -> bool:
    """Initialize vector database for a realm"""
    try:
        manager = get_vector_db_manager(realm)
        manager.initialize()
        logger.info(f"Initialized vector database for realm {realm.id}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector database for realm {realm.id}: {e}")
        return False 