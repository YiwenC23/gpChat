"""
Module for automatically updating vector database with data from PostgreSQL.
Can be used as a standalone script or imported as a module in other applications.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv

from vector_db.models.document import Document
from vector_db.utils.embeddings import get_embedding
from vector_db.core.vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', '..', 'var', 'log', 'vector_db_updates.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('vector_db.updater')


class VectorDBUpdater:
    """Tool for updating vector database with data from PostgreSQL."""
    
    def __init__(
        self,
        batch_size=None, 
        lookback_days=None,
        embedding_model=None
    ):
        """Initialize the updater with configuration parameters.
        
        Args:
            batch_size: Number of messages to process in each run
            lookback_days: How many days back to look for new messages
            embedding_model: Name of embedding model to use
        """
        # Load configuration
        self.batch_size = batch_size or int(os.getenv('BATCH_SIZE', 100))
        self.lookback_days = lookback_days or int(os.getenv('LOOKBACK_DAYS', 1))
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        
        # Initialize vector store
        self.vector_store = VectorStore()
    
    def get_db_connection(self):
        """Create and return a database connection to PostgreSQL."""
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            return conn
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            return None

    def get_new_chat_messages(self, conn, lookback_days=None):
        """Fetch new chat messages from PostgreSQL.
        
        Args:
            conn: PostgreSQL connection
            lookback_days: Number of days to look back for messages
            
        Returns:
            List of message tuples (id, sender_email, subject, content, date_sent)
        """
        lookback = lookback_days or self.lookback_days
        
        try:
            with conn.cursor() as cur:
                # Get the timestamp for lookback_days ago
                cutoff_date = datetime.now() - timedelta(days=lookback)
                
                # Query to get recent messages with sender information
                cur.execute("""
                    SELECT 
                        m.id,
                        u.email as sender_email,
                        m.subject,
                        m.content,
                        m.date_sent
                    FROM 
                        zerver_message m
                    JOIN 
                        zerver_userprofile u ON m.sender_id = u.id
                    WHERE 
                        m.date_sent >= %s
                    ORDER BY 
                        m.date_sent DESC
                    LIMIT %s
                """, (cutoff_date, self.batch_size))
                
                messages = cur.fetchall()
                logger.info(f"Retrieved {len(messages)} new messages from PostgreSQL")
                return messages
                
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []

    def create_document_from_message(self, message):
        """Create a Document object from a chat message.
        
        Args:
            message: Tuple containing message data
            
        Returns:
            Document object
        """
        msg_id, sender_email, subject, content, date_sent = message
        
        # Format the content for better context
        formatted_content = f"From: {sender_email}\nSubject: {subject}\n\n{content}"
        
        # Generate embedding for the content
        try:
            embedding = get_embedding(formatted_content, model=self.embedding_model)
        except Exception as e:
            logger.error(f"Error generating embedding for message {msg_id}: {e}")
            return None
        
        # Create document with metadata
        document = Document(
            content=formatted_content,
            embedding=embedding,
            metadata={
                "message_id": msg_id,
                "sender": sender_email,
                "subject": subject,
                "date_sent": date_sent.isoformat() if date_sent else None,
                "source": "chat_history"
            },
            created_at=datetime.now()
        )
        
        return document

    def update_vector_db(self):
        """Main function to fetch new messages and update the vector database.
        
        Returns:
            Number of documents added to vector database or None if an error occurred
        """
        # Connect to PostgreSQL
        pg_conn = self.get_db_connection()
        if not pg_conn:
            return None
        
        try:
            # Get new messages
            messages = self.get_new_chat_messages(pg_conn)
            if not messages:
                logger.info("No new messages found to process")
                return 0
            
            # Create documents from messages
            documents = []
            for message in messages:
                doc = self.create_document_from_message(message)
                if doc:
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} document objects with embeddings")
            
            # Add documents to vector database
            try:
                result = self.vector_store.add_documents(documents)
                logger.info(f"Successfully added {result} documents to vector database")
                return result
            except Exception as e:
                logger.error(f"Error adding documents to vector database: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error during vector database update: {e}")
            return None
        finally:
            pg_conn.close()


def run_update():
    """Run an update as a standalone script."""
    logger.info("Starting vector database update...")
    updater = VectorDBUpdater()
    result = updater.update_vector_db()
    if result is not None:
        logger.info(f"Vector database update completed. Added {result} documents.")
    else:
        logger.error("Vector database update failed.")
    return result


if __name__ == "__main__":
    run_update() 