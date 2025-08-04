"""
Vector Database Worker for Zulip Chat

This worker listens for new messages and stores them in a vector database
for semantic search and similarity matching.
"""

import json
import logging
from typing import Any, Dict, List
import time

from django.core.management.base import BaseCommand
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings

from zerver.models import Message
from zerver.models.realms import get_realm
from vector_db import VectorStore
from vector_db.models import Document
from vector_db.utils import get_embedding

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self):
        """Initialize the vector database connection."""
        self.vector_store = VectorStore()
        self.initialize_collection()
    
    def initialize_collection(self):
        """Initialize the messages collection in the vector database."""
        try:
            self.vector_store.create_table("zulip_messages")
            logger.info("Initialized vector database collection")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    def store_message(self, message: Message):
        """Store a message in the vector database."""
        try:
            realm = get_realm(message.realm_id)
            
            # Create document metadata
            metadata = {
                "sender": message.sender.email,
                "subject": message.subject,
                "realm": realm.string_id,
                "timestamp": str(message.date_sent),
                "message_id": str(message.id)
            }
            
            # Generate embedding for the message content
            embedding = get_embedding(message.content)
            
            # Create a document
            doc = Document(
                content=message.content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Store in the vector database
            doc_id = self.vector_store.insert(
                "zulip_messages",
                doc.content,
                doc.embedding,
                doc.metadata
            )
            
            logger.info(f"Stored message {message.id} in vector DB with ID {doc_id}")
            
        except Exception as e:
            logger.error(f"Error storing message {message.id}: {str(e)}")
            raise
    
    def search_messages(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar messages in the vector database."""
        try:
            # Generate embedding for the query
            query_embedding = get_embedding(query)
            
            # Search for similar messages
            results = self.vector_store.search(
                "zulip_messages",
                query_embedding,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}")
            raise

# Global instance of the vector DB manager
vector_db = VectorDBManager()

@receiver(post_save, sender=Message)
def on_message_save(sender, instance: Message, created: bool, **kwargs):
    """Signal handler that gets called whenever a message is saved."""
    if created:  # Only process new messages, not updates
        try:
            vector_db.store_message(instance)
        except Exception as e:
            logger.error(f"Error in on_message_save: {str(e)}")

class Command(BaseCommand):
    help = 'Run the vector database worker for storing chat messages'

    def handle(self, *args, **options):
        logger.info("Starting Zulip vector DB worker...")
        logger.info("Press Ctrl+C to exit")
        
        try:
            # Keep the process running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down vector DB worker...")
            vector_db.vector_store.close()
            logger.info("Vector DB connection closed")
