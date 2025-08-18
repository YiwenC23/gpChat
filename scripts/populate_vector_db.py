#!/usr/bin/env python3
"""
Script to populate the vector database with existing chat history.
This ensures the RAG functionality has historical context.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "zproject.settings")
import django
django.setup()

# Import needed models and libraries
from zerver.models import Message, UserProfile, Realm, Recipient
from zerver.lib.ai_agents_openai import ZulipAIAgent, get_shared_vector_store
import logging

logger = logging.getLogger(__name__)

def populate_vector_db_from_history(
    bot_email: str = "welcome-bot@zulip.com",
    realm_domain: str = None,
    max_messages: int = 1000,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Populate the vector database with existing chat history
    
    Args:
        bot_email: Email of the bot whose messages should be loaded
        realm_domain: Domain of the realm to filter by
        max_messages: Maximum number of messages to process
        skip_existing: Skip messages that are already in the vector database
        
    Returns:
        Dict with statistics about the operation
    """
    stats = {
        "total_messages_processed": 0,
        "messages_added": 0,
        "errors": 0
    }
    
    try:
        # Get bot user
        bot = UserProfile.objects.get(email=bot_email)
        logger.info(f"Found bot: {bot.full_name} ({bot.email})")
        
        # Get realm
        if realm_domain:
            realm = Realm.objects.get(string_id=realm_domain)
        else:
            realm = bot.realm
        logger.info(f"Using realm: {realm.name} ({realm.string_id})")
        
        # Initialize AI Agent for embedding generation
        ai_agent = ZulipAIAgent(realm)
        vector_store = get_shared_vector_store()
        logger.info(f"Vector store initialized with ID: {id(vector_store)}")
        
        # Get messages TO the bot
        messages_to_bot = Message.objects.filter(
            recipient__type=Recipient.PERSONAL,
            recipient__type_id=bot.id
        ).order_by('date_sent')[:max_messages]
        
        logger.info(f"Found {messages_to_bot.count()} messages sent TO the bot")
        
        # Get messages FROM the bot
        messages_from_bot = Message.objects.filter(
            sender=bot
        ).order_by('date_sent')[:max_messages]
        
        logger.info(f"Found {messages_from_bot.count()} messages sent FROM the bot")
        
        # Process messages sent to the bot
        print(f"Processing {messages_to_bot.count()} messages TO the bot...")
        for message in messages_to_bot:
            stats["total_messages_processed"] += 1
            
            try:
                # Get sender
                sender = message.sender
                
                # Generate embedding
                embedding = ai_agent.generate_embeddings(message.content)
                if not embedding:
                    logger.warning(f"Failed to generate embedding for message ID {message.id}")
                    stats["errors"] += 1
                    continue
                
                # Create timestamp
                timestamp = str(time.mktime(message.date_sent.timetuple()))
                
                # Create conversation ID based on sender and date
                conversation_id = f"conv_{sender.email}_{int(float(timestamp))}"
                
                # Store in vector database
                metadata = {
                    "sender": sender.email,
                    "user_name": sender.full_name,
                    "timestamp": timestamp,
                    "is_bot": False,
                    "message_id": str(message.id),
                    "conversation_id": conversation_id
                }
                
                success = ai_agent.store_message_in_vector_db(
                    message_content=message.content,
                    metadata=metadata
                )
                
                if success:
                    stats["messages_added"] += 1
                    print(f"Added message {stats['messages_added']} to vector DB")
                else:
                    stats["errors"] += 1
                
            except Exception as e:
                logger.error(f"Error processing message ID {message.id}: {e}")
                stats["errors"] += 1
        
        # Process messages sent from the bot
        print(f"Processing {messages_from_bot.count()} messages FROM the bot...")
        for message in messages_from_bot:
            stats["total_messages_processed"] += 1
            
            try:
                # Get recipient
                recipient_id = message.recipient.type_id
                recipient_user = UserProfile.objects.filter(id=recipient_id).first()
                
                if not recipient_user:
                    logger.warning(f"Couldn't find recipient for message ID {message.id}")
                    stats["errors"] += 1
                    continue
                
                # Generate embedding
                embedding = ai_agent.generate_embeddings(message.content)
                if not embedding:
                    logger.warning(f"Failed to generate embedding for message ID {message.id}")
                    stats["errors"] += 1
                    continue
                
                # Create timestamp
                timestamp = str(time.mktime(message.date_sent.timetuple()))
                
                # Try to find existing conversation with this user
                user_convs = ai_agent.search_user_conversations(recipient_user.email, hours=24)
                
                if user_convs:
                    # Use the most recent conversation ID
                    conversation_id = user_convs[0].get('metadata', {}).get('conversation_id')
                else:
                    # Create conversation ID based on recipient and date
                    conversation_id = f"conv_{recipient_user.email}_{int(float(timestamp))}"
                
                # Store in vector database
                metadata = {
                    "sender": bot.email,
                    "user_name": bot.full_name,
                    "timestamp": timestamp,
                    "is_bot": True,
                    "bot_response": True,
                    "message_id": str(message.id),
                    "conversation_id": conversation_id
                }
                
                success = ai_agent.store_message_in_vector_db(
                    message_content=message.content,
                    metadata=metadata
                )
                
                if success:
                    stats["messages_added"] += 1
                    print(f"Added message {stats['messages_added']} to vector DB")
                else:
                    stats["errors"] += 1
                
            except Exception as e:
                logger.error(f"Error processing message ID {message.id}: {e}")
                stats["errors"] += 1
        
        # Export vector database contents to verify
        from scripts.export_vector_db import export_vector_db_to_json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"vector_db_populated_{timestamp}.json"
        export_vector_db_to_json(export_file)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error populating vector DB: {e}")
        stats["errors"] += 1
        return stats
        

if __name__ == "__main__":
    # Get parameters from command line if provided
    if len(sys.argv) > 1:
        bot_email = sys.argv[1]
    else:
        bot_email = "welcome-bot@zulip.com"
        
    # Run the population function
    print(f"Populating vector DB with messages for {bot_email}...")
    stats = populate_vector_db_from_history(bot_email=bot_email)
    
    # Print summary
    print("\nVector DB Population Complete:")
    print(f"Total messages processed: {stats['total_messages_processed']}")
    print(f"Messages added to vector DB: {stats['messages_added']}")
    print(f"Errors encountered: {stats['errors']}")
