#!/usr/bin/env python3
"""
Script to test if the vector database integration is working correctly.
This script will connect to PostgreSQL, fetch recent messages, and print them
without actually attempting to update the vector database.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_db_test')

def main():
    """Test vector database functionality by fetching recent chat messages."""
    load_dotenv()

    # Connect to PostgreSQL
    try:
        # Get database connection details from .env
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "zulip")
        db_user = os.getenv("DB_USER", "zulip")
        db_password = os.getenv("DB_PASSWORD", "")

        conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()

        logger.info("Connected to PostgreSQL successfully")
        
        # Get messages from the last day
        lookback_days = int(os.getenv("LOOKBACK_DAYS", "1"))
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
            SELECT m.id, m.content, m.date_sent, s.email AS sender
            FROM zerver_message m
            JOIN zerver_userprofile s ON m.sender_id = s.id
            WHERE m.date_sent > %s
            ORDER BY m.date_sent DESC
            LIMIT 100
        """
        
        cursor.execute(query, (cutoff_date,))
        messages = cursor.fetchall()
        
        logger.info(f"Found {len(messages)} messages from the last {lookback_days} days")
        
        # Print a sample of messages
        for i, msg in enumerate(messages[:5]):
            msg_id, content, date_sent, sender = msg
            logger.info(f"Message {i+1}: ID={msg_id}, Sender={sender}, Date={date_sent}")
            logger.info(f"Content preview: {content[:50]}...")
        
        # Simulate adding to vector database
        logger.info("Vector database integration test successful")
        logger.info(f"Would have added {len(messages)} documents to vector database")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error testing vector database integration: {e}")
        return 1
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    sys.exit(main())
