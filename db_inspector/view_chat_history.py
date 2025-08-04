#!/usr/bin/env python3
import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB', 'zulip'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_chat_history(limit=50):
    """Fetch and display recent chat messages."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            # Query to get recent messages with sender and stream information
            cur.execute("""
                SELECT 
                    m.id,
                    m.sender_id,
                    u.email as sender_email,
                    m.recipient_id,
                    m.subject,
                    m.content,
                    m.date_sent
                FROM 
                    zerver_message m
                JOIN 
                    zerver_userprofile u ON m.sender_id = u.id
                ORDER BY 
                    m.date_sent DESC
                LIMIT %s
            """, (limit,))
            
            messages = cur.fetchall()
            
            if not messages:
                print("No messages found in the database.")
                return
            
            print("\n=== Recent Chat Messages ===\n")
            for msg in messages:
                msg_id, sender_id, sender_email, recipient_id, subject, content, date_sent = msg
                print(f"From: {sender_email}")
                print(f"Date: {date_sent}")
                print(f"Subject: {subject}")
                print("---")
                print(content[:500] + ("..." if len(content) > 500 else ""))
                print("\n" + "="*80 + "\n")
                
    except Exception as e:
        print(f"Error fetching messages: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("Fetching chat history...\n")
    get_chat_history()