#!/usr/bin/env python3
"""
Script to export chat history to a JSON file.
This is useful for analysis, debugging, and monitoring bot interactions.
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

# Import Zulip models
from zerver.models import Message, Realm, UserProfile, Recipient, Stream


def export_chat_history(
    bot_email: str = None,
    realm_domain: str = None, 
    max_messages: int = 1000,
    output_file: str = None
) -> None:
    """
    Export chat history involving a specific bot to a JSON file
    
    Args:
        bot_email: Email of the bot (e.g., 'welcome-bot@zulip.com')
        realm_domain: Domain of the realm to search in
        max_messages: Maximum number of messages to export
        output_file: Path to output file. If None, a timestamped file will be created.
    """
    # Get realm
    realm = None
    if realm_domain:
        try:
            realm = Realm.objects.get(domain=realm_domain)
            print(f"Found realm: {realm.name}")
        except Realm.DoesNotExist:
            # Try to get the first realm
            realm = Realm.objects.first()
            print(f"Realm with domain '{realm_domain}' not found, using {realm.name} instead")
    else:
        # Use the first realm
        realm = Realm.objects.first()
        print(f"Using realm: {realm.name}")
    
    if not realm:
        print("Error: No realm found")
        return
    
    # Get the bot user
    bot_user = None
    if bot_email:
        try:
            bot_user = UserProfile.objects.get(email=bot_email, realm=realm)
            print(f"Found bot user: {bot_user.full_name}")
        except UserProfile.DoesNotExist:
            print(f"Bot with email '{bot_email}' not found")
            # Try to find welcome-bot
            try:
                bot_user = UserProfile.objects.get(full_name="Welcome Bot", realm=realm)
                print(f"Using Welcome Bot instead: {bot_user.email}")
            except UserProfile.DoesNotExist:
                print("Welcome Bot not found either")
    else:
        # Try to find welcome-bot
        try:
            bot_user = UserProfile.objects.get(full_name="Welcome Bot", realm=realm)
            print(f"Using Welcome Bot: {bot_user.email}")
        except UserProfile.DoesNotExist:
            print("Welcome Bot not found")
    
    # Query messages
    messages_data = []
    
    if bot_user:
        # Get messages sent by the bot
        bot_sent_messages = Message.objects.filter(
            sender=bot_user
        ).order_by('-date_sent')[:max_messages//2]
        
        # Get messages received by the bot (sent to the bot)
        bot_received_messages = Message.objects.filter(
            recipient__type=Recipient.PERSONAL,
            recipient__type_id=bot_user.id
        ).order_by('-date_sent')[:max_messages//2]
        
        # Process sent messages
        for msg in bot_sent_messages:
            try:
                recipient = Recipient.objects.get(id=msg.recipient_id)
                recipient_email = None
                
                if recipient.type == Recipient.PERSONAL:
                    try:
                        user = UserProfile.objects.get(id=recipient.type_id)
                        recipient_email = user.email
                    except UserProfile.DoesNotExist:
                        recipient_email = "unknown"
                
                messages_data.append({
                    "id": msg.id,
                    "date": msg.date_sent.isoformat(),
                    "direction": "bot_to_user",
                    "sender": bot_user.email,
                    "sender_name": bot_user.full_name,
                    "recipient": recipient_email,
                    "content": msg.content,
                })
            except Exception as e:
                print(f"Error processing message {msg.id}: {e}")
        
        # Process received messages
        for msg in bot_received_messages:
            try:
                sender = UserProfile.objects.get(id=msg.sender_id)
                messages_data.append({
                    "id": msg.id,
                    "date": msg.date_sent.isoformat(),
                    "direction": "user_to_bot",
                    "sender": sender.email,
                    "sender_name": sender.full_name,
                    "recipient": bot_user.email,
                    "content": msg.content,
                })
            except Exception as e:
                print(f"Error processing message {msg.id}: {e}")
    else:
        print("No bot user found, exporting recent messages instead")
        recent_messages = Message.objects.all().order_by('-date_sent')[:max_messages]
        
        for msg in recent_messages:
            try:
                sender = UserProfile.objects.get(id=msg.sender_id)
                recipient = Recipient.objects.get(id=msg.recipient_id)
                recipient_name = "Unknown"
                
                if recipient.type == Recipient.PERSONAL:
                    try:
                        recipient_user = UserProfile.objects.get(id=recipient.type_id)
                        recipient_name = recipient_user.full_name
                    except UserProfile.DoesNotExist:
                        recipient_name = "Unknown user"
                elif recipient.type == Recipient.STREAM:
                    try:
                        stream = Stream.objects.get(id=recipient.type_id)
                        recipient_name = f"Stream: {stream.name}"
                    except Stream.DoesNotExist:
                        recipient_name = "Unknown stream"
                
                messages_data.append({
                    "id": msg.id,
                    "date": msg.date_sent.isoformat(),
                    "sender": sender.email,
                    "sender_name": sender.full_name,
                    "recipient_type": recipient.type,
                    "recipient_name": recipient_name,
                    "content": msg.content,
                })
            except Exception as e:
                print(f"Error processing message {msg.id}: {e}")
    
    # Sort by date
    messages_data.sort(key=lambda x: x["date"])
    
    # Prepare export data
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "bot_email": bot_user.email if bot_user else None,
        "realm": realm.name,
        "message_count": len(messages_data),
        "messages": messages_data
    }
    
    # Generate output filename if not provided
    if not output_file:
        bot_name = bot_user.full_name.lower().replace(" ", "_") if bot_user else "all_messages"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"chat_history_{bot_name}_{timestamp}.json"
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Chat history exported to {output_file}")
    print(f"Total messages: {len(messages_data)}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Export chat history to a JSON file')
    parser.add_argument('--bot', help='Bot email address')
    parser.add_argument('--realm', help='Realm domain')
    parser.add_argument('--max', type=int, default=1000, help='Maximum number of messages')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    export_chat_history(
        bot_email=args.bot,
        realm_domain=args.realm,
        max_messages=args.max,
        output_file=args.output
    )
