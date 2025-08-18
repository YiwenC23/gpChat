import re
import logging
from typing import Tuple, Optional, Any, Dict, Union, List

from django.utils.translation import gettext as _
from django.db.models import Q

from zerver.lib.string_validation import check_stream_name
from zerver.lib.streams import create_stream_if_needed
from zerver.actions.user_groups import create_user_group_in_database
from zerver.lib.streams import get_stream
from zerver.actions.streams import bulk_add_subscriptions
from zerver.models import Realm, UserProfile, UserGroup, Stream, Message, Recipient


def create_chat_group(realm: Realm, user: UserProfile, group_name: str) -> Tuple[bool, str, Optional[UserGroup]]:
    """
    Creates a user group (direct message group) with the specified name
    Returns a tuple of (success, message, user_group_object)
    """
    try:
        # Basic validation
        if not group_name or len(group_name) < 2:
            return False, _('Group name must be at least 2 characters.'), None
        
        if len(group_name) > 60:
            return False, _('Group name cannot exceed 60 characters.'), None
            
        # Create the user group
        members = [user]
        user_group = create_user_group_in_database(
            realm,
            group_name,
            members,
            description=f'Group created by welcome bot for {user.full_name}',
        )
        
        return True, _('Group "{}" created successfully!').format(group_name), user_group
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f'Error creating chat group: {str(e)}')
        return False, _('Failed to create group. Please try again.'), None


def create_channel(realm: Realm, user: UserProfile, channel_name: str, topic_keywords: Optional[str] = None) -> Tuple[bool, str, Optional[Any]]:
    """
    Creates a new channel (stream) with the specified name and optionally adds relevant users
    Returns a tuple of (success, message, stream_object)
    """
    try:
        # Basic validation
        error = check_stream_name(channel_name)
        if error:
            return False, error, None
        
        # Check if channel already exists
        try:
            existing_stream = get_stream(channel_name, realm)
            return False, _('Channel "{}" already exists.').format(channel_name), existing_stream
        except Stream.DoesNotExist:
            # Stream doesn't exist, we can create it
            pass
        
        # Default settings - public channel with history visible to subscribers
        history_public_to_subscribers = True
        invite_only = False
        
        # Create the channel
        stream, created = create_stream_if_needed(
            realm, 
            channel_name,
            invite_only=invite_only,
            history_public_to_subscribers=history_public_to_subscribers,
            acting_user=user
        )
        
        # Add relevant users to the channel if topic keywords are provided
        added_users = []
        if topic_keywords and stream and created:
            added_users = add_relevant_users_to_channel(realm, stream, topic_keywords, user)
        
        success_message = _('Channel "{}" created successfully!').format(channel_name)
        if added_users:
            success_message += f" Added {len(added_users)} relevant users: {', '.join([u.full_name for u in added_users[:3]])}{'...' if len(added_users) > 3 else ''}."
        
        return True, success_message, stream
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f'Error creating channel: {str(e)}')
        return False, _('Failed to create channel. Please try again.'), None


def detect_chat_creation_request(message: str) -> Tuple[bool, Optional[str]]:
    """
    Use AI to detect if a message is requesting to create a chat or channel
    Returns a tuple of (is_creation_request, group_name or None)
    """
    try:
        from zerver.lib.ai_agents_openai import ZulipAIAgent
        from zerver.models import Realm
        
        # Get a realm for AI agent - we need to pass it from the caller
        # For now, use a fallback but this should be improved
        realm = Realm.objects.first()  # Get any available realm
        
        ai_agent = ZulipAIAgent(realm)
        if not ai_agent.is_healthy():
            return _fallback_pattern_detection(message)
        
        prompt = f"""
        Analyze this message to determine if the user wants to create a new chat channel or group.
        Return a JSON object with:
        - is_creation_request: true if they want to create something
        - channel_name: the name they want to use (if specified)
        
        User message: "{message}"
        
        Examples:
        - "create channel python-help" → {{"is_creation_request": true, "channel_name": "python-help"}}
        - "make a new group for design" → {{"is_creation_request": true, "channel_name": "design"}}
        - "start coding channel" → {{"is_creation_request": true, "channel_name": "coding"}}
        - "how are you?" → {{"is_creation_request": false}}
        
        Return only valid JSON, no other text.
        """
        
        # Create a dummy user for AI agent (we need to pass a user)
        from zerver.models import UserProfile
        dummy_user = UserProfile.objects.filter(realm=realm, is_active=True).first()
        if not dummy_user:
            return _fallback_pattern_detection(message)
            
        response = ai_agent.chat(prompt, dummy_user)
        if response:
            import json
            try:
                result = json.loads(response.strip())
                is_creation = result.get('is_creation_request', False)
                channel_name = result.get('channel_name')
                return is_creation, channel_name
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    is_creation = result.get('is_creation_request', False)
                    channel_name = result.get('channel_name')
                    return is_creation, channel_name
        
        return _fallback_pattern_detection(message)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"AI chat creation detection failed: {e}")
        return _fallback_pattern_detection(message)


def _fallback_pattern_detection(message: str) -> Tuple[bool, Optional[str]]:
    """
    Fallback pattern-based detection for chat creation requests
    """
    patterns = [
        r'create (?:a |)(?:new |)(?:chat|group|channel) ["\']?([^"\']+)["\']?',
        r'make (?:a |)(?:new |)(?:chat|group|channel) ["\']?([^"\']+)["\']?',
        r'start (?:a |)(?:new |)(?:chat|group|channel) ["\']?([^"\']+)["\']?',
        r'create (?:a |)(?:new |)(?:chat|group|channel) called ["\']?([^"\']+)["\']?',
        r'create (?:a |)(?:new |)(?:chat|group|channel) named ["\']?([^"\']+)["\']?',
        r'create (?:a |)(?:new |)(?:chat|group|channel) ["\']?([^"\']+)["\']? for me',
    ]
    
    message_lower = message.lower()
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            return True, match.group(1).strip()
    
    return False, None


def add_relevant_users_to_channel(realm: Realm, stream: Stream, topic_keywords: str, creator: UserProfile) -> List[UserProfile]:
    """
    Find and add users who have discussed similar topics to the newly created channel
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Extract keywords from the topic
        keywords = []
        if topic_keywords:
            # Split on common separators and clean up
            raw_keywords = re.split(r'[,\s]+', topic_keywords.lower())
            keywords = [kw.strip() for kw in raw_keywords if kw.strip() and len(kw.strip()) > 2]
        
        if not keywords:
            return []
        
        # Build search query to find users who have discussed these topics
        q_objects = Q()
        for keyword in keywords:
            q_objects |= Q(content__icontains=keyword)
        
        # Find messages from users who have discussed similar topics
        relevant_messages = Message.objects.filter(
            realm=realm,
            recipient__type=Recipient.STREAM,  # Only stream messages
        ).filter(q_objects).exclude(
            sender=creator  # Exclude the channel creator
        ).exclude(
            sender__is_bot=True  # Exclude bots
        ).select_related('sender').order_by('-date_sent')[:50]
        
        # Count activity by user and find most relevant participants
        user_activity = {}
        for msg in relevant_messages:
            user_key = msg.sender.email
            if user_key not in user_activity:
                user_activity[user_key] = {
                    'user': msg.sender,
                    'message_count': 0,
                    'last_activity': msg.date_sent
                }
            user_activity[user_key]['message_count'] += 1
        
        # Sort by activity and get top relevant users
        sorted_users = sorted(
            user_activity.values(),
            key=lambda x: (x['message_count'], x['last_activity']),
            reverse=True
        )[:5]  # Limit to top 5 most relevant users
        
        # Add users to the channel
        users_to_add = [data['user'] for data in sorted_users]
        if users_to_add:
            # Use bulk_add_subscriptions to add users to the stream
            bulk_add_subscriptions(realm, [stream], users_to_add, acting_user=creator)
            logger.info(f"Added {len(users_to_add)} relevant users to channel {stream.name}")
        
        return users_to_add
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error adding relevant users to channel: {e}")
        return []
