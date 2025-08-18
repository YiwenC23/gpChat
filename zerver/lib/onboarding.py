from django.conf import settings
from django.db import transaction
from django.db.models import Count
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
import re

from zerver.actions.create_realm import setup_realm_internal_bots
from zerver.actions.message_send import (
    do_send_messages,
    internal_prep_stream_message_by_name,
    internal_send_private_message,
)
from zerver.actions.reactions import do_add_reaction
from zerver.lib.streams import create_stream_if_needed
from zerver.lib.string_validation import check_stream_name
from zerver.actions.user_groups import create_user_group_in_database
from zerver.lib.emoji import get_emoji_data
from zerver.lib.message import SendMessageRequest, remove_single_newlines
from zerver.lib.streams import get_stream
from zerver.models import Message, OnboardingUserMessage, Realm, UserProfile, UserGroup, Stream, Recipient
from zerver.models.users import get_system_bot
from django.db.models import Q
from typing import List, Dict, Any


def extract_search_terms_with_ai(query: str, realm: Realm) -> List[str]:
    """
    Use AI to intelligently extract search terms and categories from user query.
    """
    try:
        from zerver.lib.ai_agents_openai import ZulipAIAgent
        from django.conf import settings
        
        ai_agent = ZulipAIAgent(realm)
        
        prompt = f"""
        Analyze this user query and extract relevant search terms and categories: "{query}"
        
        Your task:
        1. Identify what category/topic the user is asking about (e.g., food, technology, work, entertainment, etc.)
        
        Return only the keywords as a comma-separated list, no other text.
        """
        
        # Create a dummy user for AI agent (we need to pass a user)
        from zerver.models import UserProfile
        dummy_user = UserProfile.objects.filter(realm=realm, is_active=True).first()
        if not dummy_user:
            return []
            
        response = ai_agent.chat(prompt, dummy_user)
        if response:
            # Parse the response to extract keywords
            keywords = [term.strip() for term in response.split(',') if term.strip()]
            return keywords[:5]  # Limit to 5 keywords
        
        return []
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"AI search term extraction failed: {e}")
        
        # Fallback to simple word extraction
        words = message.lower().split()
        return [word for word in words if len(word) > 3 and word.isalpha()][:5]


def analyze_user_intent_with_ai(message: str, realm: Realm) -> Dict[str, Any]:
    """
    Use AI to analyze user intent and extract relevant information from their message.
    """
    try:
        from zerver.lib.ai_agents_openai import ZulipAIAgent
        
        ai_agent = ZulipAIAgent(realm)
        if not ai_agent.is_healthy():
            return {}
        
        prompt = f"""
        Analyze this user message and determine their intent. Return a JSON object with these fields:

        - is_channel_send_request: true if user wants to send a message to a channel or person
        - is_suggestion_request: true if user wants suggestions for who to contact about a topic
        - message_to_send: the actual message content they want to send (if any)
        - target_channel: the channel name they want to send to (if specified)
        - target_user: the username they want to send to (if specified, without @)

        User message: "{message}"

        Examples:
        - "send 'hello' to #general" → {{"is_channel_send_request": true, "message_to_send": "hello", "target_channel": "general"}}
        - "send 'hi' to @john" → {{"is_channel_send_request": true, "message_to_send": "hi", "target_user": "john"}}
        - "who knows about python?" → {{"is_suggestion_request": true}}
        - "who can I send this to?" → {{"is_suggestion_request": true}}

        Return only valid JSON, no other text.
        """
        
        # Create a dummy user for AI agent (we need to pass a user)
        from zerver.models import UserProfile
        dummy_user = UserProfile.objects.filter(realm=realm, is_active=True).first()
        if not dummy_user:
            return {}
            
        response = ai_agent.chat(prompt, dummy_user)
        if response:
            import json
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        
        return {}
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"AI intent analysis failed: {e}")
        return {}


def suggest_target_channel(realm: Realm, message_content: str, sender_email: str) -> Dict[str, Any]:
    """
    Suggest the best channel to send a message to based on content and chat history.
    """
    try:
        # Use AI to extract topics and keywords from the message
        search_terms = extract_search_terms_with_ai(message_content, realm)
        
        if not search_terms:
            return {}
        
        # Build search query to find channels with similar discussions
        q_objects = Q()
        for term in search_terms:
            q_objects |= Q(content__icontains=term)
        
        # Find messages in streams on similar topics
        messages = Message.objects.filter(
            realm=realm,
            recipient__type=Recipient.STREAM,  # Only stream messages
        ).filter(q_objects).exclude(
            sender__email=sender_email  # Exclude the current user
        ).exclude(
            sender__email__endswith='welcome-bot@zulip.com'  # Exclude bots
        ).select_related('sender', 'recipient').order_by('-date_sent')[:50]
        
        # Count activity by channel to find most relevant
        channel_activity = {}
        for msg in messages:
            try:
                stream = Stream.objects.get(recipient=msg.recipient)
                stream_key = stream.name
                if stream_key not in channel_activity:
                    channel_activity[stream_key] = {
                        'stream': stream,
                        'message_count': 0,
                        'recent_messages': [],
                        'last_message_date': msg.date_sent,
                        'participants': set()
                    }
                channel_activity[stream_key]['message_count'] += 1
                channel_activity[stream_key]['participants'].add(msg.sender.full_name)
                if len(channel_activity[stream_key]['recent_messages']) < 3:
                    channel_activity[stream_key]['recent_messages'].append(msg.content[:100])
            except Stream.DoesNotExist:
                continue
        
        # Find the most active channel for this topic
        if channel_activity:
            best_channel = max(
                channel_activity.values(), 
                key=lambda x: (x['message_count'], x['last_message_date'])
            )
            
            return {
                'channel_name': best_channel['stream'].name,
                'stream_id': best_channel['stream'].id,
                'message_count': best_channel['message_count'],
                'participants': list(best_channel['participants'])[:5],
                'recent_messages': best_channel['recent_messages'],
                'last_active': best_channel['last_message_date'].strftime('%Y-%m-%d %H:%M')
            }
        
        return {}
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error suggesting target channel: {e}")
        return {}


def suggest_message_recipients(realm: Realm, message_content: str, sender_email: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Suggest relevant people to send a message to based on chat history and message content.
    """
    try:
        # Use AI to extract topics and keywords from the message
        search_terms = extract_search_terms_with_ai(message_content, realm)
        
        if not search_terms:
            return []
        
        # Build search query to find users who have discussed similar topics
        q_objects = Q()
        for term in search_terms:
            # Clean up the term and make search more specific for Python
            clean_term = term.strip('",').lower()
            if clean_term in ['python', 'learn', 'coding', 'programming', 'help']:
                q_objects |= Q(content__icontains=clean_term)
        
        # If no relevant terms found, fallback to direct keyword search
        if not q_objects:
            # Extract key topics from the message content
            content_lower = message_content.lower()
            if 'python' in content_lower:
                q_objects |= Q(content__icontains='python')
            if 'learn' in content_lower or 'help' in content_lower:
                q_objects |= Q(content__icontains='help') | Q(content__icontains='learn')
        
        # Find messages from other users on similar topics
        messages = Message.objects.filter(
            realm=realm,
            recipient__type=Recipient.STREAM,  # Only stream messages
        ).filter(q_objects).exclude(
            sender__email=sender_email  # Exclude the current user
        ).exclude(
            sender__email__endswith='welcome-bot@zulip.com'  # Exclude bots
        ).select_related('sender', 'recipient').order_by('-date_sent')[:50]
        
        # Count message frequency by user to find most active participants
        user_activity = {}
        for msg in messages:
            user_key = msg.sender.email
            if user_key not in user_activity:
                user_activity[user_key] = {
                    'user': msg.sender,
                    'message_count': 0,
                    'recent_topics': [],
                    'last_message_date': msg.date_sent
                }
            user_activity[user_key]['message_count'] += 1
            if len(user_activity[user_key]['recent_topics']) < 3:
                user_activity[user_key]['recent_topics'].append(msg.content[:100])
        
        # Sort by activity and return top suggestions
        sorted_users = sorted(
            user_activity.values(), 
            key=lambda x: (x['message_count'], x['last_message_date']), 
            reverse=True
        )[:limit]
        
        suggestions = []
        for user_data in sorted_users:
            suggestions.append({
                'user_name': user_data['user'].full_name,
                'user_email': user_data['user'].email,
                'message_count': user_data['message_count'],
                'recent_topics': user_data['recent_topics'],
                'last_active': user_data['last_message_date'].strftime('%Y-%m-%d %H:%M')
            })
        
        return suggestions
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error suggesting message recipients: {e}")
        return []


def search_messages_in_database(realm: Realm, query: str, limit: int = 10, exclude_sender_email: str = None) -> List[Dict[str, Any]]:
    """
    Search for messages in the database based on the query using AI-powered term extraction.
    Returns a list of dictionaries with message information.
    """
    try:
        # Use AI to extract intelligent search terms
        search_terms = extract_search_terms_with_ai(query, realm)
        
        if not search_terms:
            return []
        
        # Build search query
        q_objects = Q()
        for term in search_terms:
            q_objects |= Q(content__icontains=term)
        
        # Search messages in streams (not private messages)
        messages_query = Message.objects.filter(
            realm=realm,
            recipient__type=Recipient.STREAM,  # Only stream messages
        ).filter(q_objects).select_related('sender', 'recipient')
        
        # Exclude welcome bot messages and optionally the requesting user
        messages_query = messages_query.exclude(sender__email__endswith='welcome-bot@zulip.com')
        if exclude_sender_email:
            messages_query = messages_query.exclude(sender__email=exclude_sender_email)
            
        messages = messages_query.order_by('-date_sent')[:limit]
        
        results = []
        for msg in messages:
            try:
                # Get stream info
                stream = Stream.objects.get(recipient=msg.recipient)
                results.append({
                    'content': msg.content,
                    'sender': msg.sender.full_name,
                    'stream': stream.name,
                    'stream_id': stream.id,
                    'topic': msg.topic_name(),
                    'date': msg.date_sent.strftime('%Y-%m-%d %H:%M'),
                    'message_id': msg.id
                })
            except Stream.DoesNotExist:
                continue
        
        return results
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error searching messages: {e}")
        return []


def missing_any_realm_internal_bots() -> bool:
    bot_emails = [
        bot["email_template"] % (settings.INTERNAL_BOT_DOMAIN,)
        for bot in settings.REALM_INTERNAL_BOTS
    ]
    realm_count = Realm.objects.count()
    return UserProfile.objects.filter(email__in=bot_emails).values("email").alias(
        count=Count("id")
    ).filter(count=realm_count).count() != len(bot_emails)


def create_if_missing_realm_internal_bots() -> None:
    """This checks if there is any realm internal bot missing.

    If that is the case, it creates the missing realm internal bots.
    """
    if missing_any_realm_internal_bots():
        for realm in Realm.objects.all():
            setup_realm_internal_bots(realm)


def send_initial_direct_message(user: UserProfile) -> int:
    # We adjust the initial Welcome Bot direct message for education organizations.
    education_organization = user.realm.org_type in (
        Realm.ORG_TYPES["education_nonprofit"]["id"],
        Realm.ORG_TYPES["education"]["id"],
    )

    # We need to override the language in this code path, because it's
    # called from account registration, which is a pre-account API
    # request and thus may not have the user's language context yet.
    with override_language(user.default_language):
        if education_organization:
            getting_started_string = _("""
To learn more, check out our [using Zulip for a class guide]({getting_started_url})!
""").format(getting_started_url="/help/using-zulip-for-a-class")
        else:
            getting_started_string = _("""
To learn more, check out our [getting started guide]({getting_started_url})!
""").format(getting_started_url="/help/getting-started-with-zulip")

        organization_setup_string = ""
        # Add extra content on setting up a new organization for administrators.
        if user.is_realm_admin:
            if education_organization:
                organization_setup_string = _("""
We also have a guide for [setting up Zulip for a class]({organization_setup_url}).
""").format(organization_setup_url="/help/setting-up-zulip-for-a-class")
            else:
                organization_setup_string = _("""
We also have a guide for [moving your organization to Zulip]({organization_setup_url}).
""").format(organization_setup_url="/help/moving-to-zulip")

        demo_organization_warning_string = ""
        # Add extra content about automatic deletion for demo organization owners.
        if user.is_realm_owner and user.realm.demo_organization_scheduled_deletion_date is not None:
            demo_organization_warning_string = _("""
Note that this is a [demo organization]({demo_organization_help_url}) and
will be **automatically deleted** in 30 days, unless it's [converted into
a permanent organization]({convert_demo_organization_help_url}).
""").format(
                demo_organization_help_url="/help/demo-organizations",
                convert_demo_organization_help_url="/help/demo-organizations#convert-a-demo-organization-to-a-permanent-organization",
            )

        inform_about_tracked_onboarding_messages_text = ""
        if OnboardingUserMessage.objects.filter(realm_id=user.realm_id).exists():
            inform_about_tracked_onboarding_messages_text = _("""
I've kicked off some conversations to help you get started. You can find
them in your [Inbox](/#inbox).
""")

        navigation_tour_video_string = _("""
You can always come back to the [Welcome to Zulip video]({navigation_tour_video_url}) for a quick app overview.
""").format(navigation_tour_video_url=settings.NAVIGATION_TOUR_VIDEO_URL)

        content = _("""
Hello, and welcome to Zulip!👋 {inform_about_tracked_onboarding_messages_text}

{getting_started_text} {organization_setup_text}

{navigation_tour_video_text}

{demo_organization_text}

""").format(
            inform_about_tracked_onboarding_messages_text=inform_about_tracked_onboarding_messages_text,
            getting_started_text=getting_started_string,
            organization_setup_text=organization_setup_string,
            navigation_tour_video_text=navigation_tour_video_string,
            demo_organization_text=demo_organization_warning_string,
        )

    message_id = internal_send_private_message(
        get_system_bot(settings.WELCOME_BOT, user.realm_id),
        user,
        remove_single_newlines(content),
        # Note: Welcome bot doesn't trigger email/push notifications,
        # as this is intended to be seen contextually in the application.
        disable_external_notifications=True,
    )
    assert message_id is not None
    return message_id


def bot_commands(no_help_command: bool = False) -> str:
    commands = [
        "apps",
        "profile",
        "theme",
        "channels",
        "topics",
        "message formatting",
        "keyboard shortcuts",
        "create chat",
    ]
    if not no_help_command:
        commands.append("help")
    return ", ".join("`" + command + "`" for command in commands) + "."


def select_welcome_bot_response(human_response_lower: str) -> str:
    # Given the raw (pre-markdown-rendering) content for a private
    # message from the user to Welcome Bot, select the appropriate reply.
    if human_response_lower in ["app", "apps"]:
        return _("""
You can [download](/apps/) the [mobile and desktop apps](/apps/).
Zulip also works great in a browser.
""")
    elif human_response_lower == "profile":
        return _("""
Go to [Profile settings](#settings/profile) to add a [profile picture](/help/change-your-profile-picture)
and edit your [profile information](/help/edit-your-profile).
""")
    elif human_response_lower == "theme":
        return _("""
You can switch between [light and dark theme](/help/dark-theme), [pick your
favorite emoji set](/help/emoji-and-emoticons#change-your-emoji-set), [change
your language](/help/change-your-language), and otherwise customize your Zulip
experience in your [Preferences](#settings/preferences).
""")
    elif human_response_lower in ["stream", "streams", "channel", "channels"]:
        return _("""
Channels organize conversations based on who needs to see them. For example,
it's common to have a channel for each team in an organization.

[Browse and subscribe to channels]({settings_link}).
""").format(help_link="/help/introduction-to-channels", settings_link="#channels/all")
    elif human_response_lower in ["topic", "topics"]:
        return _("""
[Topics](/help/introduction-to-topics) summarize what each conversation in Zulip
is about. You can read Zulip one topic at a time, seeing each message in
context, no matter how many other conversations are going on.

When you start a conversation, label it with a new topic. For a good topic name,
think about finishing the sentence: “Hey, can we chat about…?”

Check out [Recent conversations](#recent) for a list of topics that are being
discussed.
""")
    elif human_response_lower in ["keyboard", "shortcuts", "keyboard shortcuts"]:
        return _("""
Zulip's [keyboard shortcuts](#keyboard-shortcuts) let you navigate the app
quickly and efficiently.

Press `?` any time to see a [cheat sheet](#keyboard-shortcuts).
""")
    elif human_response_lower in ["formatting", "message formatting"]:
        return _("""
You can **format** *your* `message` using the handy formatting buttons, or by
typing your formatting with Markdown.

Check out the [cheat sheet](#message-formatting) to learn about spoilers, global
times, and more.
""")
    elif human_response_lower in ["help", "?"]:
        return _("""
Here are a few messages I understand: {bot_commands}

Check out our [Getting started guide](/help/getting-started-with-zulip),
or browse the [Help center](/help/) to learn more!
""").format(bot_commands=bot_commands(no_help_command=True))
    else:
        return _("""
You can chat with me as much as you like! To
get help, try one of the following messages: {bot_commands}
""").format(bot_commands=bot_commands())


def send_welcome_bot_response(send_request: SendMessageRequest) -> None:
    """Given the send_request object for a direct message from the user
    to welcome-bot, trigger the welcome-bot reply."""
    welcome_bot = get_system_bot(settings.WELCOME_BOT, send_request.realm.id)
    human_message = send_request.message.content
    human_response_lower = human_message.lower()
    human_user = send_request.message.sender
    human_user_recipient_id = human_user.recipient_id
    assert human_user_recipient_id is not None
    
    # Import our chat creation functions
    from zerver.lib.chat_creation import detect_chat_creation_request, create_channel, create_chat_group
    from zerver.actions.message_send import internal_send_stream_message_by_name, internal_send_private_message
    
    # Check for chat creation requests first
    is_creation_request, chat_name = detect_chat_creation_request(human_message)
    if is_creation_request and chat_name:
        # Try to create the channel with topic keywords for adding relevant users
        from zerver.lib.chat_creation import create_channel
        success, message, stream = create_channel(send_request.realm, human_user, chat_name, topic_keywords=chat_name)
        
        if success and stream:
            # Add clickable channel link to the response
            channel_link = f"#**{chat_name}**"
            enhanced_message = f"{message}\n\nYou can access your new channel here: {channel_link}"
            response_message = enhanced_message
        else:
            response_message = message
        
        # Send response about channel creation
        internal_send_private_message(
            welcome_bot,
            human_user,
            remove_single_newlines(response_message),
            disable_external_notifications=True,
        )
        return
    
    # Check if AI agents are enabled and try to use OpenAI
    ai_enabled = getattr(settings, "AI_AGENTS_ENABLED", False)
    content = None
    
    # Log AI status for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Welcome Bot AI check - Enabled: {ai_enabled}, User: {human_user.full_name}, Message: {human_message}")
    
    if ai_enabled:
        try:
            from zerver.lib.ai_agents_openai import ZulipAIAgent
            from zerver.lib.openai_client import OpenAIConnectionError, OpenAIModelError
            
            # Create AI agent for welcome bot responses
            ai_agent = ZulipAIAgent(send_request.realm)
            
            # Check if AI system is healthy
            ai_healthy = ai_agent.is_healthy()
            logger.info(f"AI system health check: {ai_healthy}")
            
            if ai_healthy:
                # Build context for AI with conversation history awareness
                context = f"""
                You are the Welcome Bot in Zulip, a team collaboration platform. 
                You are helping a user named {human_user.full_name} 
                in the {send_request.realm.name} organization.
                
                Your role is to:
                1. Help users understand how to use Zulip
                2. Answer questions about Zulip features
                3. Provide helpful guidance for getting started
                4. Be friendly and welcoming
                5. Use retrieved conversation history to provide personalized, contextually relevant responses
                6. Remember previous interactions with this user when appropriate
                7. Help users create channels when they request it
                8. Search for and link to relevant messages when users ask about specific topics or content
                
                The user's message is: "{human_message}"
                
                If the user asks about specific Zulip features, provide helpful explanations.
                If they ask for help with commands, you can mention these common commands:
                - "help" or "?" - Show available commands
                - "apps" - Information about mobile/desktop apps
                - "profile" - How to edit profile settings
                - "theme" - How to customize appearance
                - "streams" or "channels" - Information about channels
                - "topics" - How topics work in Zulip
                - "keyboard shortcuts" - Navigation shortcuts
                - "formatting" - Message formatting options
                - "create chat" - Create a new channel
                
                SPECIAL COMMAND: If the user asks you to create a chat or channel, you can now do this!
                Examples of creation requests:
                - "create a chat called team-chat"
                - "make a new channel project-updates"
                - "create channel marketing"
                - "start a new group design-team"
                
                CONVERSATION HISTORY GUIDELINES:
                1. You have access to previous conversations between users and the bot.
                2. Use this conversation history to maintain continuity and context.
                3. If you see relevant past interactions in the retrieved context, reference them naturally.
                4. Don't repeat yourself if you see you've already explained something to this user.
                5. If the user is continuing a previous conversation, acknowledge that appropriately.
                6. Always prioritize directly answering the current question over repeating old information.
                
                SEARCH AND LINKING GUIDELINES:
                When users ask about finding specific content (like "anyone mention restaurants?"), you should:
                1. Search through the retrieved conversation history for relevant messages
                2. Provide direct quotes or summaries of relevant messages you find
                3. Include the sender's name and approximate time when possible
                4. If you find relevant messages, present them clearly with context
                5. If no relevant messages are found in the retrieved context, suggest using Zulip's search feature
                
                Keep your responses concise, helpful, and welcoming.
                """
                
                # Configure AI agent with welcome bot specific settings
                ai_agent.context_limit = getattr(settings, "WELCOME_BOT_VECTOR_CONTEXT_LIMIT", 3)
                ai_agent.context_threshold = getattr(settings, "WELCOME_BOT_VECTOR_THRESHOLD", 0.5)
                
                # Use AI to detect user intent instead of keyword matching
                intent_analysis = analyze_user_intent_with_ai(human_message, send_request.realm)
                
                is_channel_send_request = intent_analysis.get('is_channel_send_request', False)
                is_suggestion_request = intent_analysis.get('is_suggestion_request', False)
                message_to_send = intent_analysis.get('message_to_send')
                target_channel = intent_analysis.get('target_channel')
                target_user = intent_analysis.get('target_user')
                
                if is_channel_send_request and not is_suggestion_request:
                    
                    if message_to_send:
                        # Check if sending to a specific user via @mention
                        if target_user:
                            try:
                                # Find the user by name
                                recipient_user = UserProfile.objects.get(
                                    realm=send_request.realm,
                                    full_name__iexact=target_user,
                                    is_active=True
                                )
                                
                                # Send direct message to the user
                                internal_send_private_message(
                                    send_request.message.sender,
                                    recipient_user,
                                    message_to_send,
                                    disable_external_notifications=True,
                                )
                                content = f"✅ I've sent your message \"{message_to_send}\" directly to @{target_user}!"
                                logger.info(f"Sent direct message '{message_to_send}' to user {target_user} for user {send_request.message.sender.full_name}")
                            except UserProfile.DoesNotExist:
                                content = f"❌ User @{target_user} not found. Please check the username and try again."
                            except Exception as e:
                                content = f"❌ Failed to send message to @{target_user}: {str(e)}"
                                logger.error(f"Failed to send direct message: {e}")
                        elif target_channel:
                            # Send directly to specified channel
                            try:
                                internal_send_stream_message_by_name(
                                    send_request.realm,
                                    send_request.message.sender,
                                    target_channel,
                                    "general",  # Default topic
                                    message_to_send
                                )
                                content = f"✅ I've sent your message \"{message_to_send}\" to #{target_channel}!"
                                logger.info(f"Sent message '{message_to_send}' to specified channel {target_channel} for user {send_request.message.sender.full_name}")
                            except Exception as e:
                                content = f"❌ Failed to send message to #{target_channel}: {str(e)}"
                                logger.error(f"Failed to send message to channel: {e}")
                        else:
                            # Get channel suggestion based on message content
                            channel_suggestion = suggest_target_channel(
                                send_request.realm,
                                message_to_send,
                                send_request.message.sender.email
                            )
                            
                            if channel_suggestion:
                                # Send the message to the suggested channel
                                try:
                                    internal_send_stream_message_by_name(
                                        send_request.realm,
                                        send_request.message.sender,
                                        channel_suggestion['channel_name'],
                                        "general",  # Default topic
                                        message_to_send
                                    )
                                    
                                    channel_response = f"✅ I've sent your message \"{message_to_send}\" to #{channel_suggestion['channel_name']}!\n\n"
                                    channel_response += f"This channel has {channel_suggestion['message_count']} related messages and {len(channel_suggestion['participants'])} active participants including: {', '.join(channel_suggestion['participants'][:3])}.\n\n"
                                    channel_response += f"You can view the channel here: #**{channel_suggestion['channel_name']}**"
                                    content = channel_response
                                    logger.info(f"Sent message '{message_to_send}' to channel {channel_suggestion['channel_name']} for user {send_request.message.sender.full_name}")
                                except Exception as e:
                                    content = f"❌ Failed to send message to #{channel_suggestion['channel_name']}: {str(e)}"
                                    logger.error(f"Failed to send message to channel: {e}")
                            else:
                                content = f"I couldn't find a suitable channel for your message \"{message_to_send}\". Try being more specific about the topic or create a new channel."
                    else:
                        content = "Please specify the message you want to send. For example: 'Send \"I want pho\" to \"test\"'"
                        
                elif is_suggestion_request:
                    # Get recipient suggestions based on message content
                    suggestions = suggest_message_recipients(
                        send_request.realm, 
                        human_message, 
                        send_request.message.sender.email, 
                        limit=3
                    )
                    
                    if suggestions:
                        suggestion_response = "Here are some people who might be interested in your message based on their chat history:\n\n"
                        for i, suggestion in enumerate(suggestions, 1):
                            suggestion_response += f"{i}. **{suggestion['user_name']}** - Active in similar topics ({suggestion['message_count']} related messages)\n"
                            suggestion_response += f"   Recent topics: {', '.join(suggestion['recent_topics'][:2])}...\n"
                            suggestion_response += f"   Last active: {suggestion['last_active']}\n\n"
                        
                        suggestion_response += "You can send them a direct message or mention them in a relevant channel!"
                        content = suggestion_response
                        logger.info(f"Provided recipient suggestions for user {send_request.message.sender.full_name}")
                    else:
                        content = "I couldn't find specific people to suggest based on your message. Try being more specific about the topic, or use Zulip's search to find relevant conversations."
                else:
                    # Search for relevant messages in database, excluding the user's own messages
                    search_results = search_messages_in_database(send_request.realm, human_message, limit=5, exclude_sender_email=send_request.message.sender.email)
                    
                    # Add search results to context if found
                    if search_results:
                        search_context = "\n\nRELEVANT MESSAGES FOUND:\n"
                        for msg in search_results:
                            search_context += f"- {msg['sender']} in #{msg['stream']} ({msg['date']}): {msg['content'][:200]}...\n"
                            # Use proper Zulip message link format with stream ID
                            search_context += f"  Link: [View message](/#narrow/stream/{msg['stream_id']}-{msg['stream']}/near/{msg['message_id']})\n"
                        context += search_context
                        
                        # Create response with multiple relevant messages
                        if len(search_results) == 1:
                            ai_response_prefix = f"I found a message from {search_results[0]['sender']} in #{search_results[0]['stream']} on {search_results[0]['date']} that says: \"{search_results[0]['content'][:100]}...\"\n\n"
                            ai_response_prefix += f"You can view the message here: [View message](/#narrow/stream/{search_results[0]['stream_id']}-{search_results[0]['stream']}/near/{search_results[0]['message_id']})"
                        else:
                            ai_response_prefix = f"I found {len(search_results)} relevant messages:\n\n"
                            for i, msg in enumerate(search_results[:5], 1):  # Show up to 5 messages
                                ai_response_prefix += f"{i}. **{msg['sender']}** in #{msg['stream']} ({msg['date']}):\n"
                                ai_response_prefix += f"   \"{msg['content'][:150]}...\"\n"
                                ai_response_prefix += f"   [View message](/#narrow/stream/{msg['stream_id']}-{msg['stream']}/near/{msg['message_id']})\n\n"
                        
                        # Skip AI generation and use direct response with working links
                        content = ai_response_prefix
                        logger.info(f"Using direct response with {len(search_results)} messages found...")
                    else:
                        # Only use AI if no search results found
                        context += "\nNo relevant messages found in recent history. Suggest using Zulip's search feature."
                    
                    # Generate AI response
                    logger.info("Generating AI response with database search context...")
                    ai_response = ai_agent.chat(
                        message=send_request.message.content,
                        user=send_request.message.sender,
                        context=context,
                        agent_type="welcome_bot",
                        use_vector_context=False,  # Use database search instead
                    )
                    logger.info(f"AI response received: {ai_response[:100]}...")
                    
                    # Use AI response if it's reasonable
                    if ai_response and len(ai_response.strip()) > 10:
                        content = ai_response
                        # Log successful AI usage
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info(f"AI welcome bot used for user {send_request.message.sender.full_name}: {ai_response[:100]}...")
                    else:
                        # Log why AI response was not used
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"AI response too short or empty: '{ai_response}'")
                    
        except (OpenAIConnectionError, OpenAIModelError) as e:
            # Log error but continue with fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"AI welcome bot failed: {e}, using fallback response")
        except Exception as e:
            # Log any other errors but continue with fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unexpected error in AI welcome bot: {e}, using fallback response")
    
    # Fallback to original logic if AI is not available or fails
    if content is None:
        content = select_welcome_bot_response(human_response_lower)
    
    realm_id = send_request.realm.id
    commands = bot_commands()
    if (
        commands in content
        and Message.objects.filter(
            realm_id=realm_id,
            sender_id=welcome_bot.id,
            recipient_id=human_user_recipient_id,
            content__icontains=commands,
        ).exists()
        # Uses index 'zerver_message_realm_sender_recipient'
    ):
        # If the bot has already sent bot commands to this user and
        # if the bot does not understand the current message sent by this user then
        # do not send any message
        return

    internal_send_private_message(
        welcome_bot,
        send_request.message.sender,
        remove_single_newlines(content),
        # Note: Welcome bot doesn't trigger email/push notifications,
        # as this is intended to be seen contextually in the application.
        disable_external_notifications=True,
    )


@transaction.atomic(savepoint=False)
def send_initial_realm_messages(realm: Realm) -> None:
    # Sends the initial messages for a new organization.
    #
    # Technical note: Each stream created in the realm creation
    # process should have at least one message declared in this
    # function, to enforce the pseudo-invariant that every stream has
    # at least one message.
    welcome_bot = get_system_bot(settings.WELCOME_BOT, realm.id)

    # Content is declared here to apply translation properly.
    #
    # remove_single_newlines needs to be called on any multiline
    # strings for them to render properly.
    content1_of_moving_messages_topic_name = (
        _("""
If anything is out of place, it’s easy to [move messages]({move_content_another_topic_help_url}),
[rename]({rename_topic_help_url}) and [split]({move_content_another_topic_help_url}) topics,
or even move a topic [to a different channel]({move_content_another_channel_help_url}).
""")
    ).format(
        move_content_another_topic_help_url="/help/move-content-to-another-topic",
        rename_topic_help_url="/help/rename-a-topic",
        move_content_another_channel_help_url="/help/move-content-to-another-channel",
    )

    content2_of_moving_messages_topic_name = _("""
:point_right: Try moving this message to another topic and back.
""")

    content1_of_welcome_to_zulip_topic_name = _("""
Zulip is organized to help you communicate more efficiently. Conversations are
labeled with topics, which summarize what the conversation is about.

For example, this message is in the “{topic_name}” topic in the
#**{zulip_discussion_channel_name}** channel, as you can see in the left sidebar
and above.
""").format(
        zulip_discussion_channel_name=str(Realm.ZULIP_DISCUSSION_CHANNEL_NAME),
        topic_name=_("welcome to Zulip!"),
    )

    content2_of_welcome_to_zulip_topic_name = _("""
You can read Zulip one conversation at a time, seeing each message in context,
no matter how many other conversations are going on.
""")

    content3_of_welcome_to_zulip_topic_name = _("""
:point_right: When you're ready, check out your [Inbox](/#inbox) for other
conversations with unread messages.
""")

    content1_of_start_conversation_topic_name = _("""
To kick off a new conversation, pick a channel in the left sidebar, and click
the `+` button next to its name.
""")

    content2_of_start_conversation_topic_name = _("""
Label your conversation with a topic. Think about finishing the sentence: “Hey,
can we chat about…?”
""")

    content3_of_start_conversation_topic_name = _("""
:point_right: Try starting a new conversation in this channel.
""")

    content1_of_experiments_topic_name = (
        _("""
:point_right:  Use this topic to try out [Zulip's messaging features]({format_message_help_url}).
""")
    ).format(format_message_help_url="/help/format-your-message-using-markdown")

    content2_of_experiments_topic_name = (
        _("""
```spoiler Want to see some examples?

````python

print("code blocks")

````

- bulleted
- lists

Link to a conversation: #**{zulip_discussion_channel_name}>{topic_name}**

```
""")
    ).format(
        zulip_discussion_channel_name=str(Realm.ZULIP_DISCUSSION_CHANNEL_NAME),
        topic_name=_("welcome to Zulip!"),
    )

    content1_of_greetings_topic_name = _("""
This **greetings** topic is a great place to say “hi” :wave: to your teammates.
""")

    content2_of_greetings_topic_name = _("""
:point_right: Click on this message to start a new message in the same conversation.
""")

    welcome_messages: list[dict[str, str]] = []

    # Messages added to the "welcome messages" list last will be most
    # visible to users, since welcome messages will likely be browsed
    # via the right sidebar or recent conversations view, both of
    # which are sorted newest-first.
    #
    # Initial messages are configured below.

    # Advertising moving messages.
    welcome_messages += [
        {
            "channel_name": str(Realm.ZULIP_DISCUSSION_CHANNEL_NAME),
            "topic_name": _("moving messages"),
            "content": content,
        }
        for content in [
            content1_of_moving_messages_topic_name,
            content2_of_moving_messages_topic_name,
        ]
    ]

    # Suggestion to test messaging features.
    # Dependency on knowing how to send messages.
    welcome_messages += [
        {
            "channel_name": str(realm.ZULIP_SANDBOX_CHANNEL_NAME),
            "topic_name": _("experiments"),
            "content": content,
        }
        for content in [content1_of_experiments_topic_name, content2_of_experiments_topic_name]
    ]

    # Suggestion to start your first new conversation.
    welcome_messages += [
        {
            "channel_name": str(realm.ZULIP_SANDBOX_CHANNEL_NAME),
            "topic_name": _("start a conversation"),
            "content": content,
        }
        for content in [
            content1_of_start_conversation_topic_name,
            content2_of_start_conversation_topic_name,
            content3_of_start_conversation_topic_name,
        ]
    ]

    # Suggestion to send first message as a hi to your team.
    welcome_messages += [
        {
            "channel_name": str(Realm.DEFAULT_NOTIFICATION_STREAM_NAME),
            "topic_name": _("greetings"),
            "content": content,
        }
        for content in [content1_of_greetings_topic_name, content2_of_greetings_topic_name]
    ]

    # Main welcome message, this should be last.
    welcome_messages += [
        {
            "channel_name": str(realm.ZULIP_DISCUSSION_CHANNEL_NAME),
            "topic_name": _("welcome to Zulip!"),
            "content": content,
        }
        for content in [
            content1_of_welcome_to_zulip_topic_name,
            content2_of_welcome_to_zulip_topic_name,
            content3_of_welcome_to_zulip_topic_name,
        ]
    ]

    # End of message declarations; now we actually send them.

    messages = [
        internal_prep_stream_message_by_name(
            realm,
            welcome_bot,
            message["channel_name"],
            message["topic_name"],
            remove_single_newlines(message["content"]),
        )
        for message in welcome_messages
    ]
    message_ids = [
        sent_message_result.message_id for sent_message_result in do_send_messages(messages)
    ]

    seen_topics = set()
    onboarding_topics_first_message_ids = set()
    for index, message in enumerate(welcome_messages):
        topic_name = message["topic_name"]
        if topic_name not in seen_topics:
            onboarding_topics_first_message_ids.add(message_ids[index])
            seen_topics.add(topic_name)

    onboarding_user_messages = []
    for message_id in message_ids:
        flags = OnboardingUserMessage.flags.historical
        if message_id in onboarding_topics_first_message_ids:
            flags |= OnboardingUserMessage.flags.starred
        onboarding_user_messages.append(
            OnboardingUserMessage(realm=realm, message_id=message_id, flags=flags)
        )

    OnboardingUserMessage.objects.bulk_create(onboarding_user_messages)

    # We find the one of our just-sent greetings messages, and react to it.
    # This is a bit hacky, but works and is kinda a 1-off thing.
    greetings_message = (
        Message.objects.select_for_update()
        .filter(
            id__in=message_ids, content=remove_single_newlines(content1_of_greetings_topic_name)
        )
        .first()
    )
    assert greetings_message is not None
    emoji_data = get_emoji_data(realm.id, "wave")
    do_add_reaction(
        welcome_bot, greetings_message, "wave", emoji_data.emoji_code, emoji_data.reaction_type
    )
