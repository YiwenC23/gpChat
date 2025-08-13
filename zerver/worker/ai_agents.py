"""
Queue worker for AI Agents subsystem
Handles asynchronous AI processing tasks
"""
import logging
import time
import orjson
from typing import Any, Dict, List, Mapping
from typing_extensions import override

from django.conf import settings
from zerver.lib.ai_agents import get_ai_agent
from zerver.worker.base import QueueProcessingWorker, assign_queue
from zerver.models.ai_agents import AIAgentConfig, AIAgentInteraction, AIAgentModel, AIAgentUsageStats
from zerver.models.messages import Message
from zerver.models.realms import Realm, get_realm
from zerver.models.users import UserProfile, get_user_profile_by_id
from zerver.actions.message_send import internal_send_private_message

# For fetching message history
from zerver.lib.narrow import (
    LARGER_THAN_MAX_MESSAGE_ID,
    NarrowParameter,
    clean_narrow_for_message_fetch,
    fetch_messages,
)
from zerver.lib.message import messages_for_ids
from zerver.models.realms import MessageEditHistoryVisibilityPolicyEnum

from django.utils.timezone import now as timezone_now

logger = logging.getLogger(__name__)


def _log_interaction_and_update_stats(
    realm: Realm,
    user: UserProfile,
    model: AIAgentModel | None,
    prompt: str,
    response: str,
    was_successful: bool,
    error_message: str = "",
    response_time_ms: int = 0,
    token_usage: Dict[str, int] | None = None,
) -> None:
    """Log AI agent interaction to database and update usage stats"""
    try:
        # Ensure we always have a valid model FK (model field is non-nullable)
        resolved_model = model
        if resolved_model is None:
            try:
                conf = AIAgentConfig.objects.get(realm=realm)
                if conf.default_model:
                    resolved_model = conf.default_model
            except Exception:
                pass
        if resolved_model is None:
            # Fallback to settings default; create registry row if missing
            default_name = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
            resolved_model, _ = AIAgentModel.objects.get_or_create(
                name=default_name,
                defaults={
                    "display_name": default_name,
                    "description": "",
                    "supports_streaming": True,
                    "supports_embeddings": False,
                    "is_available": True,
                    "is_default": True,
                },
            )

        interaction = AIAgentInteraction.objects.create(
            realm=realm,
            user=user,
            model=resolved_model,
            prompt=prompt[:5000],  # Limit prompt length
            response=response[:10000],  # Limit response length
            success=was_successful,
            error_message=error_message,
        )

        # Create usage stats for this interaction
        now = timezone_now()
        AIAgentUsageStats.objects.create(
            interaction=interaction,
            realm=realm,
            user=user,
            date=now.date(),
            hour=now.hour,
            prompt_tokens=token_usage.get("prompt_tokens", 0) if token_usage else 0,
            completion_tokens=token_usage.get("completion_tokens", 0) if token_usage else 0,
            total_tokens=token_usage.get("total_tokens", 0) if token_usage else 0,
            response_time_ms=response_time_ms,
        )

    except Exception as e:
        logger.error(f"Failed to log AI agent interaction: {e}")


@assign_queue("ai_agents")
class AIAgentWorker(QueueProcessingWorker):
    """Process AI agent requests asynchronously"""

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        """Process AI agent request asynchronously (supports REST ai_agents queue and embedded-bot compatibility)"""
        start_time = time.time()
        try:
            # REST-triggered events published to the dedicated "ai_agents" queue
            if "task_type" in event:
                task_type = event.get("task_type")
                if task_type != "chat":
                    logger.info(f"Ignoring unsupported task_type={task_type}")
                    return

                realm_id = event["realm_id"]
                user_id = event["user_id"]
                prompt = event["prompt"]

                # Resolve realm and user
                realm = Realm.objects.get(id=realm_id)
                user = get_user_profile_by_id(user_id)

                # Check config
                try:
                    ai_config = AIAgentConfig.objects.get(realm=realm)
                    if not ai_config.enabled:
                        logger.warning(f"AI agents disabled for realm {realm.name}")
                        return
                except AIAgentConfig.DoesNotExist:
                    logger.warning(f"No AI agent config found for realm {realm.name}")
                    return

                ai_agent = get_ai_agent(realm)

                # Health check
                if not ai_agent.ollama.health_check():
                    logger.error("AI agent system is not healthy")
                    _log_interaction_and_update_stats(
                        realm,
                        user,
                        None,
                        prompt,
                        "AI agent system unavailable",
                        False,
                        error_message="System health check failed",
                        response_time_ms=int((time.time() - start_time) * 1000),
                    )
                    return

                # Generate response
                response_data = ai_agent.chat(
                    message=prompt,
                    user=user,
                )
                response_time_ms = int((time.time() - start_time) * 1000)
                response_text = response_data.get("response", "")
                token_usage = response_data.get("tokens", {})

                # Model (optional)
                model = None
                try:
                    conf = AIAgentConfig.objects.get(realm=realm)
                    if conf.default_model:
                        model = conf.default_model
                except Exception:
                    model = None

                # Log usage
                _log_interaction_and_update_stats(
                    realm,
                    user,
                    model,
                    prompt,
                    response_text,
                    True,
                    response_time_ms=response_time_ms,
                    token_usage=token_usage,
                )

                # Send response back as the AI bot (behave like a user)
                bot_profile = self._get_ai_bot_profile(realm)
                if bot_profile is None:
                    logger.error("AI bot user not found in realm; cannot deliver response")
                    return

                internal_send_private_message(
                    bot_profile,
                    user,
                    response_text,
                )
                logger.info(f"AI agent chat request processed successfully in {response_time_ms}ms")
                return

            # Compatibility: Embedded-bot style events if ever routed here
            if "message" in event and ("user_profile" in event or "user_profile_id" in event):
                # Normalize to have bot_profile and message
                if "user_profile" in event:
                    user_profile = event["user_profile"]
                    realm = Realm.objects.get(id=user_profile["realm_id"])
                    user = get_user_profile_by_id(user_profile["id"])
                else:
                    bot_profile = get_user_profile_by_id(event["user_profile_id"])
                    realm = bot_profile.realm
                    user = get_user_profile_by_id(event["message"]["sender_id"])
                message = event["message"]

                # Config
                try:
                    ai_config = AIAgentConfig.objects.get(realm=realm)
                    if not ai_config.enabled:
                        logger.warning(f"AI agents disabled for realm {realm.name}")
                        return
                except AIAgentConfig.DoesNotExist:
                    logger.warning(f"No AI agent config found for realm {realm.name}")
                    return

                ai_agent = get_ai_agent(realm)
                if not ai_agent.ollama.health_check():
                    logger.error("AI agent system is not healthy")
                    _log_interaction_and_update_stats(
                        realm,
                        user,
                        None,
                        message["content"],
                        "AI agent system unavailable",
                        False,
                        error_message="System health check failed",
                        response_time_ms=int((time.time() - start_time) * 1000),
                    )
                    return

                # Use existing handler for embedded chat events
                self._handle_chat_task(message, realm, user, ai_agent, ai_config, start_time)
                return

            logger.warning(f"Unknown ai_agents event schema keys={list(event.keys())}")
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"AI agent request failed: {e}")

            # Best-effort logging
            try:
                if "realm_id" in event and "user_id" in event:
                    realm = Realm.objects.get(id=event["realm_id"])
                    user = get_user_profile_by_id(event["user_id"])
                    _log_interaction_and_update_stats(
                        realm,
                        user,
                        None,
                        event.get("prompt", ""),
                        "",
                        False,
                        error_message=str(e),
                        response_time_ms=response_time_ms,
                    )
            except Exception:
                logger.error("Failed to log AI agent error")

    def _get_ai_bot_profile(self, realm: Realm) -> UserProfile | None:
        try:
            bot = UserProfile.objects.filter(
                realm=realm, is_bot=True, bot_type=UserProfile.EMBEDDED_BOT, full_name="AI Agent"
            ).first()
            if bot:
                return bot
            bot = UserProfile.objects.filter(
                realm=realm, is_bot=True, bot_type=UserProfile.EMBEDDED_BOT, delivery_email__istartswith="ai-agent-bot@"
            ).first()
            return bot
        except Exception:
            return None

    def _fetch_message_history(
        self,
        realm: Realm,
        user: UserProfile,
        stream_name: str | None = None,
        topic_name: str | None = None,
        num_messages: int = 50,
        exclude_message_id: int | None = None,
    ) -> str | None:
        """Fetch recent message history from a channel/topic or DM conversation

        Args:
            exclude_message_id: If provided, exclude this message from the history (e.g., the triggering message)
        """
        try:
            narrow = []

            # Build narrow based on message type
            if stream_name:
                narrow.append(NarrowParameter(operator="channel", operand=stream_name, negated=False))
                if topic_name:
                    narrow.append(NarrowParameter(operator="topic", operand=topic_name, negated=False))
            else:
                # For DM conversations, we could add DM narrow support here if needed
                return None

            # Clean narrow for fetching
            narrow = clean_narrow_for_message_fetch(narrow, realm, user)

            # Fetch messages
            # We fetch num_messages + 1 to account for potentially excluding the current message
            query_info = fetch_messages(
                narrow=narrow,
                user_profile=user,
                realm=realm,
                is_web_public_query=False,
                anchor=LARGER_THAN_MAX_MESSAGE_ID,
                include_anchor=False,  # Don't include the anchor (latest message) by default
                num_before=num_messages + 1,  # Fetch extra in case we need to exclude one
                num_after=0,
            )

            if not query_info.rows:
                return None

            # Get message IDs and filter out the current triggering message if specified
            result_message_ids = [row[0] for row in query_info.rows]
            if exclude_message_id and exclude_message_id in result_message_ids:
                result_message_ids.remove(exclude_message_id)

            # Limit to num_messages after exclusion
            result_message_ids = result_message_ids[:num_messages]

            user_message_flags = {msg_id: [] for msg_id in result_message_ids}

            # Get full message objects
            message_list = messages_for_ids(
                message_ids=result_message_ids,
                user_message_flags=user_message_flags,
                search_fields={},
                apply_markdown=False,
                client_gravatar=True,
                allow_empty_topic_name=False,
                message_edit_history_visibility_policy=MessageEditHistoryVisibilityPolicyEnum.none.value,
                user_profile=user,
                realm=realm,
            )

            if not message_list:
                return None

            # Format messages into context string
            intro = f"The following is the recent conversation history in channel #{stream_name}"
            if topic_name:
                intro += f", topic: {topic_name}"
            intro += ".\n\n"

            # Build conversation transcript
            messages_json = []
            for msg in message_list:
                messages_json.append({
                    "sender": msg["sender_full_name"],
                    "content": msg["content"]
                })

            formatted_conversation = orjson.dumps(messages_json, option=orjson.OPT_INDENT_2).decode()
            return intro + formatted_conversation

        except Exception as e:
            logger.error(f"Failed to fetch message history: {e}")
            return None

    def _handle_chat_task(self, message: Dict[str, Any], realm: Realm, user: UserProfile,
                            ai_agent: Any, ai_config: AIAgentConfig, start_time: float) -> None:
        """Handle chat tasks"""
        prompt = message["content"]
        context = None
        current_message_id = message.get("id")  # Get the current message ID to exclude it from history

        # Get the bot profile early
        bot_profile = self._get_ai_bot_profile(realm)
        if bot_profile is None:
            logger.error("AI bot user not found in realm; cannot deliver response")
            return

        # Check if the original message was in a stream/channel
        recipient_type = message.get("type")
        stream_name = None
        topic_name = None

        if recipient_type == "stream":
            stream_name = message.get("display_recipient")
            topic_name = message.get("subject", "")

            # Auto-join the stream BEFORE fetching history
            from zerver.models.streams import get_stream
            from zerver.models import Subscription
            from zerver.actions.streams import bulk_add_subscriptions

            try:
                stream = get_stream(stream_name, realm)

                # Check if bot is subscribed to the stream
                is_subscribed = Subscription.objects.filter(
                    user_profile=bot_profile,
                    recipient__type_id=stream.id,
                    recipient__type=2,  # Recipient.STREAM
                    active=True
                ).exists()

                if not is_subscribed:
                    logger.info(f"Auto-subscribing AI Agent to stream '{stream_name}'")
                    bulk_add_subscriptions(
                        realm=realm,
                        streams=[stream],
                        users=[bot_profile],
                        acting_user=None,
                        send_subscription_add_events=False,
                    )
            except Exception as e:
                logger.warning(f"Could not auto-subscribe AI Agent to stream '{stream_name}': {e}")

            # Fetch message history for the channel/topic, excluding the current message
            context = self._fetch_message_history(
                realm=realm,
                user=bot_profile,  # Use bot profile to ensure we can read the messages
                stream_name=stream_name,
                topic_name=topic_name,
                num_messages=50,  # Fetch last 50 messages for context
                exclude_message_id=current_message_id  # Exclude the current triggering message
            )

            if context:
                logger.info(f"Fetched message history for stream '{stream_name}', topic '{topic_name}'")
            else:
                logger.info(f"No message history found for stream '{stream_name}', topic '{topic_name}'")

        # Generate AI response with context
        response_data = ai_agent.chat(
            message=prompt,
            user=user,
            context=context  # Pass the fetched message history as context
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        # Extract response components
        response_text = response_data.get("response", "")
        token_usage = response_data.get("tokens", {})

        # Get the model used
        model = None
        if ai_config.default_model:
            model = ai_config.default_model

        # Log successful interaction
        _log_interaction_and_update_stats(
            realm,
            user,
            model,
            prompt,
            response_text,
            True,
            response_time_ms=response_time_ms,
            token_usage=token_usage,
        )

        # Send the response back
        if recipient_type == "stream":
            # Reply in the same stream and topic
            from zerver.actions.message_send import internal_send_stream_message_by_name

            internal_send_stream_message_by_name(
                realm,
                bot_profile,
                stream_name,
                topic_name,
                response_text,
            )
        else:
            # Send as a private message
            internal_send_private_message(
                bot_profile,
                user,
                response_text,
            )

        logger.info(f"AI agent chat request processed successfully in {response_time_ms}ms")
