"""
Queue worker for AI Agents subsystem
Handles asynchronous AI processing tasks
"""
import logging
import time
from typing import Any, Dict, Mapping
from typing_extensions import override

from django.conf import settings
from zerver.lib.ai_agents import get_ai_agent
from zerver.worker.base import QueueProcessingWorker, assign_queue
from zerver.models.ai_agents import AIAgentConfig, AIAgentInteraction, AIAgentModel, AIAgentUsageStats
from zerver.models.messages import Message
from zerver.models.realms import Realm, get_realm
from zerver.models.users import UserProfile, get_user_profile_by_id
from zerver.actions.message_send import internal_send_private_message

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

        # Update usage stats
        now = timezone_now()
        stats, created = AIAgentUsageStats.objects.get_or_create(
            realm=realm,
            user=user,
            date=now.date(),
            hour=now.hour,
            defaults={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "response_time_ms": 0,
            },
        )

        stats.interaction = interaction
        if token_usage:
            stats.prompt_tokens = token_usage.get("prompt_tokens", 0)
            stats.completion_tokens = token_usage.get("completion_tokens", 0)
            stats.total_tokens = token_usage.get("total_tokens", 0)
        stats.response_time_ms = response_time_ms
        stats.save()

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

    def _handle_chat_task(self, message: Dict[str, Any], realm: Realm, user: UserProfile,
                         ai_agent: Any, ai_config: AIAgentConfig, start_time: float) -> None:
        """Handle chat and ReAct tasks"""
        prompt = message["content"]

        # Generate AI response
        response_data = ai_agent.chat(
            message=prompt,
            user=user,
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

        # Send the response back to the user
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
