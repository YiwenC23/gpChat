"""
AI Agents API views for Zulip integration
Provides REST endpoints for AI agent interactions
"""
import logging
from typing import List, Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from django.utils.timezone import now as timezone_now
from pydantic import Json

from analytics.lib.counts import COUNT_STATS
from zerver.lib.exceptions import JsonableError
from zerver.lib.queue import queue_json_publish_rollback_unsafe
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint, typed_endpoint_without_parameters
from zerver.models import (
    AIAgentConfig,
    AIAgentInteraction,
    AIAgentUsageStats,
    AIAgentModel,
    Message,
    UserProfile,
)

logger = logging.getLogger(__name__)


@typed_endpoint
def chat_with_ai_agent(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    message: str,
    context_type: Optional[str] = None,
    context_id: Optional[int] = None,
) -> HttpResponse:
    """Send a message to the AI assistant and get a response
    """

    # Check if AI agents are enabled
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))

    # Check realm configuration
    try:
        ai_config = AIAgentConfig.objects.get(realm=user_profile.realm)
        if not ai_config.enabled:
            raise JsonableError(_("AI agents are not enabled for this organization."))

        # Check if AI is enabled (single agent approach)
        if not ai_config.agent_enabled:
            raise JsonableError(_("AI assistant is not enabled."))

    except AIAgentConfig.DoesNotExist:
        raise JsonableError(_("AI agents are not configured for this organization."))

    # Check user permissions (similar to topic summarization)
    if not user_profile.can_use_ai_features():
        raise JsonableError(_("Insufficient permission to use AI features."))

    # Check usage limits
    if getattr(settings, 'MAX_PER_USER_MONTHLY_AI_COST', None) is not None:
        used_credits = COUNT_STATS["ai_credit_usage::day"].current_month_accumulated_count_for_user(
            user_profile
        )
        if used_credits >= settings.MAX_PER_USER_MONTHLY_AI_COST * 1000000000:
            raise JsonableError(_("Reached monthly limit for AI credits."))

    # Check daily request limit
    today_interactions = AIAgentInteraction.objects.filter(
        realm=user_profile.realm,
        user=user_profile,
        timestamp__date=timezone_now().date()
    ).count()

    if today_interactions >= ai_config.max_requests_per_user_per_day:
        raise JsonableError(_("Daily AI request limit reached."))

    # Validate message length
    max_length = getattr(settings, 'AI_AGENTS_MAX_MESSAGE_LENGTH', 5000)
    if len(message) > max_length:
        raise JsonableError(_("Message too long. Maximum length is {max_length} characters."))

    # Queue the AI request for async processing
    event = {
        "task_type": "chat",  # Consolidated worker
        "realm_id": user_profile.realm.id,
        "user_id": user_profile.id,
        "prompt": message,
        "context_type": context_type or "",
        "context_id": context_id,
        "callback_type": "api_response",
        "request_id": f"{user_profile.id}_{int(timezone_now().timestamp())}",
    }

    queue_json_publish_rollback_unsafe("ai_agents", event)

    return json_success(request, {
        "message": "AI agent request queued successfully",
        "request_id": event["request_id"],
    })


@typed_endpoint
def generate_embeddings(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    text: str,
    purpose: str = "general",
) -> HttpResponse:
    """Generate embeddings for text"""

    # Check if AI agents are enabled
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))

    try:
        ai_config = AIAgentConfig.objects.get(realm=user_profile.realm)
        if not ai_config.enabled:
            raise JsonableError(_("AI agents are not enabled for this organization."))
    except AIAgentConfig.DoesNotExist:
        raise JsonableError(_("AI agents are not configured for this organization."))

    # Check permissions
    if not user_profile.can_use_ai_features():
        raise JsonableError(_("Insufficient permission to use AI features."))

    # Validate text length
    # TODO: Do not limit the length of the text to embed.
    max_length = getattr(settings, "AI_AGENTS_MAX_EMBEDDING_TEXT_LENGTH", 10000)
    if len(text) > max_length:
        raise JsonableError(_("Text too long for embedding. Maximum length is {max_length} characters."))

    if len(text.strip()) == 0:
        raise JsonableError(_("Cannot generate embeddings for empty text."))

    # Queue embedding request (unified worker)
    event = {
        "task_type": "embeddings",  # Consolidated worker
        "realm_id": user_profile.realm.id,
        "user_id": user_profile.id,
        "text": text,
        "purpose": purpose,
        "callback_type": "api_response",
        "request_id": f"emb_{user_profile.id}_{int(timezone_now().timestamp())}"
    }

    queue_json_publish_rollback_unsafe("embedded_bots", event)

    return json_success(request, {
        "message": "Embedding generation request queued successfully",
        "request_id": event["request_id"],
    })


@typed_endpoint_without_parameters
def list_ai_models(
    request: HttpRequest,
    user_profile: UserProfile,
) -> HttpResponse:
    """List available AI models"""

    # Check if AI agents are enabled
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))

    try:
        ai_config = AIAgentConfig.objects.get(realm=user_profile.realm)
        if not ai_config.enabled:
            raise JsonableError(_("AI agents are not enabled for this organization."))
    except AIAgentConfig.DoesNotExist:
        raise JsonableError(_("AI agents are not configured for this organization."))

    # Get available models
    models = AIAgentModel.objects.filter(is_available=True).order_by('name')

    model_data = []
    for model in models:
        model_data.append({
            "id": model.id,
            "name": model.name,
            "display_name": model.display_name,
            "description": model.description,
            "context_length": model.context_length,
            "supports_streaming": model.supports_streaming,
            "supports_embeddings": model.supports_embeddings,
            "is_default": model.is_default,
            "size_gb": model.size_gb,
        })

    return json_success(request, {
        "models": model_data,
        "default_model": ai_config.default_model.name if ai_config.default_model else None,
    })


@typed_endpoint_without_parameters
def get_ai_agent_config(
    request: HttpRequest,
    user_profile: UserProfile,
) -> HttpResponse:
    """Get AI agent configuration for the current realm"""

    # Check if AI agents are enabled
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))

    try:
        ai_config = AIAgentConfig.objects.get(realm=user_profile.realm)
    except AIAgentConfig.DoesNotExist:
        # Return default disabled config
        return json_success(request, {
            "enabled": False,
            "agent_enabled": False,
            "limits": {},
        })

    config_data = {
        "enabled": ai_config.enabled,
        "agent_enabled": ai_config.agent_enabled,
        "limits": {
            "max_requests_per_day": ai_config.max_requests_per_user_per_day,
            "max_context_length": ai_config.max_context_length,
        },
        "models": {
            "default": ai_config.default_model.name if ai_config.default_model else None,
            "chat": ai_config.chat_model.name if ai_config.chat_model else None,
            "embedding": ai_config.embedding_model.name if ai_config.embedding_model else None,
        },
        "settings": {
            "temperature": ai_config.temperature,
            "top_p": ai_config.top_p,
        }
    }

    return json_success(request, config_data)


@typed_endpoint
def get_ai_interaction_history(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    limit: int = 20,
) -> HttpResponse:
    """Get AI agent interaction history for the current user"""

    # Check if AI agents are enabled
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))

    try:
        ai_config = AIAgentConfig.objects.get(realm=user_profile.realm)
        if not ai_config.enabled:
            raise JsonableError(_("AI agents are not enabled for this organization."))
    except AIAgentConfig.DoesNotExist:
        raise JsonableError(_("AI agents are not configured for this organization."))

    # Build query
    usage_stats_query = AIAgentUsageStats.objects.filter(
        realm=user_profile.realm,
        user=user_profile,
    ).select_related("interaction")

    # Limit to prevent too large responses
    if limit > 100:
        limit = 100

    usage_stats = usage_stats_query.order_by("-id")[:limit]

    interaction_data = []
    for stats in usage_stats:
        interaction = stats.interaction
        if interaction and interaction.success:
            data = {
                "id": interaction.id,
                "prompt": interaction.prompt,
                "response": interaction.response,
                "timestamp": interaction.timestamp.isoformat(),
                "response_time_ms": stats.response_time_ms,
                "tokens": {
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.total_tokens,
                },
            }
            interaction_data.append(data)

    return json_success(request, {
        "interactions": interaction_data,
    })
