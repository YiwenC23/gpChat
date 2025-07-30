"""
AI Agents API endpoints for Zulip-Ollama integration
"""
import logging
from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from pydantic import Json

from zerver.decorator import authenticated_json_view
from zerver.lib.ai_agents import ZulipAIAgent, OllamaConnectionError, OllamaModelError
from zerver.lib.exceptions import JsonableError
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint, typed_endpoint_without_parameters
from zerver.models import UserProfile

logger = logging.getLogger(__name__)


@authenticated_json_view
@typed_endpoint
def ai_chat(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    message: str,
    context: Json[str | None] = None,
    agent_type: Json[str] = "general",
    model: Json[str | None] = None,
) -> HttpResponse:
    """Chat with AI agent using Ollama"""
    
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))
    
    if not message.strip():
        raise JsonableError(_("Message cannot be empty."))
    
    # Check message length limit
    max_length = getattr(settings, "AI_AGENTS_MAX_MESSAGE_LENGTH", 5000)
    if len(message) > max_length:
        raise JsonableError(
            _(f"Message too long. Maximum length is {max_length} characters.")
        )
    
    try:
        # Create AI agent instance
        ai_agent = ZulipAIAgent(user_profile.realm)
        
        # Check if AI system is healthy
        if not ai_agent.is_healthy():
            raise JsonableError(_("AI service is currently unavailable. Please try again later."))
        
        # Generate response
        response = ai_agent.chat(
            message=message,
            user=user_profile,
            context=context,
            agent_type=agent_type,
        )
        
        return json_success(request, {
            "response": response,
            "model": model or ai_agent.default_model,
            "agent_type": agent_type,
        })
        
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error for user {user_profile.id}: {e}")
        raise JsonableError(_("Unable to connect to AI service. Please try again later."))
        
    except OllamaModelError as e:
        logger.error(f"Ollama model error for user {user_profile.id}: {e}")
        raise JsonableError(_("AI model error. Please try again later."))
        
    except Exception as e:
        logger.error(f"Unexpected error in AI chat for user {user_profile.id}: {e}")
        raise JsonableError(_("An unexpected error occurred. Please try again later."))


@authenticated_json_view
@typed_endpoint_without_parameters
def ai_health_check(
    request: HttpRequest,
    user_profile: UserProfile,
) -> HttpResponse:
    """Check AI agent system health"""
    
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        return json_success(request, {
            "enabled": False,
            "status": "disabled",
            "message": "AI agents are not enabled on this server."
        })
    
    try:
        ai_agent = ZulipAIAgent(user_profile.realm)
        is_healthy = ai_agent.is_healthy()
        
        return json_success(request, {
            "enabled": True,
            "status": "healthy" if is_healthy else "unhealthy",
            "models": {
                "default": ai_agent.default_model,
                "embedding": ai_agent.embedding_model,
            },
            "ollama_url": getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return json_success(request, {
            "enabled": True,
            "status": "error",
            "message": str(e),
        })


@authenticated_json_view
@typed_endpoint_without_parameters
def ai_models(
    request: HttpRequest,
    user_profile: UserProfile,
) -> HttpResponse:
    """List available AI models"""
    
    if not getattr(settings, "AI_AGENTS_ENABLED", False):
        raise JsonableError(_("AI agents are not enabled on this server."))
    
    try:
        ai_agent = ZulipAIAgent(user_profile.realm)
        models = ai_agent.ollama.list_models()
        
        return json_success(request, {
            "models": models,
            "default_model": ai_agent.default_model,
            "embedding_model": ai_agent.embedding_model,
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise JsonableError(_("Unable to retrieve model information.")) 