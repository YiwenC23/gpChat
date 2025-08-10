"""
Library for provisioning AI agents for realms.

This module provides core functionality for creating and configuring
AI agent bots and their associated models for Zulip realms.
"""

import logging
from typing import Optional

from django.db import IntegrityError

from zerver.actions.create_user import do_create_user
from zerver.models import Realm, UserProfile
from zerver.models.ai_agents import AIAgentConfig, AIAgentModel

logger = logging.getLogger(__name__)


def provision_ai_agent_for_realm(
    realm: Realm,
    bot_name: str = "AI Agent",
    email_prefix: str = "ai-agent-bot",
    seed_config: bool = True,
) -> Optional[UserProfile]:
    """
    Provision an AI agent bot and configuration for a realm.

    This function creates an AI agent bot user and seeds the necessary
    AI model configuration for a realm. It is idempotent and can be
    safely called multiple times.

    Args:
        realm: The realm to provision the AI agent for.
        bot_name: The display name for the AI agent bot.
        email_prefix: The email prefix for the bot's email address.
        seed_config: Whether to seed AI model configuration.

    Returns:
        The created bot UserProfile, or None if the bot already exists
        or an error occurred.
    """
    bot_profile = None

    # Seed AI configuration if requested
    if seed_config:
        try:
            _seed_ai_config_for_realm(realm)
        except Exception as e:
            logger.error(f"Error seeding AI configuration for realm {realm.string_id}: {e}")
            # Continue with bot creation even if config seeding fails

    # Create the AI agent bot
    try:
        bot_profile = _create_ai_bot_for_realm(realm, bot_name, email_prefix)
    except Exception as e:
        logger.error(f"Error creating AI agent bot for realm {realm.string_id}: {e}")

    return bot_profile


def _seed_ai_config_for_realm(realm: Realm) -> None:
    """
    Seed AI model and configuration for a realm.

    This creates or updates the default AI model and realm configuration
    to enable the AI agent functionality.
    """
    # Create or get the default AI model
    model, model_created = AIAgentModel.objects.get_or_create(
        name="llama3.1:8b",
        defaults={
            "display_name": "Llama 3.1 8B",
            "supports_streaming": True,
            "supports_embeddings": False,
            "is_available": True,
            "is_default": True,
        },
    )

    if model_created:
        logger.info(f"Created AI model: {model.display_name}")
    else:
        logger.debug(f"AI model already exists: {model.display_name}")

    # Create or update the realm's AI configuration
    config, config_created = AIAgentConfig.objects.update_or_create(
        realm=realm,
        defaults={
            "enabled": True,
            "agent_enabled": True,
            "default_model": model,
            "chat_model": model,
        },
    )

    if config_created:
        logger.info(f"Created AI config for realm: {realm.string_id}")
    else:
        logger.info(f"Updated AI config for realm: {realm.string_id}")


def _create_ai_bot_for_realm(
    realm: Realm, bot_name: str, email_prefix: str
) -> Optional[UserProfile]:
    """
    Create an AI agent bot for a specific realm.

    Args:
        realm: The realm to create the bot in.
        bot_name: The display name for the bot.
        email_prefix: The email prefix for the bot's email address.

    Returns:
        The created bot UserProfile, or None if the bot already exists.
    """
    email = f"{email_prefix}@{realm.string_id}.zulip.com"

    # Check if bot already exists
    existing_bot = UserProfile.objects.filter(
        realm=realm,
        email__iexact=email,
        is_bot=True,
    ).first()

    if existing_bot:
        logger.info(
            f"AI agent bot '{bot_name}' already exists in realm '{realm.string_id}' "
            f"with email '{existing_bot.email}'"
        )
        return None

    try:
        bot_profile = do_create_user(
            email=email,
            password=None,
            realm=realm,
            full_name=bot_name,
            bot_type=UserProfile.EMBEDDED_BOT,
            bot_owner=None,  # System bot
            acting_user=None,  # Created by system
        )
        logger.info(
            f"Successfully created AI agent bot '{bot_name}' in realm '{realm.string_id}' "
            f"with API key '{bot_profile.api_key}'"
        )
        return bot_profile
    except IntegrityError:
        # Bot might already exist (race condition)
        logger.info(f"AI agent bot '{bot_name}' already exists in realm '{realm.string_id}'")
        return None
    except Exception as e:
        logger.error(f"Error creating AI agent bot in realm '{realm.string_id}': {e}")
        raise
