"""
AI Agent Tools for Zulip

This module defines tools that the AI agent can use to perform actions on behalf of users.
Each tool is a class that implements the AIAgentTool interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from django.db import transaction
from django.utils.translation import gettext as _

from zerver.actions.message_send import check_send_stream_message
from zerver.actions.streams import bulk_add_subscriptions
from zerver.lib.exceptions import (
    JsonableError,
    MessagesNotAllowedInEmptyTopicError,
    StreamDoesNotExistError,
    StreamWildcardMentionNotAllowedError,
    TopicsNotAllowedError,
    TopicWildcardMentionNotAllowedError,
)
from zerver.lib.streams import (
    check_stream_name,
    create_stream_if_needed,
)
from zerver.models import Realm, UserProfile
from zerver.models.clients import get_client


logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Defines a parameter for an AI agent tool."""
    name: str
    type: str  # "string", "boolean", "integer", "enum"
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[str]] = None  # For enum types
    min_value: Optional[Union[int, float]] = None  # For numeric types
    max_value: Optional[Union[int, float]] = None  # For numeric types


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: str  # "success", "error", "permission_denied", "already_exists"
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error_code: Optional[str] = None


class AIAgentTool(ABC):
    """Base class for AI agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        pass

    @abstractmethod
    def execute(
        self,
        requesting_user: UserProfile,
        realm: Realm,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            requesting_user: The user requesting the action
            realm: The realm in which to execute the action
            parameters: The parameters for the tool

        Returns:
            ToolResult with the outcome of the execution
        """
        pass

    def coerce_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce parameters to their expected types, handling common LLM output variations.
        Returns a new dict with coerced values.
        """
        coerced = {}
        for key, value in parameters.items():
            # Find the parameter definition
            param_def = None
            for param in self.parameters:
                if param.name == key:
                    param_def = param
                    break

            if param_def is None:
                # Unknown parameter, pass through as-is
                coerced[key] = value
                continue

            # Apply type coercion based on expected type
            if param_def.type == "string":
                if value is not None:
                    # Trim whitespace from strings
                    coerced[key] = str(value).strip()
                else:
                    coerced[key] = value
            elif param_def.type == "boolean":
                if isinstance(value, bool):
                    coerced[key] = value
                elif isinstance(value, str):
                    # Handle string representations of booleans
                    value_lower = value.lower().strip()
                    if value_lower in ["true", "yes", "1", "on"]:
                        coerced[key] = True
                    elif value_lower in ["false", "no", "0", "off"]:
                        coerced[key] = False
                    else:
                        coerced[key] = value  # Let validation catch this
                else:
                    coerced[key] = value
            elif param_def.type == "integer":
                if isinstance(value, int):
                    coerced[key] = value
                elif isinstance(value, str):
                    # Try to parse string as integer
                    try:
                        coerced[key] = int(value.strip())
                    except (ValueError, AttributeError):
                        coerced[key] = value  # Let validation catch this
                else:
                    coerced[key] = value
            elif param_def.type == "float":
                if isinstance(value, (int, float)):
                    coerced[key] = float(value)
                elif isinstance(value, str):
                    # Try to parse string as float
                    try:
                        coerced[key] = float(value.strip())
                    except (ValueError, AttributeError):
                        coerced[key] = value  # Let validation catch this
                else:
                    coerced[key] = value
            else:
                # For enum and other types, pass through as-is
                coerced[key] = value

        return coerced

    def validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Validate that all required parameters are present and valid.
        Returns None if valid, or an error message if invalid.
        """
        for param in self.parameters:
            param_value = parameters.get(param.name)

            # Check required parameters
            if param.required and (param_value is None or param_value == ""):
                return f"Missing required parameter: {param.name}"

            # Skip validation for optional parameters that weren't provided
            if not param.required and param_value is None:
                continue

            # Type validation
            if param.type == "string" and not isinstance(param_value, str):
                return f"Parameter {param.name} must be a string"
            elif param.type == "boolean" and not isinstance(param_value, bool):
                return f"Parameter {param.name} must be a boolean"
            elif param.type == "integer" and not isinstance(param_value, int):
                return f"Parameter {param.name} must be an integer"
            elif param.type == "float" and not isinstance(param_value, (int, float)):
                return f"Parameter {param.name} must be a number"
            elif param.type == "enum":
                if param.enum_values and param_value not in param.enum_values:
                    return f"Parameter {param.name} must be one of: {', '.join(param.enum_values)}"

            # Range validation for numeric types
            if param.type in ["integer", "float"]:
                if param.min_value is not None and param_value < param.min_value:
                    return f"Parameter {param.name} must be at least {param.min_value}"
                if param.max_value is not None and param_value > param.max_value:
                    return f"Parameter {param.name} must be at most {param.max_value}"

        return None

    def get_json_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool's parameters.
        This is used by the AI model for function calling.
        """
        properties = {}
        required_params = []

        for param in self.parameters:
            # Map parameter type to JSON Schema type
            json_type = param.type
            if param.type == "float":
                json_type = "number"
            elif param.type == "enum":
                json_type = "string"

            prop_def: Dict[str, Any] = {
                "type": json_type,
                "description": param.description
            }

            if param.type == "enum" and param.enum_values:
                prop_def["enum"] = param.enum_values

            # Only include default if it's not None (avoid "default": null for booleans)
            if param.default is not None:
                prop_def["default"] = param.default

            if param.type in ["integer", "float"]:
                if param.min_value is not None:
                    prop_def["minimum"] = param.min_value
                if param.max_value is not None:
                    prop_def["maximum"] = param.max_value

            properties[param.name] = prop_def

            if param.required:
                required_params.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required_params
        }


class CreateChannelTool(AIAgentTool):
    """Tool for creating Zulip channels (streams)."""

    @property
    def name(self) -> str:
        return "create_channel"

    @property
    def description(self) -> str:
        return (
            "Create a new Zulip channel (stream) with the specified name and settings. "
            "The requesting user will be automatically subscribed to the channel."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="The name of the channel to create",
                required=True
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Description of the channel's purpose",
                required=False,
                default=""
            ),
            ToolParameter(
                name="privacy",
                type="enum",
                description="Privacy setting for the channel",
                required=False,
                default="public",
                enum_values=["public", "private", "web_public"]
            ),
            ToolParameter(
                name="history_public_to_subscribers",
                type="boolean",
                description="Whether new subscribers can see message history (for private channels)",
                required=False,
                default=None
            )
        ]

    def _check_permissions(
        self,
        requesting_user: UserProfile,
        realm: Realm,
        privacy: str
    ) -> Optional[str]:
        """
        Check if the user has permission to create a channel with the given privacy.
        Returns None if permitted, or an error message if not.
        """
        if privacy == "public":
            if not requesting_user.can_create_public_streams(realm):
                return _(
                    "You don't have permission to create public channels in this organization. "
                    "Please contact an administrator."
                )
        elif privacy == "private":
            if not requesting_user.can_create_private_streams(realm):
                return _(
                    "You don't have permission to create private channels in this organization. "
                    "Please contact an administrator."
                )
        elif privacy == "web_public":
            if not realm.web_public_streams_enabled():
                return _("Web-public channels are not enabled in this organization.")
            if not requesting_user.can_create_web_public_streams():
                return _(
                    "You don't have permission to create web-public channels. "
                    "Only organization owners can create web-public channels."
                )

        return None

    @transaction.atomic
    def execute(
        self,
        requesting_user: UserProfile,
        realm: Realm,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute the channel creation."""

        # Coerce parameters to expected types
        coerced_params = self.coerce_parameters(parameters)

        # Validate parameters
        validation_error = self.validate_parameters(coerced_params)
        if validation_error:
            return ToolResult(
                status="error",
                message=validation_error,
                error_code="validation.missing_param" if "Missing" in validation_error else "validation.type"
            )

        # Extract parameters
        name = coerced_params["name"]  # Already trimmed by coerce_parameters
        description = coerced_params.get("description", "")
        privacy = coerced_params.get("privacy", "public")
        history_public_to_subscribers = coerced_params.get("history_public_to_subscribers")

        # Check permissions (explicit web-public gate to provide a distinct error code)
        if privacy == "web_public" and not realm.web_public_streams_enabled():
            return ToolResult(
                status="permission_denied",
                message=_("Web-public channels are not enabled in this organization."),
                error_code="permission.web_public_disabled"
            )
        permission_error = self._check_permissions(requesting_user, realm, privacy)
        if permission_error:
            return ToolResult(
                status="permission_denied",
                message=permission_error,
                error_code="permission.denied"
            )

        # Validate stream name
        try:
            check_stream_name(name)
        except JsonableError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="validation.invalid_name"
            )

        # Map privacy to stream parameters
        invite_only = privacy == "private"
        is_web_public = privacy == "web_public"

        try:
            # Create the stream
            stream, created = create_stream_if_needed(
                realm=realm,
                stream_name=name,
                invite_only=invite_only,
                is_web_public=is_web_public,
                history_public_to_subscribers=history_public_to_subscribers,
                stream_description=description,
                acting_user=requesting_user
            )

            if not created:
                # Channel already exists - include link in message
                stream_url = f"#**{name}**"
                return ToolResult(
                    status="already_exists",
                    message=_("Channel {stream_url} already exists.").format(stream_url=stream_url),
                    data={
                        "stream_id": stream.id,
                        "stream_name": stream.name,
                        "stream_url": stream_url
                    }
                )

            # Subscribe the requesting user to the new stream
            bulk_add_subscriptions(
                realm=realm,
                streams=[stream],
                users=[requesting_user],
                acting_user=requesting_user
            )

            # Generate stream mention link (Zulip's clickable format)
            stream_url = f"#**{name}**"

            return ToolResult(
                status="success",
                message=_("Successfully created channel #**{channel_name}**.").format(channel_name=name),
                data={
                    "stream_id": stream.id,
                    "stream_name": stream.name,
                    "stream_url": stream_url,
                    "description": stream.description,
                    "privacy": privacy,
                    "invite_only": invite_only,
                    "is_web_public": is_web_public
                }
            )

        except JsonableError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="stream.creation_failed"
            )
        except Exception as e:
            logger.exception(f"Unexpected error creating channel: {e}")
            return ToolResult(
                status="error",
                message=_("An unexpected error occurred while creating the channel."),
                error_code="internal.error"
            )


class CreateTopicTool(AIAgentTool):
    """Tool for creating a new topic in a Zulip channel."""

    @property
    def name(self) -> str:
        return "create_topic"

    @property
    def description(self) -> str:
        return (
            "Start a new topic in a Zulip channel (stream) by posting the first message to it."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="channel_name",
                type="string",
                description="The name of the channel where the topic will be created",
                required=True,
            ),
            ToolParameter(
                name="topic_name",
                type="string",
                description="The name of the new topic",
                required=True,
            ),
            ToolParameter(
                name="initial_message",
                type="string",
                description='Optional initial message. Defaults to: "The topic {topic_name} is created by user {user}"',
                required=False,
                default=None,
            ),
        ]

    def execute(
        self,
        requesting_user: UserProfile,
        realm: Realm,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Execute the topic creation by sending the first message."""

        # Coerce and validate parameters
        coerced = self.coerce_parameters(parameters)
        error = self.validate_parameters(coerced)
        if error:
            return ToolResult(
                status="error",
                message=error,
                error_code="validation.error",
            )

        channel_name = coerced["channel_name"].strip()
        topic_name = coerced["topic_name"].strip()
        initial_message = coerced.get("initial_message")
        if isinstance(initial_message, str):
            initial_message = initial_message.strip()

        # Enforce non-empty topic
        if not topic_name:
            return ToolResult(
                status="error",
                message=_("Topic name cannot be empty."),
                error_code="validation.empty_topic",
            )

        # Default message when empty/missing
        if not initial_message:
            user_display = requesting_user.full_name or requesting_user.delivery_email or str(requesting_user.id)
            initial_message = _("The topic {topic} is created by user {user}").format(
                topic=topic_name, user=user_display
            )

        try:
            # Send the message to create the topic
            message_id = check_send_stream_message(
                sender=requesting_user,
                client=get_client("Internal"),
                stream_name=channel_name,
                topic_name=topic_name,
                body=initial_message,
                realm=realm,
            )

            topic_url = f"#**{channel_name}>{topic_name}**"

            return ToolResult(
                status="success",
                message=_("Successfully created topic {topic_url}.").format(topic_url=topic_url),
                data={
                    "channel_name": channel_name,
                    "topic_name": topic_name,
                    "topic_url": topic_url,
                    "message_id": message_id,
                },
            )

        except StreamDoesNotExistError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="stream.not_found",
            )
        except MessagesNotAllowedInEmptyTopicError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="topics.empty_topic_forbidden",
            )
        except TopicsNotAllowedError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="topics.non_empty_topic_forbidden",
            )
        except (StreamWildcardMentionNotAllowedError, TopicWildcardMentionNotAllowedError) as e:
            # Kept for completeness in case future defaults add wildcard mentions
            return ToolResult(
                status="error",
                message=str(e),
                error_code="mention.not_allowed",
            )
        except JsonableError as e:
            return ToolResult(
                status="error",
                message=str(e),
                error_code="message.send_failed",
            )
        except Exception as e:
            logger.exception("Unexpected error creating topic: %s", e)
            return ToolResult(
                status="error",
                message=_("An unexpected error occurred while creating the topic."),
                error_code="internal.error",
            )


# Tool Registry - Easy to extend by adding new tools here
AVAILABLE_TOOLS: Dict[str, AIAgentTool] = {
    "create_channel": CreateChannelTool(),
    "create_topic": CreateTopicTool(),
}


def get_tool_by_name(name: str) -> Optional[AIAgentTool]:
    """Get a tool instance by its name."""
    return AVAILABLE_TOOLS.get(name)


def get_all_tools() -> List[AIAgentTool]:
    """Get all available tools."""
    return list(AVAILABLE_TOOLS.values())


def get_tools_json_schema() -> List[Dict[str, Any]]:
    """
    Get the JSON schema for all available tools.
    This is used for function calling with the AI model.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_json_schema()
        }
        for tool in AVAILABLE_TOOLS.values()
    ]
