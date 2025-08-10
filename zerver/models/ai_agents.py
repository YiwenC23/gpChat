"""
Django models for AI Agents subsystem
"""
from django.db import models
from django.utils.timezone import now as timezone_now
from django.core.exceptions import ValidationError

from zerver.models.realms import Realm
from zerver.models.users import UserProfile
from zerver.models.streams import Stream
from zerver.models.messages import Message

class AIAgentModel(models.Model):
    """Available AI models in the system"""

    name = models.CharField(max_length=100, unique=True)  # e.g., "llama3.1:8b"
    display_name = models.CharField(max_length=200)  # e.g., "Llama 3.1 8B"
    description = models.TextField(max_length=200, blank=True)
    model_type = models.CharField(max_length=20, default="agent")

    # Model capabilities
    context_length = models.IntegerField(default=4096)
    supports_streaming = models.BooleanField(default=True)
    supports_embeddings = models.BooleanField(default=False)

    # Model status
    is_available = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)

    # Metadata
    size_gb = models.FloatField(null=True, blank=True)
    version = models.CharField(max_length=50, blank=True)
    date_added = models.DateTimeField(default=timezone_now)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "zerver_aiagentmodel"
        indexes = [
            models.Index(fields=["is_available", "is_default"]),
        ]

    def __str__(self) -> str:
        return f"{self.display_name} ({self.name})"


class AIAgentConfig(models.Model):
    """Configuration for AI agents per realm"""

    realm = models.OneToOneField(Realm, on_delete=models.CASCADE, related_name="ai_agent_config")

    # General settings
    enabled = models.BooleanField(default=False)
    default_model = models.ForeignKey(
        AIAgentModel,
        on_delete=models.SET_NULL,
        null=True,
        related_name="default_for_realms"
    )

    # AI assistant enabled
    agent_enabled = models.BooleanField(default=True, help_text="Enable the AI assistant for this realm")

    # Usage limits
    max_requests_per_user_per_day = models.IntegerField(default=100)
    max_context_length = models.IntegerField(default=4096)

    # Model preferences
    chat_model = models.ForeignKey(
        AIAgentModel,
        on_delete=models.SET_NULL,
        null=True,
        related_name="chat_for_realms"
    )
    embedding_model = models.ForeignKey(
        AIAgentModel,
        on_delete=models.SET_NULL,
        null=True,
        related_name="embedding_for_realms"
    )

    # Advanced settings
    temperature = models.FloatField(default=0.7)
    top_p = models.FloatField(default=0.9)

    # Timestamps
    date_created = models.DateTimeField(default=timezone_now)
    last_modified = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "zerver_aiagentconfig"

    def clean(self) -> None:
        """Validate model configuration"""
        if self.temperature < 0 or self.temperature > 1:
            raise ValidationError("Temperature must be between 0 and 1")
        if self.top_p < 0 or self.top_p > 1:
            raise ValidationError("Top_p must be between 0 and 1")

    def __str__(self) -> str:
        return f"AI Config for {self.realm.name}"


class AIAgentInteraction(models.Model):
    """Log of AI agent interactions"""

    realm = models.ForeignKey(Realm, on_delete=models.CASCADE)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    model = models.ForeignKey(AIAgentModel, on_delete=models.CASCADE)

    # For simple interactions, store prompt and response
    # For conversations, these store the initial prompt and final response
    prompt = models.TextField()
    response = models.TextField()

    # Context
    context_type = models.CharField(max_length=50, blank=True)  # "message", "stream", "dm", etc.
    context_id = models.IntegerField(null=True, blank=True)  # Message ID, stream ID, etc.

    # Status
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)

    # Timestamps
    timestamp = models.DateTimeField(default=timezone_now)

    class Meta:
        db_table = "zerver_aiagentinteraction"
        indexes = [
            models.Index(fields=["realm", "user", "timestamp"]),
            models.Index(fields=["success", "timestamp"]),
        ]

    def __str__(self) -> str:
        return f"{self.user.full_name} - {self.timestamp}"


class AIAgentUsageStats(models.Model):
    """Token usage statistics for AI agents"""

    interaction = models.OneToOneField(
        AIAgentInteraction, on_delete=models.CASCADE, related_name="usage_stats"
    )
    realm = models.ForeignKey(Realm, on_delete=models.CASCADE)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, null=True, blank=True)

    # Date/Time period
    date = models.DateField()
    hour = models.IntegerField(default=0)  # 0-23 for hourly stats

    # Token usage metrics
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)

    # Performance metrics
    response_time_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "zerver_aiagentusagestats"
        # Removed unique_together constraint - OneToOneField to interaction already ensures uniqueness
        indexes = [
            models.Index(fields=["realm", "date"]),
            models.Index(fields=["user", "date"]),
        ]

    def __str__(self) -> str:
        user_str = f" - {self.user.full_name}" if self.user else ""
        return f"{self.realm.name}{user_str} - {self.date}"


class AIAgentAuditLog(models.Model):
    """Audit log for AI agent administrative actions"""

    realm = models.ForeignKey(Realm, on_delete=models.CASCADE)
    acting_user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, null=True, blank=True)

    # Event details
    event_type = models.CharField(max_length=50)
    event_time = models.DateTimeField(default=timezone_now)

    # Additional fields specific to AI agents
    model_name = models.CharField(max_length=100, blank=True)
    configuration_changes = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "zerver_aiagentauditlog"
        indexes = [
            models.Index(fields=["realm", "event_time"]),
            models.Index(fields=["event_type", "event_time"]),
        ]

    def __str__(self) -> str:
        return f"AI Agent Audit: {self.event_type} - {self.realm.name}"
