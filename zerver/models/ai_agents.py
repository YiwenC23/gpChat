"""
Django models for AI Agents subsystem
"""
from django.db import models
from django.utils.timezone import now as timezone_now
from django.core.exceptions import ValidationError

from zerver.models.realms import Realm
from zerver.models.users import UserProfile
from zerver.models import Stream


class AIAgentModel(models.Model):
    """Available AI models in the system"""
    
    name = models.CharField(max_length=100, unique=True)  # e.g., "llama3.1:8b"
    display_name = models.CharField(max_length=200)  # e.g., "Llama 3.1 8B"
    description = models.TextField(max_length=200, blank=True)
    model_type = models.CharField(max_length=50)  # "chat", "embedding", "code", etc.
    
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
            models.Index(fields=["model_type"]),
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
    
    # Agent types enabled
    enable_chat_agent = models.BooleanField(default=True)
    
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
    
    # Interaction details
    agent_type = models.CharField(max_length=50)  # "chat" agent for now.
    prompt = models.TextField()
    response = models.TextField()
    
    # Context
    context_type = models.CharField(max_length=50, blank=True)  # "message", "stream", "dm", etc.
    context_id = models.IntegerField(null=True, blank=True)  # Message ID, stream ID, etc.
    
    # Performance metrics
    token_count_input = models.IntegerField(null=True, blank=True)
    token_count_output = models.IntegerField(null=True, blank=True)
    response_time_ms = models.IntegerField(null=True, blank=True)
    
    # Status
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)
    
    # Timestamps
    timestamp = models.DateTimeField(default=timezone_now)
    
    class Meta:
        db_table = "zerver_aiagentinteraction"
        indexes = [
            models.Index(fields=["realm", "user", "timestamp"]),
            models.Index(fields=["agent_type", "timestamp"]),
            models.Index(fields=["success", "timestamp"]),
        ]
    
    def get_context_object(self):
        """Return the related context object based on context_type"""
        if self.context_type == "message":
            return Message.objects.get(id=self.context_id)
        elif self.context_type == "stream":
            return Stream.objects.get(id=self.context_id)
        return None

    def __str__(self) -> str:
        return f"{self.user.full_name} - {self.agent_type} - {self.timestamp}"


class AIAgentChannelConversation(models.Model):
    """AI Agent participation in group chat channels"""
    
    realm = models.ForeignKey(Realm, on_delete=models.CASCADE)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    
    # Channel information
    stream = models.ForeignKey(Stream, on_delete=models.CASCADE)
    topic = models.CharField(max_length=200)

    # Conversation metadata
    title = models.CharField(max_length=200, blank=True)
    agent_type = models.CharField(max_length=50)
    
    # Agent behavior settings
    auto_reply = models.BooleanField(default=False)
    mention_only = models.BooleanField(default=True)  # Only respond when mentioned

    # Context handling
    context_window_size = models.IntegerField(default=10)  # Number of messages to include as context
    include_thread_context = models.BooleanField(default=True)
    
    # State
    is_active = models.BooleanField(default=True)
    last_processed_message_id = models.IntegerField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(default=timezone_now)
    last_activity = models.DateTimeField(default=timezone_now)
    
    class Meta:
        db_table = "zerver_aiagentchannelconversation"
        unique_together = ("stream", "topic", "agent_type")
        indexes = [
            models.Index(fields=["realm", "stream", "topic"]),
            models.Index(fields=["stream", "is_active"]),
            models.Index(fields=["last_activity"]),
        ]
        
    
    def __str__(self) -> str:
        return f"AI Agent in {self.stream.name} > {self.topic}"
    
    def should_process_message(self, message: "Message") -> bool:
        """Determine if the agent should process a given message"""
        if not self.is_active:
            return False
        if self.mention_only and not self.is_mentioned(message):
            return False
        return True
    
    def is_mentioned(self, message: "Message") -> bool:
        """Check if the AI agent is mentioned in the message"""
        if self.user in message.mentioned_users.all():
            return True
        if self.stream and message.stream_id == self.stream.id:
            return message.content.startswith(f"@{self.user.full_name}") or \
                   message.content.startswith(f"@**{self.user.full_name}**")
        return False


class AIAgentMessage(models.Model):
    """Individual messages in AI agent conversations"""
    conversation = models.ForeignKey(
        "AIAgentConversation", 
        on_delete=models.CASCADE, 
        related_name="messages"
    )
    
    # Message details
    role = models.CharField(max_length=20)  # "user", "assistant", "system"
    content = models.TextField()
    
    # AI response metadata (for assistant messages)
    model = models.ForeignKey(AIAgentModel, on_delete=models.SET_NULL, null=True, blank=True)
    token_count = models.IntegerField(null=True, blank=True)
    response_time_ms = models.IntegerField(null=True, blank=True)
    
    # Timestamps
    timestamp = models.DateTimeField(default=timezone_now)
    
    class Meta:
        db_table = "zerver_aiagentmessage"
        indexes = [
            models.Index(fields=["conversation", "timestamp"]),
            models.Index(fields=["role", "timestamp"]),
        ]
        ordering = ["timestamp"]
    
    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}..."


class AIAgentUsageStats(models.Model):
    """Daily usage statistics for AI agents"""
    
    realm = models.ForeignKey(Realm, on_delete=models.CASCADE)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, null=True, blank=True)
    
    # Date
    date = models.DateField()
    
    # Usage counts
    total_interactions = models.IntegerField(default=0)
    chat_interactions = models.IntegerField(default=0)
    code_interactions = models.IntegerField(default=0)
    summarization_interactions = models.IntegerField(default=0)
    
    # Token usage
    total_tokens_input = models.IntegerField(default=0)
    total_tokens_output = models.IntegerField(default=0)
    
    # Performance
    avg_response_time_ms = models.FloatField(null=True, blank=True)
    success_rate = models.FloatField(null=True, blank=True)
    
    class Meta:
        db_table = "zerver_aiagentusagestats"
        unique_together = ("realm", "user", "date")
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
    agent_type = models.CharField(max_length=50, blank=True)
    configuration_changes = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = "zerver_aiagentauditlog"
        indexes = [
            models.Index(fields=["realm", "event_time"]),
            models.Index(fields=["event_type", "event_time"]),
        ]
    
    def __str__(self) -> str:
        return f"AI Agent Audit: {self.event_type} - {self.realm.name}"