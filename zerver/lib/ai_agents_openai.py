"""
Zulip AI Agents Integration Library with OpenAI support
Provides interface between Zulip and OpenAI API with vector database integration
"""
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Union

from django.conf import settings

from zerver.models import Realm, UserProfile
from zproject.vector_db_mock import VectorStore
from zerver.lib.openai_client import OpenAIClient

class Document:
    """Document class for vector database"""
    def __init__(self, content, embedding, metadata):
        self.content = content
        self.embedding = embedding
        self.metadata = metadata


logger = logging.getLogger(__name__)

# Shared instance of the vector store for the application
_SHARED_VECTOR_STORE = None

def get_shared_vector_store() -> VectorStore:
    """Get or create a shared instance of the vector store"""
    global _SHARED_VECTOR_STORE
    if _SHARED_VECTOR_STORE is None:
        _SHARED_VECTOR_STORE = VectorStore()
        logger.info("Created new shared vector store instance")
    return _SHARED_VECTOR_STORE


class ZulipAIAgent:
    """High-level AI agent for Zulip integration with vector database support using OpenAI"""

    def __init__(self, realm: Realm):
        self.realm = realm
        self.openai = OpenAIClient(
            api_key=getattr(settings, "OPENAI_API_KEY", None),
            base_url=getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "gpt-3.5-turbo")
        self.embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Use shared vector store instance for context retrieval
        # OpenAI Ada embeddings are 1536 dimensions
        self.vector_store = get_shared_vector_store()
        self.context_limit = getattr(settings, "AI_AGENTS_CONTEXT_LIMIT", 5)
        self.context_threshold = getattr(settings, "AI_AGENTS_CONTEXT_THRESHOLD", 0.6)
        logger.info("ZulipAIAgent initialized with shared vector store")

    def chat(
        self,
        message: str,
        user: UserProfile,
        context: Optional[str] = None,
        agent_type: str = "general",
        use_vector_context: bool = True,
    ) -> str:
        """Generate chat response using AI agent with vector database context"""

        # Build system prompt based on agent type
        system_prompt = self._get_system_prompt(agent_type, user)

        # Get vector database context if enabled
        vector_context = ""
        if use_vector_context:
            vector_context = self._get_vector_context(message, user)

        # Combine all context
        combined_context = ""
        if context:
            combined_context += f"User Context: {context}\n\n"
        if vector_context:
            combined_context += f"Similar Messages Context:\n{vector_context}\n\n"

        # Build final prompt
        if combined_context:
            prompt = f"{combined_context}User: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"

        try:
            # For welcome_bot, always use vector context and store messages
            if agent_type == "welcome_bot":
                # Store the user message in vector DB for future context
                self.store_message_in_vector_db(
                    message_content=message,
                    metadata={
                        "sender": user.email,
                        "realm": self.realm.string_id,
                        "agent_type": agent_type,
                        "timestamp": str(time.time()),
                        "user_name": user.full_name
                    }
                )
                
                # Force enable vector context for welcome_bot and enhance it
                if not vector_context and use_vector_context:
                    vector_context = self._get_vector_context(message, user)
                    if vector_context:
                        combined_context = f"Similar Messages Context:\n{vector_context}\n\n"
                        prompt = f"{combined_context}User: {message}\nAssistant:"
                
                # Add enhanced system prompt for welcome bot with vector context awareness
                if vector_context:
                    system_prompt += f"""

IMPORTANT: You have access to similar messages from the vector database. Use this context to:
1. Provide more relevant and personalized responses
2. Reference similar discussions or questions that have been asked before
3. Build upon existing conversations in the organization
4. Show understanding of the organization's communication patterns

The similar messages context above shows relevant conversations that may help you provide a better response.
"""

            response = self.openai.generate(
                model=self.default_model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
            )
            
            # Store bot response in vector DB for future context
            if agent_type == "welcome_bot":
                self.store_message_in_vector_db(
                    message_content=response.strip(),
                    metadata={
                        "sender": "welcome-bot@zulip.com",
                        "realm": self.realm.string_id,
                        "agent_type": agent_type,
                        "timestamp": str(time.time()),
                        "bot_response": True,
                        "user_name": user.full_name
                    }
                )
                logger.info(f"Stored bot response in vector DB")
            
            return response.strip()
        except Exception as e:
            logger.error(f"AI chat generation failed for realm {self.realm.id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

    def _get_vector_context(self, message: str, user: UserProfile) -> str:
        """Get relevant context from vector database with conversation history"""
        try:
            # First, try embedding search if possible
            embedding = self.generate_embeddings(message)
            if embedding:
                similar_messages = self.vector_store.search_by_vector(
                    "zulip_messages",
                    embedding,
                    limit=self.context_limit * 2,  # Retrieve more for filtering
                    threshold=self.context_threshold
                )
            else:
                # Fallback to text search
                similar_messages = self.vector_store.search_by_text(
                    "zulip_messages",
                    message,
                    limit=self.context_limit * 2,  # Retrieve more for filtering
                    threshold=self.context_threshold
                )
            
            if not similar_messages:
                return ""
            
            # Extract conversation threads by grouping by conversation IDs
            conversations = {}
            for msg in similar_messages:
                metadata = msg.get('metadata', {})
                # Use conversation_id or create one from user and timestamp
                conv_id = metadata.get('conversation_id', f"{metadata.get('sender', 'unknown')}_{metadata.get('timestamp', '0')[:5]}")
                
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(msg)
            
            # Sort messages within each conversation by timestamp
            for conv_id in conversations:
                conversations[conv_id].sort(
                    key=lambda x: float(x.get('metadata', {}).get('timestamp', '0')),
                    reverse=False  # Chronological order
                )
            
            # Format context from conversations with enhanced information
            context_parts = []
            user_email = user.email
            user_conversations = []
            bot_conversations = []
            
            # Separate user's conversations from other conversations
            for conv_id, messages in conversations.items():
                has_user_message = any(m.get('metadata', {}).get('sender') == user_email for m in messages)
                if has_user_message:
                    user_conversations.append((conv_id, messages))
                else:
                    bot_conversations.append((conv_id, messages))
            
            # Prioritize user's conversations first
            all_conversations = user_conversations + bot_conversations
            
            # Format each conversation thread
            for conv_id, messages in all_conversations[:min(3, len(all_conversations))]:
                thread_parts = []
                for msg in messages:
                    metadata = msg.get('metadata', {})
                    sender = metadata.get('sender', 'Unknown')
                    content = msg.get('content', '')
                    similarity = msg.get('similarity_score', 0)
                    user_name = metadata.get('user_name', 'Unknown User')
                    is_bot_response = metadata.get('bot_response', False)
                    timestamp = metadata.get('timestamp', '0')
                    
                    # Skip bot responses with low similarity to avoid echo
                    if is_bot_response and similarity < 0.7 and sender != user_email:
                        continue
                    
                    # Format the message
                    if is_bot_response:
                        thread_parts.append(f"Bot: {content[:300]}")
                    else:
                        thread_parts.append(f"{user_name}: {content[:300]}")
                
                if thread_parts:
                    # Add conversation header
                    first_msg = messages[0] if messages else {}
                    first_metadata = first_msg.get('metadata', {})
                    first_sender = first_metadata.get('user_name', 'Unknown')
                    first_time = time.strftime(
                        '%Y-%m-%d', 
                        time.localtime(float(first_metadata.get('timestamp', '0')))
                    )
                    
                    context_parts.append(
                        f"--- Conversation with {first_sender} on {first_time} ---\n" +
                        "\n".join(thread_parts) + "\n"
                    )
            
            # Add a summary for context
            if context_parts:
                header = f"Retrieved {len(context_parts)} relevant conversation threads:\n\n"
                return header + "\n".join(context_parts)
            else:
                return ""
            
        except Exception as e:
            logger.error(f"Failed to get vector context: {e}")
            return ""

    def _get_system_prompt(self, agent_type: str, user: UserProfile) -> str:
        """Get system prompt based on agent type"""
        base_prompt = f"""
            You are a helpful AI assistant integrated into Zulip, a team collaboration platform. \
            You are helping user {user.full_name} in the {self.realm.name} organization. \
            Be helpful, accurate, and concise in your responses.
            
            When provided with similar message context, use it to provide more relevant and contextual responses. \
            Reference the context when appropriate to show understanding of the conversation history.
        """
        
        if agent_type == "general":
            return base_prompt + "\nYou are a general-purpose assistant."
        elif agent_type == "code":
            return base_prompt + "\nYou are a coding assistant. Provide clear, well-documented code examples."
        elif agent_type == "writing":
            return base_prompt + "\nYou are a writing assistant. Help with grammar, style, and content improvement."
        else:
            return base_prompt + f"\nYou are a {agent_type} assistant."

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text similarity/search"""
        try:
            return self.openai.embed(self.embedding_model, text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def _format_messages_for_ai(self, messages: List[Dict[str, Any]]) -> str:
        """Format Zulip messages for AI processing"""
        formatted = []
        for msg in messages:
            sender = msg.get("sender_full_name", "Unknown")
            content = msg.get("content", "")
            formatted.append(f"{sender}: {content}")
        return "\n".join(formatted)

    def is_healthy(self) -> bool:
        """Check if AI agents system is operational"""
        return self.openai.health_check()

    def store_message_in_vector_db(self, message_content: str, metadata: Dict[str, Any]) -> bool:
        """Store a message in the vector database for future context retrieval"""
        try:
            # Generate embedding
            logger.info(f"Generating embedding for message: '{message_content[:30]}...'")
            embedding = self.generate_embeddings(message_content)
            if not embedding:
                logger.warning("Failed to generate embedding for message")
                return False
            else:
                logger.info(f"Generated embedding with {len(embedding)} dimensions")
            
            # Add conversation tracking
            sender_email = metadata.get('sender', '')
            user_name = metadata.get('user_name', '')
            timestamp = metadata.get('timestamp', str(time.time()))
            logger.info(f"Processing message from {user_name} ({sender_email})")
            
            # Create conversation ID if not provided
            if 'conversation_id' not in metadata:
                # Try to find existing conversation with this user in the last hour
                user_convs = self.search_user_conversations(sender_email, hours=1)
                
                if user_convs:
                    # Use the most recent conversation ID
                    metadata['conversation_id'] = user_convs[0].get('metadata', {}).get('conversation_id')
                    logger.info(f"Found existing conversation: {metadata['conversation_id']}")
                else:
                    # Generate a new conversation ID
                    metadata['conversation_id'] = f"conv_{sender_email}_{int(float(timestamp))}"
                    logger.info(f"Created new conversation: {metadata['conversation_id']}")
            
            # Create document
            doc = Document(
                content=message_content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Store in vector database
            logger.info(f"Storing message in vector DB collection 'zulip_messages'")
            success = self.vector_store.insert("zulip_messages", doc.content, doc.embedding, doc.metadata)
            if success:
                logger.info("Successfully stored message in vector DB")
                # Debug: show collections and count
                if hasattr(self.vector_store, 'collections'):
                    for coll_name, docs in self.vector_store.collections.items():
                        logger.info(f"Collection '{coll_name}' has {len(docs)} documents")
            return success
            
        except Exception as e:
            logger.error(f"Failed to store message in vector DB: {e}")
            return False
            
    def search_user_conversations(self, user_email: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Search for recent conversations with a specific user"""
        try:
            if not user_email:
                return []
                
            # Calculate cutoff time (hours ago)
            cutoff_time = time.time() - (hours * 3600)
            
            # Get all messages in the vector store for the user's realm
            if hasattr(self.vector_store, 'collections') and 'zulip_messages' in self.vector_store.collections:
                messages = self.vector_store.collections['zulip_messages']
                
                # Filter by user and time
                user_messages = [
                    msg for msg in messages
                    if msg.get('metadata', {}).get('sender') == user_email
                    and float(msg.get('metadata', {}).get('timestamp', '0')) > cutoff_time
                ]
                
                # Sort by timestamp (newest first)
                user_messages.sort(
                    key=lambda x: float(x.get('metadata', {}).get('timestamp', '0')),
                    reverse=True
                )
                
                return user_messages
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to search user conversations: {e}")
            return []

    def search_similar_messages(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar messages in the vector database"""
        try:
            return self.vector_store.search_by_text("zulip_messages", query, limit=limit)
        except Exception as e:
            logger.error(f"Failed to search similar messages: {e}")
            return []


# Singleton instance for the default OpenAI client
_default_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get singleton OpenAI client instance"""
    global _default_openai_client
    if _default_openai_client is None:
        _default_openai_client = OpenAIClient()
    return _default_openai_client


def get_ai_agent(realm: Realm) -> ZulipAIAgent:
    """Get AI agent instance for a realm"""
    return ZulipAIAgent(realm)
