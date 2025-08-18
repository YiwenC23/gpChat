"""
Zulip AI Agents Integration Library
Provides interface between Zulip and local Ollama AI models with vector database integration
"""
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from django.conf import settings
import textwrap

from zerver.models import Realm, UserProfile
from zproject.vector_db_mock import VectorStore

class Document:
    """Mock Document class for vector database"""
    def __init__(self, content, embedding, metadata):
        self.content = content
        self.embedding = embedding
        self.metadata = metadata


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama fails"""
    pass


class OllamaModelError(Exception):
    """Raised when model operations fail"""
    pass

class OllamaClient:
    """Client for interacting with local Ollama installation"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to Ollama API with error handling"""
        url = f"{self.base_url}/api/{endpoint}"
        
        # For generate requests, disable timeout.
        if endpoint == "generate":
            kwargs["timeout"] = 3000000  # or set to a very high value

        try:
            # Log the request details for debugging (only in debug mode)
            if "json" in kwargs and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Making {method} request to {url} with payload: {kwargs['json']}")
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            # Try to get more details from the response
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error response: {error_detail}")
                    # Check for specific error types
                    if "error" in error_detail:
                        error_msg = error_detail["error"]
                        if "requires more system memory" in error_msg:
                            raise OllamaModelError(f"Insufficient memory: {error_msg}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text using specified model"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }

        if system:
            payload["system"] = system
        if context:
            payload["context"] = context

        try:
            response = self._make_request("POST", "generate", json=payload)

            if stream:
                # Return streaming response
                return self._stream_response(response)
            else:
                # Return complete response
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise OllamaModelError(f"Generation failed: {e}")

    def _stream_response(self, response: requests.Response):
        """Handle streaming response from Ollama"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    def embed(self, model: str, text: str) -> List[float]:
        """Generate embeddings for text"""
        payload = {
            "model": model,
            "prompt": text,
        }

        try:
            response = self._make_request("POST", "embeddings", json=payload)
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise OllamaModelError(f"Embedding failed: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = self._make_request("GET", "tags")
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Download a model"""
        payload = {"name": model}

        try:
            response = self._make_request("POST", "pull", json=payload, timeout=3600)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    def delete_model(self, model: str) -> bool:
        """Delete a model"""
        payload = {"name": model}

        try:
            response = self._make_request("DELETE", "delete", json=payload)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to delete model {model}: {e}")
            return False

    def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            self._make_request("GET", "tags")
            return True
        except Exception:
            return False


class ZulipAIAgent:
    """High-level AI agent for Zulip integration with vector database support"""

    def __init__(self, realm: Realm):
        self.realm = realm
        self.ollama = OllamaClient(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
        self.default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
        self.embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
        
        # Initialize vector store for context retrieval
        # Use 768 dimensions for Ollama embeddings
        self.vector_store = VectorStore(embedding_dimension=768)
        self.context_limit = getattr(settings, "AI_AGENTS_CONTEXT_LIMIT", 5)
        self.context_threshold = getattr(settings, "AI_AGENTS_CONTEXT_THRESHOLD", 0.6)

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

            response = self.ollama.generate(
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
            
            return response.strip()
        except Exception as e:
            logger.error(f"AI chat generation failed for realm {self.realm.id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

    def _get_vector_context(self, message: str, user: UserProfile) -> str:
        """Get relevant context from vector database"""
        try:
            # Search for similar messages in vector database
            similar_messages = self.vector_store.search_by_text(
                "zulip_messages",
                message,
                limit=self.context_limit,
                threshold=self.context_threshold
            )
            
            if not similar_messages:
                return ""
            
            # Format context from similar messages with enhanced information
            context_parts = []
            for msg in similar_messages:
                sender = msg.get('metadata', {}).get('sender', 'Unknown')
                subject = msg.get('metadata', {}).get('subject', 'No subject')
                content = msg.get('content', '')
                similarity = msg.get('similarity_score', 0)
                user_name = msg.get('metadata', {}).get('user_name', 'Unknown User')
                is_bot_response = msg.get('metadata', {}).get('bot_response', False)
                
                # Skip bot responses with low similarity to avoid echo
                if is_bot_response and similarity < 0.8:
                    continue
                
                # Format the context entry
                if is_bot_response:
                    context_parts.append(
                        f"Previous bot response (similarity: {similarity:.2f}):\n"
                        f"To: {user_name}\n"
                        f"Content: {content[:300]}...\n"
                    )
                else:
                    context_parts.append(
                        f"Similar user message (similarity: {similarity:.2f}):\n"
                        f"From: {user_name} ({sender})\n"
                        f"Subject: {subject}\n"
                        f"Content: {content[:300]}...\n"
                    )
            
            if context_parts:
                return "\n".join(context_parts)
            else:
                return ""
            
        except Exception as e:
            logger.error(f"Failed to get vector context: {e}")
            return ""

    def _get_system_prompt(self, agent_type: str, user: UserProfile) -> str:
        """Get system prompt based on agent type"""
        base_prompt = textwrap.dedent(f"""
            You are a helpful AI assistant integrated into Zulip, a team collaboration platform. \
            You are helping user {user.full_name} in the {self.realm.name} organization. \
            Be helpful, accurate, and concise in your responses.
            
            When provided with similar message context, use it to provide more relevant and contextual responses. \
            Reference the context when appropriate to show understanding of the conversation history.
        """)
        
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
            return self.ollama.embed(self.embedding_model, text)
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
        return self.ollama.health_check()

    def store_message_in_vector_db(self, message_content: str, metadata: Dict[str, Any]) -> bool:
        """Store a message in the vector database for future context retrieval"""
        try:
            # Generate embedding
            embedding = self.generate_embeddings(message_content)
            if not embedding:
                return False
            
            # Create document
            doc = Document(
                content=message_content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Store in vector database
            self.vector_store.insert("zulip_messages", doc.content, doc.embedding, doc.metadata)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store message in vector DB: {e}")
            return False

    def search_similar_messages(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar messages in the vector database"""
        try:
            return self.vector_store.search_by_text("zulip_messages", query, limit=limit)
        except Exception as e:
            logger.error(f"Failed to search similar messages: {e}")
            return []


# Singleton instance for the default Ollama client
_default_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client instance"""
    global _default_ollama_client
    if _default_ollama_client is None:
        _default_ollama_client = OllamaClient()
    return _default_ollama_client


def get_ai_agent(realm: Realm) -> ZulipAIAgent:
    """Get AI agent instance for a realm"""
    return ZulipAIAgent(realm)
