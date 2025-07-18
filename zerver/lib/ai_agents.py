"""
Zulip AI Agents Integration Library
Provides interface between Zulip and local Ollama AI models
"""
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from django.conf import settings
import textwrap

from zerver.models import Realm, UserProfile


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama fails"""
    pass


class OllamaModelError(Exception):
    """Raised when model operations fail"""
    pass

class OllamaClient:
    """Client for interacting with local Ollama installation"""

    _shared_session = requests.Session()

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = OllamaClient._shared_session
    def _make_request(self, method: str, endpoint: str, timeout: Optional[float] = None, **kwargs) -> requests.Response:
        """
        Make HTTP request to Ollama API with error handling.

        Args:
            method: HTTP method (e.g., 'GET', 'POST').
            endpoint: API endpoint.
            timeout: Optional timeout for the request in seconds.
            **kwargs: Additional arguments for requests.Session.request.
        """
        url = f"{self.base_url}/api/{endpoint}"
        # Set default timeout if not provided
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif "timeout" not in kwargs:
            kwargs["timeout"] = 30
        
        try:
            # Log the request details for debugging (only in debug mode)
            if 'json' in kwargs and logger.isEnabledFor(logging.DEBUG):
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
    """High-level AI agent for Zulip integration"""

    def __init__(self, realm: Realm):
        self.realm = realm
        
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Invalid OLLAMA_BASE_URL setting")
        
        self.ollama = OllamaClient(base_url)
        self.default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
        self.embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "nomic-embed-text:v1.5")

    def chat(
        self,
        message: str,
        user: UserProfile,
        context: Optional[str] = None,
        agent_type: str = "general",
    ) -> str:
        """Generate chat response using AI agent"""

        # Build system prompt based on agent type
        system_prompt = self._get_system_prompt(agent_type, user)

        # Add context if provided
        if context:
            prompt = f"Context: {context}\n\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"

        try:
            response = self.ollama.generate(
                model=self.default_model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"AI chat generation failed for realm {self.realm.id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

    def _get_system_prompt(self, agent_type: str, user: UserProfile) -> str:
        """Get system prompt based on agent type"""
        base_prompt = textwrap.dedent(f"""
            You are a helpful AI assistant integrated into Zulip, a team collaboration platform. \
            You are helping user {user.full_name} in the {self.realm.name} organization. \
            Be helpful, accurate, and concise in your responses.
        """)
        return base_prompt + "\nYou are a general-purpose assistant."

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
