"""
Ollama Client for Zulip AI Agents

This module provides the low-level interface to Ollama API.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class OllamaGenerateResponse:
    """Response from Ollama generate API including token counts"""
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context: Optional[List[int]] = None

    @property
    def token_usage(self) -> Dict[str, int]:
        """Get token usage as a dictionary"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


class StreamingResponse:
    """Wrapper for streaming responses that captures token counts"""

    def __init__(self, response_iterator: Iterator[str]):
        self._iterator = response_iterator
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.context: Optional[List[int]] = None
        self._consumed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    @property
    def token_usage(self) -> Dict[str, int]:
        """Get token usage as a dictionary (available after stream is consumed)"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


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
        stream: bool = True,  # Default to True for streaming
        keep_alive: str = "10m",  # Keep model in memory for 10 minutes
        return_raw: bool = False,  # For backward compatibility
    ) -> Union[OllamaGenerateResponse, StreamingResponse, str, Iterator[str]]:
        """Generate text using specified model

        Args:
            model: The model to use for generation
            prompt: The prompt to generate from
            system: Optional system prompt
            context: Optional context from previous conversation
            temperature: Temperature for generation (0.0 to 1.0)
            stream: Whether to stream the response
            keep_alive: How long to keep the model in memory (e.g., "5m", "10m", "1h")
            return_raw: If True, return raw string (for backward compatibility)

        Returns:
            OllamaGenerateResponse with token counts for non-streaming
            StreamingResponse wrapper for streaming (with token counts available after consumption)
            Raw string/iterator if return_raw=True (for backward compatibility)
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": keep_alive,  # Add keep_alive to keep model in memory
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
                streaming_resp = self._stream_response(response)
                if return_raw:
                    # For backward compatibility, return raw iterator
                    return streaming_resp
                else:
                    # Return wrapped streaming response with token tracking
                    return streaming_resp
            else:
                # Return complete response with token counts
                result = response.json()

                if return_raw:
                    # For backward compatibility
                    return result.get("response", "")

                # Extract token counts from the response
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens

                return OllamaGenerateResponse(
                    response=result.get("response", ""),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    context=result.get("context")
                )
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise OllamaModelError(f"Generation failed: {e}")

    def _stream_response(self, response: requests.Response) -> StreamingResponse:
        """Handle streaming response from Ollama and capture token counts"""
        def generate_chunks():
            prompt_tokens = 0
            completion_tokens = 0
            context = None

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            # Capture token counts from the final message
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            context = data.get("context")
                            break
                    except json.JSONDecodeError:
                        continue

            # Store the final stats in the wrapper
            wrapper.prompt_tokens = prompt_tokens
            wrapper.completion_tokens = completion_tokens
            wrapper.total_tokens = prompt_tokens + completion_tokens
            wrapper.context = context
            wrapper._consumed = True

        # Create wrapper with the generator
        wrapper = StreamingResponse(generate_chunks())
        return wrapper

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

    def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            self._make_request("GET", "tags")
            return True
        except Exception:
            return False


# Singleton instance for the default Ollama client
_default_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client instance"""
    global _default_ollama_client
    if _default_ollama_client is None:
        _default_ollama_client = OllamaClient(
            getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        )
    return _default_ollama_client
