"""
Zulip OpenAI Integration Library
Provides interface between Zulip and OpenAI API
"""
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class OpenAIConnectionError(Exception):
    """Raised when connection to OpenAI fails"""
    pass


class OpenAIModelError(Exception):
    """Raised when model operations fail"""
    pass


class OpenAIClient:
    """Client for interacting with OpenAI API"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        self.base_url = base_url
        self.api_key = api_key or getattr(settings, "OPENAI_API_KEY", None)
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to OpenAI API with error handling"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Log the request details for debugging (only in debug mode)
            if "json" in kwargs and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Making {method} request to {url} with payload: {kwargs['json']}")
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            # Try to get more details from the response
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error response: {error_detail}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            raise OpenAIConnectionError(f"Failed to connect to OpenAI: {e}")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text using OpenAI's chat completion API"""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        try:
            response = self._make_request("POST", "chat/completions", json=payload)
            
            if stream:
                # Return streaming response
                return self._stream_response(response)
            else:
                # Return complete response
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise OpenAIModelError(f"Generation failed: {e}")

    def _stream_response(self, response: requests.Response):
        """Handle streaming response from OpenAI"""
        for line in response.iter_lines():
            if line:
                try:
                    line_text = line.decode("utf-8")
                    # Skip the "data: " prefix
                    if line_text.startswith("data: "):
                        line_text = line_text[6:]
                    
                    # Skip the [DONE] message
                    if line_text.strip() == "[DONE]":
                        break
                        
                    data = json.loads(line_text)
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error parsing stream: {e}")

    def embed(self, model: str, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI's embedding API"""
        payload = {
            "model": model,
            "input": text,
        }

        try:
            response = self._make_request("POST", "embeddings", json=payload)
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise OpenAIModelError(f"Embedding failed: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models"""
        try:
            response = self._make_request("GET", "models")
            result = response.json()
            return result["data"]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            self._make_request("GET", "models")
            return True
        except Exception:
            return False
