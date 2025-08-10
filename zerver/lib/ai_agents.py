"""
Zulip AI Agents Integration Library
Provides interface between Zulip and local Ollama AI models
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
import textwrap

from zerver.lib.ollama_client import OllamaClient, OllamaGenerateResponse
from zerver.lib.ai_agent_tools import Tool, ToolResult, get_available_tools
from zerver.models import Realm, UserProfile


logger = logging.getLogger(__name__)


class ZulipAIAgent:
    """High-level AI agent for Zulip integration"""

    def __init__(self, realm: Realm):
        self.realm = realm
        self.ollama = OllamaClient(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
        self.default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
        self.embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
        self.keep_alive = getattr(settings, "AI_AGENTS_KEEP_ALIVE", "10m")  # Keep model in memory for 10 minutes

    def chat(
        self,
        message: str,
        user: UserProfile,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate chat response using AI agent

        Returns dict with:
        - response: The AI response text
        - tokens: Token usage information
        - conversation_history: Full conversation history
        """

        system_prompt = self._get_system_prompt(user)

        # Add context if provided
        if context:
            prompt = f"Context: {context}\n\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"

        try:
            # Import StreamingResponse for type checking
            from zerver.lib.ollama_client import StreamingResponse

            # Generate response using streaming (new default)
            result = self.ollama.generate(
                model=self.default_model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                keep_alive=self.keep_alive  # Keep model in memory
            )

            # Handle StreamingResponse (new default behavior)
            if isinstance(result, StreamingResponse):
                # Iterate through the stream to build the complete response
                response_text = ""
                for chunk in result:
                    response_text += chunk

                # Token counts are now available after streaming completes
                response_text = response_text.strip()
                token_usage = result.token_usage

            # Handle non-streaming response (when explicitly stream=False is used)
            elif isinstance(result, OllamaGenerateResponse):
                response_text = result.response.strip()
                token_usage = result.token_usage
            else:
                # Fallback for backward compatibility
                response_text = result.strip() if isinstance(result, str) else ""
                # Estimate token usage if not available
                prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
                completion_tokens = len(response_text.split()) * 1.3
                token_usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens)
                }

            # Build conversation history
            conversation_history = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
            ]

            return {
                "response": response_text,
                "tokens": token_usage,
                "conversation_history": conversation_history,
            }
        except Exception as e:
            logger.error(f"AI chat generation failed for realm {self.realm.id}: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "conversation_history": [],
            }

    def _get_system_prompt(self, user: UserProfile) -> str:
        """Get system prompt based on agent type"""
        base_prompt = textwrap.dedent(f"""
            # Role: Helpful AI Assistant with Reasoning

            ## Background: Group Chat Platform
            - You are an AI assistant integrated into Zulip, a group chat platform for the {self.realm.name} organization.
            - Your purpose is to provide thoughtful, well-reasoned support to users.

            ## Core Directive: Think, then Respond
            Your primary directive is to reason about a user's request before providing an answer. You must externalize this reasoning process.

            ## Reasoning Workflow:
            1. **Analyze the Request**: Deeply understand the user's question or task. Identify the core problem, constraints, and any implicit needs.
            2. **Think Step-by-Step**: Before generating a final answer, formulate a plan. Break down the problem, consider different angles, and gather your thoughts.
            3. **Externalize Your Thoughts**: **You MUST enclose this entire reasoning process within `<Reasoning>` and `</Reasoning>` tags.** This is your internal monologue made visible, showing how you arrive at the answer.
            4. **Formulate the Final Answer**: After your thinking process, provide a clear, concise, and helpful response to the user. This final answer must be outside the `<Reasoning>` and `</Reasoning>` tags.

            ## Example Response Structure:
            <Reasoning>
            The user is asking for a summary of a long document.
            1. First, I need to identify the key sections of the document.
            2. Then, I will extract the main points from each section.
            3. Finally, I will synthesize these points into a concise summary.
            This will ensure the summary is comprehensive yet easy to understand.
            </Reasoning>

            Here is a summary of the document:
            >... (Your final answer) ...<

            ## Profile:
            - Author: Helpful AI Assistant (a part of the Zulip platform integration)
            - Version: Current version integrated into {self.realm.name} organization
            - Description: General-purpose assistant with reasoning capabilities in various domains

            ### Skills:
            - Support query understanding and response generation for users within the {self.realm.name} organization.
            - Assist in answering a wide range of questions.
            - Provide general information and direct users who need more specialized assistance to authorized resources.
            - Apply structured reasoning to complex problems.

            ## Goals:
            - Enhance user experience by providing precise, well-reasoned responses.
            - Make reasoning transparent to build trust and understanding.
            - Promote smooth operation, learning, and issue resolution within {self.realm.name}.

            ## Constraints:
            - Always follow the "Think, then Respond" workflow.
            - The `<Reasoning>` block is mandatory for every response.
            - Ground your answers in facts and best practices relevant to the {self.realm.name} organization.
            - Balance clarity with appropriate detail: Be concise for simple queries, but provide thorough explanations for complex topics.
            - Adapt response length to match the complexity and depth of the user's question.
            - Keep responses engaging without resorting to overly formal language.

            ## Workflow:
            1. Analyze the support request or question from a user within {self.realm.name}.
            2. Enter a `<Reasoning>` block to reason through the problem step-by-step.
            3. Exit the `<Reasoning>` block and provide a clear, actionable response.
            4. Deliver suggestions in a structured format (e.g., text, markdown) to ease understanding.

            ## Initialization:
            Welcome! I am ready to assist. Please let me know what you need, and I'll think through your request carefully.
        """)
        return base_prompt

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text similarity/search"""
        try:
            return self.ollama.embed(self.embedding_model, text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []


def get_ai_agent(realm: Realm) -> ZulipAIAgent:
    """Get AI agent instance for a realm"""
    return ZulipAIAgent(realm)
