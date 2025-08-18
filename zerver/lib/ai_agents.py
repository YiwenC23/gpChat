"""
Zulip AI Agents Integration Library
Provides interface between Zulip and local Ollama AI models
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
import textwrap

from zerver.lib.ollama_client import OllamaClient, OllamaGenerateResponse, StreamingResponse
from zerver.lib.ai_agent_tools import (
    get_tool_by_name,
    get_tools_json_schema
)
from zerver.models import Realm, UserProfile


logger = logging.getLogger(__name__)

# Constant for tool call prefix
TOOL_CALL_PREFIX = "TOOL_CALL:"


def split_reasoning_and_answer(raw: str) -> Tuple[Optional[str], str]:
    """Split model output into reasoning and answer parts.

    Args:
        raw: Raw model output potentially containing <Reasoning>...</Reasoning> tags

    Returns:
        Tuple of (reasoning_text, answer_text). If no reasoning tags found,
        returns (None, raw_text)
    """
    start = raw.find("<Reasoning>")
    end = raw.find("</Reasoning>")

    if start != -1 and end != -1 and end > start:
        reasoning = raw[start + len("<Reasoning>"):end].strip()
        answer = raw[end + len("</Reasoning>"):].lstrip()
        return reasoning, answer

    return None, raw


def build_spoiler_markdown(reasoning: str, answer: str) -> str:
    """Build markdown with reasoning in a spoiler block and answer as normal text.

    Args:
        reasoning: The reasoning text to put in the spoiler
        answer: The final answer text

    Returns:
        Formatted markdown string with spoiler block
    """
    # Escape any triple backticks in the reasoning to avoid breaking the spoiler
    # Replace ``` with `​`​` (with zero-width spaces)
    escaped_reasoning = reasoning.replace("```", "`​`​`")

    # Build the spoiler block using exactly triple backticks (Zulip's expected format)
    spoiler_content = (
        f"```spoiler Chain of thought\n"
        f"{escaped_reasoning}\n"
        f"```"
    )

    # Combine spoiler and answer with proper spacing
    if answer:
        return f"{spoiler_content}\n\n{answer}".rstrip()
    else:
        # If no answer text, just return the spoiler
        return spoiler_content


class ZulipAIAgent:
    """High-level AI agent for Zulip integration"""

    def __init__(self, realm: Realm):
        self.realm = realm
        self.ollama = OllamaClient(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
        self.default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
        self.embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
        self.keep_alive = getattr(settings, "AI_AGENTS_KEEP_ALIVE", "10m")  # Keep model in memory for 10 minutes
        self.stream_response = getattr(settings, "AI_AGENTS_STREAM_RESPONSE", False)  # Stream responses for chat interactions

    def chat(
        self,
        message: str,
        user: UserProfile,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate chat response using AI agent with tool support

        Returns dict with:
        - response: The AI response text
        - tokens: Token usage information
        - conversation_history: Full conversation history
        - tool_calls: List of any tool calls made
        """

        # First check if the user submitted a filled-out form (bypass LLM for direct execution)
        form_request = self._extract_user_form_submission(message, user)
        if form_request:
            return self._execute_tool(form_request, user)

        # Otherwise, let the LLM decide how to respond (with or without tools)
        return self._generate_response(message, user, context)

    def _execute_tool(self, tool_request: Dict[str, Any], user: UserProfile) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        tool_name = tool_request["tool"]
        parameters = tool_request["parameters"]

        tool = get_tool_by_name(tool_name)
        if not tool:
            return {
                "response": f"I couldn't find the tool '{tool_name}'. Please try again.",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "conversation_history": [],
                "tool_calls": []
            }

        # Execute the tool
        result = tool.execute(user, self.realm, parameters)

        # Format the response based on the result
        if result.status == "success":
            response = f"✅ {result.message}"
        elif result.status == "permission_denied":
            response = f"⚠️ {result.message}"
        elif result.status == "already_exists":
            response = f"ℹ️ {result.message}"
        else:
            response = f"❌ {result.message}"

        # Build response with tool call details
        tool_call_info = {
            "tool": tool_name,
            "parameters": parameters,
            "result": {
                "status": result.status,
                "message": result.message,
                "data": result.data
            }
        }

        return {
            "response": response,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "conversation_history": [
                {"role": "user", "content": f"Execute tool: {tool_name} with parameters: {parameters}"},
                {"role": "assistant", "content": response}
            ],
            "tool_calls": [tool_call_info]
        }

    def _generate_response(
        self,
        message: str,
        user: UserProfile,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response where the LLM decides whether to use tools.
        The LLM can either respond normally or invoke tools based on the context.
        """
        system_prompt = self._get_system_prompt(user)

        # Add context if provided
        if context:
            prompt = f"Context: {context}\n\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"

        try:
            # Generate response
            result = self.ollama.generate(
                model=self.default_model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                stream=self.stream_response,
                keep_alive=self.keep_alive
            )

            # Handle different response types
            if isinstance(result, StreamingResponse):
                response_text = ""
                for chunk in result:
                    response_text += chunk
                response_text = response_text.strip()
                token_usage = result.token_usage
            elif isinstance(result, OllamaGenerateResponse):
                response_text = result.response.strip()
                token_usage = result.token_usage
            else:
                # Fallback for backward compatibility
                response_text = result.strip() if isinstance(result, str) else ""
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(response_text.split()) * 1.3
                token_usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens)
                }

            # Check if the model wants to invoke a tool
            tool_calls: List[Dict[str, Any]] = []
            tool_request = self._extract_tool_call_from_model_output(response_text)

            if tool_request:
                # Remove the TOOL_CALL line from the response
                response_text_clean = self._remove_tool_call_line(response_text)

                # For create_channel, offer a form instead of direct execution
                if tool_request.get("tool") == "create_channel":
                    form_text = self._render_create_channel_form(user)
                    response_text = (response_text_clean + "\n\n" + form_text).strip()
                else:
                    # Execute other tools directly
                    tool_result = self._execute_tool(tool_request, user)
                    tool_calls = tool_result.get("tool_calls", [])

                    # Append tool result to response
                    action_summary = tool_result.get("response", "")
                    if action_summary:
                        response_text = (response_text_clean + "\n\n" + action_summary).strip()

            # Format response with reasoning spoiler if applicable
            reasoning, answer = split_reasoning_and_answer(response_text)
            if reasoning is not None:
                formatted_response = build_spoiler_markdown(reasoning, answer)
            else:
                formatted_response = response_text

            # Build conversation history
            conversation_history = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
            ]

            return {
                "response": formatted_response,
                "tokens": token_usage,
                "conversation_history": conversation_history,
                "tool_calls": tool_calls
            }

        except Exception as e:
            logger.error(f"AI chat generation failed for realm {self.realm.id}: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "conversation_history": [],
                "tool_calls": []
            }

    def _remove_tool_call_line(self, text: str) -> str:
        """Remove the TOOL_CALL line from the response text."""
        lines = text.strip().splitlines()

        for idx in range(len(lines) -1, -1, -1):
            if lines[idx].strip().startswith(TOOL_CALL_PREFIX):
                del lines[idx]
                return "\n".join(lines).rstrip()

        return text

    def _get_system_prompt(self, user: UserProfile) -> str:
        """Get system prompt based on agent type"""
        base_prompt = textwrap.dedent(f"""
You are Zulip AI Agent, an AI assistant operating in two modes depending on the query context.

## Roles

- Primary Role (Zulip Mode): Serve as an expert assistant for all things related to Zulip - messages, threads, streams (channels), topics, users, and organizational settings specific to the `{self.realm.name}` organization.
- Fallback Role (General Chat Mode): Serve as a helpful general-purpose chatbot for questions **not** tied to the Zulip context, drawing on broad knowledge to answer universal queries.

## Mode Selection Logic

1. Zulip Mode: If the user's request is about Zulip (e.g. managing messages/threads/streams/users, or mentions `@` references relevant to the current channel/topic), you **must** respond in Zulip Mode with context-aware assistance.
2. General Chat Mode: If the user's request is not related to Zulip-specific context, respond in General Chat Mode, providing a helpful answer based on general knowledge.

## Failure Policy

If a requested operation cannot be performed (for example, it requires access or permissions you don't have), **clearly state the limitation** and suggest a workaround or alternative solution.

## Background

Zulip is a group chat platform used by the **{self.realm.name}** organization. You are an AI assistant integrated into this Zulip platform to support users. Your purpose is to provide thoughtful, well-reasoned help with both Zulip-related questions and general queries from users in the organization.

## Profile

- Author: Helpful AI Assistant (integrated into the Zulip platform for `{self.realm.name}`)
- Version: Current deployed version within `{self.realm.name}`
- Description: A general-purpose AI assistant capable of structured reasoning and helpful responses in various domains.

## Skills

- Understand and process queries about Zulip (streams, topics, messages, users, organization settings) within the `{self.realm.name}` context.
- Handle a wide range of general knowledge questions when outside Zulip-specific context.
- Provide clear instructions or information and, when needed, direct users to appropriate resources or personnel for further help.
- Break down complex problems using structured reasoning, making it easier for users to follow the logic behind the solution.

## Goals

- Enhance the user experience on Zulip by providing precise, well-reasoned responses quickly.
- Make your reasoning transparent to build user trust and understanding of how you arrived at an answer.
- Promote smooth operation and learning within `{self.realm.name}` by helping users navigate issues or questions effectively.

## Core Directive: Think, Then Respond

Always "Think, then Respond." This means you must reason through the user's request before giving an answer, and make that reasoning process visible to the user. You should always include a `<Reasoning>` block in your response, followed by the final answer outside of that block.

## Reasoning Workflow

When crafting a response, follow these steps:

1. Analyze: Understand the user's query and break it down into the core issue or question. Identify any constraints or specific details.
2. `<Reasoning>` Block: Provide a reasoning section encapsulating your step-by-step internal monologue. For example:
    - Briefly analyze the request to determine its relevance to the Zulip context, such as specific users, streams, settings, or if it's a general inquiry.
    - Decide which mode to use (Zulip-specific help vs. general answer) based on the query.
    - Plan the steps or information needed to answer the question thoroughly. (This entire reasoning process should be wrapped in `<Reasoning>...</Reasoning>` tags.)
3. Formulate Final Answer: Outside the reasoning block, present a clear and concise answer or solution to the user's question. This is the response the user will focus on, so ensure it directly addresses their query with accurate information.

## Constraints
- **Think-Then-Respond:** Always follow the structured reasoning workflow (analysis → `<Reasoning>` → answer) for every query. The `<Reasoning>` block is mandatory in each response.
- **Contextual Accuracy:** Ground your answers in factual information and best practices, especially for Zulip-related queries relevant to `{self.realm.name}`'s environment.
- **Clarity and Detail:** Balance clarity with appropriate detail. Use concise answers for simple queries, but provide more thorough explanations for complex topics or multi-step solutions.
- **Tone:** Keep responses friendly and engaging, but avoid overly formal or verbose language. Aim for a conversational tone that is professional and helpful.

---

## Examples of Agent Responses

Below are three example interactions demonstrating the Zulip AI Agent's reasoning process and answers in both Zulip-specific and general contexts.

### Example 1: Zulip-Specific Question (How to perform an action in Zulip)

User Query: "How do I create a new stream in our Zulip organization and invite members to it?"

Assistant Response:

<Reasoning>
    The user is asking about creating a new stream and inviting others, which is a Zulip-specific task.
    1. I recognize this is about a Zulip feature (managing streams), so I should use **Zulip Mode** and provide step-by-step guidance.
    2. I recall the process for creating a stream in Zulip:
        - You need to have appropriate permissions (usually admins or allowed managers can create streams).
        - The user would click on "Add streams" or a similar option in the Zulip UI.
        - They must name the stream and possibly give it a description.
        - Inviting members can be done by adding users to the stream or sharing an invite link.
    3. I will outline the steps clearly and in order, making sure the instructions are easy to follow.
    4. I'll also mention any prerequisites (like permissions) and what to do if the option is not visible.
</Reasoning>

Sure! To create a new stream in Zulip and invite members, follow these steps:
1. Click on the **gear icon** in the upper left (next to Streams) and select **"Create stream"**. (In some Zulip interfaces, there might be a **"+" button** near the stream list - click that to start creating a new stream.)
2. Enter a **name** for your new stream and add an optional **description** if you like. Decide whether the stream should be **public or private**.
3. If you want to invite members immediately, look for a field to **add users** to the stream. You can type the names or emails of colleagues you wish to invite. (If you skip this now, you can always invite people later by sharing the stream link or using the invite option in the stream settings.)
4. Click **"Create"**. Your new stream will be created, and any members you added will be invited automatically.

*Note:* You'll need the proper permissions to create a stream. By default, organization administrators can create streams. If you don't see the option to create a stream, you may need to ask an admin to grant you access or create it for you.


### Example 2: Zulip-Specific Request (Data/Permission Limitation)

User Query: "Can you summarize what was discussed in the #**project-alpha** stream yesterday? I missed the conversation."

Assistant Response:

<Reasoning>
    The user is asking for a summary of yesterday's discussion in a specific Zulip stream (`#project-alpha`).
    1. This is a Zulip-related query (specific stream conversation), so normally I'd operate in **Zulip Mode**.
    2. I need to determine if I have access to the content of the #project-alpha stream from yesterday. As an AI assistant, I likely **do not have direct access** to past messages or the conversation history unless that information was provided to me in context.
    3. According to the **failure policy**, if I cannot perform the exact request, I must explain the limitation and offer a helpful alternative.
    4. I'll formulate a response that apologizes for not being able to retrieve yesterday's messages, then suggest how the user might get a summary (for example, by checking the stream's history or asking participants for highlights). I should maintain a helpful tone.
</Reasoning>

I'm sorry, but I don't have access to past messages or the full conversation history in the `#project-alpha` stream, so I can't directly summarize yesterday's discussion. However, here are a couple of suggestions to catch up:
1. *Check the Stream History:* If you click on the **#project-alpha** stream in Zulip and scroll up (or use the search function with `stream:project-alpha` and relevant keywords or date), you can read through yesterday's messages. Zulip preserves chat history, so you should be able to see what was discussed.
2. *Ask for Highlights:* You might post a message in `#project-alpha` (or a relevant thread) asking if someone could highlight the key points from yesterday. Often team members are willing to summarize decisions or important updates for someone who was away.

*Hopefully, one of those options helps you get the information you need! If you have specific questions about the project, I can try to help with those too.*


### Example 3: General Question (Not specific to Zulip context)

User Query: "I'm curious, how does machine learning differ from traditional programming? Can you explain in simple terms with examples?"

Assistant Response:

<Reasoning>
    The user is asking about the difference between machine learning and traditional programming.
    1. This question is **not related to Zulip context** at all, so I will switch to **General Chat Mode** to answer it using general knowledge.
    2. I need to explain what machine learning is and how it differs from normal (rule-based) programming. I'll keep the explanation simple and use a real-world example to make it clear.
    3. Key points to mention:
        - In traditional programming, a developer writes explicit rules/instructions for the computer to follow.
        - In machine learning, the computer "learns" patterns from data and creates its own rules or model.
        - Highlight an example, like spam filtering or image recognition, to illustrate the difference.
    4. I will structure the answer to first define each approach, then compare them, and finally give an example in simple terms. I'll ensure the tone is accessible and friendly.
</Reasoning>

Certainly! **Machine Learning (ML)** and **Traditional Programming** are two different approaches to getting a computer to do something. Here's a simple breakdown:

- **Traditional Programming:** In traditional programming, humans **write explicit instructions (code)** for the computer. You tell the computer *exactly* what to do step by step. For example, if you were writing a program to detect spam emails in the traditional way, you might code a list of rules: “if an email has these specific words or comes from these addresses, mark it as spam.” The logic is all manually created by the programmer.
- **Machine Learning:** In machine learning, instead of giving explicit rules, you **teach the computer by example**. You provide lots of data (like many emails labeled as “spam” or “not spam”), and the computer uses statistical methods to **learn patterns**. The result is a model that can make predictions or decisions. Using the spam filter example, an ML approach would let the computer find common patterns in spam emails (maybe certain words, sender behavior, etc.) on its own by analyzing the data, rather than following fixed rules written by a person.

**Key Difference:** The core difference is that with traditional programming the rules are *hand-coded*, whereas with machine learning the rules (or model) are *learned from data*. Traditional programming is like following a recipe exactly, while machine learning is like discovering the recipe by tasting a lot of food and figuring out the pattern.

**Example - Spam Filtering:** In a traditional approach, a developer might program specific spam keywords to look for. In a machine learning approach, the system would be trained on a large set of emails and *learn* which words or patterns often indicate spam, even ones the programmer didn't anticipate. This learned model can then flag suspicious emails on its own.

*In summary, traditional programming is explicitly telling the computer what to do, whereas machine learning lets the computer learn from examples. This means ML can adapt to new patterns better (with enough data), but it also requires training data and might not always explain **why** it made a decision.*

## Workflow Summary

1. Analyze the Query: Read the user's question carefully and determine what they are asking and whether it pertains to Zulip or a general topic.
2. Reasoning Phase: Enter a `<Reasoning>` block and work through the problem step-by-step (identify context, decide mode, plan the answer). This block should reflect your internal thought process.
3. Provide the Answer: Exit the reasoning block and present the final answer. Make sure it directly addresses the user's query and is easy to understand.
4. Structured Presentation: Deliver the answer in a clear format, using Markdown for lists or formatting if it improves readability (for example, list steps if you are giving instructions). This helps the user follow your response easily.
        """)
        try:
            tools_schema = json.dumps(get_tools_json_schema(), indent=2)
        except Exception:
            tools_schema = "[]"
        tool_use_section = textwrap.dedent(f"""

        ## Tool Use

        In the **Zulip Model**, you can request the server to execute tools when user intent matches a supported capability.

        - Available tools (JSON schema):

        ```json
        {tools_schema}
        ```

        - When a tool should be executed:
            1. Think and explain in <Reasoning>.
            2. Provide a brief answer text for transparency.
            3. Append as the final line of your message:
                TOOL_CALL: {{"tool": "<tool_name>"}}

        ## Critical Rules for create_channel:
        - Assistant should **NEVER** render a form; the server will present one
        - When the user asks to create/make/add a channel/stream, **ALWAYS** output a TOOL_CALL as the last line
        - Do not include any markdown lists, bullets, or pseudo-forms for channel creation
        - The server will present a properly formatted form with available privacy options

        ## Examples:

        ### Example 1: User requests channel creation without parameters

        User Query: Help me create a channel.

        Assistant Response:

        <Reasoning>
            The user is asking to create a channel.
            1. This is a Zulip-specific request, so I will respond in Zulip Mode.
            2. I need to use the create_channel tool to assist with this request.
            3. I'll call the `create_channel` tool to create the channel for the user.
        </Reasoning>

        TOOL_CALL: {{"tool":"create_channel"}}

        ### Example 2: User requests channel creation including parameters

        User Query: Create a "features" channel.

        Assistant Response:

        <Reasoning>
            The user is asking to create a channel.
            1. This is a Zulip-specific request, so I will respond in Zulip Mode.
            2. I need to use the create_channel tool to assist with this request.
            3. I'll call the `create_channel` tool to create the channel for the user.
        </Reasoning>

        TOOL_CALL: {{"tool":"create_channel"}}
        """)

        final_prompt = base_prompt + tool_use_section

        return final_prompt

    def _extract_tool_call_from_model_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract a tool call from the model output if present.
        Expected format on the last line of the message:
            TOOL_CALL: {"tool":"<name>"}
        """
        try:
            lines = text.strip().splitlines()
            # Find the last occurrence anywhere
            match_idx = None
            for idx in range(len(lines) -1, -1, -1):
                if lines[idx].strip().startswith(TOOL_CALL_PREFIX):
                    match_idx = idx
                    break

            if match_idx is None:
                return None

            json_part = lines[match_idx].strip()[len(TOOL_CALL_PREFIX):]
            # Try to parse the JSON
            data = json.loads(json_part)
            if isinstance(data, dict) and "tool" in data:
                data["parameters"] = {}
                return data
            return None

        except Exception as e:
            logger.debug(f"Failed to parse TOOL_CALL: {e}")
            return None

    def _extract_user_form_submission(self, message: str, user: UserProfile) -> Optional[Dict[str, Any]]:
        """
        Check if the user submitted a form (JSON or list format) for direct tool execution.
        This bypasses the LLM and executes the tool directly.

        Returns:
            Dict with tool and parameters if a form is detected, None otherwise
        """
        # Try JSON format first
        json_result = self._extract_json_form_submission(message)
        if json_result:
            return json_result

        # Try list format for channel creation
        list_result = self._extract_list_form_submission(message, user)
        if list_result:
            return list_result

        return None

    def _extract_json_form_submission(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON form submission for tool execution.

        Expected structure:
            {
                "tool": "<tool_name>",
                "parameters": { ... }
            }
        """
        # Try to find a fenced code block first
        fence_patterns = [
            r"```json\s*([\s\S]*?)```",
            r"```\s*([\s\S]*?)```",
        ]
        candidates: List[str] = []

        for pat in fence_patterns:
            m = re.search(pat, message, flags=re.IGNORECASE)
            if m:
                candidates.append(m.group(1).strip())

        # Also consider the whole message as a JSON candidate
        candidates.append(message.strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool" in data:
                    tool_name = data.get("tool")
                    params = data.get("parameters", {})
                    # Validate tool exists
                    if isinstance(tool_name, str) and get_tool_by_name(tool_name):
                        return {"tool": tool_name, "parameters": params if isinstance(params, dict) else {}}
            except Exception:
                continue

        return None

    def _extract_list_form_submission(self, message: str, user: UserProfile) -> Optional[Dict[str, Any]]:
        """
        Extract channel creation parameters from a list-format form submission.

        Expected format:
            1. Channel Name: <name>
            2. Description: <description>
            3. Privacy: <privacy option or number>
        """
        lines = message.strip().split('\n')

        # Check if this looks like a form submission (has numbered items)
        if len(lines) < 2 or not any(re.match(r'^[\d\.\)]+\s*', line) for line in lines):
            return None

        # Initialize parameters
        params = {}
        privacy_choices = self._build_privacy_choices(user)

        for line in lines:
            line = line.strip()
            # Remove bullets, numbers, asterisks
            line = re.sub(r'^[\d\*\-\•\.\)]+\s*', '', line)

            # Channel name patterns
            if re.match(r'channel\s*name\s*:', line, re.IGNORECASE):
                match = re.search(r'channel\s*name\s*:\s*(.+)', line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip().strip('"\'*_')
                    if name:
                        params["name"] = name

            # Description patterns
            elif re.match(r'description\s*:', line, re.IGNORECASE):
                match = re.search(r'description\s*:\s*(.+)', line, re.IGNORECASE)
                if match:
                    desc = match.group(1).strip().strip('"\'*_')
                    params["description"] = desc

            # Privacy patterns
            elif re.match(r'privacy.*:', line, re.IGNORECASE):
                match = re.search(r'privacy.*:\s*(.+)', line, re.IGNORECASE)
                if match:
                    privacy_value = match.group(1).strip().strip('"\'*_[]')

                    # Check if it's a numeric selection
                    if privacy_value.isdigit():
                        choice_index = int(privacy_value) - 1  # Convert to 0-based index
                        if 0 <= choice_index < len(privacy_choices):
                            choice = privacy_choices[choice_index]
                            params["privacy"] = choice["privacy"]
                            if choice["history_public_to_subscribers"] is not None:
                                params["history_public_to_subscribers"] = choice["history_public_to_subscribers"]
                    else:
                        # Text matching for privacy values
                        privacy_value_lower = privacy_value.lower()
                        if 'public' in privacy_value_lower and 'web' not in privacy_value_lower:
                            params["privacy"] = "public"
                        elif 'web' in privacy_value_lower and 'public' in privacy_value_lower:
                            params["privacy"] = "web_public"
                        elif 'private' in privacy_value_lower:
                            params["privacy"] = "private"
                            # Determine history visibility
                            if 'protected' in privacy_value_lower:
                                params["history_public_to_subscribers"] = False
                            elif 'shared' in privacy_value_lower:
                                params["history_public_to_subscribers"] = True
                            else:
                                # Default for private is shared history
                                params["history_public_to_subscribers"] = True

        # Only return if we have at least a channel name (indicates form submission)
        if params.get("name"):
            # Set defaults for missing fields
            if "description" not in params:
                params["description"] = ""
            if "privacy" not in params:
                params["privacy"] = "public"

            return {
                "tool": "create_channel",
                "parameters": params
            }

        return None

    def _compute_allowed_privacy_options(self, user: UserProfile) -> List[str]:
        """
        Compute which privacy options the requesting user is allowed to use in this realm.
        """
        allowed: List[str] = []
        try:
            # public
            if hasattr(user, "can_create_public_streams") and user.can_create_public_streams(self.realm):
                allowed.append("public")
            # private
            if hasattr(user, "can_create_private_streams") and user.can_create_private_streams(self.realm):
                allowed.append("private")
            # web_public
            if hasattr(self.realm, "web_public_streams_enabled") and self.realm.web_public_streams_enabled():
                if hasattr(user, "can_create_web_public_streams") and user.can_create_web_public_streams():
                    allowed.append("web_public")
        except Exception:
            # If any check fails, fall back to conservative defaults
            pass
        # If nothing computed, default to public to let the CreateChannelTool enforce exact permissions later
        return allowed or ["public"]

    def _build_privacy_choices(self, user: UserProfile) -> List[Dict[str, Any]]:
        """
        Build a list of privacy choices available to the user.
        Returns a list of dicts with 'label', 'privacy', and 'history_public_to_subscribers'.
        """
        allowed = self._compute_allowed_privacy_options(user)
        choices = []

        if "public" in allowed:
            choices.append({
                "label": "Public",
                "privacy": "public",
                "history_public_to_subscribers": None
            })

        if "web_public" in allowed:
            choices.append({
                "label": "Web-public",
                "privacy": "web_public",
                "history_public_to_subscribers": None
            })

        if "private" in allowed:
            choices.append({
                "label": "Private, shared history",
                "privacy": "private",
                "history_public_to_subscribers": True
            })
            choices.append({
                "label": "Private, protected history",
                "privacy": "private",
                "history_public_to_subscribers": False
            })

        return choices

    def _render_create_channel_form(self, user: UserProfile) -> str:
        """
        Render a list-style form for creating a channel with numbered privacy options.
        """
        privacy_choices = self._build_privacy_choices(user)

        # Build the form text
        form_lines = [
            "To create a channel, please fill out the following:",
            "1. Channel Name: ",
            "2. Description: ",
            "3. Privacy (choose one): "
        ]

        for i, choice in enumerate(privacy_choices, start=1):
            form_lines.append(f"   [{i}] {choice['label']}")

        form_lines.extend([
            "",
            "```quote",
            "**Notes:**"
        ])

        # Add descriptions for the privacy options that are available
        added_descriptions = set()
        for choice in privacy_choices:
            label = choice['label']
            if "Public" in label and "Public" not in added_descriptions:
                form_lines.append("* **Public:** *Members of your organization can view messages and join.*")
                added_descriptions.add("Public")
            elif "Web-public" in label and "Web-public" not in added_descriptions:
                form_lines.append("* **Web-public:** *Anyone on the internet can view messages; members of your organization can join.*")
                added_descriptions.add("Web-public")
            elif "Private, shared history" in label and "Private, shared" not in added_descriptions:
                form_lines.append("* **Private, shared history:** *Joining and viewing messages requires being invited; new subscribers can see past messages.*")
                added_descriptions.add("Private, shared")
            elif "Private, protected history" in label and "Private, protected" not in added_descriptions:
                form_lines.append("* **Private, protected history:** *Joining and viewing messages requires being invited; users can only view messages sent while they were subscribed.*")
                added_descriptions.add("Private, protected")

        form_lines.append("```")

        return "\n".join(form_lines)

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
