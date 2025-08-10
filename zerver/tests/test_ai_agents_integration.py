#!/usr/bin/env python3
"""
AI Agents Integration Test Script

This script tests the Ollama integration and AI agents functionality.

In the django shell, run with: "run ./zerver/tests/test_ai_agents_integration.py"

Note*: To login to the django shell, to go development environment and run: "./manage.py shell"
"""
import time
from typing import Any, List

from django.conf import settings
from django.utils.timezone import now as timezone_now
import orjson

from zerver.lib.ai_agents import get_ai_agent
from zerver.lib.ollama_client import OllamaClient, OllamaGenerateResponse, StreamingResponse
from zerver.models import UserProfile
from zerver.actions.message_summary import zulip_messages
from zerver.lib.narrow import NarrowParameter

# Test timing globals
ai_test_start = 0.0
ai_test_total_time = 0.0
ai_test_total_requests = 0


def ai_test_start_timer() -> None:
    """Start timing for AI request"""
    global ai_test_start
    ai_test_start = time.time()


def ai_test_finish_timer(start_time=None) -> float:
    """Finish timing for AI request"""
    global ai_test_total_time, ai_test_total_requests, ai_test_start
    ai_test_total_requests += 1
    if start_time is not None:
        elapsed_time = time.time() - start_time
    else:
        elapsed_time = time.time() - ai_test_start
    ai_test_total_time += elapsed_time
    return elapsed_time  # Return elapsed time for individual request timing


def print_separator(title: str) -> None:
    """Print a visual separator"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}\n")


def test_ollama_connection() -> bool:
    """Test basic Ollama connection and available models"""
    print_separator("Testing Ollama Connection")

    try:
        # Test with default settings
        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        client = OllamaClient(ollama_url)

        print(f"🔗 Connecting to Ollama at: {ollama_url}")

        # Test health check
        ai_test_start_timer()
        is_healthy = client.health_check()
        ai_test_finish_timer()

        print(f"✅ Health check: {'PASS' if is_healthy else 'FAIL'}")

        if not is_healthy:
            print("❌ Ollama is not responding. Make sure it's running!")
            return False

        # List available models
        ai_test_start_timer()
        models = client.list_models()
        ai_test_finish_timer()

        print(f"📋 Available models ({len(models)} found):")
        for model in models:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            size_gb = f"{size / (1024**3):.1f} GB" if size > 0 else "Unknown size"
            modified = model.get("modified_at", "Unknown")
            print(f"   - {name} ({size_gb}) - Modified: {modified}")

        return len(models) > 0

    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False


def test_chat_generation(test_messages: List[str]) -> None:
    """Test chat generation with different prompts"""
    print_separator("Testing Chat Generation with Token Counts (Using ZulipAIAgent)")

    # Get the test user and realm to instantiate the agent
    try:
        user_profile = UserProfile.objects.get(email="user11@zulipdev.com")
        realm = user_profile.realm
        agent = get_ai_agent(realm)
        print(f"🤖 Using agent for realm: {realm.name}")
    except UserProfile.DoesNotExist:
        print("❌ Test user not found. Skipping chat generation test.")
        return

    default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
    print(f"🤖 Using model: {default_model}")
    print("📡 Using ZulipAIAgent with reasoning capabilities")

    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test Chat {i} ---")
        print(f"👤 User: {message}")
        print("🤔 Reasoning...")

        try:
            # Start timing BEFORE calling chat to capture full end-to-end time
            end_to_end_start = time.time()

            # Use the agent's chat method instead of the raw client
            result = agent.chat(message=message, user=user_profile)

            response_text = result["response"]
            token_usage = result["tokens"]

            # Calculate end-to-end time
            end_to_end_time = time.time() - end_to_end_start

            # Display the complete response
            print(f"\n🤖 AI Response:")
            print(f"{response_text}")

            # Show response statistics
            print(f"\n📦 Response Statistics:")
            print(f"   - Response length: {len(response_text)} characters")
            print(f"   - Response time: {end_to_end_time:.2f}s")

            # Display the token counts
            print(f"\n📊 Token Counts:")
            print(f"   - Prompt tokens: {token_usage.get('prompt_tokens', 0)}")
            print(f"   - Completion tokens: {token_usage.get('completion_tokens', 0)}")
            print(f"   - Total tokens: {token_usage.get('total_tokens', 0)}")

            if token_usage.get('total_tokens', 0) > 0 and end_to_end_time > 0:
                tokens_per_sec = token_usage['total_tokens'] / end_to_end_time
                print(f"   - Tokens per second: {tokens_per_sec:.1f}")

            print(f"⏱️  Total response time: {end_to_end_time:.2f}s")

            # Track timing for summary
            ai_test_finish_timer(end_to_end_start)

        except Exception as e:
            print(f"❌ Error generating response: {e}")


def test_embedding_generation(client: OllamaClient, test_texts: List[str]) -> None:
    """Test embedding generation"""
    print_separator("Testing Embedding Generation")

    embedding_model = getattr(settings, "AI_AGENTS_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
    print(f"🔍 Using embedding model: {embedding_model}")

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test Embedding {i} ---")
        print(f"📝 Text: {text}")

        try:
            ai_test_start_timer()
            embeddings = client.embed(embedding_model, text)
            elapsed_time = ai_test_finish_timer()

            print(f"📊 Generated embeddings with {len(embeddings)} dimensions")
            print(f"🔢 First 5 values: {embeddings[:5]}")
            print(f"⏱️  Generation time: {(elapsed_time):.2f}s")

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")


def test_conversation_summarization(user_profile: UserProfile, narrow: List[NarrowParameter]) -> None:
    """Test conversation summarization like message_summary.py"""
    print_separator("Testing Conversation Summarization (Using ZulipAIAgent)")

    def format_zulip_messages(messages: list[dict[str, Any]]) -> str:
        intro = "The following is a chat conversation."
        topic: str | None = None
        channel: str | None = None
        if narrow and len(narrow) == 2:
            for term in narrow:
                assert not term.negated
                if term.operator == "channel":
                    channel = term.operand
                if term.operator == "topic":
                    topic = term.operand
        if channel:
            intro += f" channel: {channel}"
        if topic:
            intro += f", topic: {topic}"

        msgs = [
            {"sender": message["sender_full_name"], "content": message["content"]}
            for message in messages
        ]

        formatted_conversation = orjson.dumps(msgs, option=orjson.OPT_INDENT_2).decode()
        return formatted_conversation

    try:
        # Get the agent for this realm
        realm = user_profile.realm
        agent = get_ai_agent(realm)
        print(f"🤖 Using agent for realm: {realm.name}")

        # Get messages from the narrow (same as message_summary.py)
        print(f"📱 Fetching messages for user: {user_profile.email}")
        print(f"🎯 Narrow: {[f'{n.operator}:{n.operand}' for n in narrow]}")

        messages = zulip_messages(user_profile, narrow)

        if not messages:
            print("❌ No messages found for the given narrow")
            return

        print(f"📨 Found {len(messages)} messages")

        # Format messages for display (like in message_summary.py)
        formatted_conversation = format_zulip_messages(messages)
        print("\n--- Formatted Conversation ---")
        print(formatted_conversation)

        default_model = getattr(settings, "AI_AGENTS_DEFAULT_MODEL", "llama3.1:8b")
        print(f"🤖 Using model: {default_model}")
        print("📡 Using ZulipAIAgent with reasoning capabilities")

        # Convert messages to format expected by AI agent
        ai_messages = []
        for msg in messages:
            ai_messages.append({
                "sender_full_name": msg.get("sender_full_name", "Unknown"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
            })

        print(f"\n--- AI Summarization ---")
        print(f"🤖 Summarizing {len(ai_messages)} messages...")
        print("🤔 Reasoning...")

        end_to_end_start = time.time()

        # Use the agent's chat method with context
        summarization_prompt = f"Summarize the following conversation: {formatted_conversation}"
        result = agent.chat(message=summarization_prompt, user=user_profile)

        response_text = result["response"]
        token_usage = result["tokens"]

        # Calculate end-to-end time
        end_to_end_time = time.time() - end_to_end_start

        # Display the complete response
        print(f"\n🤖 AI Summary:")
        print(response_text)

        # Show response statistics
        print(f"\n📦 Response Statistics:")
        print(f"   - Response length: {len(response_text)} characters")
        print(f"   - Response time: {end_to_end_time:.2f}s")

        # Display the token counts
        print(f"\n📊 Token Counts:")
        print(f"   - Prompt tokens: {token_usage.get('prompt_tokens', 0)}")
        print(f"   - Completion tokens: {token_usage.get('completion_tokens', 0)}")
        print(f"   - Total tokens: {token_usage.get('total_tokens', 0)}")

        if token_usage.get('total_tokens', 0) > 0 and end_to_end_time > 0:
            tokens_per_sec = token_usage['total_tokens'] / end_to_end_time
            print(f"   - Tokens per second: {tokens_per_sec:.1f}")

        print(f"⏱️  Total response time: {end_to_end_time:.2f}s")

        # Track timing for summary
        ai_test_finish_timer(end_to_end_start)

    except Exception as e:
        print(f"❌ Error in summarization test: {e}")


def print_performance_summary() -> None:
    """Print performance summary"""
    print_separator("Performance Summary")

    print(f"📊 Total AI requests: {ai_test_total_requests}")
    print(f"⏱️  Total time: {ai_test_total_time:.2f}s")

    if ai_test_total_requests > 0:
        avg_time = ai_test_total_time / ai_test_total_requests
        print(f"📈 Average request time: {avg_time:.2f}s")
        print(f"🚀 Requests per second: {1/avg_time:.2f}")

    print(f"🕐 Test completed at: {timezone_now()}")


def main() -> None:
    """Main test function"""
    print_separator("AI Agents Integration Test Suite")
    print(f"🚀 Starting tests at: {timezone_now()}")
    print("\n📌 Note: This test suite validates the refactored AI agent architecture")

    # Test 1: Basic Ollama connection
    if not test_ollama_connection():
        print("❌ Ollama connection failed. Stopping tests.")
        return

    # Create Ollama client for further tests
    ollama_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
    client = OllamaClient(ollama_url)

    # Test 2: Chat generation
    test_chat_messages = [
        "Hello! How are you?",
        "What is Python?",
        "Explain AI agents",
        "Write a function to calculate fibonacci numbers",
    ]
    test_chat_generation(test_chat_messages)

    # Test 4: Embedding generation
    test_embedding_texts = [
        "Hello world",
        "This is a longer text for embedding generation testing",
        "Python programming language",
        "Machine learning and artificial intelligence",
    ]
    test_embedding_generation(client, test_embedding_texts)

    # Test 4: Get test user and realm
    try:
        # Try to get a test user (adjust email as needed)
        user_profile = UserProfile.objects.get(email="user11@zulipdev.com")
        if not user_profile:
            print("❌ User Profile not found. Please check the email address.")
            return

        realm = user_profile.realm
        print(f"👤 Using test user: {user_profile.email} (Realm: {realm.name})")

        # Test 5: Conversation summarization (if messages exist)
        narrow = [
            NarrowParameter(operator="channel", operand="devel", negated=False),
        ]
        test_conversation_summarization(user_profile, narrow)

    except Exception as e:
        print(f"❌ Error setting up test user/realm: {e}")

    # Final summary
    print_performance_summary()


if __name__ == "__main__":
    main()
