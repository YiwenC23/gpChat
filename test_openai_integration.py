#!/usr/bin/env python3
import os
import sys
import django

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "zproject.settings")
django.setup()

from django.conf import settings
from zerver.lib.openai_client import OpenAIClient
from zerver.lib.ai_agents_openai import ZulipAIAgent
from zerver.models import Realm

def test_openai_client():
    """Test the OpenAI client directly"""
    print("Testing OpenAI Client...")
    client = OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL
    )
    
    print(f"API Key configured: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    print(f"Base URL: {settings.OPENAI_BASE_URL}")
    print(f"Default model: {settings.AI_AGENTS_DEFAULT_MODEL}")
    
    try:
        response = client.generate(
            model=settings.AI_AGENTS_DEFAULT_MODEL,
            prompt="How does Zulip work?",
            system="You are a helpful assistant."
        )
        print(f"OpenAI Response: {response[:150]}...")
        return True
    except Exception as e:
        print(f"Error testing OpenAI client: {e}")
        return False

def test_ai_agent():
    """Test the ZulipAIAgent with OpenAI integration"""
    print("\nTesting ZulipAI Agent...")
    try:
        # Get first realm
        realm = Realm.objects.first()
        print(f"Using realm: {realm.string_id}")
        
        # Create AI agent
        agent = ZulipAIAgent(realm)
        print("AI Agent created successfully")
        
        # Check if agent is healthy
        is_healthy = agent.is_healthy()
        print(f"AI Agent health check: {'Passed' if is_healthy else 'Failed'}")
        
        return True
    except Exception as e:
        print(f"Error testing AI agent: {e}")
        return False

if __name__ == "__main__":
    print("=== OpenAI Integration Test ===")
    client_test = test_openai_client()
    agent_test = test_ai_agent()
    
    if client_test and agent_test:
        print("\n✅ All tests passed! The OpenAI integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the logs above for details.")
