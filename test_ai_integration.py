#!/usr/bin/env python3
"""
Test script to verify AI integration with Welcome Bot
"""
import os
import sys
import django
from django.conf import settings

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'zproject.settings')
django.setup()

from zerver.models import Realm, UserProfile
from zerver.lib.ai_agents import ZulipAIAgent, OllamaConnectionError, OllamaModelError

def test_ai_integration():
    """Test AI integration with Welcome Bot"""
    print("=== AI Integration Test ===")
    
    # Check if AI agents are enabled
    ai_enabled = getattr(settings, "AI_AGENTS_ENABLED", False)
    print(f"AI Agents Enabled: {ai_enabled}")
    
    if not ai_enabled:
        print("❌ AI agents are not enabled in settings")
        return False
    
    # Get default realm
    try:
        realm = Realm.objects.first()
        if not realm:
            print("❌ No realm found")
            return False
        print(f"✅ Found realm: {realm.name}")
    except Exception as e:
        print(f"❌ Error getting realm: {e}")
        return False
    
    # Test AI agent creation
    try:
        ai_agent = ZulipAIAgent(realm)
        print(f"✅ AI Agent created successfully")
        print(f"   Default model: {ai_agent.default_model}")
        print(f"   Embedding model: {ai_agent.embedding_model}")
    except Exception as e:
        print(f"❌ Error creating AI agent: {e}")
        return False
    
    # Test Ollama connection
    try:
        is_healthy = ai_agent.is_healthy()
        print(f"✅ Ollama health check: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        if not is_healthy:
            print("❌ Ollama service is not healthy")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama health: {e}")
        return False
    
    # Test model availability
    try:
        models = ai_agent.ollama.list_models()
        print(f"✅ Available models: {len(models)}")
        for model in models:
            print(f"   - {model['name']}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False
    
    # Test AI chat
    try:
        # Get a test user
        user = UserProfile.objects.filter(is_active=True).first()
        if not user:
            print("❌ No active user found for testing")
            return False
        
        print(f"✅ Testing with user: {user.full_name}")
        
        # Test simple chat
        response = ai_agent.chat(
            message="Hello, how are you?",
            user=user,
            context="You are a helpful assistant.",
            agent_type="test"
        )
        
        if response and len(response.strip()) > 10:
            print(f"✅ AI chat test successful")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"❌ AI chat returned empty or too short response")
            return False
            
    except OllamaConnectionError as e:
        print(f"❌ Ollama connection error: {e}")
        return False
    except OllamaModelError as e:
        print(f"❌ Ollama model error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in AI chat: {e}")
        return False
    
    print("\n🎉 All AI integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_ai_integration()
    sys.exit(0 if success else 1) 