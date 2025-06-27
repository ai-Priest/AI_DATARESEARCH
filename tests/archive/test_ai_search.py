#!/usr/bin/env python3
"""
Test AI Search Functionality Directly
=====================================
This script tests the AI search components directly to verify they're working.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Load environment variables
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    continue

async def test_ai_search():
    """Test AI search functionality directly"""
    print("🧪 Testing AI Search Components...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🔑 Claude API Key set: {'Yes' if os.environ.get('CLAUDE_API_KEY') else 'No'}")
    print()
    
    # Test 1: Import all components
    print("1️⃣  Testing imports...")
    try:
        from ai.optimized_research_assistant import OptimizedResearchAssistant
        from ai.ai_config_manager import AIConfigManager
        print("   ✅ Imports successful")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return
    
    # Test 2: Load configuration
    print("\n2️⃣  Testing configuration...")
    try:
        config_manager = AIConfigManager('config/ai_config.yml')
        config = config_manager.config
        llm_providers = config.get('llm_providers', {})
        claude_enabled = llm_providers.get('claude', {}).get('enabled', False)
        print(f"   ✅ Config loaded - Claude enabled: {claude_enabled}")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return
    
    # Test 3: Initialize research assistant
    print("\n3️⃣  Testing research assistant initialization...")
    try:
        research_assistant = OptimizedResearchAssistant(config)
        print("   ✅ Research assistant initialized")
    except Exception as e:
        print(f"   ❌ Research assistant error: {e}")
        return
    
    # Test 4: Test AI search query
    print("\n4️⃣  Testing AI search query...")
    try:
        query = "housing prices Singapore"
        session_id = "test-session-123"
        
        print(f"   🔍 Query: '{query}'")
        print("   ⏳ Processing (this may take 10-15 seconds)...")
        
        result = await research_assistant.process_query_optimized(query, session_id)
        
        print("   ✅ AI search completed!")
        print(f"   📝 Response: {result.get('response', 'N/A')[:100]}...")
        print(f"   📊 Recommendations: {len(result.get('recommendations', []))}")
        print(f"   🌐 Web sources: {len(result.get('web_sources', []))}")
        print(f"   ⏱️  Processing time: {result.get('processing_time', 0):.1f}s")
        
        if not result.get('fallback', True):
            print("   🎉 SUCCESS: Full AI/LLM search working!")
        else:
            print("   ⚠️  Using fallback mode")
            
        return result
        
    except Exception as e:
        print(f"   ❌ AI search error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 AI Search Test Starting...")
    print("=" * 50)
    
    result = asyncio.run(test_ai_search())
    
    print("\n" + "=" * 50)
    if result and not result.get('fallback', True):
        print("🎉 FINAL RESULT: AI/LLM search is working perfectly!")
    elif result:
        print("⚠️  FINAL RESULT: Components working but using fallback mode")
    else:
        print("❌ FINAL RESULT: AI search failed - check errors above")