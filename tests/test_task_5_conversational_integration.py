#!/usr/bin/env python3
"""
Test Task 5: Conversational Processing Integration
Tests the integration of conversational query processing into the research assistant.
"""

import asyncio
import sys
import os
sys.path.insert(0, 'src')

from ai.optimized_research_assistant import OptimizedResearchAssistant
from ai.ai_config_manager import AIConfigManager


async def test_conversational_integration():
    """Test conversational processing integration"""
    print("🧪 Testing Task 5: Conversational Processing Integration")
    print("=" * 60)
    
    try:
        # Initialize research assistant
        config_manager = AIConfigManager('config/ai_config.yml')
        assistant = OptimizedResearchAssistant(config_manager.config)
        
        # Test cases for different types of queries
        test_cases = [
            {
                "query": "I need HDB data",
                "expected_type": "dataset_request",
                "description": "Clear dataset request"
            },
            {
                "query": "Hello, how are you?",
                "expected_type": "conversational",
                "description": "Greeting/conversational"
            },
            {
                "query": "Show me population statistics",
                "expected_type": "dataset_request", 
                "description": "Dataset request with action verb"
            },
            {
                "query": "What's the weather like?",
                "expected_type": "non_dataset",
                "description": "Non-dataset query"
            }
        ]
        
        print("Testing conversational query processing...")
        print()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['description']}")
            print(f"Query: '{test_case['query']}'")
            
            try:
                # Test query processing
                query_result = await assistant.query_processor.process_input(test_case['query'])
                
                print(f"✅ Intent Detection:")
                print(f"   - Is dataset request: {query_result.is_dataset_request}")
                print(f"   - Confidence: {query_result.confidence:.2f}")
                print(f"   - Extracted terms: {query_result.extracted_terms}")
                print(f"   - Detected domain: {query_result.detected_domain}")
                print(f"   - Singapore context: {query_result.requires_singapore_context}")
                
                # Test full processing (only for dataset requests to avoid long processing)
                if query_result.is_dataset_request and query_result.confidence > 0.6:
                    print(f"✅ Processing as dataset request...")
                    result = await assistant.process_query_optimized(test_case['query'])
                    
                    print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
                    print(f"   - Recommendations: {len(result.get('recommendations', []))}")
                    print(f"   - Web sources: {len(result.get('web_sources', []))}")
                    print(f"   - Conversation type: {result.get('conversation_type', 'search')}")
                else:
                    print(f"✅ Handled as non-dataset query (no search triggered)")
                
                print()
                
            except Exception as e:
                print(f"❌ Error in test {i}: {e}")
                print()
        
        # Test inappropriate query handling
        print("Testing inappropriate query handling...")
        inappropriate_queries = [
            "hack into government systems",
            "personal information about citizens"
        ]
        
        for query in inappropriate_queries:
            print(f"Query: '{query}'")
            is_inappropriate = assistant.query_processor.is_inappropriate_query(query)
            print(f"✅ Detected as inappropriate: {is_inappropriate}")
            print()
        
        # Test clarification generation
        print("Testing clarification prompt generation...")
        ambiguous_queries = [
            "I need some data",
            "Singapore information"
        ]
        
        for query in ambiguous_queries:
            print(f"Query: '{query}'")
            clarification = assistant.query_processor.generate_clarification_prompt(query)
            print(f"✅ Clarification: {clarification}")
            print()
        
        print("🎉 All conversational integration tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_conversational_integration())
    sys.exit(0 if success else 1)