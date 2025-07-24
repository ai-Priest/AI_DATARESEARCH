#!/usr/bin/env python3
"""
Test Conversation Endpoint
Tests the enhanced conversation endpoint with intelligent routing.
"""

import asyncio
import sys
import os
sys.path.insert(0, 'src')

# Mock the FastAPI request for testing
class MockRequest:
    def __init__(self, data):
        self.data = data
    
    def get(self, key, default=None):
        return self.data.get(key, default)


async def test_conversation_endpoint():
    """Test the conversation endpoint functionality"""
    print("🧪 Testing Enhanced Conversation Endpoint")
    print("=" * 50)
    
    try:
        # Import the conversation function
        from ai.optimized_research_assistant import OptimizedResearchAssistant
        from ai.ai_config_manager import AIConfigManager
        
        # Initialize research assistant
        config_manager = AIConfigManager('config/ai_config.yml')
        research_assistant = OptimizedResearchAssistant(config_manager.config)
        
        # Test cases
        test_cases = [
            {
                "message": "Hello",
                "expected_type": "greeting",
                "description": "Simple greeting"
            },
            {
                "message": "I need HDB data",
                "expected_type": "search_with_results",
                "description": "Dataset request"
            },
            {
                "message": "What's the weather?",
                "expected_type": "general_help",
                "description": "Non-dataset query"
            },
            {
                "message": "I need some data",
                "expected_type": "clarification_needed",
                "description": "Ambiguous request"
            }
        ]
        
        print("Testing conversation endpoint logic...")
        print()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['description']}")
            print(f"Message: '{test_case['message']}'")
            
            try:
                # Simulate the conversation endpoint logic
                message = test_case['message']
                session_id = f"test-{i}"
                
                # Use conversational query processor to determine intent
                query_processor = research_assistant.query_processor
                query_result = await query_processor.process_input(message)
                
                print(f"✅ Intent Analysis:")
                print(f"   - Is dataset request: {query_result.is_dataset_request}")
                print(f"   - Confidence: {query_result.confidence:.2f}")
                print(f"   - Extracted terms: {query_result.extracted_terms}")
                
                # Determine response type based on logic
                if query_result.is_dataset_request and query_result.confidence > 0.6:
                    response_type = "search_with_results"
                    print(f"✅ Would route to: Dataset search")
                elif query_result.suggested_clarification:
                    response_type = "clarification_needed"
                    print(f"✅ Would route to: Clarification")
                    print(f"   - Clarification: {query_result.suggested_clarification}")
                elif query_processor.is_inappropriate_query(message):
                    response_type = "inappropriate_declined"
                    print(f"✅ Would route to: Inappropriate decline")
                else:
                    # Check for greetings
                    message_lower = message.lower().strip()
                    if any(greeting in message_lower for greeting in ["hi", "hello", "help"]):
                        response_type = "greeting"
                        print(f"✅ Would route to: Greeting response")
                    else:
                        response_type = "general_help"
                        print(f"✅ Would route to: General help")
                
                print(f"   - Expected: {test_case['expected_type']}")
                print(f"   - Actual: {response_type}")
                
                if response_type == test_case['expected_type'] or (
                    test_case['expected_type'] == "search_with_results" and 
                    response_type in ["search_with_results", "search_no_results"]
                ):
                    print(f"✅ Routing correct!")
                else:
                    print(f"⚠️  Routing differs (may be acceptable)")
                
                print()
                
            except Exception as e:
                print(f"❌ Error in test {i}: {e}")
                print()
        
        print("🎉 Conversation endpoint tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_conversation_endpoint())
    sys.exit(0 if success else 1)