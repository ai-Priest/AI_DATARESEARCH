"""
Test script for AI-Powered Dataset Research Assistant
Verifies all components are working correctly
"""
import asyncio
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_llm_connectivity():
    """Test connectivity to all LLM providers"""
    print("\n🔍 Testing LLM Connectivity...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        from src.ai.llm_clients import LLMManager
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        # Initialize LLM manager
        llm_manager = LLMManager(config)
        
        # Test each provider
        test_prompt = "Respond with 'Hello from [provider name]' where provider name is your model provider."
        
        providers_to_test = config_manager.get_enabled_providers()
        print(f"  Enabled providers: {providers_to_test}")
        
        for provider in providers_to_test:
            try:
                print(f"\n  Testing {provider}...")
                response = await llm_manager.complete_with_fallback(
                    prompt=test_prompt,
                    preferred_provider=provider,
                    max_tokens=50
                )
                print(f"  ✅ {provider}: Connected successfully")
                print(f"     Response: {response['content'][:100]}...")
                print(f"     Response time: {response.get('response_time', 0):.2f}s")
            except Exception as e:
                print(f"  ❌ {provider}: Connection failed - {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LLM connectivity test failed: {str(e)}")
        return False


async def test_neural_bridge():
    """Test neural model bridge functionality"""
    print("\n🧠 Testing Neural Model Bridge...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        from src.ai.neural_ai_bridge import NeuralAIBridge
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        # Initialize neural bridge
        neural_bridge = NeuralAIBridge(config)
        
        # Test query
        test_query = "housing prices singapore"
        
        print(f"  Testing neural inference for: '{test_query}'")
        
        # Get recommendations
        neural_results = await neural_bridge.get_neural_recommendations(test_query, top_k=3)
        
        print(f"  ✅ Neural inference successful")
        print(f"     Model: {neural_results['neural_metrics']['model']}")
        print(f"     NDCG@3: {neural_results['neural_metrics']['ndcg_at_3']}")
        print(f"     Inference time: {neural_results['neural_metrics']['inference_time']:.3f}s")
        print(f"     Recommendations: {len(neural_results['recommendations'])}")
        
        # Display top recommendation
        if neural_results['recommendations']:
            top_rec = neural_results['recommendations'][0]
            print(f"\n     Top recommendation:")
            print(f"     - Title: {top_rec['title']}")
            print(f"     - Confidence: {top_rec['confidence']*100:.0f}%")
            print(f"     - Source: {top_rec['source']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Neural bridge test failed: {str(e)}")
        return False


async def test_research_assistant():
    """Test complete research assistant pipeline"""
    print("\n🔬 Testing Research Assistant Pipeline...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        from src.ai.research_assistant import ResearchAssistant
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        # Initialize research assistant
        assistant = ResearchAssistant(config)
        
        # Test queries
        test_queries = [
            "transportation data singapore MRT",
            "healthcare statistics hospitals",
            "economic indicators GDP inflation"
        ]
        
        for query in test_queries:
            print(f"\n  Processing query: '{query}'")
            
            try:
                # Process query
                response = await assistant.process_query(query)
                
                print(f"  ✅ Query processed successfully")
                print(f"     Session ID: {response['session_id']}")
                print(f"     Processing time: {response['processing_time']:.2f}s")
                print(f"     Recommendations: {len(response['recommendations'])}")
                print(f"     AI Provider: {response['performance']['ai_provider']}")
                
                # Show first recommendation
                if response['recommendations']:
                    rec = response['recommendations'][0]
                    print(f"\n     Top dataset:")
                    print(f"     - {rec['dataset']['title']}")
                    print(f"     - Confidence: {rec['confidence']*100:.0f}%")
                    print(f"     - Explanation: {rec['explanation'][:100]}...")
                
                # Test refinement
                if response['conversation']['can_refine']:
                    print(f"\n     Testing refinement...")
                    refinement_response = await assistant.refine_query(
                        session_id=response['session_id'],
                        refinement="focus on recent data from 2024"
                    )
                    print(f"     ✅ Refinement successful")
                    
            except Exception as e:
                print(f"  ❌ Query processing failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Research assistant test failed: {str(e)}")
        return False


async def test_conversation_manager():
    """Test conversation session management"""
    print("\n💬 Testing Conversation Manager...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        from src.ai.conversation_manager import ConversationManager
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        # Initialize conversation manager
        conv_manager = ConversationManager(config)
        
        # Create session
        session = conv_manager.create_session()
        print(f"  ✅ Session created: {session['session_id']}")
        
        # Add to history
        test_response = {
            "recommendations": [{"dataset": {"id": "test", "title": "Test Dataset"}}],
            "processing_time": 1.5
        }
        
        success = conv_manager.add_to_history(
            session['session_id'],
            "test query",
            test_response
        )
        print(f"  ✅ History updated: {success}")
        
        # Get session summary
        summary = conv_manager.get_session_summary(session['session_id'])
        print(f"  ✅ Session summary retrieved")
        print(f"     Duration: {summary['duration_formatted']}")
        print(f"     Queries: {summary['statistics']['total_queries']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Conversation manager test failed: {str(e)}")
        return False


async def test_evaluation_metrics():
    """Test evaluation metrics system"""
    print("\n📊 Testing Evaluation Metrics...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        from src.ai.evaluation_metrics import EvaluationMetrics
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        # Initialize metrics
        metrics = EvaluationMetrics(config)
        
        # Record test feedback
        feedback_recorded = await metrics.record_feedback(
            session_id="test_session",
            query="test query",
            satisfaction_score=0.85,
            helpful_datasets=["dataset1", "dataset2"],
            feedback_text="Very helpful recommendations"
        )
        print(f"  ✅ Feedback recorded: {feedback_recorded}")
        
        # Get current metrics
        current_metrics = await metrics.get_current_metrics()
        print(f"  ✅ Metrics retrieved")
        print(f"     User satisfaction: {current_metrics['user_satisfaction']['current_rate']:.2%}")
        print(f"     Total feedback: {current_metrics['user_satisfaction']['total_feedback']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation metrics test failed: {str(e)}")
        return False


async def test_api_server():
    """Test API server endpoints (without starting server)"""
    print("\n🌐 Testing API Server Configuration...")
    
    try:
        from src.ai.ai_config_manager import AIConfigManager
        
        # Load configuration
        config_manager = AIConfigManager()
        config = config_manager.config
        
        api_config = config.get('api_server', {})
        
        print(f"  ✅ API Server configuration loaded")
        print(f"     Host: {api_config.get('host', '0.0.0.0')}")
        print(f"     Port: {api_config.get('port', 8000)}")
        print(f"     CORS enabled: {api_config.get('cors_enabled', True)}")
        print(f"     WebSocket enabled: {api_config.get('websocket', {}).get('enabled', True)}")
        
        print(f"\n  To start the API server, run:")
        print(f"     python -m src.ai.api_server")
        print(f"  Or:")
        print(f"     uvicorn src.ai.api_server:app --reload")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API server test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("🚀 AI-Powered Dataset Research Assistant - System Test")
    print("=" * 60)
    
    # Check environment variables
    print("\n🔐 Checking Environment Variables...")
    env_vars = ['MINIMAX_API_KEY', 'MISTRAL_API_KEY', 'CLAUDE_API_KEY', 'OPENAI_API_KEY']
    
    for var in env_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: Set")
        else:
            print(f"  ⚠️  {var}: Not set")
    
    # Check SDK installations
    print("\n📦 Checking SDK Installations...")
    print("API SDKs (required for your setup):")
    api_sdks = [
        ('anthropic', 'Claude SDK', True),
        ('openai', 'OpenAI SDK', True),
        ('mistralai', 'Mistral SDK', True),
        ('fastapi', 'FastAPI', True),
        ('aiohttp', 'AioHTTP', True)
    ]
    
    for module, name, required in api_sdks:
        try:
            __import__(module)
            print(f"  ✅ {name}: Installed")
        except ImportError:
            if required:
                print(f"  ❌ {name}: Not installed - run: pip install {module}")
            else:
                print(f"  ⚠️  {name}: Not installed (optional)")
    
    print("\nLocal Inference (optional - only if you have GPU):")
    optional_modules = [
        ('vllm', 'vLLM', 'For running models locally on GPU'),
        ('torch', 'PyTorch', 'Required for neural models'),
        ('transformers', 'Transformers', 'For model loading')
    ]
    
    for module, name, description in optional_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}: Installed - {description}")
        except ImportError:
            print(f"  ℹ️  {name}: Not installed - {description}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU Status: Available ({torch.cuda.get_device_name(0)})")
            print("   You CAN use vLLM for local inference if desired")
        elif torch.backends.mps.is_available():
            print("\n🎮 GPU Status: Apple Silicon (MPS) available")
            print("   Limited local inference support")
        else:
            print("\n💻 GPU Status: Not available (CPU only)")
            print("   Using API services is recommended")
    except:
        print("\n💻 GPU Status: Cannot check (PyTorch not installed)")
    
    # Run tests
    tests = [
        ("LLM Connectivity", test_llm_connectivity),
        ("Neural Bridge", test_neural_bridge),
        ("Research Assistant", test_research_assistant),
        ("Conversation Manager", test_conversation_manager),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("API Server Config", test_api_server)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for s in results.values() if s)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The AI system is ready to use.")
        print("\nNext steps:")
        print("1. Start the API server: python -m src.ai.api_server")
        print("2. Access the API at: http://localhost:8000")
        print("3. Use the WebSocket endpoint for real-time chat: ws://localhost:8000/ws")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())