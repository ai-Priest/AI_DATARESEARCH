#!/usr/bin/env python3
"""
Simple Server Starter
====================
Starts the AI research assistant server with proper environment loading.
"""

import logging
import os
import socket
import sys
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.insert(0, 'src')

# Load environment variables from .env
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    continue

# Verify key environment variables
print(f"🔑 CLAUDE_API_KEY set: {'Yes' if os.environ.get('CLAUDE_API_KEY') else 'No'}")
print(f"🔑 MISTRAL_API_KEY set: {'Yes' if os.environ.get('MISTRAL_API_KEY') else 'No'}")

# Create minimal FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Dataset Research Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components
try:
    from ai.optimized_research_assistant import OptimizedResearchAssistant
    from ai.ai_config_manager import AIConfigManager
    
    config_manager = AIConfigManager('config/ai_config.yml')
    research_assistant = OptimizedResearchAssistant(config_manager.config)
    print("✅ AI Research Assistant initialized")
except Exception as e:
    print(f"❌ AI initialization error: {e}")
    research_assistant = None

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ai_assistant": "available" if research_assistant else "unavailable",
        "claude_api_configured": bool(os.environ.get('CLAUDE_API_KEY')),
        "mistral_api_configured": bool(os.environ.get('MISTRAL_API_KEY'))
    }

@app.get("/api/metrics")
async def get_metrics():
    """Performance metrics endpoint"""
    try:
        from src.ai.performance_metrics_collector import PerformanceMetricsCollector
        
        collector = PerformanceMetricsCollector()
        
        # Get all metrics
        metrics = await collector.get_all_metrics()
        
        # Get monitoring status
        monitoring_status = collector.get_monitoring_status()
        
        # Get performance trends (last 24 hours)
        neural_trends = collector.get_performance_trends('neural_performance', 24)
        cache_trends = collector.get_performance_trends('cache_performance', 24)
        response_trends = collector.get_performance_trends('response_time', 24)
        
        # Get system health history
        health_history = collector.get_system_health_history(24)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": metrics,
            "monitoring_status": monitoring_status,
            "performance_trends": {
                "neural_performance": neural_trends[:10],  # Last 10 entries
                "cache_performance": cache_trends[:10],
                "response_time": response_trends[:10]
            },
            "system_health_history": health_history[:10],
            "cache_statistics": {
                "search_cache": {
                    "hit_rate": metrics.get('cache_performance', {}).get('search_cache_hit_rate', 0.0) / 100,
                    "total_entries": metrics.get('cache_performance', {}).get('search_cache_entries', 0)
                },
                "quality_cache": {
                    "hit_rate": metrics.get('cache_performance', {}).get('quality_cache_hit_rate', 0.0) / 100,
                    "total_entries": metrics.get('cache_performance', {}).get('quality_cache_entries', 0)
                },
                "overall_hit_rate": metrics.get('cache_performance', {}).get('overall_hit_rate', 0.0) / 100,
                "total_size_mb": metrics.get('cache_performance', {}).get('cache_size_mb', 0)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/ai-search")
async def ai_search(request: dict):
    """AI-enhanced search endpoint"""
    if not research_assistant:
        return {"error": "AI assistant not available", "fallback": True}
    
    query = request.get("query", "")
    top_k = request.get("top_k", 5)
    
    try:
        result = await research_assistant.process_query_optimized(
            query=query,
            session_id=f"api-{hash(query)}"
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "fallback": True,
            "query": query,
            "recommendations": [],
            "web_sources": []
        }

@app.post("/api/conversation")
async def conversation(request: dict):
    """Enhanced conversation endpoint with intelligent routing"""
    if not research_assistant:
        return {"error": "AI assistant not available", "fallback": True}
    
    message = request.get("message", "")
    session_id = request.get("session_id", f"conv-{hash(message)}")
    include_search = request.get("include_search", False)
    
    try:
        # Use conversational query processor to determine intent
        query_processor = research_assistant.query_processor
        query_result = await query_processor.process_input(message)
        
        # Handle different types of queries based on intent
        if query_result.is_dataset_request and query_result.confidence > 0.6:
            # High confidence dataset request - route to search
            try:
                search_result = await research_assistant.process_query_optimized(
                    query=message,
                    session_id=session_id
                )
                
                # Check if search returned actual results
                has_results = (
                    (search_result.get('recommendations') and len(search_result['recommendations']) > 0) or
                    (search_result.get('web_sources') and len(search_result['web_sources']) > 0) or
                    (search_result.get('all_results') and len(search_result['all_results']) > 0)
                )
                
                if has_results:
                    return {
                        "session_id": session_id,
                        "message": message,
                        "response": f"I found relevant datasets for '{message}'. Check the results below for details.",
                        "conversation_type": "search_with_results",
                        "search_results": search_result,
                        "can_refine": True,
                        "intent_confidence": query_result.confidence,
                        "extracted_terms": query_result.extracted_terms
                    }
                else:
                    return {
                        "session_id": session_id,
                        "message": message,
                        "response": f"I understand you're looking for data about '{' '.join(query_result.extracted_terms)}', but I couldn't find specific datasets. Try refining your search with more specific terms or check if the data exists in government portals.",
                        "conversation_type": "search_no_results",
                        "search_results": search_result,
                        "can_refine": True,
                        "intent_confidence": query_result.confidence,
                        "extracted_terms": query_result.extracted_terms,
                        "suggested_refinements": [
                            "try more specific keywords",
                            "check Singapore government data portals",
                            "look for related datasets"
                        ]
                    }
                    
            except Exception as e:
                logger.error(f"Search error in conversation: {e}")
                return {
                    "session_id": session_id,
                    "message": message,
                    "response": f"I understand you're looking for data, but I'm having trouble processing your request right now. Please try using the main search function.",
                    "conversation_type": "search_error",
                    "can_refine": True,
                    "error": str(e)
                }
        
        elif query_result.suggested_clarification:
            # Ambiguous query - ask for clarification
            return {
                "session_id": session_id,
                "message": message,
                "response": query_result.suggested_clarification,
                "conversation_type": "clarification_needed",
                "can_refine": True,
                "intent_confidence": query_result.confidence,
                "suggested_refinements": [
                    "specify the type of data you need",
                    "mention a specific topic or domain",
                    "ask about Singapore government data"
                ]
            }
        
        elif query_processor.is_inappropriate_query(message):
            # Inappropriate query - polite decline
            return {
                "session_id": session_id,
                "message": message,
                "response": "I'm designed to help you find datasets and research data. Please ask me about topics like housing data, transport statistics, population demographics, or other research-related information.",
                "conversation_type": "inappropriate_declined",
                "can_refine": True
            }
        
        else:
            # Regular conversational query - provide helpful response
            message_lower = message.lower().strip()
            
            # Handle common greetings and questions
            conversational_responses = {
                "hi": "Hello! I'm the AI Dataset Research Assistant. I can help you find datasets related to Singapore government data, global statistics, and more. Try asking for topics like 'housing data', 'transport statistics', or 'population demographics'.",
                "hello": "Hello! I'm here to help you discover relevant datasets. What kind of data are you looking for today?",
                "help": "I can help you find datasets from Singapore government portals (like data.gov.sg, LTA DataMall) and global sources (like WHO, World Bank). Just tell me what specific data you need!",
                "what can you do": "I can search through Singapore government datasets and international data sources to find relevant information. I specialize in topics like housing, transport, demographics, economics, health, and more. Try asking: 'I need housing data' or 'Show me transport statistics'.",
                "thank you": "You're welcome! Feel free to ask for any specific datasets you need.",
                "thanks": "Happy to help! Let me know what data you're looking for."
            }
            
            # Check for conversational matches
            for key, response in conversational_responses.items():
                if key in message_lower:
                    return {
                        "session_id": session_id,
                        "message": message,
                        "response": response,
                        "conversation_type": "greeting",
                        "can_refine": False
                    }
            
            # Generic helpful response for unclear queries
            return {
                "session_id": session_id,
                "message": message,
                "response": "I'm here to help you find datasets and data sources. Try asking for specific data like 'housing prices', 'transport data', 'population statistics', or any other research topic you're interested in!",
                "conversation_type": "general_help",
                "can_refine": True,
                "suggested_refinements": [
                    "ask for Singapore government data",
                    "search for specific statistics",
                    "mention a research topic"
                ]
            }
            
    except Exception as e:
        logger.error(f"Conversation processing error: {e}")
        return {
            "session_id": session_id,
            "message": message,
            "response": "I'm having trouble processing your message right now. Please try asking for specific datasets or data topics.",
            "conversation_type": "processing_error",
            "can_refine": True,
            "error": str(e)
        }

def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = sock.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(preferred_ports: List[int] = None, host: str = "0.0.0.0") -> Optional[int]:
    """Find an available port from a list of preferred ports"""
    if preferred_ports is None:
        preferred_ports = [8000, 8001, 8002, 8003, 8004, 8005]
    
    for port in preferred_ports:
        if is_port_available(port, host):
            return port
    
    return None

def get_port_config() -> List[int]:
    """Get preferred port configuration from environment or use defaults"""
    # Check for environment variable configuration
    port_range = os.environ.get('SERVER_PORT_RANGE', '8000-8005')
    
    try:
        if '-' in port_range:
            start_port, end_port = map(int, port_range.split('-'))
            return list(range(start_port, end_port + 1))
        else:
            # Single port specified
            return [int(port_range)]
    except ValueError:
        # Fallback to default range
        return [8000, 8001, 8002, 8003, 8004, 8005]

def start_server_with_port_fallback():
    """Start server with automatic port conflict resolution"""
    print("🚀 Starting AI Dataset Research Assistant Server...")
    
    # Get preferred ports
    preferred_ports = get_port_config()
    host = "0.0.0.0"
    
    # Find available port
    available_port = find_available_port(preferred_ports, host)
    
    if available_port is None:
        print("❌ Error: No available ports found in the configured range")
        print(f"   Tried ports: {preferred_ports}")
        print("   Solutions:")
        print("   1. Stop other services using these ports")
        print("   2. Set SERVER_PORT_RANGE environment variable (e.g., '9000-9005')")
        print("   3. Use 'lsof -i :PORT' to find what's using a specific port")
        sys.exit(1)
    
    # Display startup information
    if available_port != preferred_ports[0]:
        print(f"⚠️  Port {preferred_ports[0]} was busy, using port {available_port} instead")
    
    print("📍 Frontend: http://localhost:3002")
    print(f"📍 Backend: http://localhost:{available_port}") 
    print(f"📍 API Docs: http://localhost:{available_port}/docs")
    print(f"🔧 Host: {host}, Port: {available_port}")
    print()
    
    try:
        uvicorn.run(app, host=host, port=available_port, log_level="info")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("   Common solutions:")
        print("   1. Check if another process is using the port")
        print("   2. Verify network permissions")
        print("   3. Try running with different port range")
        sys.exit(1)

if __name__ == "__main__":
    start_server_with_port_fallback()