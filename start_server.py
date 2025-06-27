#!/usr/bin/env python3
"""
Simple Server Starter
====================
Starts the AI research assistant server with proper environment loading.
"""

import os
import sys
import uvicorn
from pathlib import Path

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
    """Conversation endpoint for casual queries"""
    message = request.get("message", "")
    session_id = request.get("session_id", f"conv-{hash(message)}")
    include_search = request.get("include_search", False)
    
    # Handle common conversational queries
    responses = {
        "hi": "Hello! I'm the AI Dataset Research Assistant. I can help you find datasets related to Singapore government data, global statistics, and more. Try searching for topics like 'housing', 'transport', 'population', or 'economy'.",
        "hello": "Hello! I'm here to help you discover relevant datasets. What kind of data are you looking for today?",
        "help": "I can help you find datasets from Singapore government portals (like data.gov.sg, LTA DataMall) and global sources (like WHO, World Bank). Just tell me what topic interests you!",
        "what can you do": "I can search through Singapore government datasets and international data sources to find relevant information. I specialize in topics like housing, transport, demographics, economics, health, and more.",
        "thank you": "You're welcome! Feel free to search for any datasets you need.",
        "thanks": "Happy to help! Let me know if you need to find any specific data."
    }
    
    message_lower = message.lower().strip()
    
    # Check for exact matches first
    for key, response in responses.items():
        if key in message_lower:
            return {
                "session_id": session_id,
                "message": message,
                "response": response,
                "conversation_type": "greeting",
                "can_refine": False
            }
    
    # If include_search is true or message contains data terms, do a search
    if include_search or any(term in message_lower for term in ['data', 'dataset', 'statistics', 'information']):
        try:
            result = await research_assistant.process_query_optimized(
                query=message,
                session_id=session_id
            )
            return {
                "session_id": session_id,
                "message": message,
                "response": f"I found some relevant datasets for '{message}'. Check the results panel for details.",
                "conversation_type": "search",
                "search_results": result,
                "can_refine": True
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "message": message,
                "response": f"I'd be happy to help you find datasets related to '{message}'. Try using the search function for better results.",
                "conversation_type": "fallback",
                "can_refine": True
            }
    
    # Generic helpful response
    return {
        "session_id": session_id,
        "message": message,
        "response": "I'm designed to help you find datasets and data sources. Try searching for topics like 'housing prices', 'transport data', 'population statistics', or any other data topic you're interested in!",
        "conversation_type": "redirect",
        "can_refine": True
    }

if __name__ == "__main__":
    print("🚀 Starting AI Dataset Research Assistant Server...")
    print("📍 Frontend: http://localhost:3002")
    print("📍 Backend: http://localhost:8000") 
    print("📍 API Docs: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")