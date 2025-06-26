"""
API Server for AI-Powered Dataset Research Assistant
Provides REST API and WebSocket endpoints for real-time interaction
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .evaluation_metrics import EvaluationMetrics
from .research_assistant import ResearchAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI-Powered Dataset Research Assistant",
    description="High-performance neural recommendations with AI enhancement",
    version="1.0.0"
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Research query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class RefinementRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    refinement: str = Field(..., description="Query refinement")


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="Original query")
    satisfaction_score: float = Field(..., ge=0, le=1, description="Satisfaction score (0-1)")
    helpful_datasets: List[str] = Field(default_factory=list, description="IDs of helpful datasets")
    feedback_text: Optional[str] = Field(None, description="Optional text feedback")


class SessionSummaryResponse(BaseModel):
    session_id: str
    created_at: str
    duration_formatted: str
    statistics: Dict[str, Any]
    queries: List[str]
    is_active: bool


# Global instances
research_assistant: Optional[ResearchAssistant] = None
evaluation_metrics: Optional[EvaluationMetrics] = None
config: Dict[str, Any] = {}


def load_configuration():
    """Load AI configuration from YAML file"""
    global config
    config_path = Path('config/ai_config.yml')
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global research_assistant, evaluation_metrics, config
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Initialize research assistant
        research_assistant = ResearchAssistant(config)
        logger.info("Research Assistant initialized")
        
        # Initialize evaluation metrics
        evaluation_metrics = EvaluationMetrics(config)
        logger.info("Evaluation Metrics initialized")
        
        # Set up CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get('api_server', {}).get('allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        logger.info("API Server started successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI-Powered Dataset Research Assistant",
        "version": "1.0.0",
        "neural_performance": {
            "model": "Lightweight Cross-Attention Ranker",
            "ndcg_at_3": 69.99,
            "accuracy": 92.7
        },
        "endpoints": {
            "search": "/api/search",
            "refine": "/api/refine",
            "feedback": "/api/feedback",
            "session": "/api/session/{session_id}",
            "websocket": "/ws"
        }
    }


@app.post("/api/search")
async def search_datasets(request: QueryRequest):
    """
    Main search endpoint for dataset recommendations
    
    Returns AI-enhanced recommendations from high-performing neural model
    """
    try:
        start_time = time.time()
        
        # Process query through neural + AI pipeline
        response = await research_assistant.process_query(
            query=request.query,
            session_id=request.session_id,
            context=request.context
        )
        
        # Update preferences if provided
        if request.preferences and response.get('session_id'):
            research_assistant.conversation_manager.update_preferences(
                response['session_id'],
                request.preferences
            )
        
        # Add API metadata
        response['api_metadata'] = {
            "version": "1.0.0",
            "total_time": time.time() - start_time
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refine")
async def refine_search(request: RefinementRequest):
    """
    Refine previous search based on user feedback
    
    Enables conversational interaction for improved results
    """
    try:
        response = await research_assistant.refine_query(
            session_id=request.session_id,
            refinement=request.refinement
        )
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Refinement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for continuous improvement
    
    Helps measure user satisfaction and improve recommendations
    """
    try:
        # Record feedback
        feedback_recorded = await evaluation_metrics.record_feedback(
            session_id=request.session_id,
            query=request.query,
            satisfaction_score=request.satisfaction_score,
            helpful_datasets=request.helpful_datasets,
            feedback_text=request.feedback_text
        )
        
        if feedback_recorded:
            return {
                "status": "success",
                "message": "Thank you for your feedback!"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to record feedback")
            
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session_summary(session_id: str):
    """Get summary of a conversation session"""
    try:
        summary = research_assistant.conversation_manager.get_session_summary(session_id)
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return SessionSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Session summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End a conversation session"""
    try:
        ended = research_assistant.conversation_manager.end_session(session_id)
        
        if ended:
            return {"status": "success", "message": "Session ended"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        metrics = await evaluation_metrics.get_current_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check all services
        checks = {
            "api": "healthy",
            "neural_model": "healthy" if research_assistant.neural_bridge.model else "unavailable",
            "ai_services": "healthy" if len(research_assistant.llm_manager.clients) > 0 else "no_clients",
            "active_sessions": research_assistant.conversation_manager.get_active_sessions_count()
        }
        
        # Overall health
        is_healthy = all(
            v == "healthy" for k, v in checks.items() 
            if k not in ["active_sessions"]
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "checks": checks,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# WebSocket endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversational interaction
    
    Enables streaming responses and interactive refinement
    """
    await websocket.accept()
    session_id = None
    
    try:
        # Create session for WebSocket connection
        session = research_assistant.conversation_manager.create_session()
        session_id = session['session_id']
        
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "message": "Welcome! I'm ready to help you discover datasets."
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type", "query")
            
            if message_type == "query":
                # Process query
                query = data.get("query", "")
                
                # Send processing status
                await websocket.send_json({
                    "type": "processing",
                    "message": "Analyzing your query with neural model..."
                })
                
                # Get recommendations
                response = await research_assistant.process_query(
                    query=query,
                    session_id=session_id
                )
                
                # Send results
                await websocket.send_json({
                    "type": "results",
                    "data": response
                })
                
            elif message_type == "refine":
                # Process refinement
                refinement = data.get("refinement", "")
                
                await websocket.send_json({
                    "type": "processing",
                    "message": "Refining recommendations..."
                })
                
                response = await research_assistant.refine_query(
                    session_id=session_id,
                    refinement=refinement
                )
                
                await websocket.send_json({
                    "type": "results",
                    "data": response
                })
                
            elif message_type == "feedback":
                # Process feedback
                feedback_data = data.get("feedback", {})
                
                recorded = await evaluation_metrics.record_feedback(
                    session_id=session_id,
                    **feedback_data
                )
                
                await websocket.send_json({
                    "type": "feedback_received",
                    "success": recorded
                })
                
            elif message_type == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        
    finally:
        # Clean up session if needed
        if session_id:
            research_assistant.conversation_manager.end_session(session_id)


# Development server runner
if __name__ == "__main__":
    import uvicorn

    # Get server config
    server_config = config.get('api_server', {})
    
    uvicorn.run(
        app,
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
        log_level="info"
    )