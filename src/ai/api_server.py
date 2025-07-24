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

# Import quality-first API components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ..api.quality_first_api import QualityFirstAPI, QualityFirstAPIRequest, ProgressiveResponse
from ..api.quality_validation_middleware import create_quality_validation_middleware

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
quality_first_api: Optional[QualityFirstAPI] = None
quality_middleware: Optional[Any] = None
config: Dict[str, Any] = {}


async def _enhance_response_with_quality_data(response: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Enhance response with quality scores and explanations
    
    Args:
        response: Original response from research assistant
        query: Original user query
        
    Returns:
        Enhanced response with quality data
    """
    try:
        # Extract recommendations from response
        recommendations = response.get('recommendations', [])
        
        if not recommendations:
            return response
        
        # Enhance each recommendation with quality data
        enhanced_recommendations = []
        
        for i, rec in enumerate(recommendations):
            # Calculate quality score based on various factors
            quality_score = _calculate_recommendation_quality_score(rec, query, i)
            
            # Generate quality explanation
            quality_explanation = _generate_quality_explanation(rec, quality_score, query)
            
            # Add quality validation status
            validation_status = "validated" if quality_score >= 0.7 else "needs_review"
            
            # Create enhanced recommendation
            enhanced_rec = {
                **rec,
                "quality_score": quality_score,
                "quality_explanation": quality_explanation,
                "validation_status": validation_status,
                "ranking_position": i + 1,
                "quality_factors": _get_quality_factors(rec, query),
                "confidence_level": _get_confidence_level(quality_score)
            }
            
            enhanced_recommendations.append(enhanced_rec)
        
        # Calculate overall response quality metrics
        overall_quality_metrics = _calculate_overall_quality_metrics(enhanced_recommendations)
        
        # Create enhanced response
        enhanced_response = {
            **response,
            "recommendations": enhanced_recommendations,
            "quality_metrics": overall_quality_metrics,
            "quality_summary": {
                "total_recommendations": len(enhanced_recommendations),
                "high_quality_count": sum(1 for r in enhanced_recommendations if r["quality_score"] >= 0.8),
                "validated_count": sum(1 for r in enhanced_recommendations if r["validation_status"] == "validated"),
                "average_quality": overall_quality_metrics["average_quality_score"],
                "quality_distribution": _get_quality_distribution(enhanced_recommendations)
            }
        }
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error enhancing response with quality data: {e}")
        return response


def _calculate_recommendation_quality_score(rec: Dict[str, Any], query: str, position: int) -> float:
    """Calculate quality score for a recommendation"""
    try:
        # Base score from existing relevance or confidence
        base_score = rec.get('relevance_score', rec.get('confidence', 0.5))
        
        # Position penalty (higher positions get slight penalty)
        position_factor = max(0.7, 1.0 - (position * 0.05))
        
        # Source quality factor
        source_quality = _get_source_quality_factor(rec.get('source', ''))
        
        # Explanation quality factor
        explanation_quality = _get_explanation_quality_factor(rec.get('explanation', ''))
        
        # Query relevance factor
        query_relevance = _calculate_query_relevance(rec, query)
        
        # Combine factors
        quality_score = (
            base_score * 0.4 +
            position_factor * 0.2 +
            source_quality * 0.2 +
            explanation_quality * 0.1 +
            query_relevance * 0.1
        )
        
        return min(1.0, max(0.0, quality_score))
        
    except Exception as e:
        logger.warning(f"Error calculating quality score: {e}")
        return 0.5


def _generate_quality_explanation(rec: Dict[str, Any], quality_score: float, query: str) -> str:
    """Generate explanation for quality score"""
    try:
        explanations = []
        
        if quality_score >= 0.9:
            explanations.append("Excellent match with high relevance and reliability")
        elif quality_score >= 0.8:
            explanations.append("Very good match with strong relevance")
        elif quality_score >= 0.7:
            explanations.append("Good match meeting quality standards")
        elif quality_score >= 0.6:
            explanations.append("Acceptable match with moderate relevance")
        else:
            explanations.append("Lower quality match requiring review")
        
        # Add specific quality factors
        source = rec.get('source', '')
        if 'data.gov.sg' in source.lower() or 'singstat' in source.lower():
            explanations.append("Singapore government source provides high reliability")
        
        if 'kaggle' in source.lower():
            explanations.append("Kaggle platform offers community-validated datasets")
        
        if rec.get('explanation') and len(rec['explanation']) > 50:
            explanations.append("Detailed explanation enhances understanding")
        
        return "; ".join(explanations)
        
    except Exception as e:
        logger.warning(f"Error generating quality explanation: {e}")
        return "Quality assessment completed"


def _get_quality_factors(rec: Dict[str, Any], query: str) -> Dict[str, float]:
    """Get detailed quality factors for a recommendation"""
    return {
        "source_reliability": _get_source_quality_factor(rec.get('source', '')),
        "explanation_completeness": _get_explanation_quality_factor(rec.get('explanation', '')),
        "query_relevance": _calculate_query_relevance(rec, query),
        "data_freshness": _estimate_data_freshness(rec),
        "accessibility": _estimate_accessibility(rec)
    }


def _get_confidence_level(quality_score: float) -> str:
    """Get confidence level description"""
    if quality_score >= 0.9:
        return "Very High"
    elif quality_score >= 0.8:
        return "High"
    elif quality_score >= 0.7:
        return "Medium"
    elif quality_score >= 0.6:
        return "Low"
    else:
        return "Very Low"


def _get_source_quality_factor(source: str) -> float:
    """Get quality factor based on source reliability"""
    source_lower = source.lower()
    
    # Government sources
    if any(gov in source_lower for gov in ['data.gov.sg', 'singstat', 'lta.gov.sg', 'moh.gov.sg']):
        return 0.95
    
    # Academic and research sources
    if any(academic in source_lower for academic in ['zenodo', 'arxiv', 'pubmed', 'ieee']):
        return 0.9
    
    # Established data platforms
    if any(platform in source_lower for platform in ['kaggle', 'world bank', 'un data']):
        return 0.85
    
    # Other sources
    return 0.7


def _get_explanation_quality_factor(explanation: str) -> float:
    """Get quality factor based on explanation completeness"""
    if not explanation:
        return 0.3
    
    length = len(explanation)
    if length > 100:
        return 0.9
    elif length > 50:
        return 0.8
    elif length > 20:
        return 0.7
    else:
        return 0.5


def _calculate_query_relevance(rec: Dict[str, Any], query: str) -> float:
    """Calculate relevance between recommendation and query"""
    try:
        # Simple keyword matching approach
        query_words = set(query.lower().split())
        
        # Check title/source for keyword matches
        source_words = set(rec.get('source', '').lower().split())
        title_words = set(rec.get('title', '').lower().split())
        desc_words = set(rec.get('description', '').lower().split())
        
        all_rec_words = source_words | title_words | desc_words
        
        # Calculate overlap
        overlap = len(query_words & all_rec_words)
        relevance = min(1.0, overlap / max(1, len(query_words)))
        
        return relevance
        
    except Exception as e:
        logger.warning(f"Error calculating query relevance: {e}")
        return 0.5


def _estimate_data_freshness(rec: Dict[str, Any]) -> float:
    """Estimate data freshness factor"""
    # This is a simplified estimation
    # In a real implementation, you'd check actual data timestamps
    return 0.8


def _estimate_accessibility(rec: Dict[str, Any]) -> float:
    """Estimate data accessibility factor"""
    # This is a simplified estimation
    # In a real implementation, you'd check API availability, download links, etc.
    return 0.8


def _calculate_overall_quality_metrics(recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate overall quality metrics for the response"""
    if not recommendations:
        return {
            "average_quality_score": 0.0,
            "quality_variance": 0.0,
            "validation_rate": 0.0,
            "high_quality_rate": 0.0
        }
    
    quality_scores = [r["quality_score"] for r in recommendations]
    validated_count = sum(1 for r in recommendations if r["validation_status"] == "validated")
    high_quality_count = sum(1 for r in recommendations if r["quality_score"] >= 0.8)
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    quality_variance = sum((q - avg_quality) ** 2 for q in quality_scores) / len(quality_scores)
    
    return {
        "average_quality_score": avg_quality,
        "quality_variance": quality_variance,
        "validation_rate": validated_count / len(recommendations),
        "high_quality_rate": high_quality_count / len(recommendations)
    }


def _get_quality_distribution(recommendations: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get distribution of quality levels"""
    distribution = {"very_high": 0, "high": 0, "medium": 0, "low": 0, "very_low": 0}
    
    for rec in recommendations:
        quality_score = rec["quality_score"]
        if quality_score >= 0.9:
            distribution["very_high"] += 1
        elif quality_score >= 0.8:
            distribution["high"] += 1
        elif quality_score >= 0.7:
            distribution["medium"] += 1
        elif quality_score >= 0.6:
            distribution["low"] += 1
        else:
            distribution["very_low"] += 1
    
    return distribution


async def _handle_progressive_websocket_search(websocket: WebSocket, query: str, session_id: str):
    """Handle progressive search via WebSocket"""
    try:
        # Create quality-first API request
        request = QualityFirstAPIRequest(
            query=query,
            session_id=session_id,
            progressive_loading=True,
            quality_threshold=0.7,
            max_results=10
        )
        
        # Start progressive search
        response = await quality_first_api.process_quality_first_search(request)
        
        # Send initial response
        await websocket.send_json({
            "type": "progressive_start",
            "request_id": response.request_id,
            "status": response.status,
            "initial_recommendations": [
                {
                    "source": rec.source,
                    "relevance_score": rec.relevance_score,
                    "quality_score": rec.quality_score,
                    "domain": rec.domain,
                    "explanation": rec.explanation,
                    "validation_status": rec.validation_status,
                    "quality_explanation": rec.quality_explanation
                }
                for rec in response.recommendations
            ],
            "explanation": response.explanation
        })
        
        # Poll for updates if processing
        if response.status == "processing":
            while True:
                await asyncio.sleep(0.5)  # Poll every 500ms
                
                update = await quality_first_api.get_progressive_update(response.request_id)
                if not update:
                    break
                
                # Send update
                await websocket.send_json({
                    "type": "progressive_update",
                    "request_id": update.request_id,
                    "status": update.status,
                    "recommendations": [
                        {
                            "source": rec.source,
                            "relevance_score": rec.relevance_score,
                            "quality_score": rec.quality_score,
                            "domain": rec.domain,
                            "explanation": rec.explanation,
                            "validation_status": rec.validation_status,
                            "quality_explanation": rec.quality_explanation
                        }
                        for rec in update.recommendations
                    ],
                    "quality_metrics": update.quality_metrics,
                    "processing_time": update.processing_time,
                    "explanation": update.explanation
                })
                
                # Break if complete
                if update.status in ["complete", "error", "timeout"]:
                    break
        
    except Exception as e:
        logger.error(f"Progressive WebSocket search error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Progressive search error: {str(e)}"
        })


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
    global research_assistant, evaluation_metrics, quality_first_api, quality_middleware, config
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Initialize research assistant
        research_assistant = ResearchAssistant(config)
        logger.info("Research Assistant initialized")
        
        # Initialize evaluation metrics
        evaluation_metrics = EvaluationMetrics(config)
        logger.info("Evaluation Metrics initialized")
        
        # Initialize quality-first API
        quality_first_api = QualityFirstAPI(config)
        logger.info("Quality-First API initialized")
        
        # Initialize quality validation middleware
        quality_middleware = create_quality_validation_middleware(
            min_quality_threshold=config.get('quality_threshold', 0.7),
            min_relevance_threshold=config.get('relevance_threshold', 0.6),
            max_response_time=config.get('max_response_time', 30.0),
            enable_quality_logging=config.get('enable_quality_logging', True)
        )
        logger.info("Quality Validation Middleware initialized")
        
        # Add quality validation middleware
        app.middleware("http")(quality_middleware)
        
        # Set up CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get('api_server', {}).get('allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        logger.info("API Server started successfully with quality-first enhancements")
        
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
            "quality_search": "/api/quality-search",
            "progressive_search": "/api/progressive-search/{request_id}",
            "refine": "/api/refine",
            "feedback": "/api/feedback",
            "session": "/api/session/{session_id}",
            "websocket": "/ws",
            "quality_stats": "/api/quality-stats"
        }
    }


@app.post("/api/search")
async def search_datasets(request: QueryRequest):
    """
    Enhanced search endpoint with quality scores and explanations
    
    Returns AI-enhanced recommendations with comprehensive quality validation
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
        
        # Enhance response with quality scores and explanations
        enhanced_response = await _enhance_response_with_quality_data(response, request.query)
        
        # Add comprehensive API metadata
        enhanced_response['api_metadata'] = {
            "version": "1.0.0",
            "total_time": time.time() - start_time,
            "quality_enhanced": True,
            "validation_enabled": quality_middleware is not None,
            "progressive_loading_available": True
        }
        
        return JSONResponse(content=enhanced_response)
        
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


@app.post("/api/quality-search")
async def quality_first_search(request: QualityFirstAPIRequest):
    """
    Quality-first search endpoint with progressive loading and validation
    
    Prioritizes result quality over response speed with comprehensive validation
    """
    try:
        if not quality_first_api:
            raise HTTPException(status_code=503, detail="Quality-first API not initialized")
        
        response = await quality_first_api.process_quality_first_search(request)
        
        return JSONResponse(content={
            "request_id": response.request_id,
            "query": response.query,
            "status": response.status,
            "recommendations": [
                {
                    "source": rec.source,
                    "relevance_score": rec.relevance_score,
                    "quality_score": rec.quality_score,
                    "domain": rec.domain,
                    "explanation": rec.explanation,
                    "geographic_scope": rec.geographic_scope,
                    "query_intent": rec.query_intent,
                    "validation_status": rec.validation_status,
                    "quality_explanation": rec.quality_explanation,
                    "cached_at": rec.cached_at
                }
                for rec in response.recommendations
            ],
            "quality_metrics": response.quality_metrics,
            "processing_time": response.processing_time,
            "explanation": response.explanation,
            "next_batch_available": response.next_batch_available,
            "total_expected": response.total_expected
        })
        
    except Exception as e:
        logger.error(f"Quality-first search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progressive-search/{request_id}")
async def get_progressive_update(request_id: str):
    """
    Get progressive update for an active quality-first search request
    
    Enables real-time updates for progressive result loading
    """
    try:
        if not quality_first_api:
            raise HTTPException(status_code=503, detail="Quality-first API not initialized")
        
        response = await quality_first_api.get_progressive_update(request_id)
        
        if not response:
            raise HTTPException(status_code=404, detail="Request not found or expired")
        
        return JSONResponse(content={
            "request_id": response.request_id,
            "query": response.query,
            "status": response.status,
            "recommendations": [
                {
                    "source": rec.source,
                    "relevance_score": rec.relevance_score,
                    "quality_score": rec.quality_score,
                    "domain": rec.domain,
                    "explanation": rec.explanation,
                    "geographic_scope": rec.geographic_scope,
                    "query_intent": rec.query_intent,
                    "validation_status": rec.validation_status,
                    "quality_explanation": rec.quality_explanation,
                    "cached_at": rec.cached_at
                }
                for rec in response.recommendations
            ],
            "quality_metrics": response.quality_metrics,
            "processing_time": response.processing_time,
            "explanation": response.explanation,
            "next_batch_available": response.next_batch_available,
            "total_expected": response.total_expected
        })
        
    except Exception as e:
        logger.error(f"Progressive update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/progressive-search")
async def start_progressive_search(request: QualityFirstAPIRequest):
    """
    Start a progressive search with real-time result streaming
    
    Returns initial response and enables progressive loading via WebSocket or polling
    """
    try:
        if not quality_first_api:
            raise HTTPException(status_code=503, detail="Quality-first API not initialized")
        
        # Force progressive loading for this endpoint
        request.progressive_loading = True
        
        # Start progressive search
        response = await quality_first_api.process_quality_first_search(request)
        
        # Return initial response with request ID for progressive updates
        return JSONResponse(content={
            "request_id": response.request_id,
            "query": response.query,
            "status": response.status,
            "initial_recommendations": [
                {
                    "source": rec.source,
                    "relevance_score": rec.relevance_score,
                    "quality_score": rec.quality_score,
                    "domain": rec.domain,
                    "explanation": rec.explanation,
                    "geographic_scope": rec.geographic_scope,
                    "query_intent": rec.query_intent,
                    "validation_status": rec.validation_status,
                    "quality_explanation": rec.quality_explanation,
                    "cached_at": rec.cached_at
                }
                for rec in response.recommendations
            ],
            "quality_metrics": response.quality_metrics,
            "processing_time": response.processing_time,
            "explanation": response.explanation,
            "next_batch_available": response.next_batch_available,
            "total_expected": response.total_expected,
            "progressive_update_url": f"/api/progressive-search/{response.request_id}",
            "websocket_available": True
        })
        
    except Exception as e:
        logger.error(f"Progressive search start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quality-stats")
async def get_quality_statistics():
    """Get comprehensive quality statistics from cache and validation middleware"""
    try:
        stats = {}
        
        # Get quality cache statistics
        if quality_first_api:
            cache_stats = quality_first_api.get_cache_statistics()
            stats['cache'] = cache_stats
        
        # Get quality validation statistics
        if quality_middleware:
            validation_stats = quality_middleware.get_quality_statistics()
            stats['validation'] = validation_stats
        
        # Add system-wide quality metrics
        stats['system'] = {
            'quality_first_enabled': quality_first_api is not None,
            'validation_middleware_enabled': quality_middleware is not None,
            'timestamp': time.time()
        }
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Quality statistics error: {str(e)}")
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
            "quality_first_api": "healthy" if quality_first_api else "unavailable",
            "quality_middleware": "healthy" if quality_middleware else "unavailable",
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
                progressive_loading = data.get("progressive_loading", False)
                
                # Send processing status
                await websocket.send_json({
                    "type": "processing",
                    "message": "Analyzing your query with neural model..."
                })
                
                if progressive_loading and quality_first_api:
                    # Handle progressive loading via WebSocket
                    await _handle_progressive_websocket_search(websocket, query, session_id)
                else:
                    # Standard search with quality enhancement
                    response = await research_assistant.process_query(
                        query=query,
                        session_id=session_id
                    )
                    
                    # Enhance response with quality data
                    enhanced_response = await _enhance_response_with_quality_data(response, query)
                    
                    # Send results
                    await websocket.send_json({
                        "type": "results",
                        "data": enhanced_response
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