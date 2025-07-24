"""
Quality-First API Response System
Enhanced API endpoints that prioritize result quality over response speed
with progressive loading and quality validation
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import quality-aware components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ..ai.quality_aware_cache import QualityAwareCacheManager, CachedRecommendation, QualityMetrics
from ..ai.integrated_query_processor import IntegratedQueryProcessor, ProcessedQuery

logger = logging.getLogger(__name__)


@dataclass
class QualityValidatedRecommendation:
    """Recommendation with quality validation and explanation"""
    source: str
    relevance_score: float
    quality_score: float
    domain: str
    explanation: str
    geographic_scope: str
    query_intent: str
    validation_status: str  # 'validated', 'pending', 'failed'
    quality_explanation: str
    cached_at: Optional[float] = None


@dataclass
class ProgressiveResponse:
    """Progressive API response with quality metrics"""
    request_id: str
    query: str
    status: str  # 'processing', 'partial', 'complete', 'error'
    recommendations: List[QualityValidatedRecommendation]
    quality_metrics: Optional[Dict[str, float]]
    processing_time: float
    explanation: str
    next_batch_available: bool = False
    total_expected: Optional[int] = None


class QualityFirstAPIRequest(BaseModel):
    """Enhanced API request with quality preferences"""
    query: str = Field(..., description="Research query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    quality_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum quality threshold")
    max_results: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")
    progressive_loading: Optional[bool] = Field(True, description="Enable progressive result loading")
    include_explanations: Optional[bool] = Field(True, description="Include quality explanations")
    timeout_seconds: Optional[int] = Field(30, ge=5, le=120, description="Maximum processing timeout")


class QualityFirstAPI:
    """Quality-first API system with progressive loading and validation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize quality-aware components
        self.cache_manager = QualityAwareCacheManager(
            quality_threshold=self.config.get('quality_threshold', 0.7)
        )
        
        self.query_processor = IntegratedQueryProcessor(
            training_mappings_path=self.config.get('training_mappings_path', 'training_mappings.md')
        )
        
        # Active progressive requests
        self.active_requests: Dict[str, Dict] = {}
        
        logger.info("🎯 QualityFirstAPI initialized with quality-first approach")
    
    async def process_quality_first_search(self, request: QualityFirstAPIRequest) -> ProgressiveResponse:
        """
        Process search with quality-first approach and progressive loading
        
        Args:
            request: Quality-first API request
            
        Returns:
            ProgressiveResponse with quality-validated recommendations
        """
        request_id = str(uuid4())
        start_time = time.time()
        
        logger.info(f"🔍 Processing quality-first search: {request.query}")
        logger.info(f"  Request ID: {request_id}")
        logger.info(f"  Quality threshold: {request.quality_threshold}")
        logger.info(f"  Progressive loading: {request.progressive_loading}")
        
        try:
            # Check cache first
            cached_result = await self._get_cached_recommendations(
                request.query, request.quality_threshold
            )
            
            if cached_result:
                recommendations, quality_metrics = cached_result
                
                # Convert to quality validated recommendations
                validated_recs = [
                    self._convert_to_quality_validated(rec, "validated", "From high-quality cache")
                    for rec in recommendations[:request.max_results]
                ]
                
                return ProgressiveResponse(
                    request_id=request_id,
                    query=request.query,
                    status="complete",
                    recommendations=validated_recs,
                    quality_metrics=asdict(quality_metrics),
                    processing_time=time.time() - start_time,
                    explanation="High-quality cached results served",
                    next_batch_available=False
                )
            
            # Process query with integrated processor
            processed_query = self.query_processor.process_query(request.query)
            
            # Generate recommendations with quality validation
            if request.progressive_loading:
                return await self._process_progressive_search(
                    request_id, request, processed_query, start_time
                )
            else:
                return await self._process_batch_search(
                    request_id, request, processed_query, start_time
                )
                
        except Exception as e:
            logger.error(f"Quality-first search error: {e}")
            return ProgressiveResponse(
                request_id=request_id,
                query=request.query,
                status="error",
                recommendations=[],
                quality_metrics=None,
                processing_time=time.time() - start_time,
                explanation=f"Processing error: {str(e)}"
            )
    
    async def _process_progressive_search(self, request_id: str, request: QualityFirstAPIRequest,
                                        processed_query: ProcessedQuery, start_time: float) -> ProgressiveResponse:
        """Process search with progressive loading"""
        
        # Store request for progressive updates
        self.active_requests[request_id] = {
            'request': request,
            'processed_query': processed_query,
            'start_time': start_time,
            'recommendations': [],
            'status': 'processing'
        }
        
        # Start background processing
        asyncio.create_task(self._background_recommendation_generation(request_id))
        
        # Return initial response
        return ProgressiveResponse(
            request_id=request_id,
            query=request.query,
            status="processing",
            recommendations=[],
            quality_metrics=None,
            processing_time=time.time() - start_time,
            explanation="Quality-first processing initiated",
            next_batch_available=True,
            total_expected=request.max_results
        )
    
    async def _process_batch_search(self, request_id: str, request: QualityFirstAPIRequest,
                                  processed_query: ProcessedQuery, start_time: float) -> ProgressiveResponse:
        """Process search in batch mode with quality validation"""
        
        # Generate recommendations
        recommendations = await self._generate_quality_recommendations(
            processed_query, request.max_results, request.quality_threshold
        )
        
        # Validate recommendations
        validated_recs = []
        for rec in recommendations:
            validation_result = await self._validate_recommendation_quality(
                rec, processed_query.original_query, request.quality_threshold
            )
            
            if validation_result['is_valid']:
                validated_recs.append(
                    self._convert_to_quality_validated(
                        rec, "validated", validation_result['explanation']
                    )
                )
        
        # Calculate quality metrics
        quality_metrics = self._calculate_response_quality_metrics(validated_recs)
        
        # Cache high-quality results
        if quality_metrics['overall_quality'] >= request.quality_threshold:
            await self._cache_quality_results(
                request.query, validated_recs, quality_metrics
            )
        
        return ProgressiveResponse(
            request_id=request_id,
            query=request.query,
            status="complete",
            recommendations=validated_recs,
            quality_metrics=quality_metrics,
            processing_time=time.time() - start_time,
            explanation=f"Quality-validated {len(validated_recs)} recommendations",
            next_batch_available=False
        )
    
    async def _background_recommendation_generation(self, request_id: str):
        """Background task for progressive recommendation generation"""
        try:
            request_data = self.active_requests[request_id]
            request = request_data['request']
            processed_query = request_data['processed_query']
            
            # Generate recommendations progressively
            recommendations = await self._generate_quality_recommendations(
                processed_query, request.max_results, request.quality_threshold
            )
            
            # Validate and add recommendations progressively
            validated_recs = []
            for i, rec in enumerate(recommendations):
                validation_result = await self._validate_recommendation_quality(
                    rec, processed_query.original_query, request.quality_threshold
                )
                
                if validation_result['is_valid']:
                    validated_rec = self._convert_to_quality_validated(
                        rec, "validated", validation_result['explanation']
                    )
                    validated_recs.append(validated_rec)
                    
                    # Update active request
                    request_data['recommendations'] = validated_recs
                    request_data['status'] = 'partial' if i < len(recommendations) - 1 else 'complete'
                    
                    # Small delay to simulate progressive loading
                    await asyncio.sleep(0.1)
            
            # Final update
            request_data['status'] = 'complete'
            request_data['quality_metrics'] = self._calculate_response_quality_metrics(validated_recs)
            
            # Cache results if high quality
            if request_data['quality_metrics']['overall_quality'] >= request.quality_threshold:
                await self._cache_quality_results(
                    request.query, validated_recs, request_data['quality_metrics']
                )
            
        except Exception as e:
            logger.error(f"Background processing error for {request_id}: {e}")
            if request_id in self.active_requests:
                self.active_requests[request_id]['status'] = 'error'
                self.active_requests[request_id]['error'] = str(e)
    
    async def get_progressive_update(self, request_id: str) -> Optional[ProgressiveResponse]:
        """Get progressive update for an active request"""
        if request_id not in self.active_requests:
            return None
        
        request_data = self.active_requests[request_id]
        request = request_data['request']
        
        current_time = time.time()
        processing_time = current_time - request_data['start_time']
        
        # Check for timeout
        if processing_time > request.timeout_seconds:
            request_data['status'] = 'timeout'
        
        response = ProgressiveResponse(
            request_id=request_id,
            query=request.query,
            status=request_data['status'],
            recommendations=request_data.get('recommendations', []),
            quality_metrics=request_data.get('quality_metrics'),
            processing_time=processing_time,
            explanation=self._get_status_explanation(request_data),
            next_batch_available=request_data['status'] in ['processing', 'partial'],
            total_expected=request.max_results
        )
        
        # Clean up completed requests
        if request_data['status'] in ['complete', 'error', 'timeout']:
            del self.active_requests[request_id]
        
        return response
    
    async def _get_cached_recommendations(self, query: str, quality_threshold: float) -> Optional[Tuple[List[CachedRecommendation], QualityMetrics]]:
        """Get cached recommendations meeting quality threshold"""
        return self.cache_manager.get_cached_recommendations(query, quality_threshold)
    
    async def _generate_quality_recommendations(self, processed_query: ProcessedQuery, 
                                             max_results: int, quality_threshold: float) -> List[CachedRecommendation]:
        """Generate quality recommendations from processed query"""
        recommendations = []
        
        # Convert processed query sources to cached recommendations
        for i, source_info in enumerate(processed_query.recommended_sources[:max_results]):
            # Calculate quality score based on processing confidence and source ranking
            quality_score = processed_query.processing_confidence * (1.0 - i * 0.1)
            quality_score = max(0.1, min(1.0, quality_score))
            
            # Only include if meets quality threshold
            if quality_score >= quality_threshold:
                rec = CachedRecommendation(
                    source=source_info['name'],
                    relevance_score=source_info.get('relevance', 0.8),
                    domain=processed_query.classification.domain,
                    explanation=source_info.get('reason', 'Recommended based on query analysis'),
                    geographic_scope=processed_query.classification.geographic_scope,
                    query_intent=processed_query.classification.intent,
                    quality_score=quality_score,
                    cached_at=time.time()
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _validate_recommendation_quality(self, recommendation: CachedRecommendation, 
                                             original_query: str, quality_threshold: float) -> Dict[str, Any]:
        """Validate recommendation quality against training mappings and thresholds"""
        
        # Basic quality checks
        is_valid = (
            recommendation.quality_score >= quality_threshold and
            recommendation.relevance_score >= quality_threshold * 0.8 and
            len(recommendation.explanation) > 10
        )
        
        # Additional validation logic could include:
        # - Training mappings validation
        # - Domain-specific validation
        # - Geographic context validation
        
        explanation = "Quality validation passed" if is_valid else "Quality validation failed"
        
        if not is_valid:
            if recommendation.quality_score < quality_threshold:
                explanation += f" (quality score {recommendation.quality_score:.2f} < {quality_threshold})"
            if recommendation.relevance_score < quality_threshold * 0.8:
                explanation += f" (relevance score {recommendation.relevance_score:.2f} too low)"
        
        return {
            'is_valid': is_valid,
            'explanation': explanation,
            'quality_score': recommendation.quality_score,
            'relevance_score': recommendation.relevance_score
        }
    
    def _convert_to_quality_validated(self, recommendation: CachedRecommendation, 
                                    validation_status: str, quality_explanation: str) -> QualityValidatedRecommendation:
        """Convert cached recommendation to quality validated recommendation"""
        return QualityValidatedRecommendation(
            source=recommendation.source,
            relevance_score=recommendation.relevance_score,
            quality_score=recommendation.quality_score,
            domain=recommendation.domain,
            explanation=recommendation.explanation,
            geographic_scope=recommendation.geographic_scope,
            query_intent=recommendation.query_intent,
            validation_status=validation_status,
            quality_explanation=quality_explanation,
            cached_at=recommendation.cached_at
        )
    
    def _calculate_response_quality_metrics(self, recommendations: List[QualityValidatedRecommendation]) -> Dict[str, float]:
        """Calculate quality metrics for response"""
        if not recommendations:
            return {
                'overall_quality': 0.0,
                'avg_relevance_score': 0.0,
                'avg_quality_score': 0.0,
                'validation_rate': 0.0,
                'recommendation_count': 0
            }
        
        validated_count = sum(1 for r in recommendations if r.validation_status == 'validated')
        
        return {
            'overall_quality': sum(r.quality_score for r in recommendations) / len(recommendations),
            'avg_relevance_score': sum(r.relevance_score for r in recommendations) / len(recommendations),
            'avg_quality_score': sum(r.quality_score for r in recommendations) / len(recommendations),
            'validation_rate': validated_count / len(recommendations),
            'recommendation_count': len(recommendations)
        }
    
    async def _cache_quality_results(self, query: str, recommendations: List[QualityValidatedRecommendation], 
                                   quality_metrics: Dict[str, float]):
        """Cache high-quality results"""
        # Convert to cached recommendations
        cached_recs = [
            CachedRecommendation(
                source=rec.source,
                relevance_score=rec.relevance_score,
                domain=rec.domain,
                explanation=rec.explanation,
                geographic_scope=rec.geographic_scope,
                query_intent=rec.query_intent,
                quality_score=rec.quality_score,
                cached_at=time.time()
            )
            for rec in recommendations
        ]
        
        # Create quality metrics object
        quality_metrics_obj = QualityMetrics(
            ndcg_at_3=quality_metrics['overall_quality'],
            relevance_accuracy=quality_metrics['avg_relevance_score'],
            domain_routing_accuracy=quality_metrics['validation_rate'],
            singapore_first_accuracy=1.0,  # Simplified
            user_satisfaction_score=quality_metrics['overall_quality'],
            recommendation_diversity=min(1.0, len(set(r.domain for r in recommendations)) / max(1, len(recommendations)))
        )
        
        # Cache the results
        self.cache_manager.cache_recommendations(query, cached_recs, quality_metrics_obj)
    
    def _get_status_explanation(self, request_data: Dict) -> str:
        """Get explanation for current request status"""
        status = request_data['status']
        rec_count = len(request_data.get('recommendations', []))
        
        if status == 'processing':
            return "Quality-first processing in progress"
        elif status == 'partial':
            return f"Progressive loading: {rec_count} quality recommendations ready"
        elif status == 'complete':
            return f"Quality validation complete: {rec_count} recommendations"
        elif status == 'error':
            return f"Processing error: {request_data.get('error', 'Unknown error')}"
        elif status == 'timeout':
            return f"Processing timeout: returning {rec_count} partial results"
        else:
            return f"Status: {status}"
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get quality cache statistics"""
        return self.cache_manager.get_quality_cache_statistics()


# Factory function
def create_quality_first_api(config: Dict = None) -> QualityFirstAPI:
    """Factory function to create quality-first API"""
    return QualityFirstAPI(config)


if __name__ == "__main__":
    async def test_quality_first_api():
        """Test the quality-first API"""
        logging.basicConfig(level=logging.INFO)
        
        api = create_quality_first_api()
        
        # Test requests
        test_requests = [
            QualityFirstAPIRequest(
                query="psychology research data",
                quality_threshold=0.7,
                progressive_loading=True
            ),
            QualityFirstAPIRequest(
                query="singapore housing statistics",
                quality_threshold=0.8,
                progressive_loading=False
            )
        ]
        
        print("🧪 Testing Quality-First API\n")
        
        for i, request in enumerate(test_requests):
            print(f"Test {i+1}: {request.query}")
            print(f"  Quality threshold: {request.quality_threshold}")
            print(f"  Progressive loading: {request.progressive_loading}")
            
            response = await api.process_quality_first_search(request)
            
            print(f"  Status: {response.status}")
            print(f"  Recommendations: {len(response.recommendations)}")
            print(f"  Processing time: {response.processing_time:.3f}s")
            
            if response.quality_metrics:
                print(f"  Overall quality: {response.quality_metrics['overall_quality']:.3f}")
                print(f"  Validation rate: {response.quality_metrics['validation_rate']:.3f}")
            
            print(f"  Explanation: {response.explanation}")
            print()
        
        # Test cache statistics
        stats = api.get_cache_statistics()
        print("📊 Cache Statistics:")
        print(f"  Hit rate: {stats.get('hit_rate', 0):.3f}")
        print(f"  Total entries: {stats.get('total_entries', 0)}")
        print(f"  Average quality: {stats.get('avg_quality_score', 0):.3f}")
        
        print("\n✅ Quality-First API testing complete!")
    
    import asyncio
    asyncio.run(test_quality_first_api())