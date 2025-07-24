"""
Quality Validation Middleware
Middleware for validating recommendation quality before serving to users
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class QualityValidationMiddleware:
    """Middleware to validate recommendation quality before serving"""
    
    def __init__(self, 
                 min_quality_threshold: float = 0.7,
                 min_relevance_threshold: float = 0.6,
                 max_response_time: float = 30.0,
                 enable_quality_logging: bool = True):
        
        self.min_quality_threshold = min_quality_threshold
        self.min_relevance_threshold = min_relevance_threshold
        self.max_response_time = max_response_time
        self.enable_quality_logging = enable_quality_logging
        
        # Quality tracking
        self.quality_stats = {
            'total_requests': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'avg_quality_score': 0.0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"🛡️ QualityValidationMiddleware initialized")
        logger.info(f"  Quality threshold: {min_quality_threshold}")
        logger.info(f"  Relevance threshold: {min_relevance_threshold}")
        logger.info(f"  Max response time: {max_response_time}s")
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with quality validation"""
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Only validate API responses with recommendations
        if self._should_validate_response(request):
            response = await self._validate_response_quality(request, response, start_time)
        
        return response
    
    def _should_validate_response(self, request: Request) -> bool:
        """Determine if response should be quality validated"""
        path = request.url.path
        
        # Validate search and recommendation endpoints
        validation_paths = ['/api/search', '/api/quality-search', '/api/recommendations']
        
        return any(path.startswith(vp) for vp in validation_paths)
    
    async def _validate_response_quality(self, request: Request, response: Response, start_time: float) -> Response:
        """Validate response quality and modify if necessary"""
        processing_time = time.time() - start_time
        
        try:
            # Parse response body
            if hasattr(response, 'body'):
                response_data = self._parse_response_body(response.body)
            else:
                return response
            
            # Extract recommendations from response
            recommendations = self._extract_recommendations(response_data)
            
            if not recommendations:
                return response
            
            # Validate quality
            validation_result = self._validate_recommendations_quality(
                recommendations, processing_time
            )
            
            # Update statistics
            self._update_quality_statistics(validation_result, processing_time)
            
            # Modify response based on validation
            if validation_result['quality_passed']:
                # Add quality metadata to response
                response_data = self._add_quality_metadata(response_data, validation_result)
                
                if self.enable_quality_logging:
                    logger.info(f"✅ Quality validation passed: {validation_result['overall_quality']:.3f}")
                
            else:
                # Handle quality failure
                response_data = self._handle_quality_failure(response_data, validation_result)
                
                if self.enable_quality_logging:
                    logger.warning(f"⚠️ Quality validation failed: {validation_result['failure_reason']}")
            
            # Create new response with validated data
            return JSONResponse(
                content=response_data,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            return response
    
    def _parse_response_body(self, body: bytes) -> Dict[str, Any]:
        """Parse response body to extract data"""
        try:
            import json
            return json.loads(body.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to parse response body: {e}")
            return {}
    
    def _extract_recommendations(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recommendations from response data"""
        # Handle different response formats
        if 'recommendations' in response_data:
            return response_data['recommendations']
        elif 'results' in response_data:
            return response_data['results']
        elif 'datasets' in response_data:
            return response_data['datasets']
        elif isinstance(response_data.get('data'), list):
            return response_data['data']
        else:
            return []
    
    def _validate_recommendations_quality(self, recommendations: List[Dict[str, Any]], 
                                        processing_time: float) -> Dict[str, Any]:
        """Validate quality of recommendations"""
        
        if not recommendations:
            return {
                'quality_passed': False,
                'failure_reason': 'No recommendations found',
                'overall_quality': 0.0,
                'recommendation_count': 0
            }
        
        # Check processing time
        if processing_time > self.max_response_time:
            return {
                'quality_passed': False,
                'failure_reason': f'Processing time exceeded {self.max_response_time}s',
                'overall_quality': 0.0,
                'recommendation_count': len(recommendations),
                'processing_time': processing_time
            }
        
        # Calculate quality metrics
        quality_scores = []
        relevance_scores = []
        validation_failures = []
        
        for i, rec in enumerate(recommendations):
            # Extract quality and relevance scores
            quality_score = self._extract_quality_score(rec)
            relevance_score = self._extract_relevance_score(rec)
            
            quality_scores.append(quality_score)
            relevance_scores.append(relevance_score)
            
            # Check individual recommendation quality
            if quality_score < self.min_quality_threshold:
                validation_failures.append(f"Recommendation {i+1}: quality {quality_score:.3f} < {self.min_quality_threshold}")
            
            if relevance_score < self.min_relevance_threshold:
                validation_failures.append(f"Recommendation {i+1}: relevance {relevance_score:.3f} < {self.min_relevance_threshold}")
        
        # Calculate overall metrics
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        overall_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Determine if quality validation passed
        quality_passed = (
            overall_quality >= self.min_quality_threshold and
            overall_relevance >= self.min_relevance_threshold and
            len(validation_failures) == 0
        )
        
        return {
            'quality_passed': quality_passed,
            'failure_reason': '; '.join(validation_failures) if validation_failures else None,
            'overall_quality': overall_quality,
            'overall_relevance': overall_relevance,
            'recommendation_count': len(recommendations),
            'processing_time': processing_time,
            'quality_scores': quality_scores,
            'relevance_scores': relevance_scores
        }
    
    def _extract_quality_score(self, recommendation: Dict[str, Any]) -> float:
        """Extract quality score from recommendation"""
        # Try different field names for quality score
        quality_fields = ['quality_score', 'quality', 'score', 'confidence']
        
        for field in quality_fields:
            if field in recommendation:
                try:
                    return float(recommendation[field])
                except (ValueError, TypeError):
                    continue
        
        # Fallback: estimate quality from available fields
        if 'relevance_score' in recommendation:
            return float(recommendation['relevance_score'])
        elif 'relevance' in recommendation:
            return float(recommendation['relevance'])
        else:
            return 0.5  # Default moderate quality
    
    def _extract_relevance_score(self, recommendation: Dict[str, Any]) -> float:
        """Extract relevance score from recommendation"""
        # Try different field names for relevance score
        relevance_fields = ['relevance_score', 'relevance', 'score']
        
        for field in relevance_fields:
            if field in recommendation:
                try:
                    return float(recommendation[field])
                except (ValueError, TypeError):
                    continue
        
        # Fallback: estimate from quality score
        if 'quality_score' in recommendation:
            return float(recommendation['quality_score'])
        else:
            return 0.5  # Default moderate relevance
    
    def _add_quality_metadata(self, response_data: Dict[str, Any], 
                            validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Add quality metadata to response"""
        
        # Add quality validation metadata
        response_data['quality_validation'] = {
            'status': 'passed',
            'overall_quality': validation_result['overall_quality'],
            'overall_relevance': validation_result['overall_relevance'],
            'recommendation_count': validation_result['recommendation_count'],
            'processing_time': validation_result['processing_time'],
            'quality_threshold': self.min_quality_threshold,
            'relevance_threshold': self.min_relevance_threshold
        }
        
        # Add individual quality scores to recommendations
        recommendations = self._extract_recommendations(response_data)
        if recommendations and 'quality_scores' in validation_result:
            for i, rec in enumerate(recommendations):
                if i < len(validation_result['quality_scores']):
                    rec['validated_quality_score'] = validation_result['quality_scores'][i]
                if i < len(validation_result['relevance_scores']):
                    rec['validated_relevance_score'] = validation_result['relevance_scores'][i]
                rec['quality_validation_status'] = 'passed'
        
        return response_data
    
    def _handle_quality_failure(self, response_data: Dict[str, Any], 
                              validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality validation failure"""
        
        # Add quality failure metadata
        response_data['quality_validation'] = {
            'status': 'failed',
            'failure_reason': validation_result['failure_reason'],
            'overall_quality': validation_result['overall_quality'],
            'recommendation_count': validation_result['recommendation_count'],
            'quality_threshold': self.min_quality_threshold,
            'action_taken': 'filtered_low_quality_results'
        }
        
        # Filter out low-quality recommendations
        recommendations = self._extract_recommendations(response_data)
        if recommendations and 'quality_scores' in validation_result:
            filtered_recommendations = []
            
            for i, rec in enumerate(recommendations):
                if i < len(validation_result['quality_scores']):
                    quality_score = validation_result['quality_scores'][i]
                    relevance_score = validation_result['relevance_scores'][i]
                    
                    if (quality_score >= self.min_quality_threshold * 0.8 and 
                        relevance_score >= self.min_relevance_threshold * 0.8):
                        
                        rec['validated_quality_score'] = quality_score
                        rec['validated_relevance_score'] = relevance_score
                        rec['quality_validation_status'] = 'conditionally_passed'
                        filtered_recommendations.append(rec)
                    else:
                        rec['quality_validation_status'] = 'failed'
            
            # Update recommendations in response
            if 'recommendations' in response_data:
                response_data['recommendations'] = filtered_recommendations
            elif 'results' in response_data:
                response_data['results'] = filtered_recommendations
            elif 'datasets' in response_data:
                response_data['datasets'] = filtered_recommendations
            
            # Add warning about filtered results
            response_data['warning'] = f"Filtered {len(recommendations) - len(filtered_recommendations)} low-quality recommendations"
        
        return response_data
    
    def _update_quality_statistics(self, validation_result: Dict[str, Any], processing_time: float):
        """Update quality statistics"""
        self.quality_stats['total_requests'] += 1
        
        if validation_result['quality_passed']:
            self.quality_stats['quality_passed'] += 1
        else:
            self.quality_stats['quality_failed'] += 1
        
        # Update running averages
        total = self.quality_stats['total_requests']
        
        # Update average quality score
        current_avg_quality = self.quality_stats['avg_quality_score']
        new_quality = validation_result['overall_quality']
        self.quality_stats['avg_quality_score'] = (
            (current_avg_quality * (total - 1) + new_quality) / total
        )
        
        # Update average response time
        current_avg_time = self.quality_stats['avg_response_time']
        self.quality_stats['avg_response_time'] = (
            (current_avg_time * (total - 1) + processing_time) / total
        )
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality validation statistics"""
        total = self.quality_stats['total_requests']
        
        return {
            'total_requests': total,
            'quality_pass_rate': self.quality_stats['quality_passed'] / total if total > 0 else 0.0,
            'quality_fail_rate': self.quality_stats['quality_failed'] / total if total > 0 else 0.0,
            'avg_quality_score': self.quality_stats['avg_quality_score'],
            'avg_response_time': self.quality_stats['avg_response_time'],
            'quality_threshold': self.min_quality_threshold,
            'relevance_threshold': self.min_relevance_threshold,
            'max_response_time': self.max_response_time
        }
    
    def reset_statistics(self):
        """Reset quality statistics"""
        self.quality_stats = {
            'total_requests': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'avg_quality_score': 0.0,
            'avg_response_time': 0.0
        }
        
        logger.info("📊 Quality validation statistics reset")


# Factory function
def create_quality_validation_middleware(
    min_quality_threshold: float = 0.7,
    min_relevance_threshold: float = 0.6,
    max_response_time: float = 30.0,
    enable_quality_logging: bool = True
) -> QualityValidationMiddleware:
    """Factory function to create quality validation middleware"""
    
    return QualityValidationMiddleware(
        min_quality_threshold=min_quality_threshold,
        min_relevance_threshold=min_relevance_threshold,
        max_response_time=max_response_time,
        enable_quality_logging=enable_quality_logging
    )


if __name__ == "__main__":
    # Test the middleware
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    async def test_quality_middleware():
        """Test quality validation middleware"""
        logging.basicConfig(level=logging.INFO)
        
        middleware = create_quality_validation_middleware()
        
        # Mock request and response
        class MockRequest:
            def __init__(self, path: str):
                self.url = type('URL', (), {'path': path})()
        
        # Test data
        test_responses = [
            {
                'recommendations': [
                    {'source': 'Test Source 1', 'quality_score': 0.8, 'relevance_score': 0.9},
                    {'source': 'Test Source 2', 'quality_score': 0.7, 'relevance_score': 0.8}
                ]
            },
            {
                'recommendations': [
                    {'source': 'Low Quality Source', 'quality_score': 0.4, 'relevance_score': 0.3}
                ]
            }
        ]
        
        print("🧪 Testing Quality Validation Middleware\n")
        
        for i, response_data in enumerate(test_responses):
            print(f"Test {i+1}:")
            
            # Mock response
            class MockResponse:
                def __init__(self, data):
                    import json
                    self.body = json.dumps(data).encode('utf-8')
                    self.status_code = 200
                    self.headers = {}
            
            request = MockRequest('/api/search')
            response = MockResponse(response_data)
            
            # Validate response
            start_time = time.time()
            validated_response = await middleware._validate_response_quality(
                request, response, start_time
            )
            
            print(f"  Original recommendations: {len(response_data['recommendations'])}")
            
            if hasattr(validated_response, 'body'):
                import json
                validated_data = json.loads(validated_response.body)
                
                if 'quality_validation' in validated_data:
                    validation = validated_data['quality_validation']
                    print(f"  Validation status: {validation['status']}")
                    print(f"  Overall quality: {validation.get('overall_quality', 0):.3f}")
                    
                    if 'failure_reason' in validation:
                        print(f"  Failure reason: {validation['failure_reason']}")
                
                filtered_recs = validated_data.get('recommendations', [])
                print(f"  Filtered recommendations: {len(filtered_recs)}")
            
            print()
        
        # Test statistics
        stats = middleware.get_quality_statistics()
        print("📊 Quality Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Quality pass rate: {stats['quality_pass_rate']:.3f}")
        print(f"  Average quality score: {stats['avg_quality_score']:.3f}")
        
        print("\n✅ Quality validation middleware testing complete!")
    
    asyncio.run(test_quality_middleware())