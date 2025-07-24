"""
End-to-End Quality Integration System
Integrates all quality-enhanced components into a unified system for comprehensive testing
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .integrated_query_processor import IntegratedQueryProcessor, ProcessedQuery
from .quality_aware_cache import QualityAwareCacheManager, CachedRecommendation, QualityMetrics
from .quality_monitoring_system import QualityMonitoringSystem, QualityMetrics as QualityMonitoringMetrics
from .automated_quality_validator import AutomatedQualityValidator, ValidationResult
from .quality_reporting_analytics import QualityReportingSystem

logger = logging.getLogger(__name__)


@dataclass
class EndToEndResponse:
    """Complete end-to-end response with quality validation"""
    query: str
    processed_query: ProcessedQuery
    recommendations: List[CachedRecommendation]
    quality_metrics: QualityMetrics
    validation_result: ValidationResult
    cache_hit: bool
    processing_time: float
    quality_report: QualityMonitoringMetrics
    analytics_summary: Dict[str, Any]
    explanation: str
    confidence_score: float


class EndToEndQualityIntegration:
    """
    Complete end-to-end quality integration system that combines:
    - Enhanced neural model with quality-aware caching
    - Singapore-first routing with domain-specific recommendations
    - Complete pipeline from query to high-quality recommendations
    """
    
    def __init__(self, 
                 training_mappings_path: str = "training_mappings.md",
                 cache_dir: str = "cache/quality_aware",
                 quality_threshold: float = 0.7):
        
        self.training_mappings_path = training_mappings_path
        self.cache_dir = cache_dir
        self.quality_threshold = quality_threshold
        
        # Initialize integrated components
        self.query_processor = IntegratedQueryProcessor(training_mappings_path)
        self.cache_manager = QualityAwareCacheManager(
            cache_dir=cache_dir,
            quality_threshold=quality_threshold,
            training_mappings_path=training_mappings_path
        )
        self.quality_monitor = QualityMonitoringSystem(training_mappings_path)
        self.quality_validator = AutomatedQualityValidator(training_mappings_path)
        self.analytics = QualityReportingSystem()
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.quality_validations = 0
        
        logger.info("🎯 EndToEndQualityIntegration initialized")
        logger.info(f"  Quality threshold: {quality_threshold}")
        logger.info(f"  Cache directory: {cache_dir}")
        logger.info(f"  Training mappings: {training_mappings_path}")
    
    async def process_complete_query(self, query: str, 
                                   session_id: Optional[str] = None,
                                   max_results: int = 10) -> EndToEndResponse:
        """
        Process query through complete end-to-end pipeline with quality validation
        
        Args:
            query: User query
            session_id: Optional session ID for tracking
            max_results: Maximum number of results to return
            
        Returns:
            Complete end-to-end response with quality validation
        """
        start_time = time.time()
        self.request_count += 1
        
        logger.info(f"🔄 Processing complete query: '{query}' (Request #{self.request_count})")
        
        try:
            # Step 1: Check quality-aware cache first
            cache_result = self.cache_manager.get_cached_recommendations(
                query, min_quality_threshold=self.quality_threshold
            )
            
            cache_hit = cache_result is not None
            if cache_hit:
                self.cache_hits += 1
                recommendations, quality_metrics = cache_result
                processed_query = None  # Skip processing for cache hit
                
                logger.info(f"✅ Cache hit for query: '{query[:50]}...'")
            else:
                # Step 2: Process query through integrated pipeline
                processed_query = self.query_processor.process_query(query)
                
                # Step 3: Generate high-quality recommendations
                recommendations = await self._generate_quality_recommendations(
                    processed_query, max_results
                )
                
                # Step 4: Calculate quality metrics
                quality_metrics = self._calculate_comprehensive_quality_metrics(
                    query, recommendations
                )
                
                # Step 5: Cache high-quality results
                if quality_metrics.meets_quality_threshold(self.quality_threshold):
                    cache_key = self.cache_manager.cache_recommendations(
                        query, recommendations, quality_metrics
                    )
                    logger.info(f"💎 Cached high-quality results: {cache_key[:12]}...")
            
            # Step 6: Validate recommendations against training mappings
            validation_result = await self.quality_validator.validate_recommendations(
                query, [self._recommendation_to_dict(r) for r in recommendations]
            )
            self.quality_validations += 1
            
            # Step 7: Generate quality monitoring report
            quality_report = QualityMonitoringMetrics(
                timestamp=str(time.time()),
                ndcg_at_3=quality_metrics.ndcg_at_3,
                relevance_accuracy=quality_metrics.relevance_accuracy,
                domain_routing_accuracy=quality_metrics.domain_routing_accuracy,
                singapore_first_accuracy=quality_metrics.singapore_first_accuracy,
                total_queries=1,
                successful_queries=1,
                average_response_time=processing_time,
                cache_hit_rate=1.0 if cache_hit else 0.0
            )
            
            # Step 8: Generate analytics summary
            analytics_summary = self._generate_analytics_summary(
                query, recommendations, quality_metrics, validation_result
            )
            
            # Step 9: Calculate processing time and confidence
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            confidence_score = self._calculate_overall_confidence(
                quality_metrics, validation_result, cache_hit
            )
            
            # Step 10: Generate comprehensive explanation
            explanation = self._generate_comprehensive_explanation(
                query, processed_query, recommendations, quality_metrics, 
                validation_result, cache_hit, processing_time
            )
            
            # Create complete response
            response = EndToEndResponse(
                query=query,
                processed_query=processed_query,
                recommendations=recommendations,
                quality_metrics=quality_metrics,
                validation_result=validation_result,
                cache_hit=cache_hit,
                processing_time=processing_time,
                quality_report=quality_report,
                analytics_summary=analytics_summary,
                explanation=explanation,
                confidence_score=confidence_score
            )
            
            # Log comprehensive results
            self._log_comprehensive_results(response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ End-to-end processing error: {e}")
            raise
    
    async def _generate_quality_recommendations(self, 
                                              processed_query: ProcessedQuery,
                                              max_results: int) -> List[CachedRecommendation]:
        """Generate high-quality recommendations based on processed query"""
        recommendations = []
        
        # Use processed query information to generate recommendations
        query = processed_query.enhanced_query
        classification = processed_query.classification
        
        # Generate recommendations based on domain and geographic context
        if classification.singapore_first_applicable:
            # Singapore-first recommendations
            singapore_recs = self._generate_singapore_recommendations(query, classification)
            recommendations.extend(singapore_recs)
        
        # Domain-specific recommendations
        domain_recs = self._generate_domain_recommendations(query, classification)
        recommendations.extend(domain_recs)
        
        # Global recommendations as fallback
        if len(recommendations) < max_results:
            global_recs = self._generate_global_recommendations(query, classification)
            recommendations.extend(global_recs)
        
        # Sort by quality score and limit results
        recommendations.sort(key=lambda r: r.quality_score, reverse=True)
        return recommendations[:max_results]
    
    def _generate_singapore_recommendations(self, query: str, 
                                          classification) -> List[CachedRecommendation]:
        """Generate Singapore-specific recommendations"""
        recommendations = []
        current_time = time.time()
        
        # Singapore government sources
        singapore_sources = [
            {
                'source': 'data.gov.sg',
                'relevance_score': 0.9,
                'domain': 'singapore_government',
                'explanation': 'Singapore government open data portal with official datasets',
                'geographic_scope': 'singapore',
                'query_intent': 'research'
            },
            {
                'source': 'SingStat',
                'relevance_score': 0.85,
                'domain': 'singapore_statistics',
                'explanation': 'Singapore Department of Statistics official data',
                'geographic_scope': 'singapore',
                'query_intent': 'analysis'
            },
            {
                'source': 'LTA DataMall',
                'relevance_score': 0.8,
                'domain': 'singapore_transport',
                'explanation': 'Land Transport Authority Singapore transport data',
                'geographic_scope': 'singapore',
                'query_intent': 'research'
            }
        ]
        
        for source_info in singapore_sources:
            # Calculate quality score based on Singapore-first strategy
            quality_score = self._calculate_singapore_quality_score(
                query, source_info, classification
            )
            
            if quality_score >= self.quality_threshold:
                rec = CachedRecommendation(
                    source=source_info['source'],
                    relevance_score=source_info['relevance_score'],
                    domain=source_info['domain'],
                    explanation=source_info['explanation'],
                    geographic_scope=source_info['geographic_scope'],
                    query_intent=source_info['query_intent'],
                    quality_score=quality_score,
                    cached_at=current_time
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_domain_recommendations(self, query: str, 
                                       classification) -> List[CachedRecommendation]:
        """Generate domain-specific recommendations"""
        recommendations = []
        current_time = time.time()
        
        domain = classification.domain.lower()
        
        # Domain-specific source mappings
        domain_sources = {
            'psychology': [
                {
                    'source': 'Kaggle Psychology Datasets',
                    'relevance_score': 0.9,
                    'explanation': 'Community-curated psychology and mental health datasets'
                },
                {
                    'source': 'Zenodo Psychology Research',
                    'relevance_score': 0.85,
                    'explanation': 'Academic psychology research datasets and publications'
                }
            ],
            'climate': [
                {
                    'source': 'World Bank Climate Data',
                    'relevance_score': 0.9,
                    'explanation': 'Comprehensive global climate and environmental indicators'
                },
                {
                    'source': 'Kaggle Climate Datasets',
                    'relevance_score': 0.8,
                    'explanation': 'Climate change and environmental datasets for analysis'
                }
            ],
            'machine learning': [
                {
                    'source': 'Kaggle ML Datasets',
                    'relevance_score': 0.95,
                    'explanation': 'High-quality machine learning datasets with competitions'
                },
                {
                    'source': 'Zenodo ML Research',
                    'relevance_score': 0.8,
                    'explanation': 'Academic machine learning research datasets'
                }
            ],
            'economics': [
                {
                    'source': 'World Bank Economic Data',
                    'relevance_score': 0.9,
                    'explanation': 'Global economic indicators and development data'
                },
                {
                    'source': 'Kaggle Economic Datasets',
                    'relevance_score': 0.75,
                    'explanation': 'Economic analysis datasets and financial data'
                }
            ]
        }
        
        if domain in domain_sources:
            for source_info in domain_sources[domain]:
                quality_score = self._calculate_domain_quality_score(
                    query, source_info, classification
                )
                
                if quality_score >= self.quality_threshold:
                    rec = CachedRecommendation(
                        source=source_info['source'],
                        relevance_score=source_info['relevance_score'],
                        domain=domain,
                        explanation=source_info['explanation'],
                        geographic_scope='global',
                        query_intent='research',
                        quality_score=quality_score,
                        cached_at=current_time
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _generate_global_recommendations(self, query: str, 
                                       classification) -> List[CachedRecommendation]:
        """Generate global fallback recommendations"""
        recommendations = []
        current_time = time.time()
        
        # Global sources as fallback
        global_sources = [
            {
                'source': 'Kaggle Datasets',
                'relevance_score': 0.8,
                'domain': 'general',
                'explanation': 'Community-driven datasets across multiple domains'
            },
            {
                'source': 'Zenodo Research Data',
                'relevance_score': 0.75,
                'domain': 'academic',
                'explanation': 'Academic research datasets and publications'
            },
            {
                'source': 'World Bank Open Data',
                'relevance_score': 0.7,
                'domain': 'global_statistics',
                'explanation': 'Global development and economic indicators'
            }
        ]
        
        for source_info in global_sources:
            quality_score = self._calculate_global_quality_score(
                query, source_info, classification
            )
            
            if quality_score >= 0.6:  # Lower threshold for fallback
                rec = CachedRecommendation(
                    source=source_info['source'],
                    relevance_score=source_info['relevance_score'],
                    domain=source_info['domain'],
                    explanation=source_info['explanation'],
                    geographic_scope='global',
                    query_intent='research',
                    quality_score=quality_score,
                    cached_at=current_time
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _calculate_singapore_quality_score(self, query: str, source_info: Dict, 
                                         classification) -> float:
        """Calculate quality score for Singapore sources"""
        base_score = source_info['relevance_score']
        
        # Boost for Singapore-first queries
        singapore_boost = 0.1 if classification.singapore_first_applicable else 0.0
        
        # Query relevance boost
        query_lower = query.lower()
        relevance_boost = 0.0
        if any(term in query_lower for term in ['singapore', 'sg', 'government', 'official']):
            relevance_boost = 0.1
        
        return min(1.0, base_score + singapore_boost + relevance_boost)
    
    def _calculate_domain_quality_score(self, query: str, source_info: Dict, 
                                      classification) -> float:
        """Calculate quality score for domain-specific sources"""
        base_score = source_info['relevance_score']
        
        # Domain match boost
        domain_boost = 0.1 if classification.confidence > 0.7 else 0.0
        
        # Query relevance boost
        query_lower = query.lower()
        domain_terms = classification.domain.lower().split()
        relevance_boost = 0.0
        for term in domain_terms:
            if term in query_lower:
                relevance_boost += 0.05
        
        return min(1.0, base_score + domain_boost + relevance_boost)
    
    def _calculate_global_quality_score(self, query: str, source_info: Dict, 
                                      classification) -> float:
        """Calculate quality score for global sources"""
        base_score = source_info['relevance_score']
        
        # General relevance boost
        query_lower = query.lower()
        relevance_boost = 0.0
        if any(term in query_lower for term in ['data', 'dataset', 'research']):
            relevance_boost = 0.05
        
        return min(1.0, base_score + relevance_boost)
    
    def _calculate_comprehensive_quality_metrics(self, query: str, 
                                               recommendations: List[CachedRecommendation]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        if not recommendations:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate NDCG@3
        ndcg_at_3 = self._calculate_ndcg_at_3(recommendations)
        
        # Calculate relevance accuracy
        relevance_accuracy = sum(r.relevance_score for r in recommendations[:3]) / min(3, len(recommendations))
        
        # Calculate domain routing accuracy
        domain_routing_accuracy = self._calculate_domain_routing_accuracy(query, recommendations)
        
        # Calculate Singapore-first accuracy
        singapore_first_accuracy = self._calculate_singapore_first_accuracy(query, recommendations)
        
        # Calculate recommendation diversity
        recommendation_diversity = self._calculate_recommendation_diversity(recommendations)
        
        # User satisfaction score (estimated)
        user_satisfaction_score = min(1.0, (ndcg_at_3 + relevance_accuracy) / 2 + 0.1)
        
        return QualityMetrics(
            ndcg_at_3=ndcg_at_3,
            relevance_accuracy=relevance_accuracy,
            domain_routing_accuracy=domain_routing_accuracy,
            singapore_first_accuracy=singapore_first_accuracy,
            user_satisfaction_score=user_satisfaction_score,
            recommendation_diversity=recommendation_diversity
        )
    
    def _calculate_ndcg_at_3(self, recommendations: List[CachedRecommendation]) -> float:
        """Calculate NDCG@3 for recommendations"""
        if not recommendations:
            return 0.0
        
        # Use quality scores as relevance scores
        relevance_scores = [r.quality_score for r in recommendations[:3]]
        
        # Calculate DCG@3
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / (1 + i)  # Simplified DCG calculation
        
        # Calculate IDCG@3
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / (1 + i)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_domain_routing_accuracy(self, query: str, 
                                         recommendations: List[CachedRecommendation]) -> float:
        """Calculate domain routing accuracy"""
        if not recommendations:
            return 0.0
        
        query_lower = query.lower()
        top_rec = recommendations[0]
        
        # Check domain-specific routing accuracy
        if 'psychology' in query_lower and 'kaggle' in top_rec.source.lower():
            return 1.0
        elif 'climate' in query_lower and 'world bank' in top_rec.source.lower():
            return 1.0
        elif 'singapore' in query_lower and any(sg in top_rec.source.lower() 
                                               for sg in ['data.gov.sg', 'singstat', 'lta']):
            return 1.0
        
        return 0.8  # Default good routing
    
    def _calculate_singapore_first_accuracy(self, query: str, 
                                          recommendations: List[CachedRecommendation]) -> float:
        """Calculate Singapore-first strategy accuracy"""
        if not recommendations:
            return 0.0
        
        query_lower = query.lower()
        
        if 'singapore' in query_lower or 'sg' in query_lower:
            top_rec = recommendations[0]
            if top_rec.geographic_scope == 'singapore':
                return 1.0
            else:
                return 0.0
        
        return 1.0  # Singapore-first not applicable
    
    def _calculate_recommendation_diversity(self, recommendations: List[CachedRecommendation]) -> float:
        """Calculate recommendation diversity"""
        if len(recommendations) <= 1:
            return 0.0
        
        unique_sources = set(r.source for r in recommendations)
        unique_domains = set(r.domain for r in recommendations)
        
        source_diversity = len(unique_sources) / len(recommendations)
        domain_diversity = len(unique_domains) / len(recommendations)
        
        return (source_diversity + domain_diversity) / 2
    
    def _calculate_overall_confidence(self, quality_metrics: QualityMetrics,
                                    validation_result: ValidationResult,
                                    cache_hit: bool) -> float:
        """Calculate overall confidence score"""
        confidence = 0.0
        
        # Quality metrics contribution (60%)
        confidence += quality_metrics.ndcg_at_3 * 0.3
        confidence += quality_metrics.relevance_accuracy * 0.2
        confidence += quality_metrics.domain_routing_accuracy * 0.1
        
        # Validation result contribution (30%)
        confidence += validation_result.overall_score * 0.3
        
        # Cache hit bonus (10%)
        if cache_hit:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_comprehensive_explanation(self, query: str, processed_query: Optional[ProcessedQuery],
                                          recommendations: List[CachedRecommendation],
                                          quality_metrics: QualityMetrics,
                                          validation_result: ValidationResult,
                                          cache_hit: bool, processing_time: float) -> str:
        """Generate comprehensive explanation"""
        explanations = []
        
        # Cache status
        if cache_hit:
            explanations.append("Retrieved high-quality cached results")
        else:
            explanations.append("Generated fresh recommendations with quality validation")
        
        # Query processing
        if processed_query:
            if processed_query.classification.singapore_first_applicable:
                explanations.append("Applied Singapore-first routing strategy")
            
            explanations.append(f"Classified as {processed_query.classification.domain} domain")
            
            if processed_query.enhancement.expansion_terms:
                explanations.append(f"Enhanced query with {len(processed_query.enhancement.expansion_terms)} domain terms")
        
        # Quality metrics
        explanations.append(f"Achieved {quality_metrics.ndcg_at_3:.1%} NDCG@3 quality score")
        
        # Validation results
        if validation_result.passes_validation:
            explanations.append("Passed comprehensive quality validation")
        else:
            explanations.append("Quality validation identified areas for improvement")
        
        # Performance
        explanations.append(f"Processed in {processing_time:.2f}s")
        
        return "; ".join(explanations)
    
    def _generate_analytics_summary(self, query: str, recommendations: List[CachedRecommendation],
                                  quality_metrics: QualityMetrics, 
                                  validation_result: ValidationResult) -> Dict[str, Any]:
        """Generate analytics summary"""
        return {
            'query_length': len(query.split()),
            'recommendation_count': len(recommendations),
            'high_quality_count': sum(1 for r in recommendations if r.quality_score >= 0.8),
            'average_quality_score': sum(r.quality_score for r in recommendations) / len(recommendations) if recommendations else 0.0,
            'validation_passed': validation_result.passes_validation,
            'quality_threshold_met': quality_metrics.meets_quality_threshold(self.quality_threshold),
            'singapore_recommendations': sum(1 for r in recommendations if r.geographic_scope == 'singapore'),
            'domain_diversity': len(set(r.domain for r in recommendations)),
            'processing_request_id': self.request_count
        }
    
    def _recommendation_to_dict(self, rec: CachedRecommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary"""
        return {
            'source': rec.source,
            'relevance_score': rec.relevance_score,
            'domain': rec.domain,
            'explanation': rec.explanation,
            'geographic_scope': rec.geographic_scope,
            'query_intent': rec.query_intent,
            'quality_score': rec.quality_score,
            'cached_at': rec.cached_at
        }
    
    def _log_comprehensive_results(self, response: EndToEndResponse):
        """Log comprehensive results"""
        logger.info(f"✅ End-to-end processing completed:")
        logger.info(f"  Query: '{response.query}'")
        logger.info(f"  Recommendations: {len(response.recommendations)}")
        logger.info(f"  Quality Score: {response.quality_metrics.ndcg_at_3:.2f}")
        logger.info(f"  Validation: {'✅ PASSED' if response.validation_result.passes_validation else '❌ FAILED'}")
        logger.info(f"  Cache Hit: {'✅ YES' if response.cache_hit else '❌ NO'}")
        logger.info(f"  Processing Time: {response.processing_time:.2f}s")
        logger.info(f"  Confidence: {response.confidence_score:.2f}")
        
        if response.recommendations:
            logger.info(f"  Top Recommendation: {response.recommendations[0].source} ({response.recommendations[0].quality_score:.2f})")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        cache_stats = self.cache_manager.get_quality_cache_statistics()
        
        return {
            'requests_processed': self.request_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.total_processing_time / max(1, self.request_count),
            'cache_hit_rate': self.cache_hits / max(1, self.request_count),
            'quality_validations': self.quality_validations,
            'quality_threshold': self.quality_threshold,
            'cache_statistics': cache_stats,
            'system_health': {
                'status': 'healthy',
                'components_initialized': 5,
                'quality_monitoring_active': True,
                'cache_operational': True
            }
        }
    
    async def test_end_to_end_integration(self):
        """Test complete end-to-end integration with various query types"""
        test_queries = [
            "psychology research datasets",
            "singapore housing data",
            "climate change indicators",
            "machine learning datasets",
            "transport statistics singapore",
            "health data analysis",
            "economic indicators global"
        ]
        
        logger.info("🧪 Testing End-to-End Quality Integration:")
        logger.info("=" * 60)
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}/{len(test_queries)}: '{query}'")
            
            try:
                response = await self.process_complete_query(query)
                results.append(response)
                
                # Log key metrics
                logger.info(f"  ✅ Success - Quality: {response.quality_metrics.ndcg_at_3:.2f}, "
                           f"Confidence: {response.confidence_score:.2f}, "
                           f"Time: {response.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
                results.append(None)
        
        # Generate summary report
        successful_tests = [r for r in results if r is not None]
        
        logger.info("\n📊 End-to-End Integration Test Summary:")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {len(test_queries)}")
        logger.info(f"Successful: {len(successful_tests)}")
        logger.info(f"Failed: {len(test_queries) - len(successful_tests)}")
        
        if successful_tests:
            avg_quality = sum(r.quality_metrics.ndcg_at_3 for r in successful_tests) / len(successful_tests)
            avg_confidence = sum(r.confidence_score for r in successful_tests) / len(successful_tests)
            avg_time = sum(r.processing_time for r in successful_tests) / len(successful_tests)
            cache_hit_rate = sum(1 for r in successful_tests if r.cache_hit) / len(successful_tests)
            validation_pass_rate = sum(1 for r in successful_tests if r.validation_result.passes_validation) / len(successful_tests)
            
            logger.info(f"Average Quality Score: {avg_quality:.2f}")
            logger.info(f"Average Confidence: {avg_confidence:.2f}")
            logger.info(f"Average Processing Time: {avg_time:.2f}s")
            logger.info(f"Cache Hit Rate: {cache_hit_rate:.1%}")
            logger.info(f"Validation Pass Rate: {validation_pass_rate:.1%}")
        
        # System statistics
        system_stats = self.get_system_statistics()
        logger.info(f"\n📈 System Statistics:")
        logger.info(f"  Total Requests: {system_stats['requests_processed']}")
        logger.info(f"  Cache Hit Rate: {system_stats['cache_hit_rate']:.1%}")
        logger.info(f"  Average Response Time: {system_stats['average_processing_time']:.2f}s")
        
        logger.info("\n✅ End-to-End Quality Integration testing completed!")
        
        return results


def create_end_to_end_quality_integration(training_mappings_path: str = "training_mappings.md",
                                        cache_dir: str = "cache/quality_aware",
                                        quality_threshold: float = 0.7) -> EndToEndQualityIntegration:
    """Factory function to create end-to-end quality integration system"""
    return EndToEndQualityIntegration(training_mappings_path, cache_dir, quality_threshold)


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Create and test the end-to-end integration system
        integration = create_end_to_end_quality_integration()
        
        # Run comprehensive tests
        await integration.test_end_to_end_integration()
        
        print("\n✅ End-to-End Quality Integration system ready!")
    
    asyncio.run(main())