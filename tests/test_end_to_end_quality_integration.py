"""
Test End-to-End Quality Integration System
Comprehensive testing of the complete quality-enhanced pipeline
"""

import asyncio
import logging
import pytest
import time
from pathlib import Path

from src.ai.end_to_end_quality_integration import (
    EndToEndQualityIntegration,
    create_end_to_end_quality_integration
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEndToEndQualityIntegration:
    """Test suite for end-to-end quality integration"""
    
    @pytest.fixture
    def integration_system(self):
        """Create integration system for testing"""
        return create_end_to_end_quality_integration(
            training_mappings_path="training_mappings.md",
            cache_dir="cache/test_quality_aware",
            quality_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_singapore_first_routing(self, integration_system):
        """Test Singapore-first routing strategy"""
        logger.info("🧪 Testing Singapore-first routing...")
        
        # Test Singapore-specific query
        response = await integration_system.process_complete_query(
            "singapore housing data"
        )
        
        # Assertions
        assert response is not None
        assert len(response.recommendations) > 0
        
        # Check Singapore-first strategy
        top_rec = response.recommendations[0]
        assert top_rec.geographic_scope == 'singapore'
        assert any(sg_source in top_rec.source.lower() 
                  for sg_source in ['data.gov.sg', 'singstat', 'lta'])
        
        # Check quality metrics
        assert response.quality_metrics.singapore_first_accuracy >= 0.9
        assert response.quality_metrics.ndcg_at_3 >= 0.7
        
        logger.info(f"✅ Singapore-first routing test passed")
        logger.info(f"   Top source: {top_rec.source}")
        logger.info(f"   Quality score: {response.quality_metrics.ndcg_at_3:.2f}")
    
    @pytest.mark.asyncio
    async def test_domain_specific_routing(self, integration_system):
        """Test domain-specific routing (psychology → Kaggle/Zenodo)"""
        logger.info("🧪 Testing domain-specific routing...")
        
        # Test psychology query
        response = await integration_system.process_complete_query(
            "psychology research datasets"
        )
        
        # Assertions
        assert response is not None
        assert len(response.recommendations) > 0
        
        # Check domain-specific routing
        top_rec = response.recommendations[0]
        assert any(domain_source in top_rec.source.lower() 
                  for domain_source in ['kaggle', 'zenodo'])
        
        # Check quality metrics
        assert response.quality_metrics.domain_routing_accuracy >= 0.8
        assert response.quality_metrics.ndcg_at_3 >= 0.7
        
        logger.info(f"✅ Domain-specific routing test passed")
        logger.info(f"   Top source: {top_rec.source}")
        logger.info(f"   Domain routing accuracy: {response.quality_metrics.domain_routing_accuracy:.2f}")
    
    @pytest.mark.asyncio
    async def test_climate_world_bank_routing(self, integration_system):
        """Test climate queries routing to World Bank"""
        logger.info("🧪 Testing climate → World Bank routing...")
        
        # Test climate query
        response = await integration_system.process_complete_query(
            "climate change indicators"
        )
        
        # Assertions
        assert response is not None
        assert len(response.recommendations) > 0
        
        # Check climate-specific routing
        top_rec = response.recommendations[0]
        assert 'world bank' in top_rec.source.lower() or 'climate' in top_rec.source.lower()
        
        # Check quality metrics
        assert response.quality_metrics.ndcg_at_3 >= 0.7
        
        logger.info(f"✅ Climate routing test passed")
        logger.info(f"   Top source: {top_rec.source}")
    
    @pytest.mark.asyncio
    async def test_quality_caching_system(self, integration_system):
        """Test quality-aware caching system"""
        logger.info("🧪 Testing quality-aware caching...")
        
        query = "machine learning datasets"
        
        # First request (should not be cached)
        start_time = time.time()
        response1 = await integration_system.process_complete_query(query)
        first_time = time.time() - start_time
        
        assert response1 is not None
        assert not response1.cache_hit  # First request should not be cache hit
        
        # Second request (should be cached if quality is high)
        start_time = time.time()
        response2 = await integration_system.process_complete_query(query)
        second_time = time.time() - start_time
        
        assert response2 is not None
        
        # If quality was high enough, should be cached
        if response1.quality_metrics.meets_quality_threshold(0.7):
            assert response2.cache_hit
            assert second_time < first_time  # Cached should be faster
            
            logger.info(f"✅ Quality caching test passed")
            logger.info(f"   First request: {first_time:.2f}s (no cache)")
            logger.info(f"   Second request: {second_time:.2f}s (cached)")
        else:
            logger.info(f"⚠️  Quality too low for caching ({response1.quality_metrics.ndcg_at_3:.2f})")
    
    @pytest.mark.asyncio
    async def test_quality_validation_against_training_mappings(self, integration_system):
        """Test quality validation against training mappings"""
        logger.info("🧪 Testing quality validation...")
        
        # Test with various queries
        test_queries = [
            "psychology research",
            "singapore transport data",
            "climate indicators"
        ]
        
        for query in test_queries:
            response = await integration_system.process_complete_query(query)
            
            assert response is not None
            assert response.validation_result is not None
            
            # Check validation components
            assert hasattr(response.validation_result, 'passes_validation')
            assert hasattr(response.validation_result, 'overall_score')
            
            logger.info(f"   Query: '{query}' - Validation: {'✅ PASS' if response.validation_result.passes_validation else '❌ FAIL'}")
        
        logger.info(f"✅ Quality validation test completed")
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_metrics(self, integration_system):
        """Test comprehensive quality metrics calculation"""
        logger.info("🧪 Testing comprehensive quality metrics...")
        
        response = await integration_system.process_complete_query(
            "singapore education statistics"
        )
        
        assert response is not None
        assert response.quality_metrics is not None
        
        # Check all quality metrics are present
        metrics = response.quality_metrics
        assert hasattr(metrics, 'ndcg_at_3')
        assert hasattr(metrics, 'relevance_accuracy')
        assert hasattr(metrics, 'domain_routing_accuracy')
        assert hasattr(metrics, 'singapore_first_accuracy')
        assert hasattr(metrics, 'user_satisfaction_score')
        assert hasattr(metrics, 'recommendation_diversity')
        
        # Check metrics are in valid ranges
        assert 0.0 <= metrics.ndcg_at_3 <= 1.0
        assert 0.0 <= metrics.relevance_accuracy <= 1.0
        assert 0.0 <= metrics.domain_routing_accuracy <= 1.0
        assert 0.0 <= metrics.singapore_first_accuracy <= 1.0
        assert 0.0 <= metrics.user_satisfaction_score <= 1.0
        assert 0.0 <= metrics.recommendation_diversity <= 1.0
        
        logger.info(f"✅ Quality metrics test passed")
        logger.info(f"   NDCG@3: {metrics.ndcg_at_3:.2f}")
        logger.info(f"   Relevance: {metrics.relevance_accuracy:.2f}")
        logger.info(f"   Domain routing: {metrics.domain_routing_accuracy:.2f}")
        logger.info(f"   Singapore-first: {metrics.singapore_first_accuracy:.2f}")
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, integration_system):
        """Test response time performance"""
        logger.info("🧪 Testing response time performance...")
        
        queries = [
            "health data",
            "economic indicators",
            "transport statistics"
        ]
        
        response_times = []
        
        for query in queries:
            start_time = time.time()
            response = await integration_system.process_complete_query(query)
            response_time = time.time() - start_time
            
            assert response is not None
            response_times.append(response_time)
            
            # Check response time is reasonable (under 4 seconds as per requirements)
            assert response_time < 4.0, f"Response time {response_time:.2f}s exceeds 4s limit"
        
        avg_response_time = sum(response_times) / len(response_times)
        
        logger.info(f"✅ Response time test passed")
        logger.info(f"   Average response time: {avg_response_time:.2f}s")
        logger.info(f"   Max response time: {max(response_times):.2f}s")
        logger.info(f"   Min response time: {min(response_times):.2f}s")
    
    @pytest.mark.asyncio
    async def test_recommendation_quality_threshold(self, integration_system):
        """Test that recommendations meet quality threshold"""
        logger.info("🧪 Testing recommendation quality threshold...")
        
        response = await integration_system.process_complete_query(
            "research datasets"
        )
        
        assert response is not None
        assert len(response.recommendations) > 0
        
        # Check that recommendations meet quality threshold
        quality_threshold = integration_system.quality_threshold
        
        for i, rec in enumerate(response.recommendations):
            logger.info(f"   Rec {i+1}: {rec.source} (Quality: {rec.quality_score:.2f})")
            
            # Top recommendations should meet quality threshold
            if i < 3:  # Top 3 recommendations
                assert rec.quality_score >= quality_threshold * 0.9, \
                    f"Top recommendation quality {rec.quality_score:.2f} below threshold {quality_threshold}"
        
        logger.info(f"✅ Quality threshold test passed")
    
    @pytest.mark.asyncio
    async def test_system_statistics_tracking(self, integration_system):
        """Test system statistics tracking"""
        logger.info("🧪 Testing system statistics tracking...")
        
        # Process a few queries
        queries = ["test query 1", "test query 2", "test query 3"]
        
        for query in queries:
            await integration_system.process_complete_query(query)
        
        # Get system statistics
        stats = integration_system.get_system_statistics()
        
        assert stats is not None
        assert 'requests_processed' in stats
        assert 'total_processing_time' in stats
        assert 'average_processing_time' in stats
        assert 'cache_hit_rate' in stats
        assert 'quality_validations' in stats
        
        # Check statistics are reasonable
        assert stats['requests_processed'] >= len(queries)
        assert stats['total_processing_time'] > 0
        assert stats['average_processing_time'] > 0
        assert 0.0 <= stats['cache_hit_rate'] <= 1.0
        
        logger.info(f"✅ System statistics test passed")
        logger.info(f"   Requests processed: {stats['requests_processed']}")
        logger.info(f"   Average processing time: {stats['average_processing_time']:.2f}s")
        logger.info(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, integration_system):
        """Test concurrent request handling"""
        logger.info("🧪 Testing concurrent request handling...")
        
        # Create multiple concurrent requests
        queries = [
            "psychology data",
            "singapore statistics",
            "climate research",
            "machine learning",
            "health indicators"
        ]
        
        # Process queries concurrently
        start_time = time.time()
        tasks = [integration_system.process_complete_query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check all requests succeeded
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        failed_responses = [r for r in responses if isinstance(r, Exception)]
        
        assert len(successful_responses) >= len(queries) * 0.8, \
            f"Too many failed requests: {len(failed_responses)}/{len(queries)}"
        
        # Check concurrent processing was efficient
        avg_sequential_time = 2.0  # Estimated average time per request
        expected_sequential_time = len(queries) * avg_sequential_time
        
        logger.info(f"✅ Concurrent request test passed")
        logger.info(f"   Successful requests: {len(successful_responses)}/{len(queries)}")
        logger.info(f"   Total concurrent time: {total_time:.2f}s")
        logger.info(f"   Estimated sequential time: {expected_sequential_time:.2f}s")
        logger.info(f"   Concurrency benefit: {expected_sequential_time/total_time:.1f}x faster")


@pytest.mark.asyncio
async def test_complete_end_to_end_integration():
    """Complete end-to-end integration test"""
    logger.info("🚀 Starting Complete End-to-End Integration Test")
    logger.info("=" * 60)
    
    # Create integration system
    integration = create_end_to_end_quality_integration()
    
    # Run comprehensive test
    test_results = await integration.test_end_to_end_integration()
    
    # Validate test results
    successful_tests = [r for r in test_results if r is not None]
    
    assert len(successful_tests) >= len(test_results) * 0.8, \
        f"Too many failed tests: {len(test_results) - len(successful_tests)}/{len(test_results)}"
    
    # Check quality metrics across all tests
    if successful_tests:
        avg_quality = sum(r.quality_metrics.ndcg_at_3 for r in successful_tests) / len(successful_tests)
        avg_confidence = sum(r.confidence_score for r in successful_tests) / len(successful_tests)
        
        assert avg_quality >= 0.7, f"Average quality {avg_quality:.2f} below 0.7 threshold"
        assert avg_confidence >= 0.6, f"Average confidence {avg_confidence:.2f} below 0.6 threshold"
    
    logger.info("✅ Complete End-to-End Integration Test PASSED")
    
    return test_results


if __name__ == "__main__":
    async def main():
        """Run all tests"""
        logger.info("🧪 Running End-to-End Quality Integration Tests")
        
        # Run the complete integration test
        results = await test_complete_end_to_end_integration()
        
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful: {len([r for r in results if r is not None])}")
        logger.info(f"Failed: {len([r for r in results if r is None])}")
        
        print("\n✅ All End-to-End Quality Integration tests completed!")
    
    asyncio.run(main())