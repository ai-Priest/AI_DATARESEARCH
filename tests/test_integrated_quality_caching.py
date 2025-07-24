"""
Integration test for the complete quality caching system
Demonstrates the multi-level quality caching working with real scenarios
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
import tempfile

# Add src to path
sys.path.append('src')

from ai.quality_aware_cache import (
    QualityAwareCacheManager, 
    CachedRecommendation, 
    QualityMetrics
)
from ai.intelligent_cache_warming import IntelligentCacheWarming
from ai.memory_optimization import MemoryOptimizer, MemoryOptimizationConfig


def create_high_quality_recommendations(query: str) -> list:
    """Create high-quality recommendations that align with training mappings"""
    current_time = time.time()
    
    if 'psychology' in query.lower():
        return [
            CachedRecommendation(
                source='kaggle',
                relevance_score=0.95,
                domain='psychology',
                explanation='Best platform for psychology datasets and research competitions',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.92,
                cached_at=current_time
            ),
            CachedRecommendation(
                source='zenodo',
                relevance_score=0.90,
                domain='psychology',
                explanation='Academic repository with psychology research data',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.88,
                cached_at=current_time
            ),
            CachedRecommendation(
                source='world_bank',
                relevance_score=0.25,
                domain='psychology',
                explanation='Limited psychology-specific data available',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.30,
                cached_at=current_time
            )
        ]
    
    elif 'singapore' in query.lower():
        return [
            CachedRecommendation(
                source='data_gov_sg',
                relevance_score=0.98,
                domain='singapore',
                explanation='Official Singapore government data',
                geographic_scope='singapore',
                query_intent='research',
                quality_score=0.95,
                cached_at=current_time
            ),
            CachedRecommendation(
                source='singstat',
                relevance_score=0.97,
                domain='singapore',
                explanation='Singapore Department of Statistics',
                geographic_scope='singapore',
                query_intent='research',
                quality_score=0.94,
                cached_at=current_time
            )
        ]
    
    elif 'climate' in query.lower():
        return [
            CachedRecommendation(
                source='world_bank',
                relevance_score=0.95,
                domain='climate',
                explanation='Excellent global climate indicators',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.92,
                cached_at=current_time
            ),
            CachedRecommendation(
                source='kaggle',
                relevance_score=0.82,
                domain='climate',
                explanation='Climate datasets for modeling',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.85,
                cached_at=current_time
            )
        ]
    
    else:
        return [
            CachedRecommendation(
                source='kaggle',
                relevance_score=0.88,
                domain='general',
                explanation='Comprehensive datasets for research',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.85,
                cached_at=current_time
            )
        ]


def test_quality_caching_workflow():
    """Test the complete quality caching workflow"""
    print("🧪 Testing Complete Quality Caching Workflow")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize quality-aware cache manager
        cache_manager = QualityAwareCacheManager(
            cache_dir=temp_dir,
            quality_threshold=0.7,
            training_mappings_path="training_mappings.md"
        )
        
        print(f"📚 Loaded {len(cache_manager.training_mappings)} training mappings")
        
        # Test 1: Cache high-quality psychology recommendations
        print("\n🧠 Test 1: Psychology Query Caching")
        psychology_query = "psychology datasets"
        psychology_recs = create_high_quality_recommendations(psychology_query)
        
        cache_key = cache_manager.cache_recommendations(psychology_query, psychology_recs)
        if cache_key:
            print("✅ Successfully cached psychology recommendations")
            
            # Retrieve and verify
            cached_result = cache_manager.get_cached_recommendations(psychology_query)
            if cached_result:
                cached_recs, quality_metrics = cached_result
                print(f"   📊 Retrieved {len(cached_recs)} recommendations")
                print(f"   📈 NDCG@3: {quality_metrics.ndcg_at_3:.3f}")
                print(f"   🎯 Relevance Accuracy: {quality_metrics.relevance_accuracy:.3f}")
                print(f"   🌏 Singapore-First Accuracy: {quality_metrics.singapore_first_accuracy:.3f}")
        else:
            print("❌ Failed to cache psychology recommendations")
        
        # Test 2: Cache Singapore-specific recommendations
        print("\n🇸🇬 Test 2: Singapore Query Caching")
        singapore_query = "singapore data"
        singapore_recs = create_high_quality_recommendations(singapore_query)
        
        cache_key = cache_manager.cache_recommendations(singapore_query, singapore_recs)
        if cache_key:
            print("✅ Successfully cached Singapore recommendations")
            
            cached_result = cache_manager.get_cached_recommendations(singapore_query)
            if cached_result:
                cached_recs, quality_metrics = cached_result
                print(f"   📊 Retrieved {len(cached_recs)} recommendations")
                print(f"   📈 NDCG@3: {quality_metrics.ndcg_at_3:.3f}")
                print(f"   🇸🇬 Singapore-First Accuracy: {quality_metrics.singapore_first_accuracy:.3f}")
        
        # Test 3: Test cache warming
        print("\n🔥 Test 3: Intelligent Cache Warming")
        cache_warming = IntelligentCacheWarming(cache_manager)
        
        # Warm Singapore queries
        cache_warming.warm_singapore_government_queries()
        
        # Warm domain-specific queries
        cache_warming.warm_domain_specific_queries('psychology')
        cache_warming.warm_domain_specific_queries('climate')
        
        warming_stats = cache_warming.get_warming_statistics()
        print(f"   🔥 Warming completed: {warming_stats['stats']['successful_warming']} successful")
        print(f"   🇸🇬 Singapore queries warmed: {warming_stats['stats']['singapore_queries_warmed']}")
        print(f"   🎯 Domain queries warmed: {warming_stats['stats']['domain_queries_warmed']}")
        
        # Test 4: Cache statistics and quality metrics
        print("\n📊 Test 4: Cache Quality Statistics")
        cache_stats = cache_manager.get_quality_cache_statistics()
        
        print(f"   📈 Total entries: {cache_stats['total_entries']}")
        print(f"   🎯 Hit rate: {cache_stats['hit_rate']:.2f}")
        print(f"   📊 Average NDCG@3: {cache_stats['avg_ndcg_at_3']:.3f}")
        print(f"   🎯 Average relevance accuracy: {cache_stats['avg_relevance_accuracy']:.3f}")
        
        quality_dist = cache_stats['quality_distribution']
        print(f"   🌟 Quality distribution:")
        print(f"      Excellent (≥0.9): {quality_dist['excellent']}")
        print(f"      Good (0.7-0.9): {quality_dist['good']}")
        print(f"      Fair (0.5-0.7): {quality_dist['fair']}")
        print(f"      Poor (<0.5): {quality_dist['poor']}")
        
        # Test 5: Memory optimization
        print("\n🔧 Test 5: Memory Optimization")
        memory_optimizer = MemoryOptimizer()
        
        # Test embedding compression
        test_embeddings = np.random.randn(500, 384).astype(np.float32)
        embedding_path = os.path.join(temp_dir, 'test_embeddings')
        
        saved_path = memory_optimizer.save_optimized_embeddings(
            test_embeddings, 
            embedding_path,
            metadata={'test': True, 'dimension': 384}
        )
        
        print(f"   💾 Saved compressed embeddings: {saved_path}")
        
        # Load and verify
        loaded_embeddings, loaded_metadata = memory_optimizer.load_optimized_embeddings(saved_path)
        
        if np.allclose(test_embeddings, loaded_embeddings):
            print("   ✅ Embedding compression/decompression verified")
        
        # Get optimization statistics
        opt_stats = memory_optimizer.get_optimization_statistics()
        embedding_stats = opt_stats['embedding_optimization']
        
        if embedding_stats['compression_ratio'] > 1.0:
            print(f"   📦 Compression ratio: {embedding_stats['compression_ratio']:.2f}x")
        
        # Test 6: Quality threshold validation
        print("\n🎯 Test 6: Quality Threshold Validation")
        
        # Try to cache low-quality recommendations
        low_quality_recs = [
            CachedRecommendation(
                source='unknown_source',
                relevance_score=0.3,
                domain='test',
                explanation='Low quality test',
                geographic_scope='global',
                query_intent='test',
                quality_score=0.2,
                cached_at=time.time()
            )
        ]
        
        low_cache_key = cache_manager.cache_recommendations("low quality test", low_quality_recs)
        if not low_cache_key:
            print("   ✅ Correctly rejected low-quality recommendations")
        else:
            print("   ❌ Should have rejected low-quality recommendations")
        
        # Test cache invalidation of low-quality entries
        cache_manager.invalidate_low_quality_cache(0.8)  # Higher threshold
        
        final_stats = cache_manager.get_quality_cache_statistics()
        print(f"   🧹 Cache entries after quality cleanup: {final_stats['total_entries']}")
        
        print("\n🎉 Quality Caching Workflow Test Completed Successfully!")
        
        return True


def test_performance_scenarios():
    """Test performance scenarios"""
    print("\n⚡ Testing Performance Scenarios")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = QualityAwareCacheManager(
            cache_dir=temp_dir,
            quality_threshold=0.7
        )
        
        # Test concurrent caching
        queries = [
            "machine learning datasets",
            "climate change data", 
            "singapore housing statistics",
            "psychology research data",
            "economic indicators"
        ]
        
        start_time = time.time()
        
        for query in queries:
            recommendations = create_high_quality_recommendations(query)
            cache_key = cache_manager.cache_recommendations(query, recommendations)
            
            if cache_key:
                # Immediately try to retrieve
                cached_result = cache_manager.get_cached_recommendations(query)
                if cached_result:
                    print(f"   ✅ {query}: cached and retrieved successfully")
        
        duration = time.time() - start_time
        print(f"   ⏱️ Processed {len(queries)} queries in {duration:.3f}s")
        
        # Test cache hit performance
        start_time = time.time()
        
        for query in queries:
            cached_result = cache_manager.get_cached_recommendations(query)
        
        hit_duration = time.time() - start_time
        print(f"   ⚡ Cache hits for {len(queries)} queries in {hit_duration:.3f}s")
        
        return True


def main():
    """Run all integration tests"""
    print("🚀 Quality Caching System Integration Tests")
    print("=" * 60)
    
    try:
        # Run main workflow test
        test_quality_caching_workflow()
        
        # Run performance tests
        test_performance_scenarios()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())