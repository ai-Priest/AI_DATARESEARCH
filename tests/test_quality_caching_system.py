"""
Test script for Quality-Aware Caching System
Tests the multi-level quality caching implementation
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.append('src')

from ai.quality_aware_cache import (
    QualityAwareCacheManager, 
    CachedRecommendation, 
    QualityMetrics
)
from ai.intelligent_cache_warming import IntelligentCacheWarming
from ai.memory_optimization import (
    MemoryOptimizer, 
    MemoryOptimizationConfig,
    CompressedEmbeddingStorage
)


def test_quality_aware_cache():
    """Test Quality-Aware Cache Manager"""
    print("🧪 Testing Quality-Aware Cache Manager...")
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = QualityAwareCacheManager(
            cache_dir=temp_dir,
            quality_threshold=0.7
        )
        
        # Create test recommendations
        recommendations = [
            CachedRecommendation(
                source='kaggle',
                relevance_score=0.95,
                domain='psychology',
                explanation='Best platform for psychology datasets',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.92,
                cached_at=time.time()
            ),
            CachedRecommendation(
                source='zenodo',
                relevance_score=0.88,
                domain='psychology',
                explanation='Academic psychology research repository',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.85,
                cached_at=time.time()
            )
        ]
        
        # Test caching high-quality recommendations
        query = "psychology datasets"
        cache_key = cache_manager.cache_recommendations(query, recommendations)
        
        if cache_key:
            print("✅ Successfully cached high-quality recommendations")
            
            # Test retrieval
            cached_result = cache_manager.get_cached_recommendations(query)
            if cached_result:
                cached_recs, quality_metrics = cached_result
                print(f"✅ Retrieved {len(cached_recs)} cached recommendations")
                print(f"   Quality metrics: NDCG@3={quality_metrics.ndcg_at_3:.3f}")
            else:
                print("❌ Failed to retrieve cached recommendations")
        else:
            print("❌ Failed to cache recommendations")
        
        # Test low-quality filtering
        low_quality_recs = [
            CachedRecommendation(
                source='unknown_source',
                relevance_score=0.3,
                domain='psychology',
                explanation='Low quality source',
                geographic_scope='global',
                query_intent='research',
                quality_score=0.2,
                cached_at=time.time()
            )
        ]
        
        low_quality_key = cache_manager.cache_recommendations("low quality query", low_quality_recs)
        if not low_quality_key:
            print("✅ Correctly rejected low-quality recommendations")
        else:
            print("❌ Should have rejected low-quality recommendations")
        
        # Test statistics
        stats = cache_manager.get_quality_cache_statistics()
        print(f"📊 Cache statistics: {stats['total_entries']} entries, "
              f"hit rate: {stats['hit_rate']:.2f}")


def test_cache_warming():
    """Test Intelligent Cache Warming"""
    print("\n🧪 Testing Intelligent Cache Warming...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = QualityAwareCacheManager(cache_dir=temp_dir)
        
        # Create cache warming system
        warming_config = {
            'warming_enabled': True,
            'warming_interval': 10,  # Short interval for testing
            'max_concurrent_warming': 2
        }
        
        cache_warming = IntelligentCacheWarming(cache_manager, warming_config)
        
        # Test Singapore queries warming
        cache_warming.warm_singapore_government_queries()
        print("✅ Singapore government queries warming completed")
        
        # Test domain-specific warming
        cache_warming.warm_domain_specific_queries('psychology')
        print("✅ Psychology domain warming completed")
        
        # Test statistics
        warming_stats = cache_warming.get_warming_statistics()
        print(f"📊 Warming statistics: {warming_stats['stats']['successful_warming']} successful, "
              f"{warming_stats['stats']['singapore_queries_warmed']} Singapore queries")


def test_memory_optimization():
    """Test Memory Optimization"""
    print("\n🧪 Testing Memory Optimization...")
    
    # Create test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            return self.linear2(x)
    
    model = SimpleModel()
    
    # Test memory optimizer
    config = MemoryOptimizationConfig(
        enable_quantization=True,
        quantization_bits=8,
        enable_gradient_checkpointing=True
    )
    
    optimizer = MemoryOptimizer(config)
    
    # Test model optimization
    original_size = optimizer.quantizer._calculate_model_size(model)
    print(f"📊 Original model size: {original_size:.2f}MB")
    
    # Create test data for validation
    test_data = torch.randn(100, 100)
    test_labels = torch.randint(0, 10, (100,))
    
    optimized_model = optimizer.optimize_model(model, test_data=test_data, test_labels=test_labels)
    
    optimized_size = optimizer.quantizer._calculate_model_size(optimized_model)
    compression_ratio = original_size / optimized_size if optimized_size > 0 else 1.0
    
    print(f"✅ Model optimization completed: {original_size:.2f}MB → {optimized_size:.2f}MB "
          f"({compression_ratio:.2f}x compression)")


def test_embedding_compression():
    """Test Embedding Compression"""
    print("\n🧪 Testing Embedding Compression...")
    
    config = MemoryOptimizationConfig(
        use_hdf5_storage=True,
        embedding_compression_level=6
    )
    
    storage = CompressedEmbeddingStorage(config)
    
    # Create test embeddings
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    metadata = {
        'model': 'test_model',
        'dimension': 384,
        'count': 1000
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'test_embeddings')
        
        # Test saving
        saved_path = storage.save_embeddings(embeddings, file_path, metadata)
        print(f"✅ Saved compressed embeddings to: {saved_path}")
        
        # Test loading
        loaded_embeddings, loaded_metadata = storage.load_embeddings(saved_path)
        
        # Verify data integrity
        if np.allclose(embeddings, loaded_embeddings):
            print("✅ Embedding data integrity verified")
        else:
            print("❌ Embedding data integrity check failed")
        
        if loaded_metadata and loaded_metadata.get('model') == 'test_model':
            print("✅ Metadata integrity verified")
        else:
            print("❌ Metadata integrity check failed")
        
        # Check compression statistics
        stats = storage.get_storage_statistics()
        print(f"📊 Compression statistics: {stats['compression_ratio']:.2f}x compression")


def test_integration():
    """Test integration of all components"""
    print("\n🧪 Testing System Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize all components
        cache_manager = QualityAwareCacheManager(
            cache_dir=temp_dir,
            quality_threshold=0.7
        )
        
        cache_warming = IntelligentCacheWarming(cache_manager)
        
        memory_optimizer = MemoryOptimizer()
        
        # Test workflow: warm cache -> check quality -> optimize memory
        print("🔄 Running integrated workflow...")
        
        # 1. Warm cache with popular queries
        cache_warming.warm_popular_queries()
        
        # 2. Check cache statistics
        cache_stats = cache_manager.get_quality_cache_statistics()
        print(f"📊 Cache after warming: {cache_stats['total_entries']} entries")
        
        # 3. Test memory monitoring
        memory_stats = memory_optimizer.monitor_memory_usage()
        if memory_stats:
            print(f"📊 Memory usage: {memory_stats.get('rss_mb', 0):.2f}MB")
        
        # 4. Get optimization statistics
        opt_stats = memory_optimizer.get_optimization_statistics()
        print(f"📊 Optimization config: quantization={opt_stats['config']['quantization_enabled']}")
        
        print("✅ Integration test completed successfully")


def main():
    """Run all tests"""
    print("🚀 Starting Quality Caching System Tests\n")
    
    try:
        test_quality_aware_cache()
        test_cache_warming()
        test_memory_optimization()
        test_embedding_compression()
        test_integration()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())