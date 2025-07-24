#!/usr/bin/env python3
"""
Test Performance Monitoring Integration
Tests the enhanced PerformanceMetricsCollector with monitoring system integration
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from src.ai.performance_metrics_collector import PerformanceMetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_monitoring_integration():
    """Test the monitoring integration functionality"""
    print("🧪 Testing Performance Monitoring Integration")
    print("=" * 50)
    
    # Initialize collector
    collector = PerformanceMetricsCollector()
    
    # Test 1: Check monitoring status
    print("\n1. Testing monitoring status:")
    status = collector.get_monitoring_status()
    print(f"   Cache managers connected: {status['cache_managers_connected']}")
    print(f"   Health monitor active: {status['health_monitor_active']}")
    print(f"   Metrics database available: {status['metrics_database_available']}")
    
    if status['cache_managers']:
        print("   Cache manager details:")
        for name, details in status['cache_managers'].items():
            print(f"     {name}: {details}")
    
    # Test 2: Start monitoring integration
    print("\n2. Testing monitoring integration start:")
    collector.start_monitoring_integration()
    
    # Wait a moment for monitoring to initialize
    await asyncio.sleep(2)
    
    # Test 3: Log some performance metrics
    print("\n3. Testing performance metrics logging:")
    collector.log_performance_metric('test_metric', 'response_time', 2.5, {'test': True})
    collector.log_performance_metric('test_metric', 'cache_hit_rate', 75.0)
    collector.log_system_health({
        'status': 'healthy',
        'response_time': 1.2,
        'cpu_usage': 45.0,
        'memory_usage': 60.0,
        'cache_hit_rate': 80.0,
        'error_rate': 2.0,
        'requests_per_minute': 120
    })
    print("   ✅ Performance metrics logged")
    
    # Test 4: Add response times
    print("\n4. Testing response time tracking:")
    for i in range(5):
        response_time = 1.0 + (i * 0.5)
        collector.add_response_time(response_time, f'test_query_{i}')
    print("   ✅ Response times tracked")
    
    # Test 5: Get all metrics
    print("\n5. Testing metrics collection:")
    try:
        metrics = await asyncio.wait_for(collector.get_all_metrics(), timeout=10.0)
        
        print("   Neural Performance:")
        neural = metrics.get('neural_performance', {})
        if neural:
            print(f"     NDCG@3: {neural.get('ndcg_at_3', 'N/A')}")
            print(f"     Singapore Accuracy: {neural.get('singapore_accuracy', 'N/A')}")
        else:
            print("     No neural metrics available")
        
        print("   Cache Performance:")
        cache = metrics.get('cache_performance', {})
        if cache:
            print(f"     Overall Hit Rate: {cache.get('overall_hit_rate', 'N/A')}")
            print(f"     Search Cache Hit Rate: {cache.get('search_cache_hit_rate', 'N/A')}")
            print(f"     Quality Cache Hit Rate: {cache.get('quality_cache_hit_rate', 'N/A')}")
        else:
            print("     No cache metrics available")
        
        print("   System Health:")
        health = metrics.get('system_health', {})
        if health:
            print(f"     Status: {health.get('system_status', 'N/A')}")
            print(f"     Response Time: {health.get('response_time', 'N/A')}")
            print(f"     CPU Usage: {health.get('cpu_usage', 'N/A')}")
        else:
            print("     No health metrics available")
        
    except asyncio.TimeoutError:
        print("   ⚠️ Metrics collection timed out")
    except Exception as e:
        print(f"   ❌ Error collecting metrics: {e}")
    
    # Test 6: Get performance trends
    print("\n6. Testing performance trends:")
    neural_trends = collector.get_performance_trends('neural_performance', 1)
    cache_trends = collector.get_performance_trends('cache_performance', 1)
    response_trends = collector.get_performance_trends('response_time', 1)
    test_trends = collector.get_performance_trends('test_metric', 1)
    
    print(f"   Neural trends (last hour): {len(neural_trends)} entries")
    print(f"   Cache trends (last hour): {len(cache_trends)} entries")
    print(f"   Response trends (last hour): {len(response_trends)} entries")
    print(f"   Test trends (last hour): {len(test_trends)} entries")
    
    if test_trends:
        print("   Recent test metrics:")
        for trend in test_trends[:3]:
            print(f"     {trend['metric_name']}: {trend['metric_value']} at {time.ctime(trend['timestamp'])}")
    
    # Test 7: Get system health history
    print("\n7. Testing system health history:")
    health_history = collector.get_system_health_history(1)
    print(f"   Health history entries (last hour): {len(health_history)}")
    
    if health_history:
        latest = health_history[0]
        print(f"   Latest health status: {latest['status']}")
        print(f"   Latest response time: {latest['response_time']}s")
        print(f"   Latest cache hit rate: {latest['cache_hit_rate']}%")
    
    # Test 8: Stop monitoring integration
    print("\n8. Testing monitoring integration stop:")
    collector.stop_monitoring_integration()
    
    # Final status check
    print("\n9. Final monitoring status:")
    final_status = collector.get_monitoring_status()
    print(f"   Health monitor active: {final_status['health_monitor_active']}")
    
    print("\n✅ Performance monitoring integration tests completed!")


async def test_cache_integration():
    """Test cache system integration specifically"""
    print("\n🗄️ Testing Cache System Integration")
    print("=" * 40)
    
    collector = PerformanceMetricsCollector()
    
    # Test cache performance collection
    try:
        cache_metrics = await collector.get_cache_performance()
        
        print("Cache Performance Metrics:")
        for key, value in cache_metrics.items():
            print(f"  {key}: {value}")
        
        # Test if we can get statistics from connected cache managers
        if collector.cache_managers:
            print(f"\nConnected cache managers: {list(collector.cache_managers.keys())}")
            
            for name, manager in collector.cache_managers.items():
                try:
                    if hasattr(manager, 'get_overall_statistics'):
                        stats = manager.get_overall_statistics()
                        print(f"  {name} statistics: {list(stats.keys())}")
                    elif hasattr(manager, 'get_quality_cache_statistics'):
                        stats = manager.get_quality_cache_statistics()
                        print(f"  {name} quality statistics: hit_rate={stats.get('hit_rate', 0):.2f}")
                except Exception as e:
                    print(f"  {name} error: {e}")
        else:
            print("No cache managers connected")
            
    except Exception as e:
        print(f"Cache integration test failed: {e}")


if __name__ == "__main__":
    async def main():
        await test_monitoring_integration()
        await test_cache_integration()
    
    asyncio.run(main())