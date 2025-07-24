#!/usr/bin/env python3
"""Test script to verify metrics collector works correctly"""

import sys
import os
import asyncio
sys.path.append('.')

# Set environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

from src.ai.performance_metrics_collector import PerformanceMetricsCollector

async def test_metrics():
    print("Testing PerformanceMetricsCollector...")
    
    collector = PerformanceMetricsCollector()
    
    print("\n1. Testing neural performance metrics...")
    neural_metrics = await collector.get_current_neural_performance()
    print(f"Neural metrics: {neural_metrics}")
    
    print("\n2. Testing response time metrics...")
    response_metrics = await collector.get_response_time_metrics()
    print(f"Response metrics: {response_metrics}")
    
    print("\n3. Testing cache performance metrics...")
    cache_metrics = await collector.get_cache_performance()
    print(f"Cache metrics: {cache_metrics}")
    
    print("\n4. Testing all metrics...")
    all_metrics = await collector.get_all_metrics()
    print(f"All metrics: {all_metrics}")
    
    print("\n5. Testing formatted display...")
    formatted = collector.format_metrics_for_display(all_metrics)
    print(f"Formatted: {formatted}")

if __name__ == "__main__":
    asyncio.run(test_metrics())