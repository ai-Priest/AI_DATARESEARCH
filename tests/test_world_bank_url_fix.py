#!/usr/bin/env python3
"""
Test script for World Bank URL generation fixes
"""
import sys
import os
sys.path.append('src')

from ai.web_search_engine import WebSearchEngine
import asyncio

def test_world_bank_url_generation():
    """Test various query formats for World Bank URL generation"""
    
    # Initialize web search engine
    config = {'web_search': {'timeout': 10, 'max_results': 5}}
    engine = WebSearchEngine(config)
    
    # Test queries with different topics
    test_queries = [
        "I need GDP data",
        "Looking for economic growth datasets",
        "Can you find me some education statistics please",
        "Show me population data",
        "I want poverty data",
        "Get me some health indicators",
        "Find employment data",
        "economic development",
        "gdp growth",
        "education statistics"
    ]
    
    print("Testing World Bank URL Generation:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nOriginal query: '{query}'")
        
        # Test normalization
        normalized = engine._normalize_query_for_source(query, 'world_bank')
        print(f"Normalized: '{normalized}'")
        
        # Test URL generation (synchronous version for testing)
        results = asyncio.run(engine._search_international_organizations(query, None))
        
        # Find World Bank results
        world_bank_results = [r for r in results if 'world bank' in r.get('title', '').lower()]
        
        if world_bank_results:
            for wb_result in world_bank_results:
                print(f"Generated URL: {wb_result['url']}")
                print(f"Title: {wb_result['title']}")
                if 'fallback_url' in wb_result:
                    print(f"Fallback URL: {wb_result.get('fallback_url', 'N/A')}")
        else:
            print("No World Bank results generated")
        
        print("-" * 30)

if __name__ == "__main__":
    test_world_bank_url_generation()