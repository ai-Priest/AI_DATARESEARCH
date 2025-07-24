#!/usr/bin/env python3
"""
Test script for Kaggle URL generation fixes
"""
import sys
import os
sys.path.append('src')

from ai.web_search_engine import WebSearchEngine

def test_kaggle_url_generation():
    """Test various query formats for Kaggle URL generation"""
    
    # Initialize web search engine
    config = {'web_search': {'timeout': 10, 'max_results': 5}}
    engine = WebSearchEngine(config)
    
    # Test queries with conversational language
    test_queries = [
        "I need HDB data",
        "Looking for housing datasets",
        "Can you find me some cancer data please",
        "Show me titanic survival data",
        "I want cryptocurrency price data",
        "Get me some machine learning datasets about education",
        "Find housing data",
        "cancer mortality",
        "titanic",
        "bitcoin prices"
    ]
    
    print("Testing Kaggle URL Generation:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nOriginal query: '{query}'")
        
        # Test normalization
        normalized = engine._normalize_query_for_source(query, 'kaggle')
        print(f"Normalized: '{normalized}'")
        
        # Test URL generation (synchronous version for testing)
        import asyncio
        results = asyncio.run(engine._search_kaggle_datasets(query))
        
        if results:
            kaggle_result = results[0]  # First result is the main search
            print(f"Generated URL: {kaggle_result['url']}")
            print(f"Title: {kaggle_result['title']}")
        else:
            print("No results generated")
        
        print("-" * 30)

if __name__ == "__main__":
    test_kaggle_url_generation()