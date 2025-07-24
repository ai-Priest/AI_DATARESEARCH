#!/usr/bin/env python3
"""
Comprehensive test script to validate AWS Open Data URL generation
Tests edge cases and identifies specific issues that need fixing
"""

import asyncio
import aiohttp
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.web_search_engine import WebSearchEngine

async def test_aws_comprehensive():
    """Comprehensive test of AWS Open Data URL generation"""
    
    # Initialize WebSearchEngine
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 5
        }
    }
    
    engine = WebSearchEngine(config)
    
    print("Comprehensive AWS Open Data URL Generation Test")
    print("=" * 60)
    
    # Test 1: Edge cases and problematic queries
    print("\n1. Testing Edge Cases:")
    print("-" * 30)
    
    edge_cases = [
        "",  # Empty query
        "   ",  # Whitespace only
        "a",  # Single character
        "I need some data please",  # Very conversational
        "Can you find me datasets about climate change?",  # Long conversational
        "singapore hdb data",  # Singapore-specific
        "very long query with many words that might cause issues with url encoding",
        "special!@#$%^&*()characters",  # Special characters
    ]
    
    for query in edge_cases:
        print(f"\nQuery: '{query}'")
        try:
            aws_url = engine._generate_aws_open_data_url(query)
            print(f"Generated URL: {aws_url}")
            
            validated_url = await engine._validate_aws_url(aws_url, query)
            print(f"Validated URL: {validated_url}")
            
            if aws_url != validated_url:
                print("⚠️  URL was modified during validation")
            else:
                print("✅ URL passed validation unchanged")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Test 2: URL accessibility for different search patterns
    print("\n\n2. Testing URL Accessibility:")
    print("-" * 30)
    
    test_urls = [
        ("Search with simple term", "https://registry.opendata.aws/?search=climate"),
        ("Search with multiple terms", "https://registry.opendata.aws/?search=satellite+imagery"),
        ("Search with special chars", "https://registry.opendata.aws/?search=climate%21%40%23"),
        ("Empty search", "https://registry.opendata.aws/?search="),
        ("Tag-based geospatial", "https://registry.opendata.aws/tag/geospatial/"),
        ("Tag-based climate", "https://registry.opendata.aws/tag/climate/"),
        ("Main page", "https://registry.opendata.aws/"),
        ("Invalid tag", "https://registry.opendata.aws/tag/nonexistent/"),
    ]
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10),
        headers={'User-Agent': 'AI Dataset Research Assistant/2.0'}
    ) as session:
        for description, url in test_urls:
            try:
                async with session.get(url) as response:
                    status = response.status
                    content_length = len(await response.text())
                    if status == 200:
                        print(f"✅ {description}: {url} - Status: {status}, Content: {content_length} chars")
                    else:
                        print(f"⚠️  {description}: {url} - Status: {status}")
            except Exception as e:
                print(f"❌ {description}: {url} - Error: {str(e)}")
    
    # Test 3: Integration with search platforms
    print("\n\n3. Testing Integration with Search Platforms:")
    print("-" * 30)
    
    integration_queries = [
        "climate data",
        "machine learning datasets",
        "I need weather information",
        ""
    ]
    
    for query in integration_queries:
        print(f"\nTesting integration for query: '{query}'")
        try:
            results = await engine._search_public_data_platforms(query)
            aws_results = [r for r in results if 'aws' in r.get('source', '').lower()]
            
            if aws_results:
                for result in aws_results:
                    print(f"  ✅ AWS result found:")
                    print(f"     Title: {result.get('title', 'N/A')}")
                    print(f"     URL: {result.get('url', 'N/A')}")
                    print(f"     Source: {result.get('source', 'N/A')}")
            else:
                print("  ⚠️  No AWS results found in integration")
                
        except Exception as e:
            print(f"  ❌ Integration error: {str(e)}")
    
    # Test 4: Fallback behavior
    print("\n\n4. Testing Fallback Behavior:")
    print("-" * 30)
    
    # Test what happens when search parameters fail
    print("Testing fallback when search parameters might fail...")
    
    fallback_queries = [
        "nonexistent dataset type",
        "very specific query that probably has no results",
        "test query for fallback behavior"
    ]
    
    for query in fallback_queries:
        print(f"\nQuery: '{query}'")
        
        # Generate primary URL
        primary_url = engine._generate_aws_open_data_url(query)
        print(f"Primary URL: {primary_url}")
        
        # Get tag fallback
        tag_fallback = engine._get_aws_tag_fallback_url(query)
        print(f"Tag fallback: {tag_fallback}")
        
        # Get general fallback
        general_fallback = engine._get_aws_fallback_url(query)
        print(f"General fallback: {general_fallback}")
        
        # Test validation behavior
        try:
            validated = await engine._validate_aws_url(primary_url, query)
            print(f"Validated URL: {validated}")
            
            if validated != primary_url:
                print("✅ Fallback mechanism activated")
            else:
                print("✅ Primary URL validated successfully")
                
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_aws_comprehensive())