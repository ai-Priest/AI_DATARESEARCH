#!/usr/bin/env python3
"""
Test script to identify specific AWS Open Data URL generation issues
Based on the task requirements to fix AWS URL generation
"""

import asyncio
import aiohttp
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.web_search_engine import WebSearchEngine

async def test_aws_specific_issues():
    """Test for specific AWS Open Data URL generation issues"""
    
    # Initialize WebSearchEngine
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 5
        }
    }
    
    engine = WebSearchEngine(config)
    
    print("Testing AWS Open Data URL Generation Issues")
    print("=" * 60)
    
    # Test 1: Check if search parameters are properly included
    print("\n1. Testing Search Parameter Inclusion:")
    print("-" * 40)
    
    test_queries = [
        "climate data",
        "satellite imagery",
        "I need weather data",  # Conversational
        "",  # Empty query
        "very specific query with multiple words"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        aws_url = engine._generate_aws_open_data_url(query)
        print(f"Generated URL: {aws_url}")
        
        # Check if URL includes search parameters when expected
        if query.strip():
            if "?search=" in aws_url:
                print("✅ Search parameters included")
            elif "/tag/" in aws_url:
                print("✅ Tag-based routing used")
            elif aws_url == "https://registry.opendata.aws/":
                print("⚠️  Fallback to main page (might be expected for empty/invalid queries)")
            else:
                print("❌ No search parameters or tag routing found")
        else:
            if aws_url == "https://registry.opendata.aws/":
                print("✅ Correctly fallback to main page for empty query")
            else:
                print("⚠️  Unexpected URL for empty query")
    
    # Test 2: Test fallback behavior when search parameters fail
    print("\n\n2. Testing Fallback Behavior:")
    print("-" * 40)
    
    # Simulate a scenario where search parameters might fail
    # by testing with a mock URL that would fail validation
    
    test_cases = [
        ("climate data", "Should fallback to tag or main page if search fails"),
        ("satellite imagery", "Should fallback to geospatial tag"),
        ("genomics data", "Should fallback to life-sciences tag"),
        ("unknown domain", "Should fallback to main page")
    ]
    
    for query, expected_behavior in test_cases:
        print(f"\nQuery: '{query}' - {expected_behavior}")
        
        # Get primary URL
        primary_url = engine._generate_aws_open_data_url(query)
        print(f"Primary URL: {primary_url}")
        
        # Get tag fallback
        tag_fallback = engine._get_aws_tag_fallback_url(query)
        print(f"Tag fallback: {tag_fallback}")
        
        # Get general fallback
        general_fallback = engine._get_aws_fallback_url(query)
        print(f"General fallback: {general_fallback}")
        
        # Test validation (this should work with current implementation)
        try:
            validated_url = await engine._validate_aws_url(primary_url, query)
            print(f"Validated URL: {validated_url}")
            
            if validated_url != primary_url:
                print("✅ Fallback mechanism would activate if needed")
            else:
                print("✅ Primary URL validated successfully")
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
    
    # Test 3: Test URL accessibility and result relevance
    print("\n\n3. Testing URL Accessibility and Result Relevance:")
    print("-" * 40)
    
    # Test specific URLs to see if they return relevant results
    test_urls_with_queries = [
        ("https://registry.opendata.aws/?search=climate", "climate"),
        ("https://registry.opendata.aws/?search=satellite", "satellite"),
        ("https://registry.opendata.aws/tag/geospatial/", "geospatial"),
        ("https://registry.opendata.aws/tag/climate/", "climate"),
        ("https://registry.opendata.aws/", "general")
    ]
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10),
        headers={'User-Agent': 'AI Dataset Research Assistant/2.0'}
    ) as session:
        for url, query_context in test_urls_with_queries:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        content_length = len(content)
                        
                        # Basic relevance check - look for query terms in content
                        if query_context != "general":
                            query_found = query_context.lower() in content.lower()
                            print(f"✅ {url}")
                            print(f"   Status: {response.status}, Content: {content_length} chars")
                            print(f"   Query relevance: {'Found' if query_found else 'Not found'} '{query_context}' in content")
                        else:
                            print(f"✅ {url}")
                            print(f"   Status: {response.status}, Content: {content_length} chars")
                    else:
                        print(f"⚠️  {url} - Status: {response.status}")
            except Exception as e:
                print(f"❌ {url} - Error: {str(e)}")
    
    # Test 4: Integration test with full search flow
    print("\n\n4. Testing Full Integration:")
    print("-" * 40)
    
    integration_queries = [
        "climate change data",
        "machine learning datasets",
        "satellite imagery"
    ]
    
    for query in integration_queries:
        print(f"\nTesting full search integration for: '{query}'")
        try:
            # Test the full public data platforms search
            results = await engine._search_public_data_platforms(query)
            
            # Find AWS results
            aws_results = [r for r in results if 'aws' in r.get('source', '').lower()]
            
            if aws_results:
                for result in aws_results:
                    print(f"  ✅ AWS result:")
                    print(f"     Title: {result.get('title', 'N/A')}")
                    print(f"     URL: {result.get('url', 'N/A')}")
                    print(f"     Description: {result.get('description', 'N/A')[:100]}...")
                    
                    # Validate the URL
                    url = result.get('url', '')
                    if url:
                        try:
                            async with aiohttp.ClientSession(
                                timeout=aiohttp.ClientTimeout(total=5),
                                headers={'User-Agent': 'AI Dataset Research Assistant/2.0'}
                            ) as session:
                                async with session.get(url) as response:
                                    if response.status == 200:
                                        print(f"     URL Status: ✅ {response.status}")
                                    else:
                                        print(f"     URL Status: ⚠️  {response.status}")
                        except Exception as e:
                            print(f"     URL Status: ❌ {str(e)}")
            else:
                print("  ⚠️  No AWS results found in integration")
                
        except Exception as e:
            print(f"  ❌ Integration error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_aws_specific_issues())