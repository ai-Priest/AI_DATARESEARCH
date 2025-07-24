#!/usr/bin/env python3
"""
Basic test to verify web search engine functionality after AWS URL fixes
"""
import sys
import os
sys.path.append('src')

from ai.web_search_engine import WebSearchEngine
import asyncio

async def test_basic_web_search():
    """Test basic web search functionality"""
    
    config = {'web_search': {'timeout': 10, 'max_results': 5}}
    engine = WebSearchEngine(config)
    
    print("Testing basic web search functionality:")
    print("=" * 50)
    
    # Test basic search
    results = await engine.search_web("climate data")
    
    print(f"Total results: {len(results)}")
    
    # Check for AWS results
    aws_results = [r for r in results if 'aws' in r.get('source', '').lower()]
    
    if aws_results:
        print(f"✅ Found {len(aws_results)} AWS results")
        for aws_result in aws_results:
            print(f"   - {aws_result['title']}")
            print(f"     URL: {aws_result['url']}")
    else:
        print("❌ No AWS results found")
    
    # Check for other sources
    other_sources = set()
    for result in results:
        source = result.get('source', 'unknown')
        if 'aws' not in source.lower():
            other_sources.add(source)
    
    print(f"✅ Other sources found: {', '.join(other_sources)}")
    
    return len(results) > 0 and len(aws_results) > 0

if __name__ == "__main__":
    success = asyncio.run(test_basic_web_search())
    print(f"\nBasic web search test: {'✅ PASSED' if success else '❌ FAILED'}")
    exit(0 if success else 1)