#!/usr/bin/env python3
"""
Test script to verify AWS Open Data URL generation and identify issues
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai.web_search_engine import WebSearchEngine
import aiohttp

async def test_aws_url_generation():
    """Test AWS URL generation with various queries"""
    
    # Initialize WebSearchEngine
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 5
        }
    }
    
    engine = WebSearchEngine(config)
    
    # Test queries
    test_queries = [
        "I need HDB data",
        "housing data",
        "climate data",
        "satellite imagery",
        "economic indicators",
        "health statistics",
        "transportation data",
        "machine learning datasets",
        "genomics data",
        "astronomy data",
        "general search query"
    ]
    
    print("🧪 Testing AWS Open Data URL Generation")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        # Test URL generation
        generated_url = engine._generate_aws_open_data_url(query)
        print(f"🔗 Generated URL: {generated_url}")
        
        # Test URL validation
        try:
            validated_url = await engine._validate_aws_url(generated_url, query)
            print(f"✅ Validated URL: {validated_url}")
            
            # Test URL accessibility
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'AI Dataset Research Assistant/2.0'}
            ) as session:
                try:
                    async with session.get(validated_url) as response:
                        status = response.status
                        if status == 200:
                            print(f"🟢 URL Status: {status} (OK)")
                        else:
                            print(f"🟡 URL Status: {status} (Warning)")
                except Exception as e:
                    print(f"🔴 URL Access Error: {str(e)}")
                    
        except Exception as e:
            print(f"❌ Validation Error: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_aws_url_generation())