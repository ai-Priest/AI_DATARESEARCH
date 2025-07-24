#!/usr/bin/env python3
"""
Test script to verify AWS Open Data URL generation with edge cases and complex queries
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai.web_search_engine import WebSearchEngine
import aiohttp

async def test_aws_edge_cases():
    """Test AWS URL generation with edge cases and complex conversational queries"""
    
    # Initialize WebSearchEngine
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 5
        }
    }
    
    engine = WebSearchEngine(config)
    
    # Test edge case queries
    edge_case_queries = [
        "",  # Empty query
        "   ",  # Whitespace only
        "Can you please find me some data about Singapore housing please?",  # Very conversational
        "I'm looking for datasets related to climate change and environmental data",  # Long conversational
        "Show me AWS datasets",  # Meta query about AWS itself
        "Find me some machine learning data for my project",  # Conversational with context
        "What datasets are available?",  # Generic question
        "data",  # Single word
        "singapore hdb resale prices historical data analysis",  # Very specific long query
        "!@#$%^&*()",  # Special characters
        "123456789",  # Numbers only
        "covid-19 pandemic health data statistics worldwide",  # Hyphenated terms
    ]
    
    print("🧪 Testing AWS Open Data URL Generation - Edge Cases")
    print("=" * 70)
    
    for query in edge_case_queries:
        print(f"\n📝 Query: '{query}'")
        
        # Test URL generation
        try:
            generated_url = engine._generate_aws_open_data_url(query)
            print(f"🔗 Generated URL: {generated_url}")
            
            # Test URL validation
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
            print(f"❌ Generation Error: {str(e)}")
        
        print("-" * 50)

async def test_fallback_scenarios():
    """Test fallback scenarios when search parameters fail"""
    
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 5
        }
    }
    
    engine = WebSearchEngine(config)
    
    print("\n🔄 Testing Fallback Scenarios")
    print("=" * 50)
    
    # Test with a potentially problematic URL
    test_url = "https://registry.opendata.aws/?search=invalid%20search%20that%20might%20fail"
    print(f"📝 Testing URL: {test_url}")
    
    try:
        validated_url = await engine._validate_aws_url(test_url, "invalid search that might fail")
        print(f"✅ Fallback URL: {validated_url}")
        
        # Test the fallback URL
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'AI Dataset Research Assistant/2.0'}
        ) as session:
            try:
                async with session.get(validated_url) as response:
                    status = response.status
                    if status == 200:
                        print(f"🟢 Fallback URL Status: {status} (OK)")
                    else:
                        print(f"🟡 Fallback URL Status: {status} (Warning)")
            except Exception as e:
                print(f"🔴 Fallback URL Access Error: {str(e)}")
                
    except Exception as e:
        print(f"❌ Fallback Test Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_aws_edge_cases())
    asyncio.run(test_fallback_scenarios())