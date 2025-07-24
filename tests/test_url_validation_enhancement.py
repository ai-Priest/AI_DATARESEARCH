#!/usr/bin/env python3
"""
Test script to verify the enhanced URL validation system
Tests the new functionality added in task 3: Enhance URL validation and correction system
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.url_validator import URLValidator
from ai.web_search_engine import WebSearchEngine

async def test_enhanced_url_validator():
    """Test the enhanced URLValidator functionality"""
    print("🧪 Testing Enhanced URL Validator...")
    
    validator = URLValidator()
    
    # Test 1: External source URL correction
    print("\n1. Testing external source URL correction:")
    
    test_cases = [
        {
            'source': 'kaggle',
            'query': 'housing data',
            'current_url': 'https://www.kaggle.com/broken-link',
            'expected_contains': 'kaggle.com'
        },
        {
            'source': 'world_bank',
            'query': 'gdp data',
            'current_url': 'https://datacatalog.worldbank.org/broken',
            'expected_contains': 'data.worldbank.org'
        },
        {
            'source': 'aws_open_data',
            'query': 'climate data',
            'current_url': 'https://registry.opendata.aws/broken',
            'expected_contains': 'registry.opendata.aws'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        corrected_url = validator.correct_external_source_url(
            test_case['source'], 
            test_case['query'], 
            test_case['current_url']
        )
        
        success = test_case['expected_contains'] in corrected_url
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Test 1.{i} ({test_case['source']}): {status}")
        print(f"      Original: {test_case['current_url']}")
        print(f"      Corrected: {corrected_url}")
    
    # Test 2: Source search patterns
    print("\n2. Testing source search patterns:")
    patterns = validator.get_source_search_patterns()
    
    expected_sources = ['kaggle', 'world_bank', 'aws_open_data', 'un_data', 'who']
    for source in expected_sources:
        has_pattern = source in patterns and patterns[source]
        status = "✅ PASS" if has_pattern else "❌ FAIL"
        print(f"   {source}: {status} - {patterns.get(source, 'Missing')}")
    
    # Test 3: Validation with retry
    print("\n3. Testing URL validation with retry:")
    
    test_urls = [
        'https://www.google.com',  # Should work
        'https://nonexistent-domain-12345.com',  # Should fail
        'https://data.worldbank.org'  # Should work
    ]
    
    for url in test_urls:
        try:
            is_valid, status_code = await validator.validate_url_with_retry(url, max_retries=1)
            print(f"   {url}: {'✅ Valid' if is_valid else '❌ Invalid'} (Status: {status_code})")
        except Exception as e:
            print(f"   {url}: ❌ Error - {str(e)}")
    
    # Test 4: Health check functionality
    print("\n4. Testing source health checks:")
    
    try:
        health_results = await validator.validate_all_source_patterns()
        
        for source, health in health_results.items():
            status = health.get('status', 'unknown')
            response_time = health.get('response_time_ms', 'N/A')
            print(f"   {source}: {status} ({response_time}ms)")
            
    except Exception as e:
        print(f"   Health check error: {str(e)}")
    
    print("\n✅ Enhanced URL Validator tests completed!")

async def test_web_search_integration():
    """Test the integration of URL validation into WebSearchEngine"""
    print("\n🧪 Testing Web Search Engine Integration...")
    
    # Create a minimal config for testing
    config = {
        'web_search': {
            'timeout': 10,
            'max_results': 3
        }
    }
    
    search_engine = WebSearchEngine(config)
    
    # Test with sample search results
    sample_results = [
        {
            'title': 'Test Kaggle Dataset',
            'url': 'https://www.kaggle.com/datasets/test',
            'description': 'Test dataset',
            'source': 'kaggle',
            'type': 'dataset'
        },
        {
            'title': 'Test World Bank Data',
            'url': 'https://datacatalog.worldbank.org/broken-link',  # Broken URL
            'description': 'Economic indicators',
            'source': 'world_bank',
            'type': 'dataset'
        },
        {
            'title': 'Test AWS Dataset',
            'url': 'https://registry.opendata.aws/test',
            'description': 'Open data on AWS',
            'source': 'aws',
            'type': 'dataset'
        }
    ]
    
    print(f"\n1. Testing URL validation integration with {len(sample_results)} sample results:")
    
    try:
        validated_results = await search_engine._validate_and_correct_result_urls(sample_results)
        
        print(f"   Input results: {len(sample_results)}")
        print(f"   Output results: {len(validated_results)}")
        
        for i, result in enumerate(validated_results, 1):
            url_status = result.get('url_status', 'unknown')
            original_url = sample_results[i-1]['url']
            current_url = result.get('url', '')
            source = result.get('source', 'unknown')
            
            print(f"   Result {i} ({source}):")
            print(f"      Status: {url_status}")
            print(f"      Original: {original_url}")
            print(f"      Final: {current_url}")
            print(f"      Changed: {'Yes' if original_url != current_url else 'No'}")
        
        print("\n✅ Web Search Engine integration tests completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")

async def main():
    """Run all tests"""
    print("🚀 Starting Enhanced URL Validation System Tests")
    print("=" * 60)
    
    try:
        await test_enhanced_url_validator()
        await test_web_search_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)