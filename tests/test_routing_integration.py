"""
Routing System Integration Test
Tests the enhanced query router with source priority routing
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai.enhanced_query_router import EnhancedQueryRouter
from src.ai.simple_source_router import SimpleSourceRouter
from src.api.enhanced_query_api import EnhancedQueryAPI

async def test_routing_integration():
    """Test the routing system integration"""
    print("🧪 Testing Routing System Integration\n")
    
    # Initialize components
    query_router = EnhancedQueryRouter()
    source_router = SimpleSourceRouter()
    api = EnhancedQueryAPI()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Singapore Housing Query',
            'query': 'singapore housing prices data',
            'expected_domain': 'singapore',
            'expected_singapore_first': True
        },
        {
            'name': 'Psychology Research Query',
            'query': 'psychology research datasets',
            'expected_domain': 'psychology',
            'expected_singapore_first': False
        },
        {
            'name': 'Climate Global Query',
            'query': 'global climate change indicators',
            'expected_domain': 'climate',
            'expected_singapore_first': False
        },
        {
            'name': 'Generic Housing Query',
            'query': 'housing statistics',
            'expected_domain': 'general',
            'expected_singapore_first': True
        }
    ]
    
    results = {}
    passed_tests = 0
    total_tests = 0
    
    for scenario in test_scenarios:
        print(f"📝 Testing: {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")
        
        # Step 1: Query Classification
        classification = query_router.classify_query(scenario['query'])
        
        # Step 2: Source Routing
        sources = source_router.route_query(
            scenario['query'], 
            classification.domain, 
            classification.singapore_first_applicable
        )
        
        # Step 3: API Integration
        api_result = await api.process_query(scenario['query'])
        
        # Verify results
        domain_correct = classification.domain == scenario['expected_domain']
        singapore_first_correct = classification.singapore_first_applicable == scenario['expected_singapore_first']
        
        print(f"   ✅ Domain: {classification.domain} {'✓' if domain_correct else '✗'}")
        print(f"   ✅ Singapore-first: {classification.singapore_first_applicable} {'✓' if singapore_first_correct else '✗'}")
        print(f"   ✅ Top source: {sources[0]['source'] if sources else 'None'}")
        print(f"   ✅ API processing time: {api_result['processing_time']:.3f}s")
        
        # Count tests
        total_tests += 2
        if domain_correct:
            passed_tests += 1
        if singapore_first_correct:
            passed_tests += 1
        
        # Store results
        results[scenario['name']] = {
            'query': scenario['query'],
            'classification': {
                'domain': classification.domain,
                'singapore_first': classification.singapore_first_applicable,
                'confidence': classification.confidence
            },
            'sources': sources[:3] if sources else [],
            'api_processing_time': api_result['processing_time'],
            'tests_passed': {
                'domain': domain_correct,
                'singapore_first': singapore_first_correct
            }
        }
        
        print()
    
    # Summary
    print(f"📊 Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    
    # Performance analysis
    avg_processing_time = sum(
        result['api_processing_time'] 
        for result in results.values()
    ) / len(results)
    print(f"\n📈 Performance Analysis:")
    print(f"   Average processing time: {avg_processing_time:.3f}s")
    
    # Singapore-first effectiveness
    singapore_queries = [
        result for result in results.values() 
        if result['classification']['singapore_first']
    ]
    
    if singapore_queries:
        singapore_effectiveness = sum(
            1 for result in singapore_queries 
            if result['sources'] and len(result['sources']) > 0 and (
                'gov_sg' in result['sources'][0]['source'] or 
                'singstat' in result['sources'][0]['source'] or 
                'lta' in result['sources'][0]['source']
            )
        ) / len(singapore_queries)
        print(f"   Singapore-first effectiveness: {singapore_effectiveness*100:.1f}%")
    
    print(f"\n✅ Routing System Integration test complete!")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(test_routing_integration())
    if success:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")