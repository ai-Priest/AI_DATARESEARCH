"""
Complete Routing System Integration Test
Tests the enhanced query router with source priority routing
"""
import asyncio
import logging
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai.enhanced_query_router import EnhancedQueryRouter
from src.ai.simple_source_router import SimpleSourceRouter
from src.api.enhanced_query_api import EnhancedQueryAPI

async def test_complete_routing_system():
    """Test the complete routing system with various scenarios"""
    print("🧪 Testing Complete Routing System\n")
    
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
            'expected_singapore_first': True,
            'expected_top_source': 'data_gov_sg'
        },
        {
            'name': 'Psychology Research Query',
            'query': 'psychology research datasets',
            'expected_domain': 'psychology',
            'expected_singapore_first': False,
            'expected_top_source': 'kaggle'
        },
        {
            'name': 'Climate Global Query',
            'query': 'global climate change indicators',
            'expected_domain': 'climate',
            'expected_singapore_first': False,
            'expected_top_source': 'world_bank'
        },
        {
            'name': 'Transportation Singapore Query',
            'query': 'mrt transportation data singapore',
            'expected_domain': 'transportation',
            'expected_singapore_first': True,
            'expected_top_source': 'lta_datamall'
        },
        {
            'name': 'Generic Housing Query',
            'query': 'housing statistics',
            'expected_domain': 'general',
            'expected_singapore_first': True,  # Should apply Singapore-first for generic housing
            'expected_top_source': 'data_gov_sg'
        }
    ]
    
    results = {}
    
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
        top_source_correct = sources[0]['source'] == scenario['expected_top_source'] if sources else False
        
        print(f"   ✅ Domain: {classification.domain} {'✓' if domain_correct else '✗'}")
        print(f"   ✅ Singapore-first: {classification.singapore_first_applicable} {'✓' if singapore_first_correct else '✗'}")
        print(f"   ✅ Top source: {sources[0]['source'] if sources else 'None'} {'✓' if top_source_correct else '✗'}")
        print(f"   ✅ API processing time: {api_result['processing_time']:.3f}s")
        
        # Store results
        results[scenario['name']] = {
            'query': scenario['query'],
            'classification': {
                'domain': classification.domain,
                'singapore_first': classification.singapore_first_applicable,
                'confidence': classification.confidence
            },
            'sources': sources[:3],  # Top 3 sources
            'api_result': api_result,
            'tests_passed': {
                'domain': domain_correct,
                'singapore_first': singapore_first_correct,
                'top_source': top_source_correct
            }
        }
        
        print()
    
    # Summary
    total_tests = len(test_scenarios) * 3  # 3 tests per scenario
    passed_tests = sum(
        sum(result['tests_passed'].values()) 
        for result in results.values()
    )
    
    print(f"📊 Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    
    # Save detailed results
    output_dir = Path("outputs/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "complete_routing_system_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to {output_dir / 'complete_routing_system_results.json'}")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    avg_processing_time = sum(
        result['api_result']['processing_time'] 
        for result in results.values()
    ) / len(results)
    print(f"   Average processing time: {avg_processing_time:.3f}s")
    
    # Singapore-first effectiveness
    singapore_queries = [
        result for result in results.values() 
        if result['classification']['singapore_first']
    ]
    singapore_effectiveness = sum(
        1 for result in singapore_queries 
        if any(source['source'].endswith('_sg') or 'singstat' in source['source'] or 'lta' in source['source'] 
               for source in result['sources'][:1])  # Check top source
    ) / len(singapore_queries) if singapore_queries else 0
    
    print(f"   Singapore-first effectiveness: {singapore_effectiveness*100:.1f}%")
    
    print(f"\n✅ Complete Routing System test complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_complete_routing_system())