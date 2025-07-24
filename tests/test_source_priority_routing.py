"""
Test Source Priority Routing System
Tests the implementation of task 3.2 requirements
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ai.source_priority_router import SourcePriorityRouter

class TestSourcePriorityRouting:
    """Test suite for Source Priority Routing System (Task 3.2)"""
    
    def setup_method(self):
        """Setup test router"""
        self.router = SourcePriorityRouter()
    
    def test_singapore_first_priority_for_singapore_queries(self):
        """Test requirement: Create routing logic that prioritizes data.gov.sg for Singapore queries"""
        
        # Test explicit Singapore queries
        test_cases = [
            ("singapore housing data", "singapore", True),
            ("singapore population statistics", "singapore", True),
            ("data.gov.sg datasets", "singapore", True),
            ("singapore transport", "transportation", True)
        ]
        
        for query, domain, singapore_first in test_cases:
            sources = self.router.route_query(query, domain, singapore_first)
            
            # Should have sources
            assert len(sources) > 0, f"No sources returned for query: {query}"
            
            # First source should be a Singapore government source
            top_source = sources[0]
            singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
            
            assert top_source['source'] in singapore_sources, \
                f"Top source for '{query}' should be Singapore source, got: {top_source['source']}"
            
            # Should have high relevance score
            assert top_source['relevance_score'] >= 0.8, \
                f"Singapore source should have high relevance for '{query}', got: {top_source['relevance_score']}"
            
            print(f"✅ Singapore-first verified for '{query}': {top_source['source']} ({top_source['relevance_score']:.2f})")
    
    def test_psychology_to_kaggle_routing(self):
        """Test requirement: Implement psychology→Kaggle specialized routing"""
        
        psychology_queries = [
            ("psychology research datasets", "psychology", False),
            ("mental health data", "psychology", False),
            ("behavioral psychology", "psychology", False),
            ("cognitive psychology datasets", "psychology", False)
        ]
        
        for query, domain, singapore_first in psychology_queries:
            sources = self.router.route_query(query, domain, singapore_first)
            
            # Should have sources
            assert len(sources) > 0, f"No sources returned for psychology query: {query}"
            
            # Should prioritize Kaggle or Zenodo (research platforms)
            source_names = [s['source'] for s in sources]
            
            # Kaggle should be in top 2 sources for psychology
            assert 'kaggle' in source_names[:2], \
                f"Kaggle should be in top 2 for psychology query '{query}', got: {source_names[:2]}"
            
            # Check if we have training mapping match
            if query.lower() in self.router.training_mappings:
                expected_source = self.router.training_mappings[query.lower()][0]['source']  # Get first (best) source
                assert sources[0]['source'] == expected_source, \
                    f"Training mapping not respected for '{query}'"
            
            print(f"✅ Psychology routing verified for '{query}': {source_names[:2]}")
    
    def test_climate_to_world_bank_routing(self):
        """Test requirement: Implement climate→World Bank specialized routing"""
        
        climate_queries = [
            ("climate change data", "climate", False),
            ("environmental indicators", "climate", False),
            ("temperature data", "climate", False),
            ("climate data", "climate", False)
        ]
        
        for query, domain, singapore_first in climate_queries:
            sources = self.router.route_query(query, domain, singapore_first)
            
            # Should have sources
            assert len(sources) > 0, f"No sources returned for climate query: {query}"
            
            source_names = [s['source'] for s in sources]
            
            # World Bank should be prioritized for climate queries
            assert 'world_bank' in source_names[:2], \
                f"World Bank should be in top 2 for climate query '{query}', got: {source_names[:2]}"
            
            # Check if we have training mapping match
            if query.lower() in self.router.training_mappings:
                expected_source = self.router.training_mappings[query.lower()][0]['source']  # Get first (best) source
                assert sources[0]['source'] == expected_source, \
                    f"Training mapping not respected for '{query}'"
            
            print(f"✅ Climate routing verified for '{query}': {source_names[:2]}")
    
    def test_fallback_routing_for_ambiguous_queries(self):
        """Test requirement: Add fallback routing for ambiguous or new query types"""
        
        # Test ambiguous/new queries that should trigger fallback
        ambiguous_queries = [
            ("random data analysis", "general", False),
            ("unknown research topic", "general", False),
            ("new dataset type", "general", False),
            ("obscure statistics", "general", False)
        ]
        
        for query, domain, singapore_first in ambiguous_queries:
            sources = self.router.route_query(query, domain, singapore_first)
            
            # Should still return sources (fallback working)
            assert len(sources) > 0, f"Fallback routing failed for ambiguous query: {query}"
            
            # Should have fallback routing reasons
            fallback_reasons = [s['routing_reason'] for s in sources if 'fallback' in s['routing_reason']]
            
            # At least some sources should be from fallback routing
            assert len(fallback_reasons) > 0, \
                f"No fallback routing detected for ambiguous query '{query}'"
            
            print(f"✅ Fallback routing verified for '{query}': {[s['source'] for s in sources[:3]]}")
    
    def test_training_mappings_exact_matches(self):
        """Test that exact training mappings are respected"""
        
        # Test some known training mappings
        training_test_cases = [
            ("psychology", "kaggle"),  # Should match training mapping
            ("climate data", "world_bank"),  # Should match training mapping
            ("singapore data", "data_gov_sg")  # Should match training mapping
        ]
        
        for query, expected_source in training_test_cases:
            if query in self.router.training_mappings:
                sources = self.router.route_query(query, "general", False)
                
                # First source should match training mapping
                assert sources[0]['source'] == expected_source, \
                    f"Training mapping not respected: '{query}' should route to {expected_source}, got {sources[0]['source']}"
                
                # Should have exact match reason
                assert sources[0]['routing_reason'] == 'exact_training_match', \
                    f"Should have exact training match reason for '{query}'"
                
                print(f"✅ Training mapping verified: '{query}' → {expected_source}")
    
    def test_domain_specific_quality_thresholds(self):
        """Test that domain-specific quality thresholds are applied"""
        
        # Test that high-quality domains have appropriate thresholds
        high_quality_domains = ['singapore', 'economics', 'health']
        
        for domain in high_quality_domains:
            rules = self.router.domain_routing_rules[domain]
            
            # High-quality domains should have higher thresholds
            assert rules['quality_threshold'] >= 0.8, \
                f"Domain '{domain}' should have high quality threshold, got: {rules['quality_threshold']}"
            
            print(f"✅ Quality threshold verified for {domain}: {rules['quality_threshold']}")
    
    def test_singapore_source_prioritization(self):
        """Test that Singapore sources are properly prioritized"""
        
        # Test queries that should include Singapore sources
        singapore_context_queries = [
            ("housing data", "general", False),  # Generic but should include Singapore
            ("transport statistics", "transportation", False),
            ("population data", "general", False)
        ]
        
        for query, domain, singapore_first in singapore_context_queries:
            sources = self.router.route_query(query, domain, singapore_first)
            
            # Should include Singapore sources when contextually relevant
            source_names = [s['source'] for s in sources]
            singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
            
            has_singapore_source = any(source in singapore_sources for source in source_names)
            
            # For generic queries that could benefit from Singapore data
            if self.router._should_include_singapore_sources(query):
                assert has_singapore_source, \
                    f"Should include Singapore sources for contextually relevant query '{query}'"
                
                print(f"✅ Singapore context verified for '{query}': includes Singapore sources")
    
    def test_fallback_strategies_are_applied(self):
        """Test that different fallback strategies are properly applied"""
        
        # Test different domains trigger appropriate fallback strategies
        domain_fallback_tests = [
            ("psychology", "research_platforms"),
            ("machine_learning", "competition_platforms"),
            ("climate", "official_sources"),
            ("economics", "official_statistics"),
            ("singapore", "government_sources")
        ]
        
        for domain, expected_strategy in domain_fallback_tests:
            rules = self.router.domain_routing_rules[domain]
            
            assert rules['fallback_strategy'] == expected_strategy, \
                f"Domain '{domain}' should have fallback strategy '{expected_strategy}', got: {rules['fallback_strategy']}"
            
            print(f"✅ Fallback strategy verified for {domain}: {expected_strategy}")

def run_comprehensive_routing_test():
    """Run comprehensive test of the routing system"""
    print("🧪 Running Comprehensive Source Priority Routing Tests\n")
    
    router = SourcePriorityRouter()
    
    # Test cases covering all requirements
    comprehensive_test_cases = [
        # Requirement 1: Singapore-first priority
        {
            'query': 'singapore housing data',
            'domain': 'singapore',
            'singapore_first': True,
            'expected_top_source_type': 'singapore',
            'requirement': 'Singapore-first priority'
        },
        
        # Requirement 2a: Psychology → Kaggle
        {
            'query': 'psychology research datasets',
            'domain': 'psychology',
            'singapore_first': False,
            'expected_top_sources': ['kaggle', 'zenodo'],
            'requirement': 'Psychology → Kaggle routing'
        },
        
        # Requirement 2b: Climate → World Bank
        {
            'query': 'climate change indicators',
            'domain': 'climate',
            'singapore_first': False,
            'expected_top_sources': ['world_bank'],
            'requirement': 'Climate → World Bank routing'
        },
        
        # Requirement 3: Fallback routing
        {
            'query': 'obscure research topic',
            'domain': 'general',
            'singapore_first': False,
            'expected_fallback': True,
            'requirement': 'Fallback routing for ambiguous queries'
        }
    ]
    
    print("Testing Task 3.2 Requirements:")
    print("=" * 50)
    
    for i, test_case in enumerate(comprehensive_test_cases, 1):
        query = test_case['query']
        domain = test_case['domain']
        singapore_first = test_case['singapore_first']
        requirement = test_case['requirement']
        
        print(f"\n{i}. Testing: {requirement}")
        print(f"   Query: '{query}' (Domain: {domain}, Singapore-first: {singapore_first})")
        
        # Get routing results
        sources = router.route_query(query, domain, singapore_first)
        
        if not sources:
            print(f"   ❌ FAILED: No sources returned")
            continue
        
        source_names = [s['source'] for s in sources]
        top_source = sources[0]
        
        print(f"   Results: {source_names[:3]}")
        print(f"   Top source: {top_source['source']} (relevance: {top_source['relevance_score']:.2f})")
        print(f"   Routing reason: {top_source['routing_reason']}")
        
        # Validate based on test case expectations
        success = True
        
        if 'expected_top_source_type' in test_case:
            if test_case['expected_top_source_type'] == 'singapore':
                singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
                if top_source['source'] not in singapore_sources:
                    success = False
                    print(f"   ❌ Expected Singapore source, got: {top_source['source']}")
        
        if 'expected_top_sources' in test_case:
            expected = test_case['expected_top_sources']
            if not any(source in source_names[:2] for source in expected):
                success = False
                print(f"   ❌ Expected one of {expected} in top 2, got: {source_names[:2]}")
        
        if 'expected_fallback' in test_case and test_case['expected_fallback']:
            fallback_detected = any('fallback' in s['routing_reason'] for s in sources)
            if not fallback_detected:
                success = False
                print(f"   ❌ Expected fallback routing, but none detected")
        
        if success:
            print(f"   ✅ PASSED: {requirement}")
        else:
            print(f"   ❌ FAILED: {requirement}")
    
    print("\n" + "=" * 50)
    print("✅ Comprehensive routing test completed!")

if __name__ == "__main__":
    # Run the comprehensive test
    run_comprehensive_routing_test()
    
    # Also run pytest if available
    try:
        import pytest
        print("\n🧪 Running pytest suite...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\n⚠️ pytest not available, skipping automated tests")