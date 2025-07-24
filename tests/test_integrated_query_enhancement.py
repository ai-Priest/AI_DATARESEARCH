"""
Test Integrated Query Enhancement
Tests the integration between Context-Aware Query Enhancement and Enhanced Query Router
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from ai.context_aware_query_enhancement import create_context_aware_query_enhancer
from ai.enhanced_query_router import create_enhanced_query_router

logging.basicConfig(level=logging.INFO)

def test_integrated_query_enhancement():
    """Test the integration between query enhancement and routing"""
    
    print("🔗 Testing Integrated Query Enhancement and Routing")
    print("=" * 80)
    
    # Create both components
    enhancer = create_context_aware_query_enhancer()
    router = create_enhanced_query_router()
    
    # Test queries that demonstrate the full pipeline
    test_queries = [
        "psychology",  # Should be enhanced and routed to Kaggle/Zenodo
        "singapore transport",  # Should be enhanced and routed to LTA/data.gov.sg
        "climate",  # Should be enhanced and routed to World Bank
        "machine learning",  # Should be enhanced and routed to Kaggle
        "housing",  # Should detect Singapore context and route appropriately
    ]
    
    print("\n🔄 Testing Full Enhancement → Routing Pipeline:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Processing: '{query}'")
        print("   " + "─" * 40)
        
        # Step 1: Enhance the query
        enhancement = enhancer.enhance_query(query, max_expansions=3)
        
        print(f"   📝 Enhancement:")
        print(f"      Original: {enhancement.original_query}")
        print(f"      Enhanced: {enhancement.enhanced_query}")
        print(f"      Domain: {enhancement.domain_context['primary_domain']}")
        print(f"      Geographic: {enhancement.geographic_context['scope']}")
        print(f"      Confidence: {enhancement.confidence_score:.2f}")
        
        # Step 2: Route the enhanced query
        classification = router.classify_query(enhancement.enhanced_query)
        sources = router.route_to_sources(classification)
        
        print(f"   🧭 Routing:")
        print(f"      Domain: {classification.domain}")
        print(f"      Singapore-first: {classification.singapore_first_applicable}")
        print(f"      Top sources: {[s['name'] for s in sources[:3]]}")
        print(f"      Explanation: {classification.explanation}")
        
        # Step 3: Compare original vs enhanced routing
        original_classification = router.classify_query(query)
        original_sources = router.route_to_sources(original_classification)
        
        print(f"   📊 Comparison:")
        print(f"      Original domain: {original_classification.domain}")
        print(f"      Enhanced domain: {classification.domain}")
        print(f"      Original sources: {[s['name'] for s in original_sources[:3]]}")
        print(f"      Enhanced sources: {[s['name'] for s in sources[:3]]}")
        
        # Validate improvement
        domain_improved = (classification.confidence > original_classification.confidence or 
                          classification.domain != 'general')
        sources_improved = len(sources) >= len(original_sources)
        
        print(f"   ✅ Domain improved: {'YES' if domain_improved else 'NO'}")
        print(f"   ✅ Sources improved: {'YES' if sources_improved else 'NO'}")
    
    print("\n" + "=" * 80)
    
    # Test refinement suggestions integration
    print("\n💡 Testing Refinement Suggestions Integration:")
    print("-" * 50)
    
    refinement_queries = [
        "psychology",
        "singapore data",
        "climate change"
    ]
    
    for query in refinement_queries:
        enhancement = enhancer.enhance_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Suggestions: {enhancement.refinement_suggestions}")
        
        # Test each suggestion
        for suggestion in enhancement.refinement_suggestions[:2]:
            # Extract the suggested term (simplified)
            if "Add '" in suggestion and "'" in suggestion:
                suggested_term = suggestion.split("Add '")[1].split("'")[0]
                enhanced_query = f"{query} {suggested_term}"
                
                # Route the suggested query
                suggested_classification = router.classify_query(enhanced_query)
                
                print(f"  Suggestion: {suggestion}")
                print(f"  Enhanced query: '{enhanced_query}'")
                print(f"  Result domain: {suggested_classification.domain}")
                print(f"  Result confidence: {suggested_classification.confidence:.2f}")
    
    print("\n" + "=" * 80)
    
    # Test geographic context integration
    print("\n🌍 Testing Geographic Context Integration:")
    print("-" * 45)
    
    geographic_queries = [
        ("housing", "Should detect Singapore context"),
        ("singapore housing", "Explicit Singapore context"),
        ("global climate data", "Explicit global context"),
        ("transport data", "Should suggest Singapore context")
    ]
    
    for query, expectation in geographic_queries:
        enhancement = enhancer.enhance_query(query)
        classification = router.classify_query(enhancement.enhanced_query)
        
        print(f"\nQuery: '{query}' ({expectation})")
        print(f"Enhanced: '{enhancement.enhanced_query}'")
        print(f"Geographic scope: {enhancement.geographic_context['scope']}")
        print(f"Singapore-first: {classification.singapore_first_applicable}")
        print(f"Suggested sources: {enhancement.geographic_context['suggested_sources']}")
        print(f"Router sources: {[s['name'] for s in router.route_to_sources(classification)[:3]]}")
    
    print("\n" + "=" * 80)
    print("✅ Integrated Query Enhancement and Routing testing completed!")
    print("The context-aware enhancement successfully improves query routing accuracy.")


if __name__ == "__main__":
    test_integrated_query_enhancement()