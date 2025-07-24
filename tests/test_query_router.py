"""
Test the Enhanced Query Router with specific examples from training mappings
"""

import logging
from src.ai.enhanced_query_router import create_enhanced_query_router

logging.basicConfig(level=logging.INFO)

def test_specific_queries():
    """Test queries that should match the training mappings exactly"""
    
    router = create_enhanced_query_router()
    
    # Test queries from your training mappings
    test_cases = [
        # Psychology queries (should go to Kaggle/Zenodo)
        ("psychology", "kaggle", "zenodo"),
        ("psychology research", "kaggle", "zenodo"),
        ("mental health data", "kaggle", "zenodo"),
        
        # Machine learning queries (should go to Kaggle)
        ("machine learning", "kaggle", "zenodo"),
        ("artificial intelligence", "kaggle", "zenodo"),
        ("deep learning", "kaggle", "zenodo"),
        
        # Climate queries (should go to World Bank)
        ("climate data", "world_bank", "zenodo"),
        ("climate change", "world_bank", "zenodo"),
        ("environmental data", "world_bank", "zenodo"),
        
        # Economics queries (should go to World Bank)
        ("economic data", "world_bank", "singstat"),
        ("gdp data", "world_bank", "singstat"),
        ("financial data", "world_bank", "singstat"),
        
        # Singapore queries (should prioritize Singapore sources)
        ("singapore data", "data_gov_sg", "singstat"),
        ("singapore housing", "data_gov_sg", "singstat"),
        ("singapore transport", "lta_datamall", "data_gov_sg"),
        
        # Health queries
        ("health data", "world_bank", "data_gov_sg"),
        ("medical data", "zenodo", "world_bank"),
        
        # Education queries
        ("education data", "world_bank", "data_gov_sg"),
        ("student data", "kaggle", "world_bank"),
    ]
    
    print("🧪 Testing Enhanced Query Router with Training Mapping Examples")
    print("=" * 80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for query, expected_primary, expected_secondary in test_cases:
        classification = router.classify_query(query)
        sources = router.route_to_sources(classification)
        
        # Get top 2 recommended sources
        top_sources = [s['name'] for s in sources[:2]]
        
        # Check if expected sources are in top 2
        primary_correct = expected_primary in top_sources
        secondary_correct = expected_secondary in top_sources
        
        if primary_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Display results
        status = "✅" if primary_correct else "❌"
        print(f"{status} Query: '{query}'")
        print(f"    Expected: {expected_primary} (primary), {expected_secondary} (secondary)")
        print(f"    Got: {', '.join(top_sources)}")
        print(f"    Domain: {classification.domain}")
        print(f"    Singapore-first: {classification.singapore_first_applicable}")
        print(f"    Confidence: {classification.confidence:.2f}")
        print()
    
    accuracy = correct_predictions / total_predictions * 100
    print(f"📊 Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    accuracy = test_specific_queries()
    
    if accuracy >= 70:
        print(f"🎉 Great! Router accuracy ({accuracy:.1f}%) meets quality threshold")
    else:
        print(f"⚠️ Router accuracy ({accuracy:.1f}%) needs improvement")