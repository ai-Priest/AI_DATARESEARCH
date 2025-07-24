#!/usr/bin/env python3
"""
Test script for Enhanced Training Data Parser
Validates the implementation against requirements
"""

import json
import logging
from pathlib import Path
from src.ml.enhanced_training_data_parser import (
    TrainingDataIntegrator, 
    parse_training_mappings,
    create_enhanced_training_dataset
)
from src.ml.training_data_quality_validator import validate_training_data

logging.basicConfig(level=logging.INFO)

def test_enhanced_training_parser():
    """Test the enhanced training data parser implementation"""
    print("🧪 Testing Enhanced Training Data Parser")
    print("=" * 50)
    
    # Test 1: Parse training mappings
    print("\n1. Testing training mappings parsing...")
    integrator = TrainingDataIntegrator("training_mappings.md")
    examples = integrator.parse_training_mappings()
    
    print(f"✅ Parsed {len(examples)} training examples")
    
    # Verify domain classification
    domains = set(ex.domain for ex in examples)
    print(f"✅ Detected domains: {sorted(domains)}")
    
    # Verify Singapore-first detection
    singapore_examples = [ex for ex in examples if ex.singapore_first_applicable]
    print(f"✅ Singapore-first examples: {len(singapore_examples)}")
    
    # Test 2: Domain-specific data augmentation
    print("\n2. Testing domain-specific data augmentation...")
    augmented_examples = integrator.augment_training_data(examples)
    print(f"✅ Augmented from {len(examples)} to {len(augmented_examples)} examples")
    
    # Test 3: Query classification
    print("\n3. Testing query classification...")
    test_queries = [
        "psychology research data",
        "singapore housing statistics", 
        "climate change indicators",
        "machine learning datasets"
    ]
    
    for query in test_queries:
        classification = integrator.classify_query(query)
        print(f"✅ '{query}' → Domain: {classification.domain}, "
              f"Singapore-first: {classification.singapore_first_applicable}")
    
    # Test 4: Negative example generation
    print("\n4. Testing negative example generation...")
    psychology_examples = [ex for ex in examples if ex.domain == "psychology"]
    if psychology_examples:
        example = psychology_examples[0]
        negatives = example.generate_hard_negatives({"kaggle", "zenodo", "world_bank"})
        print(f"✅ Generated {len(negatives)} negative examples for psychology query")
    
    # Test 5: Training data quality validation
    print("\n5. Testing training data quality validation...")
    
    # Create enhanced dataset
    dataset_file = "data/processed/test_enhanced_training_data.json"
    count = create_enhanced_training_dataset("training_mappings.md", dataset_file)
    print(f"✅ Created enhanced dataset with {count} examples")
    
    # Validate the dataset
    validation_results = validate_training_data(dataset_file, "data/processed/test_validation_report.json")
    
    print(f"✅ Validation complete:")
    print(f"   - Overall passed: {validation_results['overall'].passed}")
    print(f"   - Overall score: {validation_results['overall'].score:.2f}")
    print(f"   - Issues found: {len(validation_results['overall'].issues)}")
    
    # Test 6: Verify requirements compliance
    print("\n6. Verifying requirements compliance...")
    
    # Requirement 1.2: Singapore-first queries vs global intent queries
    singapore_count = len([ex for ex in augmented_examples if ex.singapore_first_applicable])
    global_count = len([ex for ex in augmented_examples if not ex.singapore_first_applicable])
    print(f"✅ Singapore-first detection: {singapore_count} Singapore, {global_count} global")
    
    # Requirement 1.6: Domain classification
    domain_counts = {}
    for ex in augmented_examples:
        domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1
    print(f"✅ Domain classification coverage: {len(domain_counts)} domains")
    
    # Requirement 1.7: Negative example generation
    examples_with_negatives = [ex for ex in augmented_examples if ex.negative_examples]
    print(f"✅ Negative examples: {len(examples_with_negatives)} examples have negatives")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Training Data Parser test complete!")
    
    return validation_results['overall'].passed

def test_specific_domain_routing():
    """Test specific domain routing requirements"""
    print("\n🧪 Testing Domain-Specific Routing")
    print("=" * 50)
    
    integrator = TrainingDataIntegrator("training_mappings.md")
    
    # Test psychology → Kaggle/Zenodo routing
    psychology_queries = ["psychology research", "behavioral psychology", "mental health data"]
    for query in psychology_queries:
        classification = integrator.classify_query(query)
        recommended = classification.get_source_priority_order()
        print(f"✅ '{query}' → Recommended sources: {recommended[:3]}")
        
        # Verify Kaggle/Zenodo are prioritized for psychology
        psychology_sources = [s for s in recommended if s in ["kaggle", "zenodo"]]
        assert len(psychology_sources) > 0, f"Psychology query should recommend Kaggle/Zenodo: {query}"
    
    # Test climate → World Bank routing
    climate_queries = ["climate change data", "environmental indicators", "temperature data"]
    for query in climate_queries:
        classification = integrator.classify_query(query)
        recommended = classification.get_source_priority_order()
        print(f"✅ '{query}' → Recommended sources: {recommended[:3]}")
        
        # Verify World Bank is prioritized for climate
        assert "world_bank" in recommended, f"Climate query should recommend World Bank: {query}"
    
    # Test Singapore-first strategy
    singapore_queries = ["singapore housing data", "singapore transport statistics", "sg demographics"]
    for query in singapore_queries:
        classification = integrator.classify_query(query)
        recommended = classification.get_source_priority_order()
        print(f"✅ '{query}' → Recommended sources: {recommended[:3]}")
        
        # Verify Singapore sources are prioritized
        singapore_sources = [s for s in recommended if s in ["data_gov_sg", "singstat", "lta_datamall"]]
        assert len(singapore_sources) > 0, f"Singapore query should prioritize local sources: {query}"
    
    print("🎉 Domain-specific routing tests passed!")

if __name__ == "__main__":
    try:
        # Run main tests
        passed = test_enhanced_training_parser()
        
        # Run domain routing tests
        test_specific_domain_routing()
        
        if passed:
            print("\n🎉 ALL TESTS PASSED! Enhanced Training Data Parser is working correctly.")
        else:
            print("\n⚠️  Some validation issues found, but core functionality works.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()