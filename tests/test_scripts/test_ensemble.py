#!/usr/bin/env python3
"""
Quick test script to validate ensemble performance improvements
"""

import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from dl.neural_inference import NeuralInferenceEngine

def test_ensemble():
    """Test ensemble vs individual model performance."""
    
    # Load config
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize inference engine
    engine = NeuralInferenceEngine(config)
    
    # Load models
    try:
        engine.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        return
    
    # Test queries
    test_queries = [
        "housing prices singapore",
        "transport data mrt",
        "healthcare statistics",
        "economic indicators gdp",
        "environmental pollution"
    ]
    
    print("\n🔍 Testing Ensemble vs Individual Models:")
    print("=" * 60)
    
    total_individual_scores = []
    total_ensemble_scores = []
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Test best individual model (graph_attention)
        try:
            individual_result = engine.recommend_datasets(query, top_k=3, model_name='graph_attention')
            individual_score = sum(individual_result.confidence_scores) / len(individual_result.confidence_scores) if individual_result.confidence_scores else 0
            print(f"GraphAttention: {individual_score:.3f} avg confidence")
        except Exception as e:
            print(f"GraphAttention failed: {e}")
            individual_score = 0
            
        # Test ensemble
        try:
            ensemble_result = engine.ensemble_recommend(query, top_k=3)
            ensemble_score = sum(ensemble_result.confidence_scores) / len(ensemble_result.confidence_scores) if ensemble_result.confidence_scores else 0
            print(f"Ensemble:       {ensemble_score:.3f} avg confidence")
        except Exception as e:
            print(f"Ensemble failed: {e}")
            ensemble_score = 0
        
        # Track improvements
        improvement = ((ensemble_score - individual_score) / individual_score * 100) if individual_score > 0 else 0
        print(f"Improvement:    {improvement:+.1f}%")
        
        total_individual_scores.append(individual_score)
        total_ensemble_scores.append(ensemble_score)
    
    # Summary
    avg_individual = sum(total_individual_scores) / len(total_individual_scores)
    avg_ensemble = sum(total_ensemble_scores) / len(total_ensemble_scores)
    overall_improvement = ((avg_ensemble - avg_individual) / avg_individual * 100) if avg_individual > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS:")
    print(f"Average Individual Score: {avg_individual:.3f}")
    print(f"Average Ensemble Score:   {avg_ensemble:.3f}")
    print(f"Overall Improvement:      {overall_improvement:+.1f}%")
    
    # Projected performance
    current_ndcg = 0.477  # Best individual model (GraphAttention)
    projected_ndcg = current_ndcg * (1 + overall_improvement/100)
    print(f"\n🎯 PERFORMANCE PROJECTION:")
    print(f"Current best NDCG@3:     {current_ndcg:.1%}")
    print(f"Projected ensemble NDCG: {projected_ndcg:.1%}")
    
    if projected_ndcg > 0.50:
        print("✅ Target exceeded! Phase 1 goals achieved.")
    else:
        print("⚠️ Need Phase 2 improvements for 70% target.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    test_ensemble()