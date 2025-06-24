#!/usr/bin/env python3
"""
Quick evaluation to get ACTUAL performance metrics
Tests the real trained models against actual ground truth
"""

import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from dl.neural_inference import NeuralInferenceEngine
from dl.deep_evaluation import DeepEvaluator

def quick_performance_test():
    """Test actual performance of trained models."""
    
    print("🔬 ACTUAL PERFORMANCE TEST")
    print("=" * 60)
    
    # Load configuration
    try:
        with open('config/dl_config.yml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Check if we have ground truth
    gt_path = Path("data/processed/intelligent_ground_truth.json")
    if not gt_path.exists():
        print(f"❌ Ground truth not found at {gt_path}")
        return
    
    # Load ground truth
    try:
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"✅ Ground truth loaded: {len(ground_truth)} scenarios")
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        return
    
    # Initialize inference engine
    try:
        inference_engine = NeuralInferenceEngine(config)
        print("✅ Inference engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return
    
    # Load models
    try:
        inference_engine.load_models()
        print("✅ Models loaded successfully")
        print(f"   Available models: {list(inference_engine.models.keys())}")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Test a few sample queries
    print("\n🧪 SAMPLE QUERY TESTS:")
    print("-" * 40)
    
    test_queries = [
        "housing prices singapore",
        "traffic data analysis", 
        "economic indicators trends",
        "population demographics"
    ]
    
    sample_results = []
    
    for query in test_queries:
        try:
            # Test individual models first
            individual_results = {}
            for model_name in inference_engine.models.keys():
                if hasattr(inference_engine, 'single_model_recommend'):
                    result = inference_engine.single_model_recommend(query, model_name=model_name)
                    if result and result.confidence_scores:
                        avg_conf = np.mean(result.confidence_scores)
                        individual_results[model_name] = avg_conf
            
            # Test ensemble
            ensemble_result = inference_engine.ensemble_recommend(query, top_k=3)
            ensemble_conf = np.mean(ensemble_result.confidence_scores) if ensemble_result.confidence_scores else 0
            
            # Test advanced ensemble if available
            advanced_conf = None
            if hasattr(inference_engine, 'advanced_ensemble_recommend'):
                try:
                    advanced_result = inference_engine.advanced_ensemble_recommend(query, top_k=3)
                    advanced_conf = np.mean(advanced_result.confidence_scores) if advanced_result.confidence_scores else 0
                except:
                    pass
            
            print(f"\nQuery: '{query}'")
            print(f"  Ensemble confidence: {ensemble_conf:.3f}")
            if advanced_conf:
                print(f"  Advanced ensemble: {advanced_conf:.3f}")
            if individual_results:
                for model, conf in individual_results.items():
                    print(f"  {model}: {conf:.3f}")
            
            sample_results.append({
                'query': query,
                'ensemble_confidence': ensemble_conf,
                'advanced_confidence': advanced_conf,
                'individual_results': individual_results
            })
            
        except Exception as e:
            print(f"  ❌ Error testing '{query}': {e}")
    
    # Calculate basic performance metrics
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    if sample_results:
        ensemble_scores = [r['ensemble_confidence'] for r in sample_results]
        avg_ensemble = np.mean(ensemble_scores)
        
        advanced_scores = [r['advanced_confidence'] for r in sample_results if r['advanced_confidence'] is not None]
        avg_advanced = np.mean(advanced_scores) if advanced_scores else None
        
        print(f"Average Ensemble Confidence: {avg_ensemble:.3f}")
        if avg_advanced:
            print(f"Average Advanced Ensemble: {avg_advanced:.3f}")
            improvement = ((avg_advanced - avg_ensemble) / avg_ensemble * 100) if avg_ensemble > 0 else 0
            print(f"Advanced vs Basic: {improvement:+.1f}% difference")
        
        # Compare to baseline and targets
        baseline_ml = 0.37  # 37% F1@3 from ML pipeline
        previous_dl = 0.318  # Previous DL average
        target = 0.70  # 70% target
        
        print(f"\nComparison to benchmarks:")
        print(f"  ML Baseline (37%): {((avg_ensemble - baseline_ml) / baseline_ml * 100):+.0f}% change")
        print(f"  Previous DL (31.8%): {((avg_ensemble - previous_dl) / previous_dl * 100):+.0f}% change")
        print(f"  Target (70%): {'✅ EXCEEDED' if avg_ensemble >= target else '❌ Below target'}")
        
        if avg_ensemble >= target:
            print(f"🎉 TARGET ACHIEVED! {avg_ensemble:.1%} vs {target:.0%} target")
        else:
            print(f"📈 Progress: {avg_ensemble:.1%} (need {target - avg_ensemble:.1%} more)")
    
    # Run ground truth evaluation if possible
    print("\n🎯 GROUND TRUTH EVALUATION:")
    print("-" * 40)
    
    try:
        evaluator = DeepEvaluator(config)
        
        # Create simple test scenarios from ground truth
        test_scenarios = []
        count = 0
        for scenario_id, scenario_data in ground_truth.items():
            if count >= 10:  # Test first 10 scenarios for speed
                break
            
            query = scenario_data.get('query', '')
            expected_datasets = scenario_data.get('expected_datasets', [])
            
            if query and expected_datasets:
                test_scenarios.append({
                    'query': query,
                    'expected': expected_datasets[:3]  # Top 3 expected
                })
                count += 1
        
        if test_scenarios:
            print(f"Testing {len(test_scenarios)} ground truth scenarios...")
            
            ndcg_scores = []
            precision_scores = []
            
            for scenario in test_scenarios:
                try:
                    # Get recommendations
                    result = inference_engine.ensemble_recommend(scenario['query'], top_k=3)
                    if result and result.recommendations:
                        recommended_ids = [r.get('dataset_id', r.get('id', '')) for r in result.recommendations]
                        expected_ids = scenario['expected']
                        
                        # Calculate simple precision@3
                        hits = len(set(recommended_ids) & set(expected_ids))
                        precision = hits / min(len(recommended_ids), 3)
                        precision_scores.append(precision)
                        
                        # Simple NDCG@3 approximation
                        dcg = sum([1.0 / np.log2(i + 2) for i, rec_id in enumerate(recommended_ids) if rec_id in expected_ids])
                        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(3, len(expected_ids)))])
                        ndcg = dcg / idcg if idcg > 0 else 0
                        ndcg_scores.append(ndcg)
                        
                except Exception as e:
                    print(f"    Error in scenario: {e}")
                    continue
            
            if ndcg_scores and precision_scores:
                avg_ndcg = np.mean(ndcg_scores)
                avg_precision = np.mean(precision_scores)
                
                print(f"📊 Ground Truth Results:")
                print(f"  Average NDCG@3: {avg_ndcg:.3f} ({avg_ndcg:.1%})")
                print(f"  Average Precision@3: {avg_precision:.3f} ({avg_precision:.1%})")
                
                # Compare to targets
                if avg_ndcg >= 0.70:
                    print(f"🎉 NDCG@3 TARGET ACHIEVED! {avg_ndcg:.1%} vs 70% target")
                    exceed_amount = (avg_ndcg - 0.70) / 0.70 * 100
                    print(f"🏆 Exceeds target by {exceed_amount:.1f}%")
                else:
                    shortfall = (0.70 - avg_ndcg) / 0.70 * 100
                    print(f"📈 NDCG@3: {avg_ndcg:.1%} (need {shortfall:.1f}% more for 70% target)")
                
                return avg_ndcg, avg_precision
            else:
                print("❌ No valid evaluation results")
        else:
            print("❌ No valid test scenarios found")
            
    except Exception as e:
        print(f"❌ Ground truth evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None

if __name__ == "__main__":
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)
    
    print("🚀 QUICK EVALUATION OF ACTUAL TRAINED MODELS")
    print("Testing real performance vs projections")
    print()
    
    ndcg, precision = quick_performance_test()
    
    if ndcg is not None:
        print(f"\n🎯 FINAL ACTUAL RESULTS:")
        print(f"NDCG@3: {ndcg:.1%}")
        print(f"Precision@3: {precision:.1%}")
        
        if ndcg >= 0.70:
            print(f"✅ BREAKTHROUGH CONFIRMED!")
        else:
            print(f"📊 Current performance documented")
    else:
        print(f"\n📋 Basic confidence testing completed")