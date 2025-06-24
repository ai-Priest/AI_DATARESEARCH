#!/usr/bin/env python3
"""
Quick evaluation script to test improved models without triggering training
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('src')

from dl.deep_evaluation import DeepEvaluator
import yaml

def quick_eval():
    """Quick evaluation of current models."""
    
    # Load config
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check available checkpoints
    models_dir = Path('models/dl')
    checkpoints = list(models_dir.glob('checkpoint_best_epoch_*.pt'))
    
    if not checkpoints:
        print("❌ No trained checkpoints found")
        return
    
    # Get latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    print(f"📂 Using checkpoint: {latest_checkpoint}")
    
    try:
        # Load checkpoint to get basic info
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        
        if 'training_results' in checkpoint:
            results = checkpoint['training_results']
            print(f"\n📊 Training Results from {latest_checkpoint}:")
            print(f"├─ Final Train Loss: {results.get('final_train_loss', 'N/A'):.4f}")
            print(f"├─ Final Val Loss: {results.get('final_val_loss', 'N/A'):.4f}")
            print(f"├─ Best Val Loss: {results.get('best_val_loss', 'N/A'):.4f}")
            print(f"└─ Epochs Trained: {results.get('epochs_trained', 'N/A')}")
            
            # Check if test results are available
            if 'test_results' in results:
                test_results = results['test_results']
                metrics = test_results.get('metrics', {})
                print(f"\n🎯 Test Performance:")
                print(f"├─ NDCG@3: {metrics.get('ndcg_at_3', 0):.1%}")
                print(f"├─ NDCG@5: {metrics.get('ndcg_at_5', 0):.1%}")
                print(f"├─ Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"└─ F1 Score: {metrics.get('f1', 0):.1%}")
                
                # Compare with baseline
                baseline_ndcg = 0.318  # Previous average
                current_ndcg = metrics.get('ndcg_at_3', 0)
                improvement = ((current_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
                
                print(f"\n📈 Performance Comparison:")
                print(f"├─ Previous Average NDCG@3: {baseline_ndcg:.1%}")
                print(f"├─ Current NDCG@3: {current_ndcg:.1%}")
                print(f"└─ Improvement: {improvement:+.1f}%")
                
                if current_ndcg > 0.5:
                    print("✅ Strong improvement! Phase 1 goals achieved.")
                elif current_ndcg > 0.4:
                    print("🎯 Good progress! Close to intermediate target.")
                else:
                    print("⚠️ Need Phase 2 improvements for 70% target.")
        
        # Check validation loss improvement
        if 'val_loss' in checkpoint:
            val_loss = checkpoint['val_loss']
            baseline_val_loss = 0.288  # Previous best validation loss
            improvement = ((baseline_val_loss - val_loss) / baseline_val_loss * 100) if baseline_val_loss > 0 else 0
            
            print(f"\n🔥 Validation Loss Analysis:")
            print(f"├─ Previous Best Val Loss: {baseline_val_loss:.4f}")
            print(f"├─ Current Val Loss: {val_loss:.4f}")
            print(f"└─ Loss Reduction: {improvement:.1f}%")
            
            if improvement > 50:
                print("🚀 Excellent loss improvement! Expect significant NDCG gains.")
        
        print(f"\n📋 Model Info:")
        print(f"├─ Device: {checkpoint.get('device', 'Unknown')}")
        print(f"├─ Mixed Precision: {checkpoint.get('mixed_precision', 'Unknown')}")
        print(f"└─ Total Parameters: {checkpoint.get('total_parameters', 'Unknown'):,}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        
    # Check if full evaluation results exist
    eval_results_path = Path('outputs/DL/evaluations/evaluation_results.json')
    if eval_results_path.exists():
        try:
            with open(eval_results_path, 'r') as f:
                eval_data = json.load(f)
            
            summary = eval_data.get('summary_metrics', {})
            print(f"\n📊 Latest Evaluation Summary:")
            print(f"├─ Average NDCG@3: {summary.get('avg_ndcg_at_3', 0):.1%}")
            print(f"├─ Best Model: {summary.get('best_model', 'Unknown')}")
            print(f"└─ Best NDCG@3: {summary.get('best_ndcg_at_3', 0):.1%}")
            
        except Exception as e:
            print(f"⚠️ Could not load evaluation results: {e}")

if __name__ == "__main__":
    quick_eval()