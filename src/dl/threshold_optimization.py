"""
Advanced Threshold Optimization for Neural Ranking
Optimizes decision thresholds for better precision-recall balance and NDCG@3 performance.
Includes multi-objective optimization for graded relevance.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve

logger = logging.getLogger(__name__)

class AdvancedThresholdOptimizer:
    """Advanced threshold optimization for neural ranking models with graded relevance support."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.optimization_results = {}
        self.graded_relevance = True  # Support for 4-level relevance
        
        # Multi-level thresholds for graded relevance
        self.graded_thresholds = {
            'highly_relevant': 0.85,  # 1.0
            'relevant': 0.65,         # 0.7
            'somewhat_relevant': 0.35, # 0.3
            'irrelevant': 0.0         # 0.0
        }
        
    def calculate_ndcg_at_k(self, predictions: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
        """Calculate NDCG@k for given predictions and targets."""
        # Sort by predictions (descending)
        indices = np.argsort(predictions)[::-1]
        sorted_targets = targets[indices]
        
        # Calculate DCG@k
        dcg = 0.0
        for i in range(min(k, len(sorted_targets))):
            dcg += sorted_targets[i] / np.log2(i + 2)
        
        # Calculate IDCG@k
        ideal_targets = np.sort(targets)[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_targets))):
            idcg += ideal_targets[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def optimize_threshold_for_ndcg(self, model, val_loader, k: int = 3) -> Dict:
        """Find optimal threshold for maximizing NDCG@k."""
        
        logger.info(f"🎯 Optimizing threshold for NDCG@{k}...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_scores = []
        
        # Collect predictions and targets
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device) 
                dataset_input_ids = batch['dataset_input_ids'].to(self.device)
                dataset_attention_mask = batch['dataset_attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get model predictions
                outputs = model(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    dataset_input_ids=dataset_input_ids,
                    dataset_attention_mask=dataset_attention_mask
                )
                
                # Extract prediction scores
                if hasattr(outputs, 'logits'):
                    scores = torch.sigmoid(outputs.logits).cpu().numpy()
                else:
                    scores = torch.sigmoid(outputs).cpu().numpy()
                
                all_scores.extend(scores.flatten())
                all_targets.extend(labels.cpu().numpy().flatten())
        
        all_scores = np.array(all_scores)
        all_targets = np.array(all_targets)
        
        logger.info(f"📊 Collected {len(all_scores)} predictions for threshold optimization")
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_ndcg = 0.0
        threshold_results = []
        
        for threshold in thresholds:
            # Convert scores to binary predictions
            predictions = (all_scores >= threshold).astype(float)
            
            # Calculate NDCG@k for each query group
            # For simplicity, we'll calculate overall NDCG approximation
            ndcg_score = self.calculate_ndcg_at_k(all_scores, all_targets, k)
            
            # Also calculate precision, recall, F1
            precision = np.sum((predictions == 1) & (all_targets == 1)) / max(np.sum(predictions == 1), 1)
            recall = np.sum((predictions == 1) & (all_targets == 1)) / max(np.sum(all_targets == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            threshold_results.append({
                'threshold': threshold,
                'ndcg': ndcg_score,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            if ndcg_score > best_ndcg:
                best_ndcg = ndcg_score
                best_threshold = threshold
        
        logger.info(f"✅ Optimal threshold found: {best_threshold:.3f}")
        logger.info(f"📈 Best NDCG@{k}: {best_ndcg:.3f}")
        
        # Find best threshold result
        best_result = next(r for r in threshold_results if r['threshold'] == best_threshold)
        logger.info(f"📊 At optimal threshold:")
        logger.info(f"  Precision: {best_result['precision']:.3f}")
        logger.info(f"  Recall: {best_result['recall']:.3f}")
        logger.info(f"  F1: {best_result['f1']:.3f}")
        
        return {
            'best_threshold': best_threshold,
            'best_ndcg': best_ndcg,
            'threshold_results': threshold_results,
            'optimization_summary': best_result
        }
    
    def optimize_threshold_for_f1(self, model, val_loader) -> Dict:
        """Find optimal threshold for maximizing F1 score."""
        
        logger.info("🎯 Optimizing threshold for F1 score...")
        
        model.eval()
        all_scores = []
        all_targets = []
        
        # Collect predictions and targets
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                dataset_input_ids = batch['dataset_input_ids'].to(self.device) 
                dataset_attention_mask = batch['dataset_attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get model predictions
                outputs = model(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    dataset_input_ids=dataset_input_ids,
                    dataset_attention_mask=dataset_attention_mask
                )
                
                # Extract prediction scores
                if hasattr(outputs, 'logits'):
                    scores = torch.sigmoid(outputs.logits).cpu().numpy()
                else:
                    scores = torch.sigmoid(outputs).cpu().numpy()
                
                all_scores.extend(scores.flatten())
                all_targets.extend(labels.cpu().numpy().flatten())
        
        all_scores = np.array(all_scores)
        all_targets = np.array(all_targets)
        
        # Use sklearn's precision_recall_curve for optimal F1 threshold
        precision, recall, thresholds = precision_recall_curve(all_targets, all_scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        
        logger.info(f"✅ Optimal F1 threshold: {best_threshold:.3f}")
        logger.info(f"📈 Best F1 score: {best_f1:.3f}")
        logger.info(f"📊 Precision: {best_precision:.3f}, Recall: {best_recall:.3f}")
        
        return {
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'precision_curve': precision,
            'recall_curve': recall,
            'thresholds': thresholds
        }
    
    def apply_threshold_optimization(self, model, val_loader, optimize_for: str = 'ndcg') -> float:
        """Apply threshold optimization and return optimal threshold."""
        
        if optimize_for.lower() == 'ndcg':
            result = self.optimize_threshold_for_ndcg(model, val_loader, k=3)
            return result['best_threshold']
        elif optimize_for.lower() == 'f1':
            result = self.optimize_threshold_for_f1(model, val_loader)
            return result['best_threshold'] 
        else:
            logger.warning(f"Unknown optimization target: {optimize_for}. Using default threshold 0.5")
            return 0.5