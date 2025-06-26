#!/usr/bin/env python3
"""
Advanced Ensemble Methods for Neural Networks
Implements sophisticated ensemble techniques to achieve 70%+ NDCG@3 performance
"""

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class AdvancedEnsemble:
    """Advanced ensemble methods for neural network predictions."""
    
    def __init__(self, config: Dict):
        self.config = config.get('ensemble', {})
        self.strategy = self.config.get('strategy', 'adaptive_stacking')
        
        # Model performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.model_weights = {}
        self.query_characteristics_cache = {}
        
        # Meta-learner for stacking
        self.meta_learner = None
        self._initialize_meta_learner()
        
        logger.info(f"🎯 Initialized AdvancedEnsemble with strategy: {self.strategy}")
    
    def _initialize_meta_learner(self):
        """Initialize meta-learner for stacking ensemble."""
        meta_type = self.config.get('stacking', {}).get('meta_learner', 'gradient_boosting')
        
        if meta_type == 'gradient_boosting':
            self.meta_learner = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif meta_type == 'random_forest':
            self.meta_learner = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
        else:  # linear
            self.meta_learner = Ridge(alpha=1.0)
    
    def analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics for adaptive weighting."""
        if query in self.query_characteristics_cache:
            return self.query_characteristics_cache[query]
        
        # Basic characteristics
        words = query.lower().split()
        word_count = len(words)
        char_count = len(query)
        
        # Length classification
        if word_count <= 2:
            length_category = 'short'
        elif word_count <= 5:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        # Complexity analysis
        complex_indicators = ['correlation', 'analysis', 'comparison', 'trend', 'relationship']
        complexity_score = sum(1 for indicator in complex_indicators if indicator in query.lower())
        complexity_category = 'complex' if complexity_score > 0 else 'simple'
        
        # Category detection
        category_keywords = {
            'housing': ['housing', 'property', 'hdb', 'rental', 'home'],
            'transport': ['transport', 'mrt', 'bus', 'traffic', 'road'],
            'economic': ['gdp', 'economic', 'inflation', 'employment', 'income'],
            'health': ['health', 'covid', 'hospital', 'disease', 'medical'],
            'environment': ['environment', 'air', 'pollution', 'energy', 'weather'],
            'education': ['education', 'school', 'university', 'student', 'learning'],
            'demographics': ['population', 'demographic', 'age', 'marriage', 'birth']
        }
        
        detected_category = 'general'
        for category, keywords in category_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_category = category
                break
        
        characteristics = {
            'word_count': word_count,
            'char_count': char_count,
            'length_category': length_category,
            'complexity_category': complexity_category,
            'complexity_score': complexity_score,
            'category': detected_category,
            'has_numbers': bool(re.search(r'\d', query)),
            'has_special_terms': any(term in query.lower() for term in ['singapore', 'sg', 'annual', 'monthly'])
        }
        
        self.query_characteristics_cache[query] = characteristics
        return characteristics
    
    def get_adaptive_weights(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """Get adaptive weights based on query characteristics."""
        characteristics = self.analyze_query_characteristics(query)
        
        # Base weights from configuration
        adaptive_config = self.config.get('adaptive_weights', {})
        
        # Select weight set based on characteristics
        length_key = f"query_length_{characteristics['length_category']}"
        complexity_key = f"query_complexity_{characteristics['complexity_category']}"
        
        length_weights = adaptive_config.get(length_key, [0.5, 0.3, 0.2])
        complexity_weights = adaptive_config.get(complexity_key, [0.5, 0.3, 0.2])
        
        # Combine length and complexity weights
        combined_weights = [(l + c) / 2 for l, c in zip(length_weights, complexity_weights)]
        
        # Map to available models
        primary_models = self.config.get('models', {}).get('primary', ['graph_attention', 'query_encoder', 'siamese_transformer'])
        
        model_weights = {}
        for i, model in enumerate(available_models):
            if model in primary_models:
                idx = primary_models.index(model)
                if idx < len(combined_weights):
                    model_weights[model] = combined_weights[idx]
                else:
                    model_weights[model] = 0.1
            else:
                model_weights[model] = 0.1  # Fallback weight
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {k: v / total_weight for k, v in model_weights.items()}
        
        # Apply performance-based adjustments
        if self.config.get('dynamic_weighting', {}).get('enabled', False):
            model_weights = self._apply_performance_adjustment(model_weights)
        
        return model_weights
    
    def _apply_performance_adjustment(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply performance-based weight adjustments."""
        if not self.performance_history:
            return base_weights
        
        adjustment_rate = self.config.get('dynamic_weighting', {}).get('adaptation_rate', 0.1)
        min_weight = self.config.get('dynamic_weighting', {}).get('min_weight', 0.05)
        
        adjusted_weights = base_weights.copy()
        
        for model_name in base_weights:
            if model_name in self.performance_history:
                recent_performance = list(self.performance_history[model_name])
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    # Boost weights for better performing models
                    performance_multiplier = 1 + (avg_performance - 0.5) * adjustment_rate
                    adjusted_weights[model_name] *= performance_multiplier
                    adjusted_weights[model_name] = max(adjusted_weights[model_name], min_weight)
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def stacking_ensemble(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Advanced stacking ensemble with meta-learning."""
        try:
            # Prepare features for meta-learner
            model_predictions = []
            model_names = []
            
            for model_name, pred in predictions.items():
                if len(pred) > 0:
                    model_predictions.append(pred)
                    model_names.append(model_name)
            
            if len(model_predictions) < 2:
                # Fallback to simple average
                return np.mean(model_predictions, axis=0) if model_predictions else np.array([0.5])
            
            # Stack predictions as features
            X_meta = np.column_stack(model_predictions)
            
            # Add query characteristics as additional features
            if self.config.get('stacking', {}).get('feature_engineering', False):
                characteristics = self.analyze_query_characteristics(query)
                query_features = np.array([
                    characteristics['word_count'],
                    characteristics['char_count'], 
                    characteristics['complexity_score'],
                    float(characteristics['has_numbers']),
                    float(characteristics['has_special_terms'])
                ])
                # Broadcast query features to match prediction length
                query_features_expanded = np.tile(query_features, (X_meta.shape[0], 1))
                X_meta = np.column_stack([X_meta, query_features_expanded])
            
            # Use meta-learner if trained, otherwise use weighted average
            if hasattr(self.meta_learner, 'predict') and hasattr(self.meta_learner, 'n_features_in_'):
                try:
                    # Ensure feature count matches
                    if X_meta.shape[1] == self.meta_learner.n_features_in_:
                        meta_predictions = self.meta_learner.predict(X_meta)
                        return np.clip(meta_predictions, 0, 1)  # Clip to valid range
                except Exception as e:
                    logger.warning(f"Meta-learner prediction failed: {e}")
            
            # Fallback to adaptive weighted average
            weights = self.get_adaptive_weights(query, model_names)
            weighted_pred = np.zeros_like(model_predictions[0])
            
            for i, model_name in enumerate(model_names):
                weight = weights.get(model_name, 1.0 / len(model_names))
                weighted_pred += weight * model_predictions[i]
            
            return weighted_pred
            
        except Exception as e:
            logger.error(f"Stacking ensemble failed: {e}")
            # Fallback to simple average
            if model_predictions:
                return np.mean(model_predictions, axis=0)
            return np.array([0.5])
    
    def voting_ensemble(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Advanced voting ensemble with confidence weighting."""
        try:
            voting_config = self.config.get('voting', {})
            method = voting_config.get('method', 'soft')
            
            if method == 'soft':
                return self._soft_voting(predictions, query)
            elif method == 'rank_based':
                return self._rank_based_voting(predictions, query)
            else:  # hard voting
                return self._hard_voting(predictions, query)
                
        except Exception as e:
            logger.error(f"Voting ensemble failed: {e}")
            # Fallback to simple average
            pred_values = list(predictions.values())
            return np.mean(pred_values, axis=0) if pred_values else np.array([0.5])
    
    def _soft_voting(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Soft voting with adaptive weights."""
        weights = self.get_adaptive_weights(query, list(predictions.keys()))
        
        weighted_pred = None
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1.0 / len(predictions))
            if weighted_pred is None:
                weighted_pred = weight * pred
            else:
                weighted_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0 and weighted_pred is not None:
            return weighted_pred / total_weight
        
        return np.mean(list(predictions.values()), axis=0)
    
    def _rank_based_voting(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Rank-based voting ensemble."""
        # Convert predictions to rankings
        rankings = {}
        for model_name, pred in predictions.items():
            rankings[model_name] = np.argsort(np.argsort(pred)[::-1])
        
        # Combine rankings with weights
        weights = self.get_adaptive_weights(query, list(predictions.keys()))
        
        combined_ranks = np.zeros_like(list(rankings.values())[0])
        for model_name, ranks in rankings.items():
            weight = weights.get(model_name, 1.0 / len(rankings))
            combined_ranks += weight * ranks
        
        # Convert back to scores (inverse ranking)
        max_rank = len(combined_ranks)
        return (max_rank - combined_ranks) / max_rank
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Hard voting ensemble."""
        # Convert to binary predictions (top-k)
        k = min(3, len(list(predictions.values())[0]))
        binary_predictions = {}
        
        for model_name, pred in predictions.items():
            top_k_indices = np.argsort(pred)[-k:]
            binary_pred = np.zeros_like(pred)
            binary_pred[top_k_indices] = 1
            binary_predictions[model_name] = binary_pred
        
        # Sum votes
        vote_sum = np.sum(list(binary_predictions.values()), axis=0)
        
        # Normalize to probabilities
        return vote_sum / len(binary_predictions)
    
    def ensemble_predict(self, predictions: Dict[str, np.ndarray], query: str) -> np.ndarray:
        """Main ensemble prediction method."""
        if not predictions:
            return np.array([0.5])
        
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        try:
            if self.strategy == 'adaptive_stacking':
                result = self.stacking_ensemble(predictions, query)
            elif self.strategy == 'voting':
                result = self.voting_ensemble(predictions, query)
            elif self.strategy == 'stacking':
                result = self.stacking_ensemble(predictions, query)
            else:  # weighted_average
                result = self._soft_voting(predictions, query)
            
            # Apply post-processing optimizations
            if self.config.get('query_optimization', {}).get('confidence_calibration', False):
                result = self._calibrate_confidence(result, query)
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            # Ultimate fallback
            return np.mean(list(predictions.values()), axis=0)
    
    def _calibrate_confidence(self, predictions: np.ndarray, query: str) -> np.ndarray:
        """Calibrate confidence scores based on query characteristics."""
        characteristics = self.analyze_query_characteristics(query)
        
        # Boost confidence for simple, well-understood queries
        if characteristics['complexity_category'] == 'simple' and characteristics['category'] != 'general':
            confidence_boost = 1.1
        else:
            confidence_boost = 1.0
        
        # Apply length normalization
        if self.config.get('query_optimization', {}).get('length_normalization', False):
            length_factor = min(1.2, 1.0 + 0.1 * (characteristics['word_count'] - 2) / 3)
            confidence_boost *= length_factor
        
        calibrated = predictions * confidence_boost
        return np.clip(calibrated, 0, 1)  # Ensure valid probability range
    
    def update_performance(self, model_name: str, performance_score: float):
        """Update performance history for dynamic weighting."""
        self.performance_history[model_name].append(performance_score)
        
    def get_ensemble_explanation(self, query: str, model_weights: Dict[str, float]) -> str:
        """Generate explanation for ensemble decision."""
        characteristics = self.analyze_query_characteristics(query)
        top_model = max(model_weights.items(), key=lambda x: x[1])
        
        explanation = f"Advanced ensemble analysis for '{query}':\n"
        explanation += f"├─ Query characteristics: {characteristics['length_category']} length, {characteristics['complexity_category']} complexity\n"
        explanation += f"├─ Category: {characteristics['category']}\n" 
        explanation += f"├─ Strategy: {self.strategy}\n"
        explanation += f"├─ Primary model: {top_model[0]} ({top_model[1]:.1%} weight)\n"
        explanation += f"└─ Models combined: {len(model_weights)} neural architectures"
        
        return explanation