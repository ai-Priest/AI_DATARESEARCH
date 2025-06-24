"""
Behavioral ML Evaluation System
Implements proper ML metrics for user behavior-based recommendation systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)

class BehavioralMLEvaluator:
    """ML evaluation system for user behavior-based recommendation systems."""
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        
    def prepare_ml_features(self, user_sessions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert user behavior sessions into ML training features.
        
        Returns:
            X: Feature matrix (session features)
            y: Target labels (session success/engagement)
        """
        features = []
        labels = []
        
        for session in user_sessions:
            # Extract behavioral features
            session_features = [
                session.get('duration_minutes', 0),
                len(session.get('events', [])),
                len(session.get('clicked_items', [])),
                len(session.get('viewed_items', [])),
                session.get('bounce_rate', 0),
                session.get('refinement_count', 0),
                # Add success signals as features for next-session prediction
                session.get('success_signals', {}).get('engagement_score', 0),
                session.get('success_signals', {}).get('search_efficiency', 0),
            ]
            
            # Target: Did user have a successful session?
            success_label = 1 if session.get('success_signals', {}).get('converted', False) else 0
            
            features.append(session_features)
            labels.append(success_label)
        
        return np.array(features), np.array(labels)
    
    def train_engagement_predictor(self, user_sessions: List[Dict]) -> Dict[str, float]:
        """
        Train ML models to predict user engagement.
        
        Returns:
            Dictionary of model performance metrics
        """
        logger.info("🤖 Training engagement prediction models")
        
        X, y = self.prepare_ml_features(user_sessions)
        
        if len(X) < 2:
            logger.warning("⚠️ Insufficient data for ML training")
            return {"error": "insufficient_data", "samples": len(X)}
        
        # Split data for validation
        if len(X) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
        else:
            # Use all data for training if very small dataset
            X_train, X_test, y_train, y_test = X, X, y, y
        
        results = {}
        
        # Train Random Forest
        try:
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)
            
            if len(np.unique(y_test)) > 1:
                rf_proba = rf_model.predict_proba(X_test)[:, 1]
                rf_auc = roc_auc_score(y_test, rf_proba)
                results['random_forest_auc'] = rf_auc
            
            rf_accuracy = rf_model.score(X_test, y_test)
            results['random_forest_accuracy'] = rf_accuracy
            
            self.models['random_forest'] = rf_model
            
        except Exception as e:
            logger.warning(f"Random Forest training failed: {e}")
            results['random_forest_accuracy'] = 0.5
        
        # Train Logistic Regression
        try:
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)
            
            if len(np.unique(y_test)) > 1:
                lr_proba = lr_model.predict_proba(X_test)[:, 1]
                lr_auc = roc_auc_score(y_test, lr_proba)
                results['logistic_regression_auc'] = lr_auc
            
            lr_accuracy = lr_model.score(X_test, y_test)
            results['logistic_regression_accuracy'] = lr_accuracy
            
            self.models['logistic_regression'] = lr_model
            
        except Exception as e:
            logger.warning(f"Logistic Regression training failed: {e}")
            results['logistic_regression_accuracy'] = 0.5
        
        return results
    
    def calculate_ranking_metrics(self, recommendations: List[Dict], 
                                user_behavior: Dict) -> Dict[str, float]:
        """
        Calculate ranking-based ML metrics (NDCG, MAP, MRR).
        
        Args:
            recommendations: List of recommended items with scores
            user_behavior: User's actual behavior (clicks, views)
        
        Returns:
            Dictionary of ranking metrics
        """
        # Extract actual user interactions
        clicked_items = set(item.get('text', '') for item in user_behavior.get('clicked_items', []))
        viewed_items = set(user_behavior.get('viewed_items', []))
        relevant_items = clicked_items.union(viewed_items)
        
        if not recommendations or not relevant_items:
            return {
                'ndcg_at_3': 0.0,
                'ndcg_at_5': 0.0,
                'map_score': 0.0,
                'mrr_score': 0.0
            }
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        relevance_scores = []
        for rec in recommendations[:5]:  # Top 5 recommendations
            item_title = rec.get('title', '')
            is_relevant = 1 if item_title in relevant_items else 0
            relevance_scores.append(is_relevant)
        
        # Calculate NDCG@k
        def calculate_ndcg(relevance_scores: List[int], k: int) -> float:
            if k > len(relevance_scores):
                k = len(relevance_scores)
            
            if k == 0:
                return 0.0
            
            # DCG calculation
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
            
            # IDCG calculation (perfect ranking)
            ideal_relevance = sorted(relevance_scores[:k], reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            
            return dcg / idcg if idcg > 0 else 0.0
        
        # Calculate MAP
        def calculate_map(relevance_scores: List[int]) -> float:
            if not any(relevance_scores):
                return 0.0
            
            precisions = []
            relevant_count = 0
            
            for i, rel in enumerate(relevance_scores):
                if rel == 1:
                    relevant_count += 1
                    precision = relevant_count / (i + 1)
                    precisions.append(precision)
            
            return np.mean(precisions) if precisions else 0.0
        
        # Calculate MRR
        def calculate_mrr(relevance_scores: List[int]) -> float:
            for i, rel in enumerate(relevance_scores):
                if rel == 1:
                    return 1.0 / (i + 1)
            return 0.0
        
        return {
            'ndcg_at_3': calculate_ndcg(relevance_scores, 3),
            'ndcg_at_5': calculate_ndcg(relevance_scores, 5),
            'map_score': calculate_map(relevance_scores),
            'mrr_score': calculate_mrr(relevance_scores)
        }
    
    def calculate_ctr_metrics(self, recommendations: List[Dict], 
                            user_sessions: List[Dict]) -> Dict[str, float]:
        """
        Calculate Click-Through Rate and position-based metrics.
        """
        total_recommendations = 0
        total_clicks = 0
        position_performance = {1: {'shown': 0, 'clicked': 0},
                              2: {'shown': 0, 'clicked': 0},
                              3: {'shown': 0, 'clicked': 0}}
        
        for session in user_sessions:
            clicked_items = set(item.get('text', '') for item in session.get('clicked_items', []))
            
            for pos, rec in enumerate(recommendations[:3], 1):
                total_recommendations += 1
                position_performance[pos]['shown'] += 1
                
                if rec.get('title', '') in clicked_items:
                    total_clicks += 1
                    position_performance[pos]['clicked'] += 1
        
        # Calculate CTR
        overall_ctr = (total_clicks / total_recommendations) if total_recommendations > 0 else 0.0
        
        # Calculate position-specific CTR
        position_ctr = {}
        for pos in [1, 2, 3]:
            shown = position_performance[pos]['shown']
            clicked = position_performance[pos]['clicked']
            position_ctr[f'ctr_position_{pos}'] = (clicked / shown) if shown > 0 else 0.0
        
        return {
            'overall_ctr': overall_ctr,
            **position_ctr,
            'total_recommendations_shown': total_recommendations,
            'total_clicks': total_clicks
        }
    
    def evaluate_recommendation_system(self, recommendation_engine, 
                                     user_sessions: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive ML evaluation of recommendation system using behavioral data.
        """
        logger.info("📊 Starting comprehensive ML evaluation")
        
        # Train engagement prediction models
        engagement_results = self.train_engagement_predictor(user_sessions)
        
        # Evaluate ranking performance
        ranking_results = []
        ctr_data = []
        
        for session in user_sessions:
            if session.get('search_intent'):
                try:
                    # Generate recommendations
                    recommendations = recommendation_engine.get_recommendations(
                        session['search_intent'], top_k=5, method='hybrid'
                    )
                    
                    # Calculate ranking metrics for this session
                    ranking_metrics = self.calculate_ranking_metrics(recommendations, session)
                    ranking_results.append(ranking_metrics)
                    
                    # Collect CTR data
                    ctr_data.append((recommendations, session))
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate session: {e}")
        
        # Aggregate ranking metrics
        if ranking_results:
            avg_ranking_metrics = {
                metric: np.mean([result[metric] for result in ranking_results])
                for metric in ranking_results[0].keys()
            }
        else:
            avg_ranking_metrics = {
                'ndcg_at_3': 0.0,
                'ndcg_at_5': 0.0,
                'map_score': 0.0,
                'mrr_score': 0.0
            }
        
        # Calculate CTR metrics
        if ctr_data and len(ctr_data) > 0:
            sample_recommendations = ctr_data[0][0]  # Use first session's recommendations
            ctr_metrics = self.calculate_ctr_metrics(sample_recommendations, user_sessions)
        else:
            ctr_metrics = {
                'overall_ctr': 0.0,
                'ctr_position_1': 0.0,
                'ctr_position_2': 0.0,
                'ctr_position_3': 0.0,
                'total_recommendations_shown': 0,
                'total_clicks': 0
            }
        
        # Combine all results
        ml_evaluation_results = {
            'engagement_prediction': engagement_results,
            'ranking_metrics': avg_ranking_metrics,
            'ctr_metrics': ctr_metrics,
            'sessions_evaluated': len(user_sessions),
            'total_ranking_evaluations': len(ranking_results)
        }
        
        # Calculate overall ML performance score
        ml_score = self._calculate_overall_ml_score(ml_evaluation_results)
        ml_evaluation_results['overall_ml_score'] = ml_score
        
        logger.info(f"✅ ML evaluation complete. Overall ML Score: {ml_score:.1%}")
        
        return ml_evaluation_results
    
    def _calculate_overall_ml_score(self, results: Dict) -> float:
        """Calculate overall ML performance score from all metrics."""
        
        # Weight different components
        weights = {
            'engagement_prediction': 0.3,
            'ranking_performance': 0.4,
            'ctr_performance': 0.3
        }
        
        # Engagement prediction score
        engagement_score = results.get('engagement_prediction', {}).get('random_forest_accuracy', 0.5)
        
        # Ranking performance score (average of NDCG@3 and MAP)
        ranking_metrics = results.get('ranking_metrics', {})
        ranking_score = (ranking_metrics.get('ndcg_at_3', 0) + ranking_metrics.get('map_score', 0)) / 2
        
        # CTR performance score
        ctr_score = results.get('ctr_metrics', {}).get('overall_ctr', 0)
        
        # Weighted combination
        overall_score = (
            engagement_score * weights['engagement_prediction'] +
            ranking_score * weights['ranking_performance'] +
            ctr_score * weights['ctr_performance']
        )
        
        return overall_score