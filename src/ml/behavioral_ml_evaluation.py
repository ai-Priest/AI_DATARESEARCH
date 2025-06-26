"""
Behavioral ML Evaluation System
Implements proper ML metrics for user behavior-based recommendation systems.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class BehavioralMLEvaluator:
    """ML evaluation system for user behavior-based recommendation systems."""
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.semantic_model = None
        self._init_semantic_model()
        
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
    
    def _init_semantic_model(self):
        """Initialize semantic similarity model for cross-domain matching."""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Semantic similarity model initialized")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, using keyword matching")
            self.semantic_model = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize semantic model: {e}")
            self.semantic_model = None
    
    def _extract_semantic_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text for semantic matching."""
        if not text:
            return []
        
        # Clean and normalize text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _calculate_semantic_similarity(self, user_items: List[str], recommendations: List[Dict]) -> List[float]:
        """Calculate semantic similarity between user items and recommendations."""
        if not self.semantic_model or not user_items or not recommendations:
            return [0.0] * len(recommendations)
        
        try:
            # Extract text from user items and recommendations
            user_texts = [item for item in user_items if item]
            rec_texts = [rec.get('title', '') + ' ' + rec.get('description', '') for rec in recommendations]
            
            if not user_texts or not any(rec_texts):
                return [0.0] * len(recommendations)
            
            # Generate embeddings
            user_embeddings = self.semantic_model.encode(user_texts)
            rec_embeddings = self.semantic_model.encode(rec_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(user_embeddings, rec_embeddings)
            
            # Return max similarity for each recommendation
            max_similarities = similarity_matrix.max(axis=0)
            return max_similarities.tolist()
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return self._calculate_keyword_similarity(user_items, recommendations)
    
    def _calculate_keyword_similarity(self, user_items: List[str], recommendations: List[Dict]) -> List[float]:
        """Fallback keyword-based similarity calculation."""
        # Extract keywords from user items
        user_keywords = set()
        for item in user_items:
            user_keywords.update(self._extract_semantic_keywords(item))
        
        if not user_keywords:
            return [0.0] * len(recommendations)
        
        similarities = []
        for rec in recommendations:
            # Extract keywords from recommendation
            rec_text = rec.get('title', '') + ' ' + rec.get('description', '')
            rec_keywords = set(self._extract_semantic_keywords(rec_text))
            
            # Calculate Jaccard similarity
            if rec_keywords:
                intersection = user_keywords.intersection(rec_keywords)
                union = user_keywords.union(rec_keywords)
                similarity = len(intersection) / len(union) if union else 0.0
            else:
                similarity = 0.0
            
            similarities.append(similarity)
        
        return similarities
    
    def _extract_enhanced_search_intent(self, session: Dict) -> str:
        """Extract enhanced search intent from user behavior patterns."""
        intent_signals = []
        
        # Extract from clicked items
        for item in session.get('clicked_items', []):
            text = item.get('target', {}).get('text', '')
            if text and text.lower() not in ['filters', 'filter', ' ', '']:
                intent_signals.append(text)
        
        # Extract from viewed items (if they contain meaningful text)
        for item_id in session.get('viewed_items', []):
            if isinstance(item_id, str) and len(item_id) > 3:
                intent_signals.append(f"item_{item_id}")
        
        # Extract from URL patterns
        for event in session.get('events', []):
            if isinstance(event, dict):
                url = event.get('url', '')
                if '/search/' in url:
                    intent_signals.append('search')
                elif '/item/' in url or '/product/' in url:
                    intent_signals.append('item_view')
        
        # Create enhanced intent
        if intent_signals:
            return ' '.join(intent_signals[:5])  # Limit to first 5 signals
        else:
            return session.get('search_intent', 'general search')
    
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
        
        # Check class distribution before splitting
        unique_classes = np.unique(y)
        
        # Split data for validation
        if len(X) >= 4 and len(unique_classes) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            # Use all data for training if very small dataset or single class
            X_train, X_test, y_train, y_test = X, X, y, y
            if len(unique_classes) == 1:
                logger.warning(f"⚠️ Only one class found in data: {unique_classes[0]}. ML models will use baseline predictions.")
        
        results = {}
        
        # Train Random Forest
        try:
            if len(unique_classes) > 1:
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                
                if len(np.unique(y_test)) > 1:
                    rf_proba = rf_model.predict_proba(X_test)[:, 1]
                    rf_auc = roc_auc_score(y_test, rf_proba)
                    results['random_forest_auc'] = rf_auc
                
                rf_accuracy = rf_model.score(X_test, y_test)
                results['random_forest_accuracy'] = rf_accuracy
                
                self.models['random_forest'] = rf_model
            else:
                # Single class - use baseline prediction
                baseline_accuracy = (unique_classes[0] == y_test).mean()
                results['random_forest_accuracy'] = baseline_accuracy
                logger.info(f"Random Forest baseline accuracy: {baseline_accuracy:.3f}")
            
        except Exception as e:
            logger.warning(f"Random Forest training failed: {e}")
            results['random_forest_accuracy'] = 0.5
        
        # Train Logistic Regression
        try:
            if len(unique_classes) > 1:
                lr_model = LogisticRegression(random_state=42, max_iter=1000)
                lr_model.fit(X_train, y_train)
                
                if len(np.unique(y_test)) > 1:
                    lr_proba = lr_model.predict_proba(X_test)[:, 1]
                    lr_auc = roc_auc_score(y_test, lr_proba)
                    results['logistic_regression_auc'] = lr_auc
                
                lr_accuracy = lr_model.score(X_test, y_test)
                results['logistic_regression_accuracy'] = lr_accuracy
                
                self.models['logistic_regression'] = lr_model
            else:
                # Single class - use baseline prediction
                baseline_accuracy = (unique_classes[0] == y_test).mean()
                results['logistic_regression_accuracy'] = baseline_accuracy
                logger.info(f"Logistic Regression baseline accuracy: {baseline_accuracy:.3f}")
            
        except Exception as e:
            logger.warning(f"Logistic Regression training failed: {e}")
            results['logistic_regression_accuracy'] = 0.5
        
        return results
    
    def calculate_ranking_metrics(self, recommendations: List[Dict], 
                                user_behavior: Dict) -> Dict[str, float]:
        """
        Calculate ranking-based ML metrics (NDCG, MAP, MRR) with semantic similarity.
        
        Args:
            recommendations: List of recommended items with scores
            user_behavior: User's actual behavior (clicks, views)
        
        Returns:
            Dictionary of ranking metrics
        """
        # Extract actual user interactions
        clicked_items = [item.get('target', {}).get('text', '') for item in user_behavior.get('clicked_items', [])]
        viewed_items = [str(item) for item in user_behavior.get('viewed_items', [])]
        user_items = [item for item in clicked_items + viewed_items if item and item.strip()]
        
        if not recommendations or not user_items:
            return {
                'ndcg_at_3': 0.0,
                'ndcg_at_5': 0.0,
                'map_score': 0.0,
                'mrr_score': 0.0
            }
        
        # Calculate semantic similarities
        similarities = self._calculate_semantic_similarity(user_items, recommendations)
        
        # Convert similarities to relevance scores (threshold-based)
        relevance_threshold = 0.3  # Minimum similarity to be considered relevant
        relevance_scores = []
        for i, rec in enumerate(recommendations[:5]):  # Top 5 recommendations
            similarity = similarities[i] if i < len(similarities) else 0.0
            
            # Graded relevance based on similarity score
            if similarity >= 0.7:
                relevance = 1.0  # Highly relevant
            elif similarity >= 0.5:
                relevance = 0.7  # Moderately relevant
            elif similarity >= relevance_threshold:
                relevance = 0.3  # Somewhat relevant
            else:
                relevance = 0.0  # Not relevant
                
            relevance_scores.append(relevance)
        
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
    
    def calculate_ctr_metrics(self, recommendation_engine, 
                            user_sessions: List[Dict]) -> Dict[str, float]:
        """
        Calculate Click-Through Rate and position-based metrics with session-specific recommendations.
        """
        total_recommendations = 0
        total_clicks = 0
        position_performance = {1: {'shown': 0, 'clicked': 0},
                              2: {'shown': 0, 'clicked': 0},
                              3: {'shown': 0, 'clicked': 0}}
        
        for session in user_sessions:
            # Extract user interactions
            clicked_items = [item.get('target', {}).get('text', '') for item in session.get('clicked_items', [])]
            clicked_items = [item for item in clicked_items if item and item.strip()]
            
            if not clicked_items:
                continue
            
            # Generate session-specific recommendations
            try:
                search_intent = self._extract_enhanced_search_intent(session)
                session_recommendations = recommendation_engine.get_recommendations(
                    search_intent, top_k=3, method='hybrid'
                )
                
                if not session_recommendations:
                    continue
                
                # Calculate semantic similarities for this session
                similarities = self._calculate_semantic_similarity(clicked_items, session_recommendations)
                
                for pos, (rec, similarity) in enumerate(zip(session_recommendations[:3], similarities), 1):
                    total_recommendations += 1
                    position_performance[pos]['shown'] += 1
                    
                    # Consider it a "click" if semantic similarity is high enough
                    if similarity >= 0.4:  # Threshold for considering it a relevant click
                        total_clicks += 1
                        position_performance[pos]['clicked'] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for session: {e}")
                continue
        
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
        
        # Evaluate ranking performance with enhanced search intent
        ranking_results = []
        
        for session in user_sessions:
            try:
                # Generate enhanced search intent
                search_intent = self._extract_enhanced_search_intent(session)
                
                # Generate recommendations
                recommendations = recommendation_engine.get_recommendations(
                    search_intent, top_k=5, method='hybrid'
                )
                
                if recommendations:
                    # Calculate ranking metrics for this session
                    ranking_metrics = self.calculate_ranking_metrics(recommendations, session)
                    ranking_results.append(ranking_metrics)
                
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
        
        # Calculate CTR metrics with session-specific recommendations
        try:
            ctr_metrics = self.calculate_ctr_metrics(recommendation_engine, user_sessions)
        except Exception as e:
            logger.warning(f"CTR calculation failed: {e}")
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