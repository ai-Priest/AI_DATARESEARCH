"""
Global ML Evaluator for Dataset Discovery
Creates universal evaluation scenarios and metrics for global dataset recommendation systems.
Works with any regional datasets to provide domain-agnostic evaluation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DatasetDiscoveryEvaluator:
    """Global evaluator for dataset discovery recommendation systems."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.semantic_model = None
        self._init_semantic_model()
        
        # Global domain evaluation scenarios (generalizable across any region/country)
        self.evaluation_scenarios = {
            'housing_research': {
                'query': 'housing prices trends real estate market data',
                'expected_categories': ['housing', 'property', 'urban', 'real estate'],
                'key_terms': ['housing', 'property', 'real estate', 'prices', 'rental', 'market'],
                'user_type': 'researcher'
            },
            'transport_analysis': {
                'query': 'public transport usage traffic patterns mobility',
                'expected_categories': ['transport', 'mobility', 'traffic'],
                'key_terms': ['transport', 'traffic', 'mobility', 'transit', 'vehicle', 'commute'],
                'user_type': 'analyst'
            },
            'economic_indicators': {
                'query': 'gdp employment economic indicators statistics',
                'expected_categories': ['economics', 'employment', 'finance'],
                'key_terms': ['gdp', 'employment', 'economy', 'trade', 'statistics', 'finance'],
                'user_type': 'economist'
            },
            'demographics_study': {
                'query': 'population demographics census age distribution',
                'expected_categories': ['demographics', 'population', 'social'],
                'key_terms': ['population', 'age', 'demographics', 'census', 'social', 'community'],
                'user_type': 'demographer'
            },
            'urban_planning': {
                'query': 'urban planning development land use zoning',
                'expected_categories': ['urban', 'planning', 'development'],
                'key_terms': ['urban', 'planning', 'development', 'land', 'zoning', 'city'],
                'user_type': 'planner'
            },
            'environmental_data': {
                'query': 'environmental data climate air quality pollution',
                'expected_categories': ['environment', 'climate', 'pollution'],
                'key_terms': ['environment', 'climate', 'air quality', 'pollution', 'emissions', 'sustainability'],
                'user_type': 'environmental_scientist'
            },
            'health_research': {
                'query': 'health statistics disease surveillance medical data',
                'expected_categories': ['health', 'medical', 'disease'],
                'key_terms': ['health', 'medical', 'disease', 'hospital', 'patient', 'healthcare'],
                'user_type': 'health_researcher'
            },
            'education_analysis': {
                'query': 'education statistics school enrollment performance',
                'expected_categories': ['education', 'academic', 'school'],
                'key_terms': ['education', 'school', 'student', 'academic', 'learning', 'university'],
                'user_type': 'education_analyst'
            }
        }
        
    def _init_semantic_model(self):
        """Initialize semantic similarity model."""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Semantic model initialized for domain evaluation")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize semantic model: {e}")
            self.semantic_model = None
    
    def calculate_domain_relevance(self, query: str, dataset: Dict) -> float:
        """Calculate domain-specific relevance score."""
        
        query_lower = query.lower()
        title = str(dataset.get('title', '')).lower()
        description = str(dataset.get('description', '')).lower()
        category = str(dataset.get('category', '')).lower()
        
        # Semantic similarity if model available
        if self.semantic_model:
            try:
                query_embedding = self.semantic_model.encode([query])
                dataset_text = f"{title} {description} {category}"
                dataset_embedding = self.semantic_model.encode([dataset_text])
                
                semantic_score = cosine_similarity(query_embedding, dataset_embedding)[0][0]
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
                semantic_score = 0.0
        else:
            semantic_score = 0.0
        
        # Keyword matching
        query_words = set(query_lower.split())
        title_words = set(title.split())
        desc_words = set(description.split())
        cat_words = set(category.split())
        
        title_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
        desc_overlap = len(query_words.intersection(desc_words)) / max(len(query_words), 1)
        cat_overlap = len(query_words.intersection(cat_words)) / max(len(query_words), 1)
        
        # Weighted combination
        keyword_score = (title_overlap * 0.5 + desc_overlap * 0.3 + cat_overlap * 0.2)
        
        # Combine semantic and keyword scores
        if semantic_score > 0:
            combined_score = (semantic_score * 0.7 + keyword_score * 0.3)
        else:
            combined_score = keyword_score
        
        return min(1.0, combined_score)
    
    def evaluate_recommendations_for_scenario(self, scenario_name: str, 
                                            recommendation_engine, 
                                            datasets_df: pd.DataFrame) -> Dict:
        """Evaluate recommendations for a specific domain scenario."""
        
        scenario = self.evaluation_scenarios[scenario_name]
        query = scenario['query']
        expected_categories = scenario['expected_categories']
        key_terms = scenario['key_terms']
        
        logger.info(f"🎯 Evaluating scenario: {scenario_name}")
        logger.info(f"📝 Query: {query}")
        
        # Get recommendations - try multiple method names for compatibility
        try:
            if hasattr(recommendation_engine, 'get_recommendations'):
                recommendations = recommendation_engine.get_recommendations(query, top_k=10, method='hybrid')
                if isinstance(recommendations, dict) and 'recommendations' in recommendations:
                    recs = recommendations['recommendations']
                else:
                    recs = recommendations
            elif hasattr(recommendation_engine, 'recommend_datasets'):
                result = recommendation_engine.recommend_datasets(query, method='hybrid', top_k=10)
                recs = result.get('recommendations', [])
            else:
                logger.error("No compatible recommendation method found")
                return {'error': 'No compatible recommendation method found'}
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {'error': str(e)}
        
        if not recs:
            return {'error': 'No recommendations returned'}
        
        # Calculate domain-specific metrics
        relevance_scores = []
        category_matches = 0
        keyword_matches = 0
        
        for i, rec in enumerate(recs[:5]):  # Top 5 for evaluation
            # Calculate domain relevance
            relevance = self.calculate_domain_relevance(query, rec)
            relevance_scores.append(relevance)
            
            # Check category alignment
            rec_category = str(rec.get('category', '')).lower()
            if any(exp_cat in rec_category for exp_cat in expected_categories):
                category_matches += 1
            
            # Check keyword presence
            rec_text = f"{rec.get('title', '')} {rec.get('description', '')}".lower()
            if any(term in rec_text for term in key_terms):
                keyword_matches += 1
        
        # Calculate metrics
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        category_precision = category_matches / len(recs[:5])
        keyword_coverage = keyword_matches / len(recs[:5])
        
        # Calculate NDCG@3 for this scenario
        ndcg_3 = self._calculate_ndcg_at_k(relevance_scores, k=3)
        
        return {
            'scenario': scenario_name,
            'query': query,
            'num_recommendations': len(recs),
            'avg_relevance': avg_relevance,
            'category_precision': category_precision,
            'keyword_coverage': keyword_coverage,
            'ndcg_at_3': ndcg_3,
            'top_relevance_scores': relevance_scores[:5],
            'recommendations_evaluated': min(5, len(recs))
        }
    
    def _calculate_ndcg_at_k(self, relevance_scores: List[float], k: int = 3) -> float:
        """Calculate NDCG@k for relevance scores."""
        if not relevance_scores or k == 0:
            return 0.0
        
        k = min(k, len(relevance_scores))
        
        # DCG calculation
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
        
        # IDCG calculation (perfect ranking)
        ideal_relevance = sorted(relevance_scores[:k], reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_with_synthetic_behavior(self, recommendation_engine, 
                                       synthetic_behavior_path: str) -> Dict:
        """Evaluate using synthetic dataset discovery behavior."""
        
        logger.info("🔄 Evaluating with synthetic dataset discovery behavior...")
        
        # Load synthetic behavior data
        with open(synthetic_behavior_path, 'r') as f:
            behavior_data = json.load(f)
        
        sessions = behavior_data['sessions']
        logger.info(f"📊 Loaded {len(sessions)} synthetic sessions")
        
        session_results = []
        overall_metrics = {
            'total_sessions': len(sessions),
            'successful_sessions': 0,
            'avg_relevance': 0,
            'avg_satisfaction': 0,
            'avg_ndcg_3': 0,
            'recommendation_accuracy': 0
        }
        
        for session in sessions:
            query = session['search_intent']
            user_interactions = session['user_interactions']
            clicked_datasets = user_interactions['clicked_datasets']
            
            try:
                # Get recommendations for this query - try multiple method names for compatibility
                if hasattr(recommendation_engine, 'get_recommendations'):
                    recommendations = recommendation_engine.get_recommendations(query, top_k=5, method='hybrid')
                    if isinstance(recommendations, dict) and 'recommendations' in recommendations:
                        recs = recommendations['recommendations']
                    else:
                        recs = recommendations
                elif hasattr(recommendation_engine, 'recommend_datasets'):
                    result = recommendation_engine.recommend_datasets(query, method='hybrid', top_k=5)
                    recs = result.get('recommendations', [])
                else:
                    logger.warning("No compatible recommendation method found")
                    recs = []
                
                # Calculate relevance for each recommendation
                relevance_scores = []
                for rec in recs:
                    relevance = self.calculate_domain_relevance(query, rec)
                    relevance_scores.append(relevance)
                
                # Calculate session-specific metrics
                session_ndcg = self._calculate_ndcg_at_k(relevance_scores, k=3)
                session_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
                
                # Check if recommendations align with user clicks
                recommendation_ids = [rec.get('id', rec.get('dataset_id', '')) for rec in recs]
                clicked_ids = [click.get('dataset_id', '') for click in clicked_datasets]
                
                # Calculate intersection (recommendation accuracy)
                intersection = len(set(recommendation_ids).intersection(set(clicked_ids)))
                rec_accuracy = intersection / max(len(recommendation_ids), 1)
                
                session_result = {
                    'query': query,
                    'user_type': session['user_type'],
                    'domain': session['domain'],
                    'ndcg_3': session_ndcg,
                    'avg_relevance': session_relevance,
                    'recommendation_accuracy': rec_accuracy,
                    'user_satisfaction': user_interactions['success_indicators']['satisfaction_score'],
                    'clicked_count': len(clicked_datasets),
                    'recommendations_count': len(recs)
                }
                
                session_results.append(session_result)
                
                # Update overall metrics
                if user_interactions['success_indicators']['found_relevant']:
                    overall_metrics['successful_sessions'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to evaluate session: {e}")
                continue
        
        # Calculate final metrics
        if session_results:
            overall_metrics['avg_relevance'] = np.mean([s['avg_relevance'] for s in session_results])
            overall_metrics['avg_satisfaction'] = np.mean([s['user_satisfaction'] for s in session_results])
            overall_metrics['avg_ndcg_3'] = np.mean([s['ndcg_3'] for s in session_results])
            overall_metrics['recommendation_accuracy'] = np.mean([s['recommendation_accuracy'] for s in session_results])
            overall_metrics['success_rate'] = overall_metrics['successful_sessions'] / overall_metrics['total_sessions']
        
        logger.info(f"📈 Evaluation Results:")
        logger.info(f"  NDCG@3: {overall_metrics['avg_ndcg_3']:.3f}")
        logger.info(f"  Average Satisfaction: {overall_metrics['avg_satisfaction']:.3f}")
        logger.info(f"  Success Rate: {overall_metrics.get('success_rate', 0):.3f}")
        
        return {
            'overall_metrics': overall_metrics,
            'session_results': session_results,
            'evaluation_type': 'synthetic_dataset_discovery'
        }
    
    def run_comprehensive_domain_evaluation(self, recommendation_engine, 
                                          datasets_df: pd.DataFrame) -> Dict:
        """Run comprehensive domain-specific evaluation."""
        
        logger.info("🎯 Starting comprehensive domain-specific evaluation...")
        
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_type': 'domain_specific_dataset_discovery',
            'scenario_results': {},
            'synthetic_behavior_results': {},
            'overall_performance': {}
        }
        
        # Evaluate each scenario
        scenario_scores = []
        for scenario_name in self.evaluation_scenarios:
            scenario_result = self.evaluate_recommendations_for_scenario(
                scenario_name, recommendation_engine, datasets_df
            )
            results['scenario_results'][scenario_name] = scenario_result
            
            if 'error' not in scenario_result:
                scenario_scores.append(scenario_result['ndcg_at_3'])
        
        # Evaluate with synthetic behavior
        synthetic_behavior_path = "data/processed/synthetic_dataset_discovery_behavior.json"
        if Path(synthetic_behavior_path).exists():
            synthetic_results = self.evaluate_with_synthetic_behavior(
                recommendation_engine, synthetic_behavior_path
            )
            results['synthetic_behavior_results'] = synthetic_results
        
        # Calculate overall performance
        if scenario_scores:
            results['overall_performance'] = {
                'scenario_avg_ndcg_3': np.mean(scenario_scores),
                'scenario_count': len(scenario_scores),
                'synthetic_ndcg_3': synthetic_results.get('overall_metrics', {}).get('avg_ndcg_3', 0),
                'synthetic_accuracy': synthetic_results.get('overall_metrics', {}).get('recommendation_accuracy', 0),
                'combined_score': (np.mean(scenario_scores) + 
                                 synthetic_results.get('overall_metrics', {}).get('avg_ndcg_3', 0)) / 2
            }
        
        logger.info("✅ Comprehensive domain evaluation complete!")
        
        return results

def run_domain_specific_evaluation():
    """Run domain-specific evaluation for the ML system."""
    
    # This would need to be integrated with the actual recommendation engine
    logger.info("🎯 Domain-specific evaluation system ready!")
    logger.info("📊 Use DatasetDiscoveryEvaluator with your recommendation engine")
    
    return "Evaluation system created - integrate with recommendation engine"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_domain_specific_evaluation()