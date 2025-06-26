"""
Enhanced ML Pipeline Integration
Integrates all new components: query expansion, user feedback, explanations, 
progressive search, and dataset previews into the ML pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset_preview_generator import DatasetPreviewGenerator
from .explanation_engine import RecommendationExplainer
from .model_evaluation import create_comprehensive_evaluator

# Import existing components
from .model_training import EnhancedRecommendationEngine
from .progressive_search import ProgressiveSearchEngine

# Import new components
from .query_expansion import QueryExpander, initialize_query_expander
from .user_behavior_evaluation import run_user_behavior_evaluation
from .user_feedback_system import FeedbackDrivenModelImprover, UserFeedbackCollector

logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with all new components integrated."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enhancement_config = config.get('enhancements', {})
        
        # Core components
        self.recommendation_engine = None
        self.datasets_df = None
        
        # New enhancement components
        self.query_expander = None
        self.feedback_collector = None
        self.explanation_engine = None
        self.progressive_search = None
        self.preview_generator = None
        
        # Enhancement enablement flags
        self.query_expansion_enabled = self.enhancement_config.get('query_expansion', {}).get('enabled', True)
        self.user_feedback_enabled = self.enhancement_config.get('user_feedback', {}).get('enabled', True)
        self.explanations_enabled = self.enhancement_config.get('explanations', {}).get('enabled', True)
        self.progressive_search_enabled = self.enhancement_config.get('progressive_search', {}).get('enabled', True)
        self.preview_cards_enabled = self.enhancement_config.get('preview_cards', {}).get('enabled', True)
        
        logger.info("🚀 Enhanced ML Pipeline initialized")
    
    def initialize_enhancement_components(self, datasets_df: pd.DataFrame):
        """Initialize all enhancement components."""
        self.datasets_df = datasets_df
        
        logger.info("🔧 Initializing enhancement components...")
        
        # 1. Query Expansion System
        if self.query_expansion_enabled:
            try:
                logger.info("📈 Initializing Query Expansion...")
                self.query_expander = QueryExpander()
                
                # Build vocabularies from datasets
                self.query_expander.build_domain_vocabulary(datasets_df)
                self.query_expander.build_keyword_associations(datasets_df)
                self.query_expander.build_category_keywords(datasets_df)
                self.query_expander.build_singapore_terms()
                
                logger.info("✅ Query Expansion initialized")
            except Exception as e:
                logger.error(f"❌ Query Expansion initialization failed: {e}")
                self.query_expansion_enabled = False
        
        # 2. User Feedback System
        if self.user_feedback_enabled:
            try:
                logger.info("📝 Initializing User Feedback System...")
                feedback_config = self.enhancement_config.get('user_feedback', {})
                feedback_file = feedback_config.get('feedback_file', 'data/feedback/user_feedback.json')
                
                self.feedback_collector = UserFeedbackCollector(feedback_file)
                logger.info("✅ User Feedback System initialized")
            except Exception as e:
                logger.error(f"❌ User Feedback System initialization failed: {e}")
                self.user_feedback_enabled = False
        
        # 3. Explanation Engine
        if self.explanations_enabled:
            try:
                logger.info("💡 Initializing Explanation Engine...")
                self.explanation_engine = RecommendationExplainer()
                logger.info("✅ Explanation Engine initialized")
            except Exception as e:
                logger.error(f"❌ Explanation Engine initialization failed: {e}")
                self.explanations_enabled = False
        
        # 4. Progressive Search
        if self.progressive_search_enabled:
            try:
                logger.info("🔍 Initializing Progressive Search...")
                self.progressive_search = ProgressiveSearchEngine()
                self.progressive_search.initialize_from_datasets(datasets_df)
                logger.info("✅ Progressive Search initialized")
            except Exception as e:
                logger.error(f"❌ Progressive Search initialization failed: {e}")
                self.progressive_search_enabled = False
        
        # 5. Dataset Preview Generator
        if self.preview_cards_enabled:
            try:
                logger.info("🎨 Initializing Dataset Preview Generator...")
                self.preview_generator = DatasetPreviewGenerator()
                logger.info("✅ Dataset Preview Generator initialized")
            except Exception as e:
                logger.error(f"❌ Dataset Preview Generator initialization failed: {e}")
                self.preview_cards_enabled = False
        
        logger.info(f"🎯 Enhancement components initialized: "
                   f"QE={self.query_expansion_enabled}, "
                   f"UF={self.user_feedback_enabled}, "
                   f"EX={self.explanations_enabled}, "
                   f"PS={self.progressive_search_enabled}, "
                   f"PV={self.preview_cards_enabled}")
    
    def enhance_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        """Enhance query using query expansion system."""
        if not self.query_expansion_enabled or not self.query_expander:
            return {
                'original_query': query,
                'expanded_query': query,
                'expansion_terms': [],
                'expansion_sources': [],
                'enhancement_applied': False
            }
        
        try:
            expansion_result = self.query_expander.expand_query(query, max_expansions)
            expansion_result['enhancement_applied'] = True
            
            logger.info(f"🔍 Query expanded: '{query}' → '{expansion_result['expanded_query']}'")
            return expansion_result
        
        except Exception as e:
            logger.error(f"❌ Query expansion failed: {e}")
            return {
                'original_query': query,
                'expanded_query': query,
                'expansion_terms': [],
                'expansion_sources': [],
                'enhancement_applied': False
            }
    
    def get_enhanced_recommendations(self, query: str, method: str = 'hybrid', 
                                   top_k: int = 10, user_context: Dict = None) -> Dict[str, Any]:
        """Get enhanced recommendations with all improvements."""
        
        # 1. Query Enhancement
        query_enhancement = self.enhance_query(query)
        enhanced_query = query_enhancement['expanded_query']
        
        # 2. Get base recommendations
        if not self.recommendation_engine:
            raise ValueError("Recommendation engine not initialized")
        
        # Use enhanced query for recommendations - call appropriate method based on type
        if method == 'tfidf':
            recommendations_list = self.recommendation_engine.recommend_datasets_tfidf(enhanced_query, top_k=top_k)
        elif method == 'semantic':
            recommendations_list = self.recommendation_engine.recommend_datasets_semantic(enhanced_query, top_k=top_k)
        elif method == 'hybrid':
            recommendations_list = self.recommendation_engine.recommend_datasets_hybrid(enhanced_query, top_k=top_k)
        else:
            # Default to hybrid
            recommendations_list = self.recommendation_engine.recommend_datasets_hybrid(enhanced_query, top_k=top_k)
        
        # Convert list format to dict format
        recommendations = {
            'recommendations': recommendations_list,
            'method': method,
            'query': enhanced_query,
            'total_found': len(recommendations_list)
        }
        
        # 3. Add explanations to each recommendation
        if self.explanations_enabled and self.explanation_engine:
            for i, rec in enumerate(recommendations.get('recommendations', [])):
                try:
                    explanation = self.explanation_engine.explain_recommendation(
                        query=query,  # Use original query for explanation
                        recommended_dataset=rec,
                        all_datasets=self.datasets_df,
                        similarity_score=rec.get('score'),
                        user_context=user_context
                    )
                    rec['explanation'] = explanation
                except Exception as e:
                    logger.warning(f"⚠️ Explanation generation failed for recommendation {i}: {e}")
                    rec['explanation'] = {'available': False, 'error': str(e)}
        
        # 4. Add preview cards to each recommendation
        if self.preview_cards_enabled and self.preview_generator:
            for i, rec in enumerate(recommendations.get('recommendations', [])):
                try:
                    preview_card = self.preview_generator.generate_preview_card(
                        dataset=rec,
                        similarity_score=rec.get('score'),
                        explanation=rec.get('explanation'),
                        user_context=user_context
                    )
                    rec['preview_card'] = preview_card
                except Exception as e:
                    logger.warning(f"⚠️ Preview card generation failed for recommendation {i}: {e}")
                    rec['preview_card'] = {'available': False, 'error': str(e)}
        
        # 5. Compile enhanced result
        enhanced_result = {
            'query_info': {
                'original_query': query,
                'enhanced_query': enhanced_query,
                'query_enhancement': query_enhancement
            },
            'recommendations': recommendations.get('recommendations', []),
            'method_used': method,
            'total_found': len(recommendations.get('recommendations', [])),
            'enhancements_applied': {
                'query_expansion': query_enhancement.get('enhancement_applied', False),
                'explanations': self.explanations_enabled,
                'preview_cards': self.preview_cards_enabled
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': recommendations.get('processing_time_ms', 0)
            }
        }
        
        return enhanced_result
    
    def get_progressive_search_suggestions(self, partial_query: str, max_suggestions: int = 8) -> List[Dict]:
        """Get progressive search suggestions."""
        if not self.progressive_search_enabled or not self.progressive_search:
            return []
        
        try:
            return self.progressive_search.get_autocomplete_suggestions(partial_query, max_suggestions)
        except Exception as e:
            logger.error(f"❌ Progressive search failed: {e}")
            return []
    
    def get_query_refinement_suggestions(self, query: str, num_results: int) -> List[Dict]:
        """Get query refinement suggestions."""
        if not self.progressive_search_enabled or not self.progressive_search:
            return []
        
        try:
            return self.progressive_search.get_query_refinement_suggestions(query, num_results)
        except Exception as e:
            logger.error(f"❌ Query refinement failed: {e}")
            return []
    
    def record_user_interaction(self, user_id: str, session_id: str, 
                              query: str, results: List[Dict],
                              interaction_type: str, dataset_id: str = None,
                              rating: int = None) -> Optional[str]:
        """Record user interaction for feedback system."""
        if not self.user_feedback_enabled or not self.feedback_collector:
            return None
        
        try:
            return self.feedback_collector.record_search_interaction(
                user_id=user_id,
                session_id=session_id,
                query=query,
                results=results,
                interaction_type=interaction_type,
                dataset_id=dataset_id,
                rating=rating
            )
        except Exception as e:
            logger.error(f"❌ User interaction recording failed: {e}")
            return None
    
    def record_query_expansion_feedback(self, user_id: str, session_id: str,
                                      original_query: str, expanded_query: str,
                                      expansion_helpful: bool,
                                      preferred_terms: List[str] = None):
        """Record feedback on query expansion."""
        if not self.user_feedback_enabled or not self.feedback_collector:
            return
        
        try:
            self.feedback_collector.record_query_expansion_feedback(
                user_id=user_id,
                session_id=session_id,
                original_query=original_query,
                expanded_query=expanded_query,
                expansion_helpful=expansion_helpful,
                preferred_terms=preferred_terms
            )
        except Exception as e:
            logger.error(f"❌ Query expansion feedback recording failed: {e}")
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Get personalization insights for a user."""
        if not self.user_feedback_enabled or not self.feedback_collector:
            return {'message': 'User feedback system not enabled'}
        
        try:
            return self.feedback_collector.get_personalization_insights(user_id)
        except Exception as e:
            logger.error(f"❌ User insights generation failed: {e}")
            return {'error': str(e)}
    
    def get_feedback_driven_improvements(self) -> List[Dict]:
        """Get suggestions for model improvements based on user feedback."""
        if not self.user_feedback_enabled or not self.feedback_collector:
            return []
        
        try:
            improver = FeedbackDrivenModelImprover(self.feedback_collector)
            return improver.suggest_model_improvements()
        except Exception as e:
            logger.error(f"❌ Feedback-driven improvements failed: {e}")
            return []
    
    def evaluate_with_enhancements(self, behavior_file: str = None) -> Dict:
        """Evaluate the enhanced ML pipeline."""
        if not self.recommendation_engine:
            raise ValueError("Recommendation engine not initialized")
        
        logger.info("📊 Running enhanced evaluation...")
        
        # Standard evaluation using user behavior if available
        if behavior_file and Path(behavior_file).exists():
            logger.info("🎯 Running user behavior evaluation with enhancements...")
            
            # Create enhanced wrapper for evaluation
            enhanced_recommender = EnhancedRecommendationWrapper(
                base_recommender=self.recommendation_engine,
                enhanced_pipeline=self
            )
            
            evaluation_results = run_user_behavior_evaluation(enhanced_recommender, behavior_file)
            
            # Add enhancement metrics
            enhancement_metrics = {
                'components_enabled': {
                    'query_expansion': self.query_expansion_enabled,
                    'user_feedback': self.user_feedback_enabled,
                    'explanations': self.explanations_enabled,
                    'progressive_search': self.progressive_search_enabled,
                    'preview_cards': self.preview_cards_enabled
                },
                'expected_improvements': {
                    'query_expansion_benefit': 'Improved semantic matching and recall',
                    'user_feedback_benefit': 'Personalized recommendations over time',
                    'explanations_benefit': 'Enhanced user trust and understanding',
                    'progressive_search_benefit': 'Faster query formulation',
                    'preview_cards_benefit': 'Better dataset evaluation before selection'
                }
            }
            
            evaluation_results['enhancement_analysis'] = enhancement_metrics
            return evaluation_results
        
        else:
            logger.warning("⚠️ No user behavior data available for enhanced evaluation")
            return {'error': 'User behavior data not available'}
    
    def save_enhanced_models(self, models_dir: str):
        """Save all enhanced components."""
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving enhanced models to {models_dir}")
        
        # Save base recommendation engine
        if self.recommendation_engine:
            self.recommendation_engine.save_models(models_dir)
        
        # Save enhancement components
        enhancement_metadata = {
            'components_enabled': {
                'query_expansion': self.query_expansion_enabled,
                'user_feedback': self.user_feedback_enabled,
                'explanations': self.explanations_enabled,
                'progressive_search': self.progressive_search_enabled,
                'preview_cards': self.preview_cards_enabled
            },
            'config': self.enhancement_config,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save query expander vocabulary if available
        if self.query_expansion_enabled and self.query_expander:
            query_expansion_data = {
                'domain_vocabulary': self.query_expander.domain_vocabulary,
                'keyword_associations': self.query_expander.keyword_associations,
                'category_keywords': self.query_expander.category_keywords,
                'singapore_terms': self.query_expander.singapore_terms,
                'abbreviation_map': self.query_expander.abbreviation_map
            }
            
            with open(models_path / 'query_expansion_data.json', 'w') as f:
                json.dump(query_expansion_data, f, indent=2)
        
        # Save progressive search vocabulary if available
        if self.progressive_search_enabled and self.progressive_search:
            progressive_search_data = {
                'vocabulary': list(self.progressive_search.vocabulary),
                'category_terms': self.progressive_search.category_terms,
                'singapore_terms': list(self.progressive_search.singapore_terms),
                'abbreviation_map': self.progressive_search.abbreviation_map
            }
            
            with open(models_path / 'progressive_search_data.json', 'w') as f:
                json.dump(progressive_search_data, f, indent=2)
        
        # Save user feedback if available
        if self.user_feedback_enabled and self.feedback_collector:
            self.feedback_collector.save_feedback()
        
        # Save enhancement metadata
        with open(models_path / 'enhancement_metadata.json', 'w') as f:
            json.dump(enhancement_metadata, f, indent=2)
        
        logger.info("✅ All enhanced models saved successfully")


class EnhancedRecommendationWrapper:
    """Wrapper to make enhanced pipeline compatible with existing evaluation."""
    
    def __init__(self, base_recommender, enhanced_pipeline):
        self.base_recommender = base_recommender
        self.enhanced_pipeline = enhanced_pipeline
        
        # Delegate all base attributes
        for attr in ['datasets_df', 'tfidf_matrix', 'semantic_embeddings', 'hybrid_alpha']:
            if hasattr(base_recommender, attr):
                setattr(self, attr, getattr(base_recommender, attr))
    
    def recommend_datasets(self, query: str, method: str = 'hybrid', top_k: int = 10) -> Dict:
        """Enhanced recommendation method."""
        # Use enhanced pipeline
        enhanced_result = self.enhanced_pipeline.get_enhanced_recommendations(
            query=query, method=method, top_k=top_k
        )
        
        # Convert to format expected by evaluation
        return {
            'recommendations': enhanced_result['recommendations'],
            'method': method,
            'query': query,
            'total_found': enhanced_result['total_found'],
            'processing_time_ms': enhanced_result['metadata']['processing_time_ms'],
            'enhancements_applied': enhanced_result['enhancements_applied']
        }
    
    def get_recommendations(self, query: str, method: str = 'hybrid', top_k: int = 10) -> List[Dict]:
        """Compatibility method for behavioral evaluation - returns list of recommendations."""
        result = self.recommend_datasets(query, method, top_k)
        return result.get('recommendations', [])


def create_enhanced_ml_pipeline(config: Dict) -> EnhancedMLPipeline:
    """Factory function to create enhanced ML pipeline."""
    return EnhancedMLPipeline(config)


def integrate_enhancements_into_existing_pipeline(base_recommender, datasets_df: pd.DataFrame, config: Dict) -> EnhancedMLPipeline:
    """Integrate enhancements into existing recommendation engine."""
    
    # Create enhanced pipeline
    enhanced_pipeline = EnhancedMLPipeline(config)
    
    # Set the base recommender
    enhanced_pipeline.recommendation_engine = base_recommender
    
    # Initialize enhancement components
    enhanced_pipeline.initialize_enhancement_components(datasets_df)
    
    logger.info("🎯 Successfully integrated enhancements into existing pipeline")
    return enhanced_pipeline