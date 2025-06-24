# Model Inference Module - Real-time Query Processing & Recommendations
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import logging
from functools import lru_cache
import hashlib

# Setup logging
logger = logging.getLogger(__name__)


class ProductionRecommendationEngine:
    """
    Production-ready recommendation engine for real-time inference
    with caching, optimization, and comprehensive query processing.
    """
    
    def __init__(self, config: Dict, models_dir: str = "models"):
        """Initialize production recommendation engine"""
        self.config = config
        self.inference_config = config.get('inference', {})
        self.models_dir = Path(models_dir)
        
        # Performance settings
        self.response_timeout = self.inference_config.get('response_timeout', 2.0)
        self.cache_enabled = self.inference_config.get('real_time', {}).get('cache_results', True)
        self.cache_duration = self.inference_config.get('real_time', {}).get('cache_duration', 300)
        
        # Query processing settings
        self.query_config = self.inference_config.get('query_processing', {})
        self.default_top_k = self.inference_config.get('default_top_k', 5)
        self.max_top_k = self.inference_config.get('max_top_k', 20)
        self.min_threshold = self.inference_config.get('min_similarity_threshold', 0.01)
        
        # Initialize model components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.datasets_df = None
        self.hybrid_config = None
        
        # Cache for recommendations
        self._recommendation_cache = {}
        self._cache_timestamps = {}
        
        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        
        # Load models
        self._load_trained_models()
        
        logger.info("🚀 ProductionRecommendationEngine initialized and ready")
    
    def _load_trained_models(self):
        """Load all trained models from disk"""
        try:
            logger.info("📂 Loading trained models...")
            
            # Load TF-IDF components
            tfidf_vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
            if tfidf_vectorizer_path.exists():
                with open(tfidf_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("✅ TF-IDF vectorizer loaded")
                
                # Load TF-IDF matrix
                tfidf_matrix_paths = [
                    self.models_dir / "tfidf_matrix.npz",
                    self.models_dir / "tfidf_matrix.npy"
                ]
                
                for matrix_path in tfidf_matrix_paths:
                    if matrix_path.exists():
                        if matrix_path.suffix == '.npz':
                            self.tfidf_matrix = np.load(matrix_path)['matrix']
                        else:
                            self.tfidf_matrix = np.load(matrix_path)
                        logger.info(f"✅ TF-IDF matrix loaded: {self.tfidf_matrix.shape}")
                        break
            
            # Load semantic components
            semantic_embeddings_paths = [
                self.models_dir / "semantic_embeddings.npz",
                self.models_dir / "semantic_embeddings.npy"
            ]
            
            for embeddings_path in semantic_embeddings_paths:
                if embeddings_path.exists():
                    if embeddings_path.suffix == '.npz':
                        self.semantic_embeddings = np.load(embeddings_path)['embeddings']
                    else:
                        self.semantic_embeddings = np.load(embeddings_path)
                    logger.info(f"✅ Semantic embeddings loaded: {self.semantic_embeddings.shape}")
                    break
            
            # Load semantic model
            hybrid_config_path = self.models_dir / "hybrid_weights.pkl"
            if hybrid_config_path.exists():
                with open(hybrid_config_path, 'rb') as f:
                    self.hybrid_config = pickle.load(f)
                
                # Initialize semantic model
                model_name = self.hybrid_config.get('semantic_model_name', 'all-MiniLM-L6-v2')
                device = self.hybrid_config.get('device', 'cpu')
                
                try:
                    self.semantic_model = SentenceTransformer(model_name, device=device)
                    logger.info(f"✅ Semantic model loaded: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load semantic model {model_name}: {e}")
                    # Fallback
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    logger.info("✅ Fallback semantic model loaded")
            
            # Load datasets metadata
            metadata_path = self.models_dir / "datasets_metadata.csv"
            if metadata_path.exists():
                self.datasets_df = pd.read_csv(metadata_path)
                logger.info(f"✅ Dataset metadata loaded: {len(self.datasets_df)} datasets")
            
            # Validate loaded models
            self._validate_loaded_models()
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _validate_loaded_models(self):
        """Validate that all required models are loaded properly"""
        validation_results = {
            'tfidf_vectorizer': self.tfidf_vectorizer is not None,
            'tfidf_matrix': self.tfidf_matrix is not None,
            'semantic_model': self.semantic_model is not None,
            'semantic_embeddings': self.semantic_embeddings is not None,
            'datasets_df': self.datasets_df is not None and len(self.datasets_df) > 0,
            'hybrid_config': self.hybrid_config is not None
        }
        
        missing_components = [name for name, loaded in validation_results.items() if not loaded]
        
        if missing_components:
            logger.warning(f"⚠️ Missing model components: {missing_components}")
            logger.warning("Some recommendation methods may not be available")
        else:
            logger.info("✅ All model components validated successfully")
        
        # Check data consistency
        if self.datasets_df is not None and self.tfidf_matrix is not None:
            if len(self.datasets_df) != self.tfidf_matrix.shape[0]:
                logger.warning(f"⚠️ Data inconsistency: {len(self.datasets_df)} datasets vs {self.tfidf_matrix.shape[0]} TF-IDF rows")
        
        if self.datasets_df is not None and self.semantic_embeddings is not None:
            if len(self.datasets_df) != self.semantic_embeddings.shape[0]:
                logger.warning(f"⚠️ Data inconsistency: {len(self.datasets_df)} datasets vs {self.semantic_embeddings.shape[0]} embedding rows")
    
    def _get_cache_key(self, query: str, method: str, top_k: int) -> str:
        """Generate cache key for query"""
        key_string = f"{method}:{query}:{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if not self.cache_enabled or cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        return (time.time() - cache_time) < self.cache_duration
    
    def _cache_recommendation(self, cache_key: str, recommendations: List[Dict]):
        """Cache recommendation result"""
        if self.cache_enabled:
            self._recommendation_cache[cache_key] = recommendations
            self._cache_timestamps[cache_key] = time.time()
            
            # Clean old cache entries (simple cleanup)
            if len(self._recommendation_cache) > 1000:  # Limit cache size
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:100]  # Remove oldest 100 entries
                
                for key in oldest_keys:
                    self._recommendation_cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query with optional enhancements"""
        try:
            # Basic cleaning
            processed_query = query.strip().lower()
            
            # Query expansion (if enabled)
            if self.query_config.get('expand_query', False):
                # Add common synonyms or expansions
                query_expansions = {
                    'housing': 'housing property real estate',
                    'transport': 'transport transportation traffic',
                    'health': 'health medical healthcare',
                    'economic': 'economic financial economy',
                    'education': 'education academic school university'
                }
                
                for term, expansion in query_expansions.items():
                    if term in processed_query:
                        processed_query = f"{processed_query} {expansion}"
            
            # Spell correction (if enabled)
            if self.query_config.get('spell_correction', False):
                # Simple spell correction could be implemented here
                # For now, just return the processed query
                pass
            
            return processed_query
            
        except Exception as e:
            logger.warning(f"⚠️ Query preprocessing failed: {e}")
            return query.strip()
    
    def recommend_datasets(
        self, 
        query: str, 
        method: str = 'hybrid',
        top_k: Optional[int] = None,
        include_explanations: bool = None,
        timeout: Optional[float] = None
    ) -> Dict:
        """
        Main recommendation interface with comprehensive features
        
        Args:
            query: Search query
            method: Recommendation method ('tfidf', 'semantic', 'hybrid')
            top_k: Number of recommendations to return
            include_explanations: Include explanations in results
            timeout: Custom timeout for this request
            
        Returns:
            Dictionary with recommendations and metadata
        """
        start_time = time.time()
        
        try:
            # Parameter validation and defaults
            top_k = top_k or self.default_top_k
            top_k = min(top_k, self.max_top_k)  # Enforce maximum
            include_explanations = include_explanations if include_explanations is not None else self.inference_config.get('include_explanations', True)
            timeout = timeout or self.response_timeout
            
            # Query preprocessing
            processed_query = self._preprocess_query(query)
            
            # Check cache
            cache_key = self._get_cache_key(processed_query, method, top_k)
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                cached_result = self._recommendation_cache[cache_key]
                return self._format_response(
                    cached_result, processed_query, method, start_time, from_cache=True
                )
            
            # Get recommendations based on method
            recommendations = []
            
            if method == 'tfidf' and self.tfidf_vectorizer is not None:
                recommendations = self._recommend_tfidf(processed_query, top_k)
            elif method == 'semantic' and self.semantic_model is not None:
                recommendations = self._recommend_semantic(processed_query, top_k)
            elif method == 'hybrid' and self.tfidf_vectorizer is not None and self.semantic_model is not None:
                recommendations = self._recommend_hybrid(processed_query, top_k)
            else:
                # Fallback to available method
                if self.tfidf_vectorizer is not None:
                    recommendations = self._recommend_tfidf(processed_query, top_k)
                    method = 'tfidf'
                elif self.semantic_model is not None:
                    recommendations = self._recommend_semantic(processed_query, top_k)
                    method = 'semantic'
                else:
                    raise ValueError("No recommendation models available")
            
            # Apply timeout check
            if time.time() - start_time > timeout:
                logger.warning(f"⚠️ Query timeout exceeded: {time.time() - start_time:.2f}s")
                return self._format_error_response("Query timeout exceeded", start_time)
            
            # Cache successful result
            self._cache_recommendation(cache_key, recommendations)
            
            # Add explanations if requested
            if include_explanations:
                recommendations = self._add_explanations(recommendations, processed_query, method)
            
            # Update performance tracking
            self.query_count += 1
            self.total_response_time += (time.time() - start_time)
            
            return self._format_response(recommendations, processed_query, method, start_time)
            
        except Exception as e:
            logger.error(f"❌ Recommendation failed: {e}")
            return self._format_error_response(str(e), start_time)
    
    def _recommend_tfidf(self, query: str, top_k: int) -> List[Dict]:
        """TF-IDF based recommendations"""
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Apply threshold
            valid_indices = np.where(similarities > self.min_threshold)[0]
            
            if len(valid_indices) == 0:
                return []
            
            # Get top k recommendations
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
            
            recommendations = []
            for idx in sorted_indices:
                dataset = self.datasets_df.iloc[idx]
                recommendations.append({
                    'dataset_id': dataset['dataset_id'],
                    'title': dataset['title'],
                    'description': self._truncate_description(dataset['description']),
                    'category': dataset.get('category', 'unknown'),
                    'source': dataset.get('source', 'unknown'),
                    'quality_score': dataset.get('quality_score', 0.5),
                    'similarity_score': float(similarities[idx]),
                    'method': 'TF-IDF',
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ TF-IDF recommendation failed: {e}")
            return []
    
    def _recommend_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Semantic based recommendations"""
        try:
            # Encode query
            query_embedding = self.semantic_model.encode([query], normalize_embeddings=True)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()
            
            # Apply threshold
            valid_indices = np.where(similarities > self.min_threshold)[0]
            
            if len(valid_indices) == 0:
                return []
            
            # Get top k recommendations
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
            
            recommendations = []
            for idx in sorted_indices:
                dataset = self.datasets_df.iloc[idx]
                recommendations.append({
                    'dataset_id': dataset['dataset_id'],
                    'title': dataset['title'],
                    'description': self._truncate_description(dataset['description']),
                    'category': dataset.get('category', 'unknown'),
                    'source': dataset.get('source', 'unknown'),
                    'quality_score': dataset.get('quality_score', 0.5),
                    'similarity_score': float(similarities[idx]),
                    'method': 'Semantic',
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Semantic recommendation failed: {e}")
            return []
    
    def _recommend_hybrid(self, query: str, top_k: int) -> List[Dict]:
        """Hybrid recommendations combining TF-IDF and semantic"""
        try:
            # Get both types of recommendations
            tfidf_recs = self._recommend_tfidf(query, top_k * 2)
            semantic_recs = self._recommend_semantic(query, top_k * 2)
            
            # Get hybrid alpha from config
            alpha = self.hybrid_config.get('alpha', 0.6) if self.hybrid_config else 0.6
            
            # Combine scores
            combined_scores = {}
            
            # Process TF-IDF recommendations
            for rec in tfidf_recs:
                dataset_id = rec['dataset_id']
                combined_scores[dataset_id] = {
                    'data': rec,
                    'tfidf_score': rec['similarity_score'],
                    'semantic_score': 0.0
                }
            
            # Process semantic recommendations
            for rec in semantic_recs:
                dataset_id = rec['dataset_id']
                if dataset_id in combined_scores:
                    combined_scores[dataset_id]['semantic_score'] = rec['similarity_score']
                else:
                    combined_scores[dataset_id] = {
                        'data': rec,
                        'tfidf_score': 0.0,
                        'semantic_score': rec['similarity_score']
                    }
            
            # Calculate hybrid scores
            hybrid_recommendations = []
            for dataset_id, scores in combined_scores.items():
                hybrid_score = alpha * scores['tfidf_score'] + (1 - alpha) * scores['semantic_score']
                
                if hybrid_score > self.min_threshold:
                    rec = scores['data'].copy()
                    rec['similarity_score'] = hybrid_score
                    rec['method'] = 'Hybrid'
                    rec['tfidf_component'] = scores['tfidf_score']
                    rec['semantic_component'] = scores['semantic_score']
                    rec['hybrid_alpha'] = alpha
                    hybrid_recommendations.append(rec)
            
            # Sort and return top k
            hybrid_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Update ranks
            for i, rec in enumerate(hybrid_recommendations[:top_k]):
                rec['rank'] = i + 1
            
            return hybrid_recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Hybrid recommendation failed: {e}")
            return []
    
    def _add_explanations(self, recommendations: List[Dict], query: str, method: str) -> List[Dict]:
        """Add explanations to recommendations"""
        try:
            for rec in recommendations:
                explanation_parts = []
                
                # Method-specific explanations
                if method == 'tfidf':
                    explanation_parts.append(f"Matched based on keyword similarity (score: {rec['similarity_score']:.3f})")
                elif method == 'semantic':
                    explanation_parts.append(f"Matched based on semantic meaning (score: {rec['similarity_score']:.3f})")
                elif method == 'hybrid':
                    tfidf_comp = rec.get('tfidf_component', 0)
                    semantic_comp = rec.get('semantic_component', 0)
                    alpha = rec.get('hybrid_alpha', 0.6)
                    explanation_parts.append(
                        f"Combined keyword ({tfidf_comp:.3f}) and semantic ({semantic_comp:.3f}) "
                        f"similarity with α={alpha:.1f}"
                    )
                
                # Quality-based explanation
                quality = rec.get('quality_score', 0.5)
                if quality >= 0.8:
                    explanation_parts.append("High-quality dataset")
                elif quality >= 0.6:
                    explanation_parts.append("Good-quality dataset")
                
                # Source credibility
                source = rec.get('source', '').lower()
                if any(term in source for term in ['gov.sg', 'government', 'ministry']):
                    explanation_parts.append("Official government source")
                elif any(term in source for term in ['world bank', 'un', 'who', 'oecd']):
                    explanation_parts.append("International organization source")
                
                rec['explanation'] = ". ".join(explanation_parts) + "."
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to add explanations: {e}")
            return recommendations
    
    def _truncate_description(self, description: str, max_length: int = 200) -> str:
        """Truncate description to specified length"""
        if pd.isna(description):
            return "No description available"
        
        desc_str = str(description)
        if len(desc_str) > max_length:
            return desc_str[:max_length] + "..."
        return desc_str
    
    def _format_response(
        self, 
        recommendations: List[Dict], 
        query: str, 
        method: str, 
        start_time: float,
        from_cache: bool = False
    ) -> Dict:
        """Format comprehensive response"""
        response_time = time.time() - start_time
        
        return {
            'query': query,
            'method': method,
            'recommendations': recommendations,
            'metadata': {
                'total_found': len(recommendations),
                'response_time_ms': round(response_time * 1000, 2),
                'from_cache': from_cache,
                'timestamp': time.time(),
                'model_version': self.config.get('version', '1.0')
            },
            'performance': {
                'cache_hit_rate': self.cache_hits / max(self.query_count, 1),
                'average_response_time_ms': round((self.total_response_time / max(self.query_count, 1)) * 1000, 2),
                'total_queries': self.query_count
            }
        }
    
    def _format_error_response(self, error_message: str, start_time: float) -> Dict:
        """Format error response"""
        return {
            'error': error_message,
            'recommendations': [],
            'metadata': {
                'total_found': 0,
                'response_time_ms': round((time.time() - start_time) * 1000, 2),
                'from_cache': False,
                'timestamp': time.time()
            }
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'status': 'operational',
            'models_loaded': {
                'tfidf_vectorizer': self.tfidf_vectorizer is not None,
                'tfidf_matrix': self.tfidf_matrix is not None,
                'semantic_model': self.semantic_model is not None,
                'semantic_embeddings': self.semantic_embeddings is not None,
                'datasets_metadata': self.datasets_df is not None
            },
            'dataset_info': {
                'total_datasets': len(self.datasets_df) if self.datasets_df is not None else 0,
                'categories': self.datasets_df['category'].nunique() if self.datasets_df is not None and 'category' in self.datasets_df.columns else 0,
                'sources': self.datasets_df['source'].nunique() if self.datasets_df is not None and 'source' in self.datasets_df.columns else 0
            },
            'performance_stats': {
                'total_queries': self.query_count,
                'cache_hit_rate': self.cache_hits / max(self.query_count, 1),
                'average_response_time_ms': round((self.total_response_time / max(self.query_count, 1)) * 1000, 2) if self.query_count > 0 else 0,
                'cache_size': len(self._recommendation_cache)
            },
            'configuration': {
                'default_top_k': self.default_top_k,
                'max_top_k': self.max_top_k,
                'cache_enabled': self.cache_enabled,
                'cache_duration_seconds': self.cache_duration,
                'response_timeout_seconds': self.response_timeout
            }
        }
    
    def clear_cache(self):
        """Clear recommendation cache"""
        self._recommendation_cache.clear()
        self._cache_timestamps.clear()
        logger.info("🗑️ Recommendation cache cleared")
    
    def batch_recommend(
        self, 
        queries: List[str], 
        method: str = 'hybrid',
        top_k: int = 5
    ) -> List[Dict]:
        """Process multiple queries in batch"""
        results = []
        
        for query in queries:
            try:
                result = self.recommend_datasets(query, method, top_k)
                results.append(result)
            except Exception as e:
                logger.warning(f"⚠️ Batch query failed for '{query}': {e}")
                results.append(self._format_error_response(str(e), time.time()))
        
        return results


def create_production_engine(config: Dict, models_dir: str = "models") -> ProductionRecommendationEngine:
    """Factory function to create production recommendation engine"""
    return ProductionRecommendationEngine(config, models_dir)


class RecommendationAPI:
    """Simple API wrapper for easy integration"""
    
    def __init__(self, config: Dict, models_dir: str = "models"):
        self.engine = create_production_engine(config, models_dir)
    
    def search(self, query: str, method: str = 'hybrid', limit: int = 5) -> Dict:
        """Simple search interface"""
        return self.engine.recommend_datasets(
            query=query,
            method=method, 
            top_k=limit,
            include_explanations=True
        )
    
    def status(self) -> Dict:
        """Get system status"""
        return self.engine.get_system_status()
    
    def clear_cache(self):
        """Clear cache"""
        self.engine.clear_cache()