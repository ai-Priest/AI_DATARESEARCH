# Model Evaluation Module - Comprehensive Supervised & Unsupervised Evaluation
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    silhouette_score, calinski_harabasz_score,
    accuracy_score, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from difflib import SequenceMatcher
import re
from sentence_transformers import SentenceTransformer

# Setup logging
logger = logging.getLogger(__name__)


class ComprehensiveModelEvaluator:
    """
    Comprehensive evaluation framework for recommendation models including
    supervised evaluation (ground truth) and unsupervised evaluation (internal metrics).
    """
    
    def __init__(self, config: Dict):
        """Initialize comprehensive evaluator"""
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.supervised_config = self.eval_config.get('supervised', {})
        self.unsupervised_config = self.eval_config.get('unsupervised', {})
        self.viz_config = config.get('visualization', {})
        
        # Evaluation parameters
        self.k_values = self.supervised_config.get('k_values', [1, 3, 5, 10])
        self.metrics = self.supervised_config.get('metrics', ['precision@k', 'recall@k', 'f1@k'])
        
        # Results storage
        self.supervised_results = {}
        self.unsupervised_results = {}
        self.cross_validation_results = {}
        
        # Semantic evaluation components
        self.semantic_model = None
        self.ground_truth_embeddings_cache = {}
        
        logger.info("📊 ComprehensiveModelEvaluator initialized")
    
    def _initialize_semantic_model(self):
        """Initialize semantic model for embedding-based evaluation"""
        if self.semantic_model is None:
            try:
                # Use the same model as specified in config
                model_name = self.config.get('models', {}).get('semantic', {}).get('model', 'all-mpnet-base-v2')
                logger.info(f"🔄 Loading semantic model for evaluation: {model_name}")
                self.semantic_model = SentenceTransformer(model_name)
                logger.info(f"✅ Semantic evaluation model loaded: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load semantic model: {e}")
                self.semantic_model = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text"""
        if self.semantic_model is None:
            self._initialize_semantic_model()
        
        if self.semantic_model is None:
            return None
        
        try:
            # Normalize text first
            normalized_text = self._normalize_title(text)
            if not normalized_text:
                return None
            
            # Get embedding
            embedding = self.semantic_model.encode([normalized_text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.warning(f"⚠️ Failed to get embedding for '{text}': {e}")
            return None
    
    def _precompute_ground_truth_embeddings(self, ground_truth: Dict):
        """Precompute embeddings for all ground truth titles"""
        logger.info("🔄 Precomputing ground truth embeddings for semantic evaluation")
        
        all_titles = set()
        for scenario_data in ground_truth.values():
            complementary = scenario_data.get('complementary', [])
            all_titles.update(complementary)
        
        embeddings_computed = 0
        for title in all_titles:
            if title not in self.ground_truth_embeddings_cache:
                embedding = self._get_embedding(title)
                if embedding is not None:
                    self.ground_truth_embeddings_cache[title] = embedding
                    embeddings_computed += 1
        
        logger.info(f"✅ Precomputed {embeddings_computed} ground truth embeddings")
    
    def _normalize_title(self, title: str) -> str:
        """Normalize dataset title for better matching"""
        if not title:
            return ""
        
        # Convert to lowercase and clean
        normalized = title.lower().strip()
        
        # Remove common punctuation and standardize spacing
        normalized = re.sub(r'[,\-_(){}[\]|]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Common abbreviation expansions
        abbreviations = {
            'lta': 'land transport authority',
            'ura': 'urban redevelopment authority',
            'moh': 'ministry of health',
            'mom': 'ministry of manpower',
            'gdp': 'gross domestic product',
            'bop': 'balance of payments',
            'sg': 'singapore',
            'stats': 'statistics'
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
        
        return normalized
    
    def _fuzzy_match(self, title1: str, title2: str, threshold: float = 0.75) -> bool:
        """Check if two titles are similar using fuzzy matching"""
        if not title1 or not title2:
            return False
            
        # Normalize both titles
        norm1 = self._normalize_title(title1)
        norm2 = self._normalize_title(title2)
        
        # Check exact match first
        if norm1 == norm2:
            return True
        
        # Use fuzzy matching
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold
    
    def _find_matches(self, recommendations: List[str], expected: set, fuzzy_threshold: float = 0.75) -> set:
        """Find matches between recommendations and expected using semantic embedding similarity"""
        matches = set()
        
        # Get semantic similarity threshold from config
        semantic_threshold = self.supervised_config.get('semantic_fallback_threshold', 0.7)
        use_semantic_evaluation = self.supervised_config.get('use_semantic_evaluation', True)
        
        if use_semantic_evaluation and self.semantic_model is not None:
            # Use embedding-based semantic matching (primary method)
            matches = self._find_semantic_matches(recommendations, expected, semantic_threshold)
            
            # Log the matching method being used
            if len(matches) > 0:
                logger.debug(f"🎯 Semantic matching found {len(matches)} matches")
        else:
            # Fallback to fuzzy matching if semantic evaluation is disabled
            matches = self._find_fuzzy_matches(recommendations, expected, fuzzy_threshold)
            logger.debug(f"🔤 Fuzzy matching found {len(matches)} matches")
        
        return matches
    
    def _find_semantic_matches(self, recommendations: List[str], expected: set, threshold: float = 0.7) -> set:
        """Find matches using semantic embeddings (much more accurate)"""
        matches = set()
        
        for rec_title in recommendations:
            # Get embedding for recommendation
            rec_embedding = self._get_embedding(rec_title)
            if rec_embedding is None:
                continue
            
            best_similarity = 0.0
            best_match = None
            
            # Compare with all expected titles
            for exp_title in expected:
                # Get cached embedding for expected title
                exp_embedding = self.ground_truth_embeddings_cache.get(exp_title)
                if exp_embedding is None:
                    # Compute on-the-fly if not cached
                    exp_embedding = self._get_embedding(exp_title)
                    if exp_embedding is not None:
                        self.ground_truth_embeddings_cache[exp_title] = exp_embedding
                
                if exp_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity([rec_embedding], [exp_embedding])[0][0]
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = exp_title
            
            if best_match is not None:
                matches.add(best_match)
                logger.debug(f"🎯 Matched '{rec_title}' → '{best_match}' (similarity: {best_similarity:.3f})")
        
        return matches
    
    def _find_fuzzy_matches(self, recommendations: List[str], expected: set, threshold: float = 0.75) -> set:
        """Find matches using fuzzy string matching (fallback method)"""
        matches = set()
        
        for rec_title in recommendations:
            for exp_title in expected:
                if self._fuzzy_match(rec_title, exp_title, threshold):
                    matches.add(exp_title)
                    break
        
        return matches
    
    def _semantic_match(self, title1: str, title2: str, threshold: float = 0.7) -> bool:
        """Check semantic similarity between two titles using embeddings"""
        try:
            # Simple keyword-based semantic similarity as fallback
            # Normalize and extract key terms
            norm1 = set(self._normalize_title(title1).split())
            norm2 = set(self._normalize_title(title2).split())
            
            if not norm1 or not norm2:
                return False
            
            # Calculate Jaccard similarity
            intersection = len(norm1.intersection(norm2))
            union = len(norm1.union(norm2))
            
            jaccard_sim = intersection / union if union > 0 else 0.0
            return jaccard_sim >= threshold
            
        except Exception:
            return False
    
    def evaluate_all_methods(
        self, 
        recommender, 
        ground_truth: Dict,
        datasets_df: pd.DataFrame
    ) -> Dict:
        """Complete evaluation pipeline for all recommendation methods"""
        try:
            logger.info("🔄 Starting comprehensive evaluation pipeline")
            
            # 1. Supervised Evaluation
            supervised_results = self.evaluate_supervised(recommender, ground_truth)
            
            # 2. Unsupervised Evaluation  
            unsupervised_results = self.evaluate_unsupervised(recommender, datasets_df)
            
            # 3. Cross-Validation (if enabled)
            cv_results = {}
            if self.supervised_config.get('cross_validation', {}).get('enabled', False):
                cv_results = self.perform_cross_validation(recommender, ground_truth)
            
            # 4. Statistical Significance Testing
            significance_results = {}
            if self.supervised_config.get('significance_testing', {}).get('enabled', False):
                significance_results = self.test_statistical_significance(supervised_results)
            
            # 5. Performance Analysis
            performance_analysis = self.analyze_performance(
                supervised_results, unsupervised_results, datasets_df
            )
            
            # Combine all results
            comprehensive_results = {
                'supervised_evaluation': supervised_results,
                'unsupervised_evaluation': unsupervised_results,
                'cross_validation': cv_results,
                'significance_testing': significance_results,
                'performance_analysis': performance_analysis,
                'summary': self._generate_summary(supervised_results, unsupervised_results)
            }
            
            logger.info("✅ Comprehensive evaluation completed successfully")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive evaluation failed: {e}")
            raise
    
    def evaluate_supervised(self, recommender, ground_truth: Dict) -> Dict:
        """Supervised evaluation using ground truth scenarios"""
        try:
            logger.info(f"🎯 Evaluating {len(ground_truth)} ground truth scenarios")
            
            # Initialize semantic model and precompute embeddings for semantic evaluation
            self._initialize_semantic_model()
            if self.semantic_model is not None:
                self._precompute_ground_truth_embeddings(ground_truth)
            
            methods = ['tfidf', 'semantic', 'hybrid']
            results = {method: {} for method in methods}
            scenario_details = {}
            
            # Evaluate each scenario
            for scenario_name, scenario_data in ground_truth.items():
                primary = scenario_data.get('primary', '')
                expected_complementary = set(scenario_data.get('complementary', []))
                
                if not primary or not expected_complementary:
                    logger.warning(f"⚠️ Skipping invalid scenario: {scenario_name}")
                    continue
                
                scenario_results = {}
                
                # Evaluate each method
                for method in methods:
                    try:
                        # Get recommendations
                        if method == 'tfidf':
                            recommendations = recommender.recommend_datasets_tfidf(
                                primary, top_k=max(self.k_values)
                            )
                        elif method == 'semantic':
                            recommendations = recommender.recommend_datasets_semantic(
                                primary, top_k=max(self.k_values)
                            )
                        elif method == 'hybrid':
                            recommendations = recommender.recommend_datasets_hybrid(
                                primary, top_k=max(self.k_values)
                            )
                        
                        # Calculate metrics for different k values
                        method_metrics = self._calculate_ranking_metrics(
                            recommendations, expected_complementary, self.k_values
                        )
                        
                        scenario_results[method] = method_metrics
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Method {method} failed for scenario {scenario_name}: {e}")
                        scenario_results[method] = {f"{metric}@{k}": 0.0 
                                                 for metric in ['precision', 'recall', 'f1'] 
                                                 for k in self.k_values}
                
                scenario_details[scenario_name] = {
                    'query': primary,
                    'expected_count': len(expected_complementary),
                    'confidence': scenario_data.get('confidence', 0.0),
                    'results': scenario_results
                }
            
            # Calculate average metrics across all scenarios
            average_metrics = self._calculate_average_metrics(scenario_details)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_supervised_metrics(scenario_details)
            
            supervised_results = {
                'scenario_details': scenario_details,
                'average_metrics': average_metrics,
                'additional_metrics': additional_metrics,
                'evaluation_summary': {
                    'total_scenarios': len(scenario_details),
                    'valid_scenarios': len([s for s in scenario_details.values() if s['expected_count'] > 0]),
                    'best_performing_method': self._identify_best_method(average_metrics)
                }
            }
            
            logger.info(f"✅ Supervised evaluation completed for {len(scenario_details)} scenarios")
            return supervised_results
            
        except Exception as e:
            logger.error(f"❌ Supervised evaluation failed: {e}")
            return {}
    
    def evaluate_unsupervised(self, recommender, datasets_df: pd.DataFrame) -> Dict:
        """Unsupervised evaluation using internal metrics"""
        try:
            logger.info("🔍 Conducting unsupervised evaluation")
            
            unsupervised_results = {}
            
            # 1. Similarity Distribution Analysis
            similarity_analysis = self._analyze_similarity_distributions(recommender)
            unsupervised_results['similarity_analysis'] = similarity_analysis
            
            # 2. Clustering Analysis
            clustering_analysis = self._analyze_clustering_quality(recommender, datasets_df)
            unsupervised_results['clustering_analysis'] = clustering_analysis
            
            # 3. Recommendation Diversity Analysis
            diversity_analysis = self._analyze_recommendation_diversity(recommender)
            unsupervised_results['diversity_analysis'] = diversity_analysis
            
            # 4. Coverage and Novelty Analysis
            coverage_analysis = self._analyze_coverage_and_novelty(recommender, datasets_df)
            unsupervised_results['coverage_analysis'] = coverage_analysis
            
            # 5. Recommendation Confidence Analysis
            confidence_analysis = self._analyze_recommendation_confidence(recommender)
            unsupervised_results['confidence_analysis'] = confidence_analysis
            
            # 6. Model Stability Analysis
            stability_analysis = self._analyze_model_stability(recommender)
            unsupervised_results['stability_analysis'] = stability_analysis
            
            logger.info("✅ Unsupervised evaluation completed")
            return unsupervised_results
            
        except Exception as e:
            logger.error(f"❌ Unsupervised evaluation failed: {e}")
            return {}
    
    def _calculate_ranking_metrics(
        self, 
        recommendations: List[Dict], 
        expected: set, 
        k_values: List[int]
    ) -> Dict:
        """Calculate precision@k, recall@k, F1@k, and NDCG@k with fuzzy matching"""
        metrics = {}
        
        # Extract recommended titles
        recommended_titles = [rec['title'] for rec in recommendations]
        
        # Use configurable fuzzy matching threshold
        fuzzy_threshold = self.supervised_config.get('fuzzy_matching_threshold', 0.75)
        
        for k in k_values:
            # Get top-k recommendations
            top_k_recommendations = recommended_titles[:k]
            
            # Find matches using fuzzy matching
            matched_expected = self._find_matches(
                top_k_recommendations, expected, fuzzy_threshold
            )
            
            # Calculate metrics based on fuzzy matches
            precision = len(matched_expected) / len(top_k_recommendations) if top_k_recommendations else 0.0
            recall = len(matched_expected) / len(expected) if expected else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate NDCG@k with fuzzy matching
            ndcg = self._calculate_ndcg_at_k_fuzzy(recommended_titles, expected, k, fuzzy_threshold)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
            metrics[f'ndcg@{k}'] = ndcg
        
        # Calculate Mean Reciprocal Rank (MRR) with fuzzy matching
        mrr = self._calculate_mrr_fuzzy(recommended_titles, expected, fuzzy_threshold)
        metrics['mrr'] = mrr
        
        return metrics
    
    def _calculate_ndcg_at_k(self, recommended_titles: List[str], expected: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        if not expected or k == 0:
            return 0.0
        
        # Calculate DCG@k
        dcg = 0.0
        for i, title in enumerate(recommended_titles[:k]):
            if title in expected:
                # Relevance = 1 if relevant, 0 otherwise
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@k (Ideal DCG)
        ideal_dcg = 0.0
        for i in range(min(k, len(expected))):
            ideal_dcg += 1.0 / np.log2(i + 2)
        
        # Calculate NDCG@k
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        return ndcg
    
    def _calculate_ndcg_at_k_fuzzy(self, recommended_titles: List[str], expected: set, k: int, fuzzy_threshold: float = 0.75) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k with fuzzy matching"""
        if not expected or k == 0:
            return 0.0
        
        # Calculate DCG@k with fuzzy matching
        dcg = 0.0
        for i, title in enumerate(recommended_titles[:k]):
            # Check if this title matches any expected title using fuzzy matching
            for exp_title in expected:
                if self._fuzzy_match(title, exp_title, fuzzy_threshold):
                    relevance = 1.0
                    dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
                    break  # Each recommendation can only contribute once
        
        # Calculate IDCG@k (Ideal DCG) - same as before
        ideal_dcg = 0.0
        for i in range(min(k, len(expected))):
            ideal_dcg += 1.0 / np.log2(i + 2)
        
        # Calculate NDCG@k
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        return ndcg
    
    def _calculate_mrr_fuzzy(self, recommended_titles: List[str], expected: set, fuzzy_threshold: float = 0.75) -> float:
        """Calculate Mean Reciprocal Rank with fuzzy matching"""
        for i, title in enumerate(recommended_titles):
            for exp_title in expected:
                if self._fuzzy_match(title, exp_title, fuzzy_threshold):
                    return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_mrr(self, recommended_titles: List[str], expected: set) -> float:
        """Calculate Mean Reciprocal Rank (legacy exact matching)"""
        for i, title in enumerate(recommended_titles):
            if title in expected:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_average_metrics(self, scenario_details: Dict) -> Dict:
        """Calculate average metrics across all scenarios"""
        methods = ['tfidf', 'semantic', 'hybrid']
        average_metrics = {method: {} for method in methods}
        
        for method in methods:
            # Collect all metric values across scenarios
            metric_collections = {}
            
            for scenario_data in scenario_details.values():
                method_results = scenario_data['results'].get(method, {})
                
                for metric_name, value in method_results.items():
                    if metric_name not in metric_collections:
                        metric_collections[metric_name] = []
                    metric_collections[metric_name].append(value)
            
            # Calculate averages
            for metric_name, values in metric_collections.items():
                if values:
                    average_metrics[method][metric_name] = np.mean(values)
                    average_metrics[method][f'{metric_name}_std'] = np.std(values)
                    average_metrics[method][f'{metric_name}_min'] = np.min(values)
                    average_metrics[method][f'{metric_name}_max'] = np.max(values)
        
        return average_metrics
    
    def _calculate_additional_supervised_metrics(self, scenario_details: Dict) -> Dict:
        """Calculate additional supervised metrics"""
        additional_metrics = {}
        
        # Success rate (percentage of scenarios with F1@3 > 0.5)
        success_rates = {}
        for method in ['tfidf', 'semantic', 'hybrid']:
            successful_scenarios = 0
            total_scenarios = 0
            
            for scenario_data in scenario_details.values():
                f1_3 = scenario_data['results'].get(method, {}).get('f1@3', 0.0)
                if f1_3 > 0.5:
                    successful_scenarios += 1
                total_scenarios += 1
            
            success_rates[method] = successful_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        
        additional_metrics['success_rates'] = success_rates
        
        # Performance by scenario confidence
        confidence_performance = {}
        high_confidence_scenarios = [
            s for s in scenario_details.values() if s.get('confidence', 0) >= 0.8
        ]
        
        if high_confidence_scenarios:
            for method in ['tfidf', 'semantic', 'hybrid']:
                f1_scores = [
                    s['results'].get(method, {}).get('f1@3', 0.0) 
                    for s in high_confidence_scenarios
                ]
                confidence_performance[method] = np.mean(f1_scores) if f1_scores else 0.0
        
        additional_metrics['high_confidence_performance'] = confidence_performance
        
        return additional_metrics
    
    def _identify_best_method(self, average_metrics: Dict) -> str:
        """Identify the best performing method based on F1@3"""
        best_method = 'hybrid'  # Default
        best_f1_3 = 0.0
        
        for method, metrics in average_metrics.items():
            f1_3 = metrics.get('f1@3', 0.0)
            if f1_3 > best_f1_3:
                best_f1_3 = f1_3
                best_method = method
        
        return best_method
    
    def _analyze_similarity_distributions(self, recommender) -> Dict:
        """Analyze similarity score distributions"""
        try:
            analysis = {}
            
            # Sample queries for analysis
            test_queries = [
                "singapore housing market analysis",
                "transport traffic data",
                "health statistics singapore",
                "economic development indicators",
                "sustainable development goals"
            ]
            
            for method in ['tfidf', 'semantic', 'hybrid']:
                similarities = []
                
                for query in test_queries:
                    try:
                        if method == 'tfidf':
                            recs = recommender.recommend_datasets_tfidf(query, top_k=20)
                        elif method == 'semantic':
                            recs = recommender.recommend_datasets_semantic(query, top_k=20)
                        elif method == 'hybrid':
                            recs = recommender.recommend_datasets_hybrid(query, top_k=20)
                        
                        similarities.extend([rec['similarity_score'] for rec in recs])
                    except:
                        continue
                
                if similarities:
                    analysis[method] = {
                        'mean': np.mean(similarities),
                        'std': np.std(similarities),
                        'min': np.min(similarities),
                        'max': np.max(similarities),
                        'percentiles': {
                            '25th': np.percentile(similarities, 25),
                            '50th': np.percentile(similarities, 50),
                            '75th': np.percentile(similarities, 75),
                            '90th': np.percentile(similarities, 90)
                        }
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Similarity distribution analysis failed: {e}")
            return {}
    
    def _analyze_clustering_quality(self, recommender, datasets_df: pd.DataFrame) -> Dict:
        """Analyze clustering quality using embeddings"""
        try:
            analysis = {}
            
            # Use semantic embeddings for clustering analysis
            if hasattr(recommender, 'semantic_embeddings') and recommender.semantic_embeddings is not None:
                embeddings = recommender.semantic_embeddings
                
                # Determine optimal number of clusters (3-10)
                silhouette_scores = []
                ch_scores = []
                k_range = range(3, min(11, len(datasets_df) // 2))
                
                for k in k_range:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(embeddings)
                        
                        silhouette_avg = silhouette_score(embeddings, cluster_labels)
                        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
                        
                        silhouette_scores.append(silhouette_avg)
                        ch_scores.append(ch_score)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Clustering failed for k={k}: {e}")
                        silhouette_scores.append(0.0)
                        ch_scores.append(0.0)
                
                analysis = {
                    'k_range': list(k_range),
                    'silhouette_scores': silhouette_scores,
                    'calinski_harabasz_scores': ch_scores,
                    'optimal_k': k_range[np.argmax(silhouette_scores)] if silhouette_scores else 5,
                    'best_silhouette': max(silhouette_scores) if silhouette_scores else 0.0,
                    'best_ch_score': max(ch_scores) if ch_scores else 0.0
                }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Clustering analysis failed: {e}")
            return {}
    
    def _analyze_recommendation_diversity(self, recommender) -> Dict:
        """Analyze diversity of recommendations"""
        try:
            analysis = {}
            
            # Test queries
            test_queries = [
                "housing analysis singapore",
                "economic development indicators", 
                "transport planning data",
                "health statistics research",
                "environmental sustainability"
            ]
            
            for method in ['tfidf', 'semantic', 'hybrid']:
                diversity_scores = []
                
                for query in test_queries:
                    try:
                        if method == 'tfidf':
                            recs = recommender.recommend_datasets_tfidf(query, top_k=10)
                        elif method == 'semantic':
                            recs = recommender.recommend_datasets_semantic(query, top_k=10)
                        elif method == 'hybrid':
                            recs = recommender.recommend_datasets_hybrid(query, top_k=10)
                        
                        # Calculate intra-list diversity (diversity within recommendations)
                        categories = [rec.get('category', 'unknown') for rec in recs]
                        sources = [rec.get('source', 'unknown') for rec in recs]
                        
                        category_diversity = len(set(categories)) / len(categories) if categories else 0
                        source_diversity = len(set(sources)) / len(sources) if sources else 0
                        
                        avg_diversity = (category_diversity + source_diversity) / 2
                        diversity_scores.append(avg_diversity)
                        
                    except:
                        continue
                
                if diversity_scores:
                    analysis[method] = {
                        'average_diversity': np.mean(diversity_scores),
                        'diversity_std': np.std(diversity_scores),
                        'min_diversity': np.min(diversity_scores),
                        'max_diversity': np.max(diversity_scores)
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Diversity analysis failed: {e}")
            return {}
    
    def _analyze_coverage_and_novelty(self, recommender, datasets_df: pd.DataFrame) -> Dict:
        """Analyze coverage and novelty of recommendations"""
        try:
            analysis = {}
            
            # Coverage analysis: what percentage of datasets are ever recommended
            all_dataset_ids = set(datasets_df['dataset_id'].astype(str))
            recommended_dataset_ids = set()
            
            test_queries = [
                "singapore data analysis", "economic indicators", "transport data",
                "health statistics", "education research", "environmental data",
                "population demographics", "urban planning", "technology adoption"
            ]
            
            for query in test_queries:
                for method in ['tfidf', 'semantic', 'hybrid']:
                    try:
                        if method == 'tfidf':
                            recs = recommender.recommend_datasets_tfidf(query, top_k=10)
                        elif method == 'semantic':
                            recs = recommender.recommend_datasets_semantic(query, top_k=10)
                        elif method == 'hybrid':
                            recs = recommender.recommend_datasets_hybrid(query, top_k=10)
                        
                        for rec in recs:
                            recommended_dataset_ids.add(str(rec['dataset_id']))
                            
                    except:
                        continue
            
            coverage = len(recommended_dataset_ids) / len(all_dataset_ids) if all_dataset_ids else 0.0
            
            # Novelty analysis: how often do we recommend less popular datasets
            if 'quality_score' in datasets_df.columns:
                high_quality_datasets = set(
                    datasets_df[datasets_df['quality_score'] >= 0.8]['dataset_id'].astype(str)
                )
                recommended_high_quality = recommended_dataset_ids.intersection(high_quality_datasets)
                novelty_score = len(recommended_high_quality) / len(recommended_dataset_ids) if recommended_dataset_ids else 0.0
            else:
                novelty_score = 0.5  # Default
            
            analysis = {
                'coverage': coverage,
                'total_datasets': len(all_dataset_ids),
                'recommended_datasets': len(recommended_dataset_ids),
                'novelty_score': novelty_score
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Coverage analysis failed: {e}")
            return {}
    
    def _analyze_recommendation_confidence(self, recommender) -> Dict:
        """Analyze confidence scores of recommendations"""
        try:
            analysis = {}
            
            test_queries = [
                "singapore housing data", "economic development",
                "transport infrastructure", "health outcomes"
            ]
            
            for method in ['tfidf', 'semantic', 'hybrid']:
                confidence_scores = []
                
                for query in test_queries:
                    try:
                        if method == 'tfidf':
                            recs = recommender.recommend_datasets_tfidf(query, top_k=5)
                        elif method == 'semantic':
                            recs = recommender.recommend_datasets_semantic(query, top_k=5)
                        elif method == 'hybrid':
                            recs = recommender.recommend_datasets_hybrid(query, top_k=5)
                        
                        scores = [rec['similarity_score'] for rec in recs]
                        if scores:
                            # Confidence based on score range and distribution
                            score_range = max(scores) - min(scores) if len(scores) > 1 else 0
                            avg_score = np.mean(scores)
                            
                            # Higher average score and good score range indicate confidence
                            confidence = avg_score * (1 + score_range)
                            confidence_scores.append(confidence)
                            
                    except:
                        continue
                
                if confidence_scores:
                    analysis[method] = {
                        'average_confidence': np.mean(confidence_scores),
                        'confidence_std': np.std(confidence_scores),
                        'min_confidence': np.min(confidence_scores),
                        'max_confidence': np.max(confidence_scores)
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Confidence analysis failed: {e}")
            return {}
    
    def _analyze_model_stability(self, recommender) -> Dict:
        """Analyze stability of recommendations across similar queries"""
        try:
            analysis = {}
            
            # Test with slight query variations
            query_variations = [
                ["singapore housing", "singapore housing data", "housing singapore"],
                ["economic indicators", "economic data", "economic statistics"],
                ["transport data", "transportation data", "transport statistics"]
            ]
            
            for method in ['tfidf', 'semantic', 'hybrid']:
                stability_scores = []
                
                for variations in query_variations:
                    all_recommendations = []
                    
                    for query in variations:
                        try:
                            if method == 'tfidf':
                                recs = recommender.recommend_datasets_tfidf(query, top_k=5)
                            elif method == 'semantic':
                                recs = recommender.recommend_datasets_semantic(query, top_k=5)
                            elif method == 'hybrid':
                                recs = recommender.recommend_datasets_hybrid(query, top_k=5)
                            
                            rec_ids = {rec['dataset_id'] for rec in recs}
                            all_recommendations.append(rec_ids)
                            
                        except:
                            all_recommendations.append(set())
                    
                    # Calculate stability as intersection of recommendations
                    if len(all_recommendations) >= 2:
                        intersection = set.intersection(*all_recommendations) if all_recommendations else set()
                        union = set.union(*all_recommendations) if all_recommendations else set()
                        
                        stability = len(intersection) / len(union) if union else 0.0
                        stability_scores.append(stability)
                
                if stability_scores:
                    analysis[method] = {
                        'average_stability': np.mean(stability_scores),
                        'stability_std': np.std(stability_scores),
                        'min_stability': np.min(stability_scores),
                        'max_stability': np.max(stability_scores)
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Stability analysis failed: {e}")
            return {}
    
    def perform_cross_validation(self, recommender, ground_truth: Dict) -> Dict:
        """Perform cross-validation evaluation"""
        try:
            logger.info("🔄 Performing cross-validation")
            
            cv_config = self.supervised_config.get('cross_validation', {})
            n_folds = cv_config.get('folds', 5)
            
            # Convert ground truth to list for easier splitting
            scenarios = list(ground_truth.items())
            np.random.shuffle(scenarios)
            
            fold_size = len(scenarios) // n_folds
            cv_results = {'tfidf': [], 'semantic': [], 'hybrid': []}
            
            for fold in range(n_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else len(scenarios)
                
                # Test set for this fold
                test_scenarios = dict(scenarios[start_idx:end_idx])
                
                # Evaluate on test set
                fold_results = self.evaluate_supervised(recommender, test_scenarios)
                avg_metrics = fold_results.get('average_metrics', {})
                
                # Store F1@3 scores for each method
                for method in ['tfidf', 'semantic', 'hybrid']:
                    f1_3 = avg_metrics.get(method, {}).get('f1@3', 0.0)
                    cv_results[method].append(f1_3)
            
            # Calculate cross-validation statistics
            cv_summary = {}
            for method in ['tfidf', 'semantic', 'hybrid']:
                scores = cv_results[method]
                if scores:
                    cv_summary[method] = {
                        'mean_f1@3': np.mean(scores),
                        'std_f1@3': np.std(scores),
                        'min_f1@3': np.min(scores),
                        'max_f1@3': np.max(scores),
                        'scores': scores
                    }
            
            logger.info("✅ Cross-validation completed")
            return cv_summary
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return {}
    
    def test_statistical_significance(self, supervised_results: Dict) -> Dict:
        """Test statistical significance of performance differences"""
        try:
            logger.info("📈 Testing statistical significance")
            
            avg_metrics = supervised_results.get('average_metrics', {})
            scenario_details = supervised_results.get('scenario_details', {})
            
            # Extract F1@3 scores for each method across scenarios
            method_scores = {'tfidf': [], 'semantic': [], 'hybrid': []}
            
            for scenario_data in scenario_details.values():
                for method in ['tfidf', 'semantic', 'hybrid']:
                    f1_3 = scenario_data['results'].get(method, {}).get('f1@3', 0.0)
                    method_scores[method].append(f1_3)
            
            # Perform pairwise t-tests
            significance_results = {}
            method_pairs = [('hybrid', 'tfidf'), ('hybrid', 'semantic'), ('tfidf', 'semantic')]
            
            for method1, method2 in method_pairs:
                scores1 = method_scores[method1]
                scores2 = method_scores[method2]
                
                if len(scores1) > 1 and len(scores2) > 1:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    significance_results[f'{method1}_vs_{method2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                            (np.var(scores1) + np.var(scores2)) / 2
                        ) if np.var(scores1) > 0 or np.var(scores2) > 0 else 0.0
                    }
            
            return significance_results
            
        except Exception as e:
            logger.warning(f"⚠️ Statistical significance testing failed: {e}")
            return {}
    
    def analyze_performance(
        self, 
        supervised_results: Dict, 
        unsupervised_results: Dict,
        datasets_df: pd.DataFrame
    ) -> Dict:
        """Comprehensive performance analysis"""
        try:
            analysis = {}
            
            # Overall performance assessment
            avg_metrics = supervised_results.get('average_metrics', {})
            best_method = self._identify_best_method(avg_metrics)
            
            analysis['overall_assessment'] = {
                'best_method': best_method,
                'target_achievement': self._assess_target_achievement(avg_metrics),
                'strengths': self._identify_strengths(supervised_results, unsupervised_results),
                'areas_for_improvement': self._identify_improvements(supervised_results, unsupervised_results)
            }
            
            # Performance by query type (if available)
            analysis['query_type_performance'] = self._analyze_query_type_performance(supervised_results)
            
            # Data quality impact
            analysis['data_quality_impact'] = self._analyze_data_quality_impact(datasets_df, supervised_results)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Performance analysis failed: {e}")
            return {}
    
    def _assess_target_achievement(self, avg_metrics: Dict) -> Dict:
        """Assess whether performance targets are met"""
        target_config = self.eval_config.get('benchmarking', {})
        target_f1 = target_config.get('target_f1_score', 0.90)
        min_acceptable = target_config.get('minimum_acceptable', 0.70)
        
        assessment = {}
        
        for method in ['tfidf', 'semantic', 'hybrid']:
            f1_3 = avg_metrics.get(method, {}).get('f1@3', 0.0)
            
            if f1_3 >= target_f1:
                status = 'TARGET_ACHIEVED'
            elif f1_3 >= min_acceptable:
                status = 'ACCEPTABLE'
            else:
                status = 'BELOW_EXPECTATIONS'
            
            assessment[method] = {
                'f1@3_score': f1_3,
                'target_score': target_f1,
                'status': status,
                'gap_to_target': target_f1 - f1_3
            }
        
        return assessment
    
    def _identify_strengths(self, supervised_results: Dict, unsupervised_results: Dict) -> List[str]:
        """Identify model strengths"""
        strengths = []
        
        avg_metrics = supervised_results.get('average_metrics', {})
        
        # Check for high performance
        for method in ['tfidf', 'semantic', 'hybrid']:
            f1_3 = avg_metrics.get(method, {}).get('f1@3', 0.0)
            if f1_3 >= 0.7:
                strengths.append(f"High {method.upper()} performance (F1@3: {f1_3:.3f})")
        
        # Check diversity
        diversity_analysis = unsupervised_results.get('diversity_analysis', {})
        for method, metrics in diversity_analysis.items():
            if metrics.get('average_diversity', 0) >= 0.7:
                strengths.append(f"Good recommendation diversity for {method.upper()}")
        
        # Check coverage
        coverage_analysis = unsupervised_results.get('coverage_analysis', {})
        if coverage_analysis.get('coverage', 0) >= 0.5:
            strengths.append("Good dataset coverage in recommendations")
        
        return strengths
    
    def _identify_improvements(self, supervised_results: Dict, unsupervised_results: Dict) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        avg_metrics = supervised_results.get('average_metrics', {})
        
        # Check for low performance
        for method in ['tfidf', 'semantic', 'hybrid']:
            f1_3 = avg_metrics.get(method, {}).get('f1@3', 0.0)
            if f1_3 < 0.5:
                improvements.append(f"Improve {method.upper()} model performance (F1@3: {f1_3:.3f})")
        
        # Check stability
        stability_analysis = unsupervised_results.get('stability_analysis', {})
        for method, metrics in stability_analysis.items():
            if metrics.get('average_stability', 0) < 0.5:
                improvements.append(f"Improve {method.upper()} recommendation stability")
        
        return improvements
    
    def _analyze_query_type_performance(self, supervised_results: Dict) -> Dict:
        """Analyze performance by query type"""
        # This would require more detailed scenario classification
        # For now, return basic analysis
        return {"note": "Query type analysis requires scenario categorization"}
    
    def _analyze_data_quality_impact(self, datasets_df: pd.DataFrame, supervised_results: Dict) -> Dict:
        """Analyze impact of data quality on performance"""
        try:
            if 'quality_score' not in datasets_df.columns:
                return {"note": "Quality score not available for analysis"}
            
            avg_quality = datasets_df['quality_score'].mean()
            high_quality_count = (datasets_df['quality_score'] >= 0.8).sum()
            
            return {
                'average_dataset_quality': avg_quality,
                'high_quality_datasets': high_quality_count,
                'total_datasets': len(datasets_df),
                'quality_impact': "High data quality likely contributes to good performance" if avg_quality >= 0.7 else "Data quality may be limiting performance"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_summary(self, supervised_results: Dict, unsupervised_results: Dict) -> Dict:
        """Generate comprehensive evaluation summary"""
        try:
            summary = {}
            
            # Supervised performance summary
            avg_metrics = supervised_results.get('average_metrics', {})
            summary['supervised_performance'] = {
                method: {
                    'f1@3': metrics.get('f1@3', 0.0),
                    'precision@3': metrics.get('precision@3', 0.0),
                    'recall@3': metrics.get('recall@3', 0.0)
                }
                for method, metrics in avg_metrics.items()
            }
            
            # Best performing method
            best_method = self._identify_best_method(avg_metrics)
            best_f1 = avg_metrics.get(best_method, {}).get('f1@3', 0.0)
            
            summary['best_method'] = {
                'method': best_method,
                'f1@3_score': best_f1,
                'performance_level': 'Excellent' if best_f1 >= 0.8 else 'Good' if best_f1 >= 0.6 else 'Needs Improvement'
            }
            
            # Unsupervised insights
            diversity_analysis = unsupervised_results.get('diversity_analysis', {})
            coverage_analysis = unsupervised_results.get('coverage_analysis', {})
            
            summary['unsupervised_insights'] = {
                'diversity_score': np.mean([
                    metrics.get('average_diversity', 0) 
                    for metrics in diversity_analysis.values()
                ]) if diversity_analysis else 0.0,
                'coverage_score': coverage_analysis.get('coverage', 0.0),
                'total_scenarios_evaluated': len(supervised_results.get('scenario_details', {}))
            }
            
            return summary
            
        except Exception as e:
            logger.warning(f"⚠️ Summary generation failed: {e}")
            return {}


def create_comprehensive_evaluator(config: Dict) -> ComprehensiveModelEvaluator:
    """Factory function to create comprehensive evaluator"""
    return ComprehensiveModelEvaluator(config)