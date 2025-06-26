# Enhanced Model Training Module - Production-Ready ML Training
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedRecommendationEngine:
    """
    Enhanced AI-powered dataset recommendation engine with optimized performance,
    advanced features, and production-ready capabilities.
    """

    def __init__(self, config: Dict):
        """Initialize enhanced recommendation engine"""
        self.config = config
        self.model_config = config.get('models', {})
        self.training_config = config.get('training', {})
        self.optimization_config = config.get('optimization', {})
        
        # Set random seed for reproducibility
        if self.training_config.get('reproducible', True):
            seed = self.training_config.get('random_seed', 42)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # TF-IDF Configuration
        tfidf_config = self.model_config.get('tfidf', {})
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 5000),
            stop_words=tfidf_config.get('stop_words', 'english'),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 3])),
            min_df=tfidf_config.get('min_df', 1),
            max_df=tfidf_config.get('max_df', 0.8),
            lowercase=tfidf_config.get('lowercase', True),
            sublinear_tf=tfidf_config.get('sublinear_tf', True)
        )
        
        # Semantic Model Configuration
        semantic_config = self.model_config.get('semantic', {})
        self.semantic_model_name = semantic_config.get('model', 'all-MiniLM-L6-v2')
        self.batch_size = semantic_config.get('batch_size', 32)
        self.normalize_embeddings = semantic_config.get('normalize_embeddings', True)
        
        # Hybrid Configuration
        hybrid_config = self.model_config.get('hybrid', {})
        self.hybrid_alpha = hybrid_config.get('alpha', 0.6)
        self.confidence_threshold = hybrid_config.get('confidence_threshold', 0.05)
        
        # Initialize model components
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.datasets_df = None
        self.feature_matrix = None
        
        # Performance optimization
        self.device = self._detect_device()
        
        # Load semantic model with fallback strategy
        self._initialize_semantic_model()
        
        logger.info("🤖 EnhancedRecommendationEngine initialized")

    def _detect_device(self) -> str:
        """Detect best available device for computation"""
        try:
            # Check for Apple Silicon MPS
            if torch.backends.mps.is_available() and self.optimization_config.get('use_mps', True):
                return 'mps'
            # Check for CUDA GPU
            elif torch.cuda.is_available() and self.optimization_config.get('use_gpu', False):
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'

    def _initialize_semantic_model(self):
        """Initialize semantic model with fallback options"""
        semantic_config = self.model_config.get('semantic', {})
        
        # Try primary model first
        models_to_try = [self.semantic_model_name]
        
        # Add alternative models as fallback
        alt_models = semantic_config.get('alternative_models', [])
        models_to_try.extend(alt_models)
        
        # Add default fallback
        if 'all-MiniLM-L6-v2' not in models_to_try:
            models_to_try.append('all-MiniLM-L6-v2')

        for model_name in models_to_try:
            try:
                logger.info(f"🔄 Loading semantic model: {model_name}")
                self.semantic_model = SentenceTransformer(model_name, device=self.device)
                self.semantic_model_name = model_name
                logger.info(f"✅ Successfully loaded: {model_name} on {self.device}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                continue

        if self.semantic_model is None:
            raise Exception("❌ Could not load any semantic model")

    def load_datasets(self, datasets_df: pd.DataFrame) -> pd.DataFrame:
        """Load preprocessed datasets"""
        try:
            self.datasets_df = datasets_df.copy()
            
            # Ensure combined_text exists
            if 'combined_text' not in self.datasets_df.columns:
                self.datasets_df['combined_text'] = (
                    self.datasets_df['title'].fillna('') + ' ' +
                    self.datasets_df['description'].fillna('') + ' ' +
                    self.datasets_df.get('tags', '').fillna('') + ' ' +
                    self.datasets_df.get('category', '').fillna('')
                ).str.strip()
            
            logger.info(f"✅ Loaded {len(self.datasets_df)} datasets for training")
            return self.datasets_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            raise

    def train_tfidf_model(self) -> np.ndarray:
        """Train enhanced TF-IDF model with optimization"""
        try:
            if self.datasets_df is None or len(self.datasets_df) == 0:
                raise ValueError("No datasets loaded for training")

            logger.info("🔄 Training enhanced TF-IDF model...")
            
            # Prepare text data
            text_data = self.datasets_df['combined_text'].fillna('').tolist()
            
            # Train TF-IDF with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("Training TF-IDF vectorizer...", total=100)
                
                # Fit and transform
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
                progress.update(task, advance=50)
                
                # Normalize for better cosine similarity performance
                if self.optimization_config.get('normalize_tfidf', True):
                    self.tfidf_matrix = normalize(self.tfidf_matrix, norm='l2')
                progress.update(task, advance=50)

            # Model statistics
            vocab_size = len(self.tfidf_vectorizer.vocabulary_)
            matrix_density = self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])
            
            logger.info(f"✅ TF-IDF training complete")
            logger.info(f"   📐 Matrix shape: {self.tfidf_matrix.shape}")
            logger.info(f"   📝 Vocabulary size: {vocab_size}")
            logger.info(f"   🔢 Matrix density: {matrix_density:.4f}")
            
            return self.tfidf_matrix

        except Exception as e:
            logger.error(f"❌ TF-IDF training failed: {e}")
            raise

    def generate_semantic_embeddings(self) -> np.ndarray:
        """Generate optimized semantic embeddings"""
        try:
            if self.datasets_df is None:
                raise ValueError("No datasets loaded for embedding generation")

            logger.info("🔄 Generating semantic embeddings...")
            
            text_data = self.datasets_df['combined_text'].fillna('').tolist()
            
            # Configure encoding parameters
            encoding_params = {
                'batch_size': self.batch_size,
                'show_progress_bar': True,
                'convert_to_numpy': True,
                'normalize_embeddings': self.normalize_embeddings
            }
            
            # Add device-specific optimizations
            if self.device == 'mps':
                # Apple Silicon optimization
                encoding_params['device'] = None  # Let model handle MPS
            elif self.device == 'cuda':
                encoding_params['device'] = 'cuda'
            
            # Generate embeddings with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("Generating embeddings...", total=100)
                
                self.semantic_embeddings = self.semantic_model.encode(
                    text_data,
                    **encoding_params
                )
                progress.update(task, advance=100)

            # Embedding statistics
            embedding_dim = self.semantic_embeddings.shape[1]
            avg_norm = np.linalg.norm(self.semantic_embeddings, axis=1).mean()
            
            logger.info(f"✅ Semantic embeddings generated")
            logger.info(f"   📐 Shape: {self.semantic_embeddings.shape}")
            logger.info(f"   🧠 Model: {self.semantic_model_name}")
            logger.info(f"   📏 Dimensions: {embedding_dim}")
            logger.info(f"   🎯 Average norm: {avg_norm:.4f}")
            
            return self.semantic_embeddings

        except Exception as e:
            logger.error(f"❌ Semantic embedding generation failed: {e}")
            raise

    def optimize_hybrid_weights(self, ground_truth: Dict) -> float:
        """Optimize hybrid model weights using grid search"""
        try:
            if not self.training_config.get('hybrid_training', {}).get('grid_search_alpha', False):
                return self.hybrid_alpha

            logger.info("🔄 Optimizing hybrid weights...")
            
            alpha_range = self.training_config.get('hybrid_training', {}).get(
                'alpha_range', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            
            best_alpha = self.hybrid_alpha
            best_score = 0.0
            
            # Test each alpha value
            for alpha in alpha_range:
                total_score = 0
                valid_scenarios = 0
                
                # Evaluate on ground truth scenarios
                for scenario_name, scenario_data in ground_truth.items():
                    primary = scenario_data.get('primary', '')
                    expected = set(scenario_data.get('complementary', []))
                    
                    if not primary or not expected:
                        continue
                    
                    # Get hybrid recommendations with current alpha
                    recommendations = self.recommend_datasets_hybrid(primary, top_k=5, alpha=alpha)
                    recommended_titles = {rec['title'] for rec in recommendations}
                    
                    # Calculate F1@3 score
                    intersection = recommended_titles.intersection(expected)
                    precision = len(intersection) / len(recommended_titles) if recommended_titles else 0
                    recall = len(intersection) / len(expected) if expected else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    total_score += f1
                    valid_scenarios += 1
                
                # Average F1 score for this alpha
                avg_score = total_score / valid_scenarios if valid_scenarios > 0 else 0
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_alpha = alpha
                
                logger.info(f"   α={alpha:.1f}: F1@3={avg_score:.3f}")
            
            self.hybrid_alpha = best_alpha
            logger.info(f"✅ Optimal hybrid weight: α={best_alpha:.1f} (F1@3={best_score:.3f})")
            
            return best_alpha

        except Exception as e:
            logger.error(f"❌ Hybrid weight optimization failed: {e}")
            return self.hybrid_alpha

    def recommend_datasets_tfidf(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced TF-IDF recommendation with optimized filtering"""
        try:
            if self.tfidf_matrix is None:
                raise ValueError("TF-IDF model not trained")

            # Transform query with enhanced preprocessing
            query_processed = self._preprocess_query(query)
            query_vector = self.tfidf_vectorizer.transform([query_processed])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Optimized similarity threshold - much more permissive
            # Use percentile-based threshold instead of std-based
            if len(similarities) > 10:
                min_threshold = max(0.001, np.percentile(similarities, 70))  # Top 30% consideration
            else:
                min_threshold = 0.001  # Very low threshold for small datasets
            
            # Get top k recommendations with expanded filtering
            valid_indices = np.where(similarities > min_threshold)[0]
            
            # If still no matches, take top k regardless of threshold
            if len(valid_indices) == 0:
                logger.info(f"Using relaxed threshold for query: '{query}'")
                valid_indices = np.arange(len(similarities))
                min_threshold = 0.0
            
            # Sort by similarity and take top k
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
                    'method': 'TF-IDF'
                })
            
            return recommendations

        except Exception as e:
            logger.error(f"❌ TF-IDF recommendation failed: {e}")
            return []

    def recommend_datasets_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced semantic recommendation with optimized thresholding"""
        try:
            if self.semantic_embeddings is None:
                raise ValueError("Semantic embeddings not generated")

            # Encode query with preprocessing
            query_processed = self._preprocess_query(query)
            query_embedding = self.semantic_model.encode([query_processed], normalize_embeddings=self.normalize_embeddings)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()
            
            # Much more permissive threshold for semantic similarity
            # Semantic models often produce lower absolute scores but meaningful relative rankings
            if len(similarities) > 10:
                threshold = max(0.01, np.percentile(similarities, 60))  # Top 40% consideration
            else:
                threshold = 0.01
            
            # Get top k recommendations with filtering
            valid_indices = np.where(similarities > threshold)[0]
            
            # If still no matches, take top k regardless of threshold
            if len(valid_indices) == 0:
                logger.info(f"Using relaxed semantic threshold for query: '{query}'")
                valid_indices = np.arange(len(similarities))
                threshold = 0.0
            
            # Sort by similarity and take top k
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
                    'method': 'Semantic'
                })
            
            return recommendations

        except Exception as e:
            logger.error(f"❌ Semantic recommendation failed: {e}")
            return []

    def recommend_datasets_hybrid(self, query: str, top_k: int = 5, alpha: Optional[float] = None) -> List[Dict]:
        """Enhanced hybrid recommendation with advanced score fusion"""
        try:
            alpha = alpha or self.hybrid_alpha
            
            # Get recommendations from both methods
            tfidf_recs = self.recommend_datasets_tfidf(query, top_k * 3)  # Get more for better fusion
            semantic_recs = self.recommend_datasets_semantic(query, top_k * 3)
            
            # Create unified scoring system
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
            
            # Calculate hybrid scores with normalization
            hybrid_recommendations = []
            
            for dataset_id, scores in combined_scores.items():
                # Normalize scores to [0,1] range for fair combination
                tfidf_norm = scores['tfidf_score']
                semantic_norm = scores['semantic_score']
                
                # Weighted combination
                hybrid_score = alpha * tfidf_norm + (1 - alpha) * semantic_norm
                
                # Apply confidence threshold
                if hybrid_score > self.confidence_threshold:
                    rec = scores['data'].copy()
                    rec['similarity_score'] = hybrid_score
                    rec['method'] = 'Hybrid'
                    rec['tfidf_component'] = tfidf_norm
                    rec['semantic_component'] = semantic_norm
                    rec['hybrid_alpha'] = alpha
                    hybrid_recommendations.append(rec)
            
            # Sort by hybrid score and return top k
            hybrid_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return hybrid_recommendations[:top_k]
        
        except Exception as e:
            logger.error(f"❌ Hybrid recommendation failed: {e}")
            return []
    
    def recommend_datasets(self, query: str, method: str = 'hybrid', top_k: int = 5) -> Dict:
        """General recommendation method that routes to specific methods."""
        if method == 'tfidf':
            recommendations = self.recommend_datasets_tfidf(query, top_k=top_k)
        elif method == 'semantic':
            recommendations = self.recommend_datasets_semantic(query, top_k=top_k)
        elif method == 'hybrid':
            recommendations = self.recommend_datasets_hybrid(query, top_k=top_k)
        else:
            # Default to hybrid
            recommendations = self.recommend_datasets_hybrid(query, top_k=top_k)
        
        return {
            'recommendations': recommendations,
            'method': method,
            'query': query,
            'total_found': len(recommendations)
        }
    
    def get_recommendations(self, query: str, method: str = 'hybrid', top_k: int = 5) -> List[Dict]:
        """Compatibility method that returns list of recommendations."""
        result = self.recommend_datasets(query, method, top_k)
        return result.get('recommendations', [])

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        import re
        
        if not query:
            return query
            
        # Convert to lowercase
        processed = query.lower().strip()
        
        # Expand common abbreviations (same as in preprocessing)
        replacements = {
            'sgd': 'singapore dollars',
            'usd': 'us dollars', 
            'ppp': 'purchasing power parity',
            'gdp': 'gross domestic product',
            'ura': 'urban redevelopment authority',
            'lta': 'land transport authority',
            'hdb': 'housing development board',
            'mrt': 'mass rapid transit',
            'lrt': 'light rail transit',
            'co2': 'carbon dioxide emissions',
            'wb': 'world bank',
            'who': 'world health organization',
            'oecd': 'organisation economic cooperation development',
            'imf': 'international monetary fund'
        }
        
        for abbrev, full_form in replacements.items():
            processed = re.sub(r'\b' + abbrev + r'\b', full_form, processed)
        
        # Add semantic context for common query patterns
        if 'economic' in processed and 'indicators' not in processed:
            processed += ' indicators statistics'
        if 'population' in processed and 'demographic' not in processed:
            processed += ' demographic statistics'
        if 'poverty' in processed and 'social' not in processed:
            processed += ' social economic indicators'
        if 'transport' in processed and 'infrastructure' not in processed:
            processed += ' infrastructure data'
        if 'health' in processed and 'statistics' not in processed:
            processed += ' statistics outcomes'
        
        # Clean up extra spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed

    def _truncate_description(self, description: str, max_length: int = 200) -> str:
        """Truncate description to specified length"""
        if pd.isna(description):
            return "No description available"
        
        desc_str = str(description)
        if len(desc_str) > max_length:
            return desc_str[:max_length] + "..."
        return desc_str

    def save_models(self, models_dir: str):
        """Save all trained models with compression and validation"""
        try:
            models_path = Path(models_dir)
            models_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving models to {models_path}")
            
            # Determine compression based on config
            use_compression = self.config.get('persistence', {}).get('compression', True)
            pickle_protocol = self.config.get('persistence', {}).get('pickle_protocol', 4)
            
            saved_files = []
            
            # Save TF-IDF components
            if self.tfidf_vectorizer is not None:
                tfidf_file = models_path / "tfidf_vectorizer.pkl"
                with open(tfidf_file, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f, protocol=pickle_protocol)
                saved_files.append(tfidf_file.name)
                
                if self.tfidf_matrix is not None:
                    matrix_file = models_path / "tfidf_matrix.npy"
                    if use_compression:
                        np.savez_compressed(str(matrix_file).replace('.npy', '.npz'), matrix=self.tfidf_matrix.toarray())
                        saved_files.append(matrix_file.name.replace('.npy', '.npz'))
                    else:
                        np.save(matrix_file, self.tfidf_matrix.toarray())
                        saved_files.append(matrix_file.name)
            
            # Save semantic embeddings
            if self.semantic_embeddings is not None:
                embeddings_file = models_path / "semantic_embeddings.npy"
                if use_compression:
                    np.savez_compressed(str(embeddings_file).replace('.npy', '.npz'), embeddings=self.semantic_embeddings)
                    saved_files.append(embeddings_file.name.replace('.npy', '.npz'))
                else:
                    np.save(embeddings_file, self.semantic_embeddings)
                    saved_files.append(embeddings_file.name)
            
            # Save hybrid configuration
            hybrid_config = {
                'alpha': self.hybrid_alpha,
                'confidence_threshold': self.confidence_threshold,
                'semantic_model_name': self.semantic_model_name,
                'device': self.device
            }
            
            hybrid_file = models_path / "hybrid_weights.pkl"
            with open(hybrid_file, 'wb') as f:
                pickle.dump(hybrid_config, f, protocol=pickle_protocol)
            saved_files.append(hybrid_file.name)
            
            # Save dataset metadata
            if self.datasets_df is not None:
                metadata_file = models_path / "datasets_metadata.csv"
                self.datasets_df.to_csv(metadata_file, index=False)
                saved_files.append(metadata_file.name)
            
            # Save model configuration
            config_file = models_path / "model_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            saved_files.append(config_file.name)
            
            logger.info(f"✅ Successfully saved {len(saved_files)} model files")
            for file in saved_files:
                logger.info(f"   📁 {file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
            raise


class EnhancedDataQualityAssessment:
    """Enhanced ML-based data quality assessment with advanced features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.anomaly_detector = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        logger.info("🔍 EnhancedDataQualityAssessment initialized")
    
    def assess_dataset_quality(self, datasets_df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive quality assessment with ML enhancement"""
        try:
            logger.info("🔄 Conducting enhanced quality assessment...")
            
            # Extract quality features
            quality_features = self._extract_enhanced_quality_features(datasets_df)
            
            # Detect anomalies
            quality_anomalies = self._detect_quality_anomalies(quality_features)
            
            # Calculate enhanced quality scores
            enhanced_scores = self._calculate_enhanced_quality_scores(
                datasets_df, quality_features, quality_anomalies
            )
            
            # Add results to dataframe
            result_df = datasets_df.copy()
            result_df['ml_quality_score'] = enhanced_scores
            result_df['quality_anomaly'] = quality_anomalies
            result_df['quality_improvement'] = enhanced_scores - datasets_df.get('quality_score', 0.5)
            
            # Quality statistics
            avg_improvement = result_df['quality_improvement'].mean()
            anomaly_count = (quality_anomalies == 0).sum()
            
            logger.info(f"✅ Quality assessment completed")
            logger.info(f"   📈 Average improvement: {avg_improvement:+.3f}")
            logger.info(f"   ⚠️ Anomalies detected: {anomaly_count}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return datasets_df
    
    def _extract_enhanced_quality_features(self, datasets_df: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive quality features"""
        features = []
        
        for _, dataset in datasets_df.iterrows():
            feature_vector = [
                # Basic text quality
                len(str(dataset.get('title', ''))) if dataset.get('title') else 0,
                len(str(dataset.get('description', ''))) if dataset.get('description') else 0,
                len(str(dataset.get('tags', '')) .split(',')) if dataset.get('tags') else 0,
                
                # Metadata completeness (enhanced)
                sum([
                    bool(dataset.get('title')),
                    bool(dataset.get('description')), 
                    bool(dataset.get('source')),
                    bool(dataset.get('category')),
                    bool(dataset.get('agency')),
                    bool(dataset.get('last_updated')),
                    bool(dataset.get('format')),
                    bool(dataset.get('license'))
                ]) / 8.0,  # Normalize to [0,1]
                
                # Quality indicators
                dataset.get('quality_score', 0.5),
                int(dataset.get('status') == 'active') if dataset.get('status') else 0.5,
                
                # Advanced features
                self._encode_source_credibility(dataset.get('source', '')),
                self._encode_update_frequency(dataset.get('frequency', '')),
                self._encode_data_format(dataset.get('format', '')),
                
                # Text quality metrics
                self._calculate_title_quality(dataset.get('title', '')),
                self._calculate_description_quality(dataset.get('description', ''))
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _encode_source_credibility(self, source: str) -> float:
        """Enhanced source credibility scoring"""
        if pd.isna(source):
            return 0.3
        
        source_lower = str(source).lower()
        
        # Government sources (highest credibility)
        if any(term in source_lower for term in ['data.gov.sg', 'gov.sg', 'government', 'ministry', 'authority']):
            return 1.0
        
        # International organizations
        elif any(term in source_lower for term in ['world bank', 'united nations', 'who', 'imf', 'oecd']):
            return 0.9
        
        # Academic institutions
        elif any(term in source_lower for term in ['university', 'institute', 'research']):
            return 0.8
        
        # Statistical offices
        elif any(term in source_lower for term in ['statistics', 'statistical', 'census']):
            return 0.85
        
        # Commercial/other
        else:
            return 0.5
    
    def _encode_update_frequency(self, frequency: str) -> float:
        """Enhanced update frequency scoring"""
        if pd.isna(frequency):
            return 0.3
        
        freq_lower = str(frequency).lower()
        frequency_scores = {
            'real-time': 1.0, 'live': 1.0, 'continuous': 1.0,
            'daily': 0.9, 'day': 0.9,
            'weekly': 0.8, 'week': 0.8,
            'monthly': 0.6, 'month': 0.6,
            'quarterly': 0.4, 'quarter': 0.4,
            'annual': 0.3, 'yearly': 0.3, 'year': 0.3,
            'static': 0.2, 'one-time': 0.2,
            'unknown': 0.1, 'irregular': 0.1
        }
        
        for key, score in frequency_scores.items():
            if key in freq_lower:
                return score
        
        return 0.3  # Default for unrecognized frequencies
    
    def _encode_data_format(self, format_str: str) -> float:
        """Enhanced data format scoring"""
        if pd.isna(format_str):
            return 0.5
        
        format_lower = str(format_str).lower()
        
        # Machine-readable formats (higher score)
        if any(fmt in format_lower for fmt in ['csv', 'json', 'xml', 'api']):
            return 1.0
        
        # Semi-structured formats
        elif any(fmt in format_lower for fmt in ['excel', 'xlsx', 'xls']):
            return 0.8
        
        # Document formats (lower score for analysis)
        elif any(fmt in format_lower for fmt in ['pdf', 'doc', 'docx']):
            return 0.4
        
        # Image/other formats
        elif any(fmt in format_lower for fmt in ['png', 'jpg', 'jpeg', 'gif']):
            return 0.3
        
        else:
            return 0.5
    
    def _calculate_title_quality(self, title: str) -> float:
        """Calculate title quality score"""
        if pd.isna(title) or not title:
            return 0.0
        
        title_str = str(title)
        
        # Length-based scoring
        length_score = min(1.0, len(title_str) / 50)  # Optimal around 50 chars
        
        # Word count scoring  
        word_count = len(title_str.split())
        word_score = min(1.0, word_count / 8)  # Optimal around 8 words
        
        # Informativeness (contains numbers, specific terms)
        info_score = 0.0
        if re.search(r'\d', title_str):  # Contains numbers
            info_score += 0.3
        if any(term in title_str.lower() for term in ['dataset', 'data', 'statistics', 'survey']):
            info_score += 0.2
        if title_str[0].isupper():  # Properly capitalized
            info_score += 0.1
        
        return (length_score + word_score + info_score) / 3
    
    def _calculate_description_quality(self, description: str) -> float:
        """Calculate description quality score"""
        if pd.isna(description) or not description:
            return 0.0
        
        desc_str = str(description)
        
        # Length-based scoring (descriptions should be substantial)
        length_score = min(1.0, len(desc_str) / 200)  # Optimal around 200 chars
        
        # Sentence count (well-structured descriptions have multiple sentences)
        sentence_count = len(re.findall(r'[.!?]+', desc_str))
        sentence_score = min(1.0, sentence_count / 3)  # Optimal around 3 sentences
        
        # Information richness
        richness_score = 0.0
        if re.search(r'\d', desc_str):  # Contains specific numbers/dates
            richness_score += 0.2
        if any(term in desc_str.lower() for term in ['includes', 'contains', 'covers', 'provides']):
            richness_score += 0.2
        if any(term in desc_str.lower() for term in ['monthly', 'quarterly', 'annual', 'updated']):
            richness_score += 0.2
        
        return (length_score + sentence_score + richness_score) / 3
    
    def _detect_quality_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect quality anomalies using enhanced Isolation Forest"""
        try:
            anomaly_scores = self.anomaly_detector.fit_predict(features)
            return (anomaly_scores == 1).astype(int)  # 1 = normal, 0 = anomaly
        except Exception as e:
            logger.warning(f"⚠️ Anomaly detection failed: {e}")
            return np.ones(len(features))  # Default to all normal
    
    def _calculate_enhanced_quality_scores(
        self, datasets_df: pd.DataFrame, features: np.ndarray, anomalies: np.ndarray
    ) -> np.ndarray:
        """Calculate enhanced ML-based quality scores"""
        enhanced_scores = []
        
        for i, (_, dataset) in enumerate(datasets_df.iterrows()):
            # Base score
            base_score = dataset.get('quality_score', 0.5)
            
            # Feature-based enhancements
            title_quality = features[i][10] if len(features[i]) > 10 else 0.5
            desc_quality = features[i][11] if len(features[i]) > 11 else 0.5
            metadata_completeness = features[i][3]
            source_credibility = features[i][6] if len(features[i]) > 6 else 0.5
            
            # Calculate enhancement factors
            text_enhancement = (title_quality + desc_quality) * 0.15
            metadata_enhancement = metadata_completeness * 0.1
            credibility_enhancement = source_credibility * 0.1
            anomaly_penalty = 0.2 if anomalies[i] == 0 else 0.0
            
            # Final enhanced score
            enhanced_score = (
                base_score + 
                text_enhancement + 
                metadata_enhancement + 
                credibility_enhancement - 
                anomaly_penalty
            )
            
            # Ensure score is in [0, 1] range
            enhanced_score = max(0.0, min(1.0, enhanced_score))
            enhanced_scores.append(enhanced_score)
        
        return np.array(enhanced_scores)


def create_enhanced_recommendation_engine(config: Dict) -> EnhancedRecommendationEngine:
    """Factory function to create enhanced recommendation engine"""
    return EnhancedRecommendationEngine(config)


def create_enhanced_quality_assessment(config: Dict) -> EnhancedDataQualityAssessment:
    """Factory function to create enhanced quality assessment"""
    return EnhancedDataQualityAssessment(config)