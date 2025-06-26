"""
Multi-Modal Search Engine
Phase 2.2: Advanced search capabilities combining text, metadata, relationships, and temporal patterns
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class MultiModalSearchEngine:
    """
    Advanced multi-modal search engine that combines:
    - Text semantic search
    - Metadata filtering
    - Relationship analysis
    - Temporal pattern matching
    - Quality scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_config = config.get('multimodal_search', {})
        
        # Initialize search components
        self.semantic_model = None
        self.tfidf_vectorizer = None
        self.relationship_graph = None
        self.temporal_index = {}
        
        # Search weights - prioritize relevance over quality
        self.weights = self.search_config.get('weights', {
            'semantic_similarity': 0.50,  # Increased for better relevance
            'keyword_match': 0.30,        # Increased for exact matches
            'metadata_relevance': 0.15,
            'relationship_score': 0.05,   # Reduced
            'temporal_relevance': 0.00,   # Disabled for now
            'quality_boost': 0.00         # Disabled to prevent irrelevant high-quality results
        })
        
        # Load datasets and initialize indices
        self._load_datasets()
        self._initialize_search_indices()
        
        logger.info("🔍 MultiModalSearchEngine initialized with advanced capabilities")
    
    def _load_datasets(self):
        """Load and prepare datasets for multi-modal search."""
        try:
            datasets_path = Path("data/processed")
            
            # Load datasets
            singapore_datasets = pd.read_csv(datasets_path / "singapore_datasets.csv")
            global_datasets = pd.read_csv(datasets_path / "global_datasets.csv")
            self.datasets = pd.concat([singapore_datasets, global_datasets], ignore_index=True)
            
            # Load relationships if available
            try:
                with open(datasets_path / "dataset_relationships.json", 'r') as f:
                    self.relationships = json.load(f)
            except FileNotFoundError:
                self.relationships = {}
                logger.warning("No relationship data found, creating empty relationships")
            
            logger.info(f"📊 Loaded {len(self.datasets)} datasets for multi-modal search")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def _initialize_search_indices(self):
        """Initialize all search indices and models."""
        try:
            # 1. Initialize semantic search model
            self._initialize_semantic_search()
            
            # 2. Initialize keyword search
            self._initialize_keyword_search()
            
            # 3. Build relationship graph
            self._build_relationship_graph()
            
            # 4. Create temporal index
            self._create_temporal_index()
            
            # 5. Prepare metadata index
            self._prepare_metadata_index()
            
            logger.info("✅ All search indices initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing search indices: {e}")
            # Continue with limited functionality
    
    def _initialize_semantic_search(self):
        """Initialize semantic search using sentence transformers."""
        try:
            model_name = self.search_config.get('semantic_model', 'all-MiniLM-L6-v2')
            self.semantic_model = SentenceTransformer(model_name)
            
            # Create semantic embeddings for all datasets
            dataset_texts = []
            for _, row in self.datasets.iterrows():
                text = f"{row.get('title', '')} {row.get('description', '')} {row.get('tags', '')}"
                dataset_texts.append(text.strip())
            
            self.semantic_embeddings = self.semantic_model.encode(dataset_texts)
            
            logger.info(f"🧠 Semantic search initialized with {model_name}")
            
        except Exception as e:
            logger.warning(f"Semantic search initialization failed: {e}")
            self.semantic_model = None
            self.semantic_embeddings = None
    
    def _initialize_keyword_search(self):
        """Initialize TF-IDF based keyword search."""
        try:
            # Prepare text corpus
            corpus = []
            for _, row in self.datasets.iterrows():
                text = f"{row.get('title', '')} {row.get('description', '')} {row.get('tags', '')}"
                corpus.append(text.lower())
            
            # Initialize TF-IDF vectorizer - more permissive for better keyword matching
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),  # Include 3-grams for better phrase matching
                min_df=1,            # More permissive - allow terms that appear once
                max_df=0.9           # Less restrictive on common terms
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            logger.info("📝 Keyword search (TF-IDF) initialized")
            
        except Exception as e:
            logger.warning(f"Keyword search initialization failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _build_relationship_graph(self):
        """Build graph representation of dataset relationships."""
        try:
            self.relationship_graph = nx.Graph()
            
            # Add all datasets as nodes
            for _, row in self.datasets.iterrows():
                self.relationship_graph.add_node(
                    row['dataset_id'],
                    title=row.get('title', ''),
                    category=row.get('category', ''),
                    source=row.get('source', '')
                )
            
            # Add edges from relationships
            for rel_data in self.relationships.get('relationships', []):
                dataset1 = rel_data.get('dataset1')
                dataset2 = rel_data.get('dataset2')
                strength = rel_data.get('strength', 0.5)
                rel_type = rel_data.get('type', 'related')
                
                if dataset1 and dataset2:
                    self.relationship_graph.add_edge(
                        dataset1, dataset2,
                        weight=strength,
                        type=rel_type
                    )
            
            logger.info(f"🕸️  Relationship graph built with {len(self.relationship_graph.nodes)} nodes, {len(self.relationship_graph.edges)} edges")
            
        except Exception as e:
            logger.warning(f"Relationship graph building failed: {e}")
            self.relationship_graph = nx.Graph()
    
    def _create_temporal_index(self):
        """Create temporal index for time-based searches."""
        try:
            for _, row in self.datasets.iterrows():
                dataset_id = row['dataset_id']
                
                # Extract temporal information
                temporal_info = {
                    'last_updated': self._parse_date(row.get('last_updated')),
                    'created_date': self._parse_date(row.get('created_date')),
                    'coverage_start': self._parse_date(row.get('coverage_start')),
                    'coverage_end': self._parse_date(row.get('coverage_end')),
                    'frequency': row.get('frequency', 'unknown')
                }
                
                self.temporal_index[dataset_id] = temporal_info
            
            logger.info("📅 Temporal index created")
            
        except Exception as e:
            logger.warning(f"Temporal index creation failed: {e}")
            self.temporal_index = {}
    
    def _prepare_metadata_index(self):
        """Prepare metadata index for filtering and relevance scoring."""
        try:
            # Create metadata categories
            self.metadata_categories = {
                'sources': self.datasets['source'].value_counts().to_dict(),
                'agencies': self.datasets['agency'].value_counts().to_dict(),
                'categories': self.datasets['category'].value_counts().to_dict(),
                'formats': self.datasets['format'].value_counts().to_dict(),
                'licenses': self.datasets['license'].value_counts().to_dict()
            }
            
            # Create quality distribution
            self.quality_stats = {
                'mean': self.datasets['quality_score'].mean(),
                'std': self.datasets['quality_score'].std(),
                'quartiles': self.datasets['quality_score'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            logger.info("📋 Metadata index prepared")
            
        except Exception as e:
            logger.warning(f"Metadata index preparation failed: {e}")
    
    def search(self, 
               query: str,
               filters: Optional[Dict[str, Any]] = None,
               top_k: int = 10,
               search_mode: str = 'comprehensive') -> List[Dict[str, Any]]:
        """
        Perform multi-modal search with comprehensive ranking.
        
        Args:
            query: Search query
            filters: Optional filters (source, category, date_range, etc.)
            top_k: Number of results to return
            search_mode: 'comprehensive', 'semantic', 'keyword', 'metadata'
            
        Returns:
            List of ranked search results with multi-modal scores
        """
        # Expand Singapore-specific abbreviations and terms
        singapore_expansions = {
            'hdb': 'hdb housing development board flat public housing',
            'coe': 'coe certificate of entitlement vehicle car',
            'cpf': 'cpf central provident fund retirement savings',
            'mrt': 'mrt mass rapid transit train subway transport',
            'lta': 'lta land transport authority traffic',
            'ura': 'ura urban redevelopment authority planning',
            'bto': 'bto build to order hdb flat',
            'sbf': 'sbf sale of balance flats hdb',
            'resale': 'resale hdb flat property housing'
        }
        
        # Enhance query with expansions
        query_lower = query.lower()
        enhanced_query = query
        for abbr, expansion in singapore_expansions.items():
            if abbr in query_lower:
                enhanced_query = f"{query} {expansion}"
                logger.info(f"🔍 Query enhanced: '{query}' → '{enhanced_query}'")
                break
        
        logger.info(f"🔍 Multi-modal search: '{enhanced_query[:50]}...' (mode: {search_mode})")
        
        try:
            # Apply filters first
            filtered_datasets = self._apply_filters(filters)
            
            if filtered_datasets.empty:
                return []
            
            # Calculate different similarity scores using enhanced query
            scores = self._calculate_multimodal_scores(enhanced_query, filtered_datasets, search_mode)
            
            # Combine scores and rank
            final_scores = self._combine_scores(scores)
            
            # Get top results
            top_indices = np.argsort(final_scores)[::-1][:top_k]
            
            # Format results
            results = []
            for idx in top_indices:
                dataset_row = filtered_datasets.iloc[idx]
                result = self._format_search_result(
                    dataset_row, 
                    final_scores[idx], 
                    scores, 
                    idx,
                    query  # Keep original query for display
                )
                results.append(result)
            
            logger.info(f"✅ Found {len(results)} multi-modal results")
            return results
            
        except Exception as e:
            logger.error(f"Multi-modal search error: {e}")
            return []
    
    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Apply filters to datasets."""
        if not filters:
            return self.datasets.copy()
        
        filtered = self.datasets.copy()
        
        # Source filter
        if 'source' in filters:
            sources = filters['source'] if isinstance(filters['source'], list) else [filters['source']]
            filtered = filtered[filtered['source'].isin(sources)]
        
        # Category filter
        if 'category' in filters:
            categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
            filtered = filtered[filtered['category'].isin(categories)]
        
        # Date range filter
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            # Filter by last_updated date
            filtered = filtered[
                (pd.to_datetime(filtered['last_updated'], errors='coerce') >= pd.to_datetime(start_date)) &
                (pd.to_datetime(filtered['last_updated'], errors='coerce') <= pd.to_datetime(end_date))
            ]
        
        # Quality threshold
        if 'min_quality' in filters:
            filtered = filtered[filtered['quality_score'] >= filters['min_quality']]
        
        # Format filter
        if 'format' in filters:
            formats = filters['format'] if isinstance(filters['format'], list) else [filters['format']]
            filtered = filtered[filtered['format'].isin(formats)]
        
        logger.debug(f"Applied filters: {len(self.datasets)} → {len(filtered)} datasets")
        return filtered
    
    def _calculate_multimodal_scores(self, 
                                   query: str, 
                                   datasets: pd.DataFrame,
                                   search_mode: str) -> Dict[str, np.ndarray]:
        """Calculate different types of similarity scores."""
        scores = {}
        
        if search_mode in ['comprehensive', 'semantic']:
            scores['semantic'] = self._calculate_semantic_similarity(query, datasets)
        
        if search_mode in ['comprehensive', 'keyword']:
            scores['keyword'] = self._calculate_keyword_similarity(query, datasets)
        
        if search_mode in ['comprehensive', 'metadata']:
            scores['metadata'] = self._calculate_metadata_relevance(query, datasets)
            scores['relationship'] = self._calculate_relationship_scores(query, datasets)
            scores['temporal'] = self._calculate_temporal_relevance(query, datasets)
            scores['quality'] = self._calculate_quality_boost(datasets)
        
        return scores
    
    def _calculate_semantic_similarity(self, query: str, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate semantic similarity scores."""
        if not self.semantic_model or self.semantic_embeddings is None:
            return np.zeros(len(datasets))
        
        try:
            # Get query embedding
            query_embedding = self.semantic_model.encode([query])
            
            # Get relevant embeddings for filtered datasets
            dataset_indices = datasets.index.tolist()
            relevant_embeddings = self.semantic_embeddings[dataset_indices]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, relevant_embeddings)[0]
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return np.zeros(len(datasets))
    
    def _calculate_keyword_similarity(self, query: str, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate keyword-based similarity scores."""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return np.zeros(len(datasets))
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query.lower()])
            
            # Get relevant TF-IDF vectors for filtered datasets
            dataset_indices = datasets.index.tolist()
            relevant_vectors = self.tfidf_matrix[dataset_indices]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, relevant_vectors)[0]
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Keyword similarity calculation failed: {e}")
            return np.zeros(len(datasets))
    
    def _calculate_metadata_relevance(self, query: str, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate metadata-based relevance scores."""
        scores = np.zeros(len(datasets))
        query_lower = query.lower()
        
        for i, (_, row) in enumerate(datasets.iterrows()):
            score = 0.0
            
            # Category relevance
            category = str(row.get('category', '')).lower()
            if category and category in query_lower:
                score += 0.3
            
            # Source credibility boost
            source = str(row.get('source', '')).lower()
            if 'gov.sg' in source:
                score += 0.2
            elif 'government' in source.lower():
                score += 0.15
            
            # Agency relevance
            agency = str(row.get('agency', '')).lower()
            if agency and agency in query_lower:
                score += 0.2
            
            # Format preference (structured data gets boost)
            data_format = str(row.get('format', '')).lower()
            if data_format in ['csv', 'json', 'xml', 'api']:
                score += 0.1
            
            # License openness
            license_type = str(row.get('license', '')).lower()
            if 'open' in license_type or 'public' in license_type:
                score += 0.1
            
            scores[i] = min(1.0, score)
        
        return scores
    
    def _calculate_relationship_scores(self, query: str, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate relationship-based relevance scores."""
        if not self.relationship_graph:
            return np.zeros(len(datasets))
        
        scores = np.zeros(len(datasets))
        
        try:
            # Find datasets that might be central in the relationship graph
            for i, (_, row) in enumerate(datasets.iterrows()):
                dataset_id = row['dataset_id']
                
                if dataset_id in self.relationship_graph:
                    # Calculate centrality-based score
                    degree_centrality = nx.degree_centrality(self.relationship_graph).get(dataset_id, 0)
                    betweenness_centrality = nx.betweenness_centrality(self.relationship_graph).get(dataset_id, 0)
                    
                    # Combine centrality measures
                    scores[i] = 0.6 * degree_centrality + 0.4 * betweenness_centrality
            
        except Exception as e:
            logger.warning(f"Relationship score calculation failed: {e}")
        
        return scores
    
    def _calculate_temporal_relevance(self, query: str, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate temporal relevance scores."""
        scores = np.zeros(len(datasets))
        
        # Extract temporal keywords from query
        current_year = datetime.now().year
        temporal_keywords = {
            'recent': 1.0,
            'latest': 1.0,
            'current': 1.0,
            str(current_year): 1.0,
            str(current_year-1): 0.8,
            str(current_year-2): 0.6,
            'historical': 0.3,
            'trend': 0.7
        }
        
        query_temporal_weight = 0.0
        for keyword, weight in temporal_keywords.items():
            if keyword in query.lower():
                query_temporal_weight = max(query_temporal_weight, weight)
        
        if query_temporal_weight == 0.0:
            return scores
        
        for i, (_, row) in enumerate(datasets.iterrows()):
            dataset_id = row['dataset_id']
            temporal_info = self.temporal_index.get(dataset_id, {})
            
            # Calculate recency score
            last_updated = temporal_info.get('last_updated')
            if last_updated:
                days_old = (datetime.now() - last_updated).days
                if days_old < 30:
                    recency_score = 1.0
                elif days_old < 365:
                    recency_score = 0.8
                elif days_old < 365 * 2:
                    recency_score = 0.5
                else:
                    recency_score = 0.2
            else:
                recency_score = 0.5  # Default for unknown dates
            
            scores[i] = query_temporal_weight * recency_score
        
        return scores
    
    def _calculate_quality_boost(self, datasets: pd.DataFrame) -> np.ndarray:
        """Calculate quality-based boost scores."""
        quality_scores = datasets['quality_score'].values
        
        # Normalize quality scores to 0-1 range for boost
        if self.quality_stats and self.quality_stats['std'] > 0:
            normalized_scores = (quality_scores - self.quality_stats['mean']) / self.quality_stats['std']
            # Convert to 0-1 range with sigmoid-like function
            boost_scores = 1 / (1 + np.exp(-normalized_scores))
        else:
            boost_scores = quality_scores / quality_scores.max() if quality_scores.max() > 0 else quality_scores
        
        return boost_scores
    
    def _combine_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine different similarity scores using weighted average."""
        if not scores:
            return np.array([])
        
        # Get the length from any score array
        score_length = len(next(iter(scores.values())))
        combined_scores = np.zeros(score_length)
        
        # Map score keys to weight keys
        score_to_weight_mapping = {
            'semantic': 'semantic_similarity',
            'keyword': 'keyword_match',
            'metadata': 'metadata_relevance',
            'relationship': 'relationship_score',
            'temporal': 'temporal_relevance',
            'quality': 'quality_boost'
        }
        
        total_weight = 0.0
        for score_type, score_array in scores.items():
            weight_key = score_to_weight_mapping.get(score_type, score_type)
            weight = self.weights.get(weight_key, 0.0)
            if weight > 0 and len(score_array) == score_length:
                combined_scores += weight * score_array
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_scores /= total_weight
        
        return combined_scores
    
    def _format_search_result(self, 
                            dataset_row: pd.Series, 
                            final_score: float,
                            individual_scores: Dict[str, np.ndarray],
                            index: int,
                            query: str) -> Dict[str, Any]:
        """Format a single search result with detailed scoring information."""
        
        # Extract individual scores and handle NaN values
        score_breakdown = {}
        for score_type, score_array in individual_scores.items():
            if index < len(score_array):
                score_val = score_array[index]
                # Handle NaN values
                if np.isnan(score_val) or np.isinf(score_val):
                    score_breakdown[score_type] = 0.0
                else:
                    score_breakdown[score_type] = float(score_val)
        
        # Handle NaN values in final score
        if np.isnan(final_score) or np.isinf(final_score):
            final_score = 0.0
        
        # Handle NaN values in quality score
        quality_score = dataset_row.get('quality_score', 0.0)
        if pd.isna(quality_score) or np.isnan(quality_score) or np.isinf(quality_score):
            quality_score = 0.0
        
        return {
            'dataset_id': dataset_row.get('dataset_id', ''),
            'title': dataset_row.get('title', 'Untitled Dataset'),
            'description': dataset_row.get('description', 'No description available'),
            'source': dataset_row.get('source', ''),
            'category': dataset_row.get('category', ''),
            'quality_score': float(quality_score),
            'last_updated': dataset_row.get('last_updated', ''),
            'format': dataset_row.get('format', ''),
            'url': dataset_row.get('url', ''),
            'multimodal_score': float(final_score),
            'score_breakdown': score_breakdown,
            'search_metadata': {
                'query': query,
                'search_timestamp': datetime.now().isoformat(),
                'search_mode': 'multimodal',
                'weights_used': self.weights
            }
        }
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date string into datetime object."""
        if pd.isna(date_str) or not date_str:
            return None
        
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def get_search_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get search suggestions based on partial query."""
        suggestions = []
        
        try:
            partial_lower = partial_query.lower()
            
            # Extract common keywords from dataset titles and descriptions
            all_text = ' '.join(self.datasets['title'].fillna('') + ' ' + self.datasets['description'].fillna(''))
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Find words that start with the partial query
            matching_words = [word for word in set(words) if word.startswith(partial_lower) and len(word) > len(partial_lower)]
            
            # Sort by frequency and take top suggestions
            word_freq = {word: words.count(word) for word in matching_words}
            suggestions = sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:max_suggestions]
            
        except Exception as e:
            logger.warning(f"Search suggestions generation failed: {e}")
        
        return suggestions
    
    def get_related_datasets(self, dataset_id: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """Get datasets related to a specific dataset."""
        related = []
        
        try:
            if self.relationship_graph and dataset_id in self.relationship_graph:
                # Get direct neighbors
                neighbors = list(self.relationship_graph.neighbors(dataset_id))
                
                # Sort by edge weight
                neighbor_weights = []
                for neighbor in neighbors:
                    weight = self.relationship_graph[dataset_id][neighbor].get('weight', 0.5)
                    neighbor_weights.append((neighbor, weight))
                
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                
                # Format related datasets
                for neighbor_id, weight in neighbor_weights[:max_related]:
                    neighbor_row = self.datasets[self.datasets['dataset_id'] == neighbor_id]
                    if not neighbor_row.empty:
                        dataset_info = neighbor_row.iloc[0]
                        related.append({
                            'dataset_id': neighbor_id,
                            'title': dataset_info.get('title', ''),
                            'description': dataset_info.get('description', ''),
                            'relationship_strength': float(weight),
                            'relationship_type': self.relationship_graph[dataset_id][neighbor_id].get('type', 'related')
                        })
            
        except Exception as e:
            logger.warning(f"Related datasets retrieval failed: {e}")
        
        return related


def create_multimodal_search_config() -> Dict[str, Any]:
    """Create configuration for multi-modal search."""
    return {
        'multimodal_search': {
            'semantic_model': 'all-MiniLM-L6-v2',
            'weights': {
                'semantic_similarity': 0.50,  # Increased for better relevance
                'keyword_match': 0.30,        # Increased for exact matches
                'metadata_relevance': 0.15,
                'relationship_score': 0.05,   # Reduced
                'temporal_relevance': 0.00,   # Disabled for now
                'quality_boost': 0.00         # Disabled to prevent irrelevant high-quality results
            },
            'max_results': 50,
            'min_score_threshold': 0.1
        }
    }