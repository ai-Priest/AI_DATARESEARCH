"""
Graded Relevance Scoring System
Implements 4-level relevance scoring: 0.0 (irrelevant), 0.3 (somewhat relevant), 0.7 (relevant), 1.0 (highly relevant)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GradedRelevanceScorer:
    """
    Advanced graded relevance scoring system for query-dataset pairs.
    
    Scoring Schema:
    - 1.0: Highly Relevant - Perfect match, direct answer to query
    - 0.7: Relevant - Good match, useful for query
    - 0.3: Somewhat Relevant - Partial match, tangentially related
    - 0.0: Irrelevant - No meaningful relationship
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.relevance_config = config.get('graded_relevance', {})
        
        # Relevance thresholds
        self.thresholds = {
            'highly_relevant': self.relevance_config.get('highly_relevant_threshold', 0.85),
            'relevant': self.relevance_config.get('relevant_threshold', 0.65),
            'somewhat_relevant': self.relevance_config.get('somewhat_relevant_threshold', 0.35)
        }
        
        # Domain-specific keywords for enhanced scoring
        self.domain_keywords = {
            'housing': ['hdb', 'housing', 'property', 'resale', 'rental', 'bto', 'condo', 'prices'],
            'transportation': ['mrt', 'bus', 'lrt', 'transport', 'traffic', 'taxi', 'grab', 'coe'],
            'healthcare': ['hospital', 'clinic', 'health', 'medical', 'doctor', 'patient', 'disease'],
            'economics': ['gdp', 'inflation', 'employment', 'wage', 'economy', 'trade', 'finance'],
            'education': ['school', 'university', 'student', 'education', 'exam', 'curriculum'],
            'demographics': ['population', 'age', 'birth', 'death', 'marriage', 'citizen', 'resident']
        }
        
        logger.info("🎯 GradedRelevanceScorer initialized with 4-level scoring system")
    
    def score_relevance(self, query: str, dataset: Dict) -> float:
        """
        Score relevance between query and dataset using multiple criteria.
        
        Args:
            query: User search query
            dataset: Dataset metadata dictionary
            
        Returns:
            Relevance score (0.0, 0.3, 0.7, 1.0)
        """
        try:
            # Extract dataset text features
            title = dataset.get('title', '').lower()
            description = dataset.get('description', '').lower()
            keywords = dataset.get('keywords', [])
            if isinstance(keywords, str):
                keywords = keywords.lower()
            elif isinstance(keywords, list):
                keywords = ' '.join([str(k).lower() for k in keywords])
            else:
                keywords = ''
            
            # Combine all text
            dataset_text = f"{title} {description} {keywords}"
            query_lower = query.lower()
            
            # Calculate multiple relevance signals
            signals = self._calculate_relevance_signals(query_lower, dataset, dataset_text)
            
            # Combine signals for final score
            relevance_score = self._combine_signals(signals)
            
            # Convert to graded levels
            graded_score = self._convert_to_graded_score(relevance_score)
            
            logger.debug(f"Query: '{query[:50]}...' → Dataset: '{title[:50]}...' → Score: {graded_score}")
            
            return graded_score
            
        except Exception as e:
            logger.warning(f"Error scoring relevance: {e}")
            return 0.0
    
    def _calculate_relevance_signals(self, query: str, dataset: Dict, dataset_text: str) -> Dict[str, float]:
        """Calculate multiple relevance signals."""
        signals = {}
        
        # 1. Exact keyword matching
        signals['exact_match'] = self._exact_keyword_match(query, dataset_text)
        
        # 2. Semantic similarity (TF-IDF based)
        signals['semantic_similarity'] = self._semantic_similarity(query, dataset_text)
        
        # 3. Domain relevance
        signals['domain_relevance'] = self._domain_relevance(query, dataset_text)
        
        # 4. Title relevance (higher weight)
        signals['title_relevance'] = self._title_relevance(query, dataset.get('title', ''))
        
        # 5. Quality score boost
        signals['quality_boost'] = dataset.get('quality_score', 0.5)
        
        # 6. Source credibility
        signals['source_credibility'] = self._source_credibility(dataset.get('source', ''))
        
        return signals
    
    def _exact_keyword_match(self, query: str, dataset_text: str) -> float:
        """Calculate exact keyword matching score."""
        query_words = set(re.findall(r'\b\w+\b', query))
        dataset_words = set(re.findall(r'\b\w+\b', dataset_text))
        
        if not query_words:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(query_words & dataset_words)
        union = len(query_words | dataset_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _semantic_similarity(self, query: str, dataset_text: str) -> float:
        """Calculate semantic similarity using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            texts = [query, dataset_text]
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0.0, similarity)
        except:
            return 0.0
    
    def _domain_relevance(self, query: str, dataset_text: str) -> float:
        """Calculate domain-specific relevance."""
        max_relevance = 0.0
        
        for domain, keywords in self.domain_keywords.items():
            query_domain_score = sum(1 for keyword in keywords if keyword in query) / len(keywords)
            dataset_domain_score = sum(1 for keyword in keywords if keyword in dataset_text) / len(keywords)
            
            # Both query and dataset should be in the same domain
            domain_relevance = min(query_domain_score, dataset_domain_score) * 2
            max_relevance = max(max_relevance, domain_relevance)
        
        return min(1.0, max_relevance)
    
    def _title_relevance(self, query: str, title: str) -> float:
        """Calculate title-specific relevance (weighted higher)."""
        title_lower = title.lower()
        query_words = set(re.findall(r'\b\w+\b', query))
        title_words = set(re.findall(r'\b\w+\b', title_lower))
        
        if not query_words:
            return 0.0
            
        # Higher weight for title matches
        intersection = len(query_words & title_words)
        return min(1.0, intersection / len(query_words) * 1.5)
    
    def _source_credibility(self, source: str) -> float:
        """Calculate source credibility score."""
        credible_sources = {
            'data.gov.sg': 1.0,
            'singstat.gov.sg': 1.0,
            'moh.gov.sg': 0.9,
            'lta.gov.sg': 0.9,
            'ura.gov.sg': 0.9,
            'government': 0.8,
            'university': 0.7,
            'research': 0.6
        }
        
        source_lower = source.lower()
        for key, score in credible_sources.items():
            if key in source_lower:
                return score
        
        return 0.5  # Default score
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine multiple signals into a single relevance score."""
        weights = {
            'exact_match': 0.25,
            'semantic_similarity': 0.20,
            'domain_relevance': 0.20,
            'title_relevance': 0.25,
            'quality_boost': 0.05,
            'source_credibility': 0.05
        }
        
        combined_score = sum(
            signals.get(signal, 0.0) * weight 
            for signal, weight in weights.items()
        )
        
        return min(1.0, combined_score)
    
    def _convert_to_graded_score(self, relevance_score: float) -> float:
        """Convert continuous score to graded levels."""
        if relevance_score >= self.thresholds['highly_relevant']:
            return 1.0  # Highly relevant
        elif relevance_score >= self.thresholds['relevant']:
            return 0.7  # Relevant
        elif relevance_score >= self.thresholds['somewhat_relevant']:
            return 0.3  # Somewhat relevant
        else:
            return 0.0  # Irrelevant
    
    def generate_graded_training_data(self, 
                                    existing_data_path: str, 
                                    output_path: str,
                                    num_samples: int = 2000) -> None:
        """
        Generate enhanced training data with graded relevance scores.
        
        Args:
            existing_data_path: Path to existing training data
            output_path: Path to save enhanced data
            num_samples: Number of samples to generate
        """
        logger.info(f"🔄 Generating graded training data with {num_samples} samples...")
        
        try:
            # Load existing data
            with open(existing_data_path, 'r') as f:
                existing_data = json.load(f)
            
            # Load datasets
            datasets_path = Path("data/processed")
            singapore_datasets = pd.read_csv(datasets_path / "singapore_datasets.csv")
            global_datasets = pd.read_csv(datasets_path / "global_datasets.csv")
            all_datasets = pd.concat([singapore_datasets, global_datasets], ignore_index=True)
            
            # Generate graded samples
            graded_samples = []
            
            # Enhance existing scenarios with graded scoring
            for scenario in existing_data.get('ground_truth_scenarios', []):
                query = scenario['search_query']
                
                for dataset_id in scenario['relevant_datasets']:
                    dataset_row = all_datasets[all_datasets['dataset_id'] == dataset_id]
                    if not dataset_row.empty:
                        dataset = dataset_row.iloc[0].to_dict()
                        graded_score = self.score_relevance(query, dataset)
                        
                        graded_samples.append({
                            'query': query,
                            'dataset_id': dataset_id,
                            'relevance_score': graded_score,
                            'query_type': scenario.get('query_type', 'general'),
                            'domain': scenario.get('domain', 'general')
                        })
            
            # Generate additional diverse samples
            additional_samples = self._generate_diverse_samples(all_datasets, num_samples - len(graded_samples))
            graded_samples.extend(additional_samples)
            
            # Save enhanced training data
            enhanced_data = {
                'metadata': {
                    'version': '2.0',
                    'created_date': pd.Timestamp.now().isoformat(),
                    'total_samples': len(graded_samples),
                    'graded_levels': [0.0, 0.3, 0.7, 1.0],
                    'scoring_method': 'multi_signal_graded'
                },
                'training_samples': graded_samples
            }
            
            with open(output_path, 'w') as f:
                json.dump(enhanced_data, f, indent=2)
            
            # Log statistics
            score_distribution = {}
            for sample in graded_samples:
                score = sample['relevance_score']
                score_distribution[score] = score_distribution.get(score, 0) + 1
            
            logger.info(f"✅ Enhanced training data saved to {output_path}")
            logger.info(f"📊 Score distribution: {score_distribution}")
            logger.info(f"📈 Total samples: {len(graded_samples)}")
            
        except Exception as e:
            logger.error(f"Error generating graded training data: {e}")
            raise
    
    def _generate_diverse_samples(self, datasets: pd.DataFrame, num_samples: int) -> List[Dict]:
        """Generate diverse query-dataset pairs for training."""
        samples = []
        
        # Query templates for different domains
        query_templates = {
            'housing': [
                "housing prices in {location}",
                "HDB resale data {year}",
                "property market trends",
                "rental prices {area}",
                "housing affordability statistics"
            ],
            'transportation': [
                "MRT ridership data",
                "bus routes and schedules",
                "traffic congestion {area}",
                "transport statistics {year}",
                "public transport usage"
            ],
            'healthcare': [
                "hospital statistics {year}",
                "healthcare utilization data",
                "disease prevalence singapore",
                "medical services {area}",
                "health outcomes data"
            ],
            'economics': [
                "GDP growth {year}",
                "employment statistics singapore",
                "inflation rates {year}",
                "trade data singapore",
                "economic indicators"
            ]
        }
        
        # Generate samples across domains
        for i in range(num_samples):
            domain = np.random.choice(list(query_templates.keys()))
            template = np.random.choice(query_templates[domain])
            
            # Fill template with random values
            query = template.format(
                location=np.random.choice(['singapore', 'central', 'north', 'south', 'east', 'west']),
                year=np.random.choice(['2023', '2024', '2022', '2021']),
                area=np.random.choice(['singapore', 'cbd', 'jurong', 'tampines', 'woodlands'])
            )
            
            # Select random dataset
            dataset_row = datasets.sample(1).iloc[0]
            dataset = dataset_row.to_dict()
            
            # Score relevance
            graded_score = self.score_relevance(query, dataset)
            
            samples.append({
                'query': query,
                'dataset_id': dataset['dataset_id'],
                'relevance_score': graded_score,
                'query_type': 'generated',
                'domain': domain
            })
        
        return samples


def create_graded_relevance_config() -> Dict:
    """Create configuration for graded relevance scoring."""
    return {
        'graded_relevance': {
            'highly_relevant_threshold': 0.85,
            'relevant_threshold': 0.65,
            'somewhat_relevant_threshold': 0.35,
            'scoring_weights': {
                'exact_match': 0.25,
                'semantic_similarity': 0.20,
                'domain_relevance': 0.20,
                'title_relevance': 0.25,
                'quality_boost': 0.05,
                'source_credibility': 0.05
            }
        }
    }