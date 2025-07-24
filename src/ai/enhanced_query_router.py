"""
Enhanced Query Router with Singapore-First Strategy
Implements intelligent query classification and domain-specific routing using the trained quality-first model
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import torch

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Query classification result with routing information"""
    original_query: str
    domain: str  # psychology, climate, technology, singapore, etc.
    geographic_scope: str  # local, singapore, global, international
    intent: str  # research, analysis, comparison, exploration
    confidence: float
    singapore_first_applicable: bool
    recommended_sources: List[str]
    explanation: str
    
    def get_source_priority_order(self) -> List[str]:
        """Get prioritized list of sources for this query"""
        return self.recommended_sources


class EnhancedQueryRouter:
    """Enhanced query router with Singapore-first strategy and domain-specific routing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Load trained quality-first model if available
        self.model = None
        self.model_loaded = False
        self._load_quality_model()
        
        # Domain definitions based on training mappings
        self.domain_definitions = {
            'psychology': {
                'keywords': ['psychology', 'mental health', 'behavioral', 'cognitive', 'psychological'],
                'preferred_sources': ['kaggle', 'zenodo'],
                'singapore_first': False,
                'confidence_boost': 0.1
            },
            'machine_learning': {
                'keywords': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural'],
                'preferred_sources': ['kaggle', 'zenodo'],
                'singapore_first': False,
                'confidence_boost': 0.1
            },
            'climate': {
                'keywords': ['climate', 'weather', 'environmental', 'temperature', 'climate change'],
                'preferred_sources': ['world_bank', 'zenodo'],
                'singapore_first': False,
                'confidence_boost': 0.15
            },
            'economics': {
                'keywords': ['economic', 'gdp', 'financial', 'trade', 'poverty', 'economy'],
                'preferred_sources': ['world_bank', 'singstat'],
                'singapore_first': False,
                'confidence_boost': 0.1
            },
            'singapore': {
                'keywords': ['singapore', 'sg', 'hdb', 'mrt', 'lta', 'singstat', 'data.gov.sg'],
                'preferred_sources': ['data_gov_sg', 'singstat', 'lta_datamall'],
                'singapore_first': True,
                'confidence_boost': 0.2
            },
            'health': {
                'keywords': ['health', 'medical', 'healthcare', 'disease', 'hospital'],
                'preferred_sources': ['world_bank', 'zenodo', 'data_gov_sg'],
                'singapore_first': True,  # Health queries often benefit from local data
                'confidence_boost': 0.1
            },
            'education': {
                'keywords': ['education', 'student', 'university', 'school', 'learning'],
                'preferred_sources': ['world_bank', 'zenodo', 'data_gov_sg'],
                'singapore_first': True,  # Education queries often benefit from local data
                'confidence_boost': 0.1
            },
            'transport': {
                'keywords': ['transport', 'traffic', 'bus', 'mrt', 'taxi', 'grab', 'vehicle'],
                'preferred_sources': ['lta_datamall', 'data_gov_sg', 'kaggle'],
                'singapore_first': True,
                'confidence_boost': 0.15
            },
            'housing': {
                'keywords': ['housing', 'property', 'real estate', 'hdb', 'rental', 'resale'],
                'preferred_sources': ['data_gov_sg', 'singstat', 'world_bank'],
                'singapore_first': True,
                'confidence_boost': 0.2
            }
        }
        
        # Source quality mapping based on training results
        self.source_quality_scores = {
            'kaggle': 0.92,      # High for ML/Psychology
            'zenodo': 0.88,      # High for academic research
            'world_bank': 0.95,  # Highest for economics/climate
            'data_gov_sg': 0.96, # Highest for Singapore data
            'singstat': 0.94,    # High for Singapore statistics
            'lta_datamall': 0.93, # High for Singapore transport
            'aws_opendata': 0.65, # Medium quality
            'data_un': 0.70,     # Medium quality
            'github': 0.75       # Medium for code/datasets
        }
        
        # Singapore-first generic terms (from your updated training mappings)
        self.singapore_first_terms = {
            'housing', 'property', 'real estate', 'rental',
            'population', 'demographics', 'census', 'migration',
            'economy', 'gdp', 'inflation', 'employment', 'wages', 'business',
            'transport', 'traffic', 'mrt', 'bus', 'roads',
            'health', 'healthcare', 'hospitals', 'medical',
            'education', 'schools', 'students', 'university',
            'environment', 'weather', 'air quality', 'urban planning', 'parks'
        }
        
        logger.info(f"🧭 EnhancedQueryRouter initialized")
        logger.info(f"  Model loaded: {self.model_loaded}")
        logger.info(f"  Domains: {len(self.domain_definitions)}")
        logger.info(f"  Sources: {len(self.source_quality_scores)}")
    
    def _load_quality_model(self):
        """Load the trained quality-first model if available"""
        try:
            model_path = Path("models/dl/quality_first/best_quality_model.pt")
            if model_path.exists():
                # Import model class
                try:
                    from ..dl.quality_first_neural_model import QualityAwareRankingModel
                except ImportError:
                    import sys
                    sys.path.append('src/dl')
                    from quality_first_neural_model import QualityAwareRankingModel
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Create model with same config
                model_config = {
                    'embedding_dim': 256,
                    'hidden_dim': 128,
                    'num_domains': 8,
                    'num_sources': 10,
                    'vocab_size': 10000
                }
                
                self.model = QualityAwareRankingModel(model_config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.model_loaded = True
                
                logger.info(f"✅ Quality-first model loaded from {model_path}")
            else:
                logger.info(f"⚠️ Quality-first model not found at {model_path}")
                
        except Exception as e:
            logger.warning(f"Could not load quality-first model: {e}")
            self.model_loaded = False
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query intent, domain, and geographic scope"""
        logger.debug(f"🔍 Classifying query: '{query}'")
        
        query_lower = query.lower().strip()
        
        # Use rule-based classification as primary (more reliable for now)
        domain, singapore_first, confidence = self._rule_based_classify(query_lower)
        
        # Use neural model as validation/enhancement if available
        if self.model_loaded and self.model and confidence < 0.8:
            try:
                neural_domain, neural_singapore = self._neural_classify(query)
                # Only use neural results if rule-based confidence is low
                if confidence < 0.6:
                    domain = neural_domain
                    singapore_first = neural_singapore
                    confidence = 0.7  # Medium confidence from neural model
            except Exception as e:
                logger.debug(f"Neural classification failed: {e}")
        
        # Determine geographic scope
        geographic_scope = self._determine_geographic_scope(query_lower, domain, singapore_first)
        
        # Determine intent
        intent = self._determine_query_intent(query_lower)
        
        # Get recommended sources
        recommended_sources = self._get_recommended_sources(domain, singapore_first, query_lower)
        
        # Generate explanation
        explanation = self._generate_explanation(query, domain, singapore_first, recommended_sources)
        
        classification = QueryClassification(
            original_query=query,
            domain=domain,
            geographic_scope=geographic_scope,
            intent=intent,
            confidence=confidence,
            singapore_first_applicable=singapore_first,
            recommended_sources=recommended_sources,
            explanation=explanation
        )
        
        logger.info(f"✅ Query classified: {domain} | Singapore-first: {singapore_first} | Confidence: {confidence:.2f}")
        
        return classification
    
    def _neural_classify(self, query: str) -> Tuple[str, bool]:
        """Use trained neural model for classification"""
        try:
            with torch.no_grad():
                domain, singapore_first = self.model.predict_domain_and_singapore(query)
                return domain, singapore_first
        except Exception as e:
            logger.warning(f"Neural classification failed: {e}")
            # Fallback to rule-based
            return self._rule_based_classify(query.lower())[:2]
    
    def _rule_based_classify(self, query_lower: str) -> Tuple[str, bool, float]:
        """Rule-based classification as fallback"""
        best_domain = 'general'
        best_score = 0.0
        singapore_first = False
        
        # Check for explicit Singapore mentions
        if 'singapore' in query_lower or any(sg_term in query_lower for sg_term in ['sg', 'hdb', 'mrt', 'lta']):
            singapore_first = True
            if 'singapore' in query_lower:
                best_domain = 'singapore'
                best_score = 0.9
        
        # Check for Singapore-first generic terms
        if not singapore_first:
            for term in self.singapore_first_terms:
                if term in query_lower:
                    # Check if it's explicitly global
                    if not any(global_word in query_lower for global_word in ['global', 'international', 'worldwide']):
                        singapore_first = True
                        break
        
        # Domain classification - improved logic
        for domain, definition in self.domain_definitions.items():
            score = 0.0
            keyword_matches = 0
            
            for keyword in definition['keywords']:
                if keyword in query_lower:
                    keyword_matches += 1
                    # Weight longer keywords more heavily
                    keyword_weight = len(keyword.split()) * 0.5 + 0.5
                    score += keyword_weight
                    
                    # Exact match bonus
                    if query_lower.strip() == keyword:
                        score += 1.0
                    
                    # Partial word match bonus
                    if keyword in query_lower.split():
                        score += 0.3
            
            # Only consider domains with actual keyword matches
            if keyword_matches > 0:
                # Normalize by number of keywords but give bonus for multiple matches
                base_score = score / len(definition['keywords'])
                match_bonus = min(keyword_matches / len(definition['keywords']), 1.0) * 0.2
                score = base_score + match_bonus
                
                # Apply confidence boost
                score += definition.get('confidence_boost', 0.0)
                
                if score > best_score:
                    best_score = score
                    best_domain = domain
        
        # Confidence based on score
        confidence = min(0.95, 0.5 + best_score)
        
        return best_domain, singapore_first, confidence
    
    def _determine_geographic_scope(self, query_lower: str, domain: str, singapore_first: bool) -> str:
        """Determine geographic scope of the query"""
        if singapore_first or 'singapore' in query_lower:
            return 'singapore'
        elif any(word in query_lower for word in ['global', 'international', 'worldwide']):
            return 'global'
        elif domain in ['climate', 'economics']:
            return 'global'  # These domains are typically global
        else:
            return 'general'
    
    def _determine_query_intent(self, query_lower: str) -> str:
        """Determine the intent behind the query"""
        if any(word in query_lower for word in ['research', 'study', 'analysis', 'analyze']):
            return 'research'
        elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['data', 'dataset', 'download', 'access']):
            return 'data_access'
        elif any(word in query_lower for word in ['find', 'search', 'look', 'need']):
            return 'exploration'
        else:
            return 'exploration'
    
    def _get_recommended_sources(self, domain: str, singapore_first: bool, query_lower: str) -> List[str]:
        """Get recommended sources based on domain and Singapore-first strategy"""
        recommended = []
        
        # Get domain-specific sources
        if domain in self.domain_definitions:
            domain_sources = self.domain_definitions[domain]['preferred_sources'].copy()
            recommended.extend(domain_sources)
        
        # Apply Singapore-first strategy
        if singapore_first:
            singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
            # Move Singapore sources to front
            for sg_source in singapore_sources:
                if sg_source in recommended:
                    recommended.remove(sg_source)
                recommended.insert(0, sg_source)
            
            # Add Singapore sources if not already present
            for sg_source in singapore_sources:
                if sg_source not in recommended:
                    recommended.insert(0, sg_source)
        
        # Add high-quality general sources if not enough recommendations
        general_sources = ['world_bank', 'kaggle', 'zenodo']
        for source in general_sources:
            if source not in recommended and len(recommended) < 5:
                recommended.append(source)
        
        # Sort by quality score (keeping Singapore-first order)
        if not singapore_first:
            recommended.sort(key=lambda x: self.source_quality_scores.get(x, 0.5), reverse=True)
        
        return recommended[:5]  # Limit to top 5 sources
    
    def _generate_explanation(self, query: str, domain: str, singapore_first: bool, sources: List[str]) -> str:
        """Generate explanation for the routing decision"""
        explanations = []
        
        # Domain explanation
        if domain != 'general':
            explanations.append(f"Classified as {domain} query")
        
        # Singapore-first explanation
        if singapore_first:
            explanations.append("Singapore-first strategy applied - prioritizing local government sources")
        
        # Source explanation
        if sources:
            top_source = sources[0]
            if top_source in ['data_gov_sg', 'singstat', 'lta_datamall']:
                explanations.append(f"Recommending {top_source} as primary Singapore government source")
            elif top_source == 'world_bank':
                explanations.append(f"Recommending World Bank for global {domain} data")
            elif top_source == 'kaggle':
                explanations.append(f"Recommending Kaggle for {domain} datasets and competitions")
            elif top_source == 'zenodo':
                explanations.append(f"Recommending Zenodo for academic {domain} research")
        
        return "; ".join(explanations) if explanations else "General routing applied"
    
    def route_to_sources(self, classification: QueryClassification) -> List[Dict[str, any]]:
        """Route to appropriate data sources based on classification"""
        sources = []
        
        for i, source_name in enumerate(classification.recommended_sources):
            source_info = {
                'name': source_name,
                'priority': i + 1,
                'quality_score': self.source_quality_scores.get(source_name, 0.5),
                'relevance_reason': self._get_source_relevance_reason(source_name, classification),
                'url_pattern': self._get_source_url_pattern(source_name),
                'singapore_source': source_name in ['data_gov_sg', 'singstat', 'lta_datamall']
            }
            sources.append(source_info)
        
        return sources
    
    def _get_source_relevance_reason(self, source_name: str, classification: QueryClassification) -> str:
        """Get reason why this source is relevant for the query"""
        domain = classification.domain
        singapore_first = classification.singapore_first_applicable
        
        reasons = {
            'kaggle': f"Excellent for {domain} datasets and machine learning competitions",
            'zenodo': f"Academic repository with high-quality {domain} research data",
            'world_bank': f"Authoritative source for global {domain} indicators and statistics",
            'data_gov_sg': "Official Singapore government open data portal",
            'singstat': "Singapore Department of Statistics - authoritative local data",
            'lta_datamall': "Singapore Land Transport Authority - official transport data",
            'aws_opendata': "AWS Open Data registry with diverse datasets",
            'data_un': "United Nations data portal for global statistics",
            'github': "Code repositories and datasets from the developer community"
        }
        
        base_reason = reasons.get(source_name, "General data source")
        
        if singapore_first and source_name in ['data_gov_sg', 'singstat', 'lta_datamall']:
            return f"{base_reason} (Singapore-first priority)"
        
        return base_reason
    
    def _get_source_url_pattern(self, source_name: str) -> str:
        """Get URL pattern for the source"""
        patterns = {
            'kaggle': 'https://www.kaggle.com/datasets',
            'zenodo': 'https://zenodo.org/search',
            'world_bank': 'https://data.worldbank.org',
            'data_gov_sg': 'https://data.gov.sg/datasets',
            'singstat': 'https://www.singstat.gov.sg/find-data',
            'lta_datamall': 'https://datamall.lta.gov.sg/content/datamall/en.html',
            'aws_opendata': 'https://registry.opendata.aws',
            'data_un': 'https://data.un.org',
            'github': 'https://github.com/search?q=dataset'
        }
        
        return patterns.get(source_name, '')
    
    def apply_singapore_first_strategy(self, query: str) -> bool:
        """Determine if Singapore-first strategy should be applied"""
        classification = self.classify_query(query)
        return classification.singapore_first_applicable
    
    def get_domain_specific_sources(self, domain: str) -> List[str]:
        """Get specialized sources for specific domains"""
        if domain in self.domain_definitions:
            return self.domain_definitions[domain]['preferred_sources']
        return ['kaggle', 'zenodo', 'world_bank']  # Default sources
    
    def test_routing_examples(self):
        """Test the router with example queries"""
        test_queries = [
            "psychology research data",
            "singapore housing prices",
            "climate change data",
            "machine learning datasets",
            "economic indicators",
            "transport data",
            "global health statistics",
            "education statistics"
        ]
        
        logger.info("🧪 Testing query routing examples:")
        
        for query in test_queries:
            classification = self.classify_query(query)
            sources = self.route_to_sources(classification)
            
            logger.info(f"  Query: '{query}'")
            logger.info(f"    Domain: {classification.domain}")
            logger.info(f"    Singapore-first: {classification.singapore_first_applicable}")
            logger.info(f"    Top sources: {[s['name'] for s in sources[:3]]}")
            logger.info(f"    Explanation: {classification.explanation}")
            logger.info("")


def create_enhanced_query_router(config: Dict = None) -> EnhancedQueryRouter:
    """Factory function to create enhanced query router"""
    return EnhancedQueryRouter(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the enhanced query router
    router = create_enhanced_query_router()
    
    # Test with example queries
    router.test_routing_examples()
    
    print("✅ Enhanced Query Router testing completed!")