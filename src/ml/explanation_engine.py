"""
Explanation Engine for Dataset Recommendations
Provides human-readable explanations for why datasets were recommended.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class RecommendationExplainer:
    """Generates explanations for dataset recommendations."""
    
    def __init__(self):
        self.explanation_templates = {
            'keyword_match': "Contains keywords: {keywords}",
            'category_match': "Similar category: {category}",
            'description_similarity': "Similar content: {description_snippet}",
            'source_match': "Same data source: {source}",
            'quality_match': "High quality dataset (score: {quality:.1f})",
            'temporal_match': "Similar time period: {temporal_info}",
            'geographic_match': "Same geographic area: {location}",
            'domain_expertise': "Recommended based on Singapore government data expertise",
            'user_preference': "Matches your previous interests in {domain}",
            'semantic_similarity': "Semantically related content"
        }
    
    def explain_recommendation(self, query: str, recommended_dataset: Dict, 
                             all_datasets: pd.DataFrame, 
                             similarity_score: float = None,
                             user_context: Dict = None) -> Dict:
        """
        Generate comprehensive explanation for why a dataset was recommended.
        
        Args:
            query: User's search query
            recommended_dataset: The recommended dataset
            all_datasets: All available datasets for comparison
            similarity_score: Computed similarity score
            user_context: User preferences and history
            
        Returns:
            Dictionary with explanation components
        """
        logger.info(f"🔍 Generating explanation for dataset: {recommended_dataset.get('title', 'Unknown')}")
        
        explanation = {
            'dataset_title': recommended_dataset.get('title', 'Unknown Dataset'),
            'similarity_score': similarity_score or 0.0,
            'primary_reasons': [],
            'secondary_reasons': [],
            'match_details': {},
            'confidence_level': 'medium',
            'explanation_text': '',
            'technical_details': {}
        }
        
        # Extract query terms for analysis
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        # 1. Keyword matching analysis
        keyword_explanation = self._analyze_keyword_matches(
            query_terms, recommended_dataset, explanation
        )
        
        # 2. Category similarity analysis
        category_explanation = self._analyze_category_similarity(
            query, recommended_dataset, all_datasets, explanation
        )
        
        # 3. Content similarity analysis
        content_explanation = self._analyze_content_similarity(
            query, recommended_dataset, explanation
        )
        
        # 4. Quality and credibility analysis
        quality_explanation = self._analyze_quality_factors(
            recommended_dataset, explanation
        )
        
        # 5. Geographic and domain relevance
        domain_explanation = self._analyze_domain_relevance(
            query, recommended_dataset, explanation
        )
        
        # 6. User preference matching (if available)
        if user_context:
            user_explanation = self._analyze_user_preferences(
                recommended_dataset, user_context, explanation
            )
        
        # 7. Generate confidence level
        explanation['confidence_level'] = self._calculate_confidence_level(explanation)
        
        # 8. Create human-readable explanation
        explanation['explanation_text'] = self._generate_explanation_text(explanation)
        
        logger.info(f"✅ Generated explanation with {len(explanation['primary_reasons'])} primary reasons")
        return explanation
    
    def _analyze_keyword_matches(self, query_terms: set, dataset: Dict, 
                                explanation: Dict) -> Dict:
        """Analyze keyword matches between query and dataset."""
        
        # Check title matches
        title = str(dataset.get('title', '')).lower()
        title_words = set(re.findall(r'\b\w{3,}\b', title))
        title_matches = query_terms.intersection(title_words)
        
        # Check description matches
        description = str(dataset.get('description', '')).lower()
        desc_words = set(re.findall(r'\b\w{3,}\b', description))
        desc_matches = query_terms.intersection(desc_words)
        
        # Check tag matches
        tags = str(dataset.get('tags', '')).lower()
        tag_words = set(re.findall(r'\b\w{3,}\b', tags))
        tag_matches = query_terms.intersection(tag_words)
        
        all_matches = title_matches.union(desc_matches).union(tag_matches)
        
        if all_matches:
            explanation['primary_reasons'].append(
                self.explanation_templates['keyword_match'].format(
                    keywords=', '.join(sorted(all_matches))
                )
            )
            explanation['match_details']['keyword_matches'] = {
                'title_matches': list(title_matches),
                'description_matches': list(desc_matches),
                'tag_matches': list(tag_matches),
                'total_matches': len(all_matches)
            }
            explanation['technical_details']['keyword_overlap_ratio'] = len(all_matches) / len(query_terms) if query_terms else 0
        
        return explanation
    
    def _analyze_category_similarity(self, query: str, dataset: Dict, 
                                   all_datasets: pd.DataFrame, explanation: Dict) -> Dict:
        """Analyze category-based similarity."""
        
        dataset_category = str(dataset.get('category', '')).lower()
        
        if not dataset_category or dataset_category == 'nan':
            return explanation
        
        # Check if query relates to this category
        category_keywords = {
            'transport': ['transport', 'traffic', 'lta', 'mrt', 'bus', 'taxi', 'vehicle'],
            'housing': ['housing', 'hdb', 'property', 'residential', 'flat', 'resale'],
            'environment': ['environment', 'weather', 'climate', 'pollution', 'green', 'sustainability'],
            'population': ['population', 'demographics', 'residents', 'citizens', 'census'],
            'economy': ['economy', 'economic', 'gdp', 'trade', 'business', 'finance', 'budget'],
            'health': ['health', 'healthcare', 'medical', 'hospital', 'disease'],
            'education': ['education', 'school', 'student', 'university', 'learning']
        }
        
        query_lower = query.lower()
        relevant_categories = []
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_categories.append(category)
        
        # Check if dataset category matches query intent
        if any(cat in dataset_category for cat in relevant_categories):
            explanation['primary_reasons'].append(
                self.explanation_templates['category_match'].format(
                    category=dataset_category.title()
                )
            )
            explanation['match_details']['category_relevance'] = {
                'dataset_category': dataset_category,
                'matching_categories': relevant_categories
            }
        
        return explanation
    
    def _analyze_content_similarity(self, query: str, dataset: Dict, 
                                  explanation: Dict) -> Dict:
        """Analyze content-based similarity."""
        
        description = str(dataset.get('description', ''))
        
        if len(description) < 20 or description.lower() == 'nan':
            return explanation
        
        # Extract key phrases from description that might relate to query
        query_words = re.findall(r'\b\w{4,}\b', query.lower())
        desc_sentences = re.split(r'[.!?]+', description)
        
        relevant_sentences = []
        for sentence in desc_sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Take the most relevant sentence (first match)
            best_sentence = relevant_sentences[0]
            if len(best_sentence) > 100:
                best_sentence = best_sentence[:97] + "..."
            
            explanation['secondary_reasons'].append(
                self.explanation_templates['description_similarity'].format(
                    description_snippet=f'"{best_sentence}"'
                )
            )
            explanation['match_details']['content_similarity'] = {
                'relevant_sentences': relevant_sentences[:3],
                'description_length': len(description)
            }
        
        return explanation
    
    def _analyze_quality_factors(self, dataset: Dict, explanation: Dict) -> Dict:
        """Analyze quality and credibility factors."""
        
        quality_score = dataset.get('quality_score', 0)
        source = str(dataset.get('source', '')).lower()
        
        # High quality datasets
        if quality_score >= 0.8:
            explanation['secondary_reasons'].append(
                self.explanation_templates['quality_match'].format(
                    quality=quality_score
                )
            )
            explanation['match_details']['quality_indicators'] = {
                'quality_score': quality_score,
                'quality_tier': 'high'
            }
        
        # Government/official sources
        gov_indicators = ['gov', 'government', 'ministry', 'authority', 'board', 'singapore']
        if any(indicator in source for indicator in gov_indicators):
            explanation['secondary_reasons'].append(
                self.explanation_templates['source_match'].format(
                    source=dataset.get('source', 'Official Source')
                )
            )
            explanation['match_details']['source_credibility'] = {
                'source_type': 'government',
                'credibility': 'high'
            }
        
        return explanation
    
    def _analyze_domain_relevance(self, query: str, dataset: Dict, 
                                explanation: Dict) -> Dict:
        """Analyze Singapore domain relevance."""
        
        singapore_indicators = ['singapore', 'sg', 'republic of singapore']
        query_lower = query.lower()
        
        # Check if query is Singapore-specific
        singapore_query = any(indicator in query_lower for indicator in singapore_indicators)
        
        # Check if dataset is Singapore-related
        dataset_text = f"{dataset.get('title', '')} {dataset.get('description', '')} {dataset.get('source', '')}".lower()
        singapore_dataset = any(indicator in dataset_text for indicator in singapore_indicators)
        
        if singapore_query and singapore_dataset:
            explanation['primary_reasons'].append(
                self.explanation_templates['domain_expertise']
            )
            explanation['match_details']['geographic_relevance'] = {
                'query_location_specific': True,
                'dataset_location_match': True,
                'location': 'Singapore'
            }
        elif singapore_dataset and not singapore_query:
            # Dataset is Singapore-specific but query wasn't - still relevant for local context
            explanation['secondary_reasons'].append(
                self.explanation_templates['geographic_match'].format(
                    location='Singapore'
                )
            )
        
        return explanation
    
    def _analyze_user_preferences(self, dataset: Dict, user_context: Dict, 
                                explanation: Dict) -> Dict:
        """Analyze match with user preferences and history."""
        
        if not user_context:
            return explanation
        
        # Check user's preferred domains
        preferred_terms = user_context.get('preferred_query_terms', [])
        if preferred_terms:
            dataset_text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
            
            matching_preferences = []
            for term, count in preferred_terms[:5]:  # Top 5 preferences
                if term.lower() in dataset_text:
                    matching_preferences.append(term)
            
            if matching_preferences:
                explanation['secondary_reasons'].append(
                    self.explanation_templates['user_preference'].format(
                        domain=', '.join(matching_preferences)
                    )
                )
                explanation['match_details']['user_preference_match'] = {
                    'matching_terms': matching_preferences,
                    'user_engagement_score': user_context.get('engagement_score', 0)
                }
        
        return explanation
    
    def _calculate_confidence_level(self, explanation: Dict) -> str:
        """Calculate overall confidence level of the recommendation."""
        
        primary_count = len(explanation['primary_reasons'])
        secondary_count = len(explanation['secondary_reasons'])
        similarity_score = explanation.get('similarity_score', 0)
        
        # Calculate confidence based on multiple factors
        confidence_score = 0
        
        # Primary reasons contribute most
        confidence_score += primary_count * 0.4
        
        # Secondary reasons contribute moderately  
        confidence_score += secondary_count * 0.2
        
        # Similarity score contributes
        confidence_score += similarity_score * 0.4
        
        # Map to confidence levels
        if confidence_score >= 0.8:
            return 'very_high'
        elif confidence_score >= 0.6:
            return 'high'
        elif confidence_score >= 0.4:
            return 'medium'
        elif confidence_score >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_explanation_text(self, explanation: Dict) -> str:
        """Generate human-readable explanation text."""
        
        title = explanation['dataset_title']
        confidence = explanation['confidence_level']
        primary_reasons = explanation['primary_reasons']
        secondary_reasons = explanation['secondary_reasons']
        
        # Start with confidence indicator
        confidence_phrases = {
            'very_high': f'"{title}" is an excellent match for your query.',
            'high': f'"{title}" is a good match for your query.',
            'medium': f'"{title}" is a reasonable match for your query.',
            'low': f'"{title}" might be relevant to your query.',
            'very_low': f'"{title}" has limited relevance to your query.'
        }
        
        text = confidence_phrases.get(confidence, f'"{title}" was recommended for your query.')
        
        # Add primary reasons
        if primary_reasons:
            if len(primary_reasons) == 1:
                text += f" {primary_reasons[0]}."
            else:
                text += f" Key reasons: {'; '.join(primary_reasons)}."
        
        # Add secondary reasons if space permits
        if secondary_reasons and len(text) < 200:
            additional = secondary_reasons[0]  # Add most important secondary reason
            text += f" Additionally, {additional.lower()}."
        
        return text
    
    def explain_ranking(self, query: str, ranked_results: List[Dict], 
                       top_n: int = 5) -> List[Dict]:
        """
        Explain why results are ranked in their order.
        
        Args:
            query: User's search query  
            ranked_results: List of ranked recommendation results
            top_n: Number of top results to explain
            
        Returns:
            List of explanations for top N results
        """
        logger.info(f"🔍 Explaining ranking for top {top_n} results")
        
        explanations = []
        
        for i, result in enumerate(ranked_results[:top_n]):
            ranking_explanation = {
                'rank': i + 1,
                'dataset_title': result.get('title', 'Unknown'),
                'score': result.get('score', 0),
                'ranking_factors': [],
                'comparison_notes': []
            }
            
            # Explain why this rank
            if i == 0:
                ranking_explanation['ranking_factors'].append("Highest overall similarity score")
            elif i < 3:
                ranking_explanation['ranking_factors'].append("Strong relevance to your query")
            else:
                ranking_explanation['ranking_factors'].append("Good potential match")
            
            # Compare with previous result if not first
            if i > 0:
                prev_score = ranked_results[i-1].get('score', 0)
                current_score = result.get('score', 0)
                score_diff = prev_score - current_score
                
                if score_diff < 0.1:
                    ranking_explanation['comparison_notes'].append("Very similar relevance to higher-ranked result")
                elif score_diff < 0.2:
                    ranking_explanation['comparison_notes'].append("Slightly lower relevance than previous result")
                else:
                    ranking_explanation['comparison_notes'].append("Notably lower relevance than higher-ranked results")
            
            explanations.append(ranking_explanation)
        
        return explanations


def demo_explanation_engine():
    """Demonstrate the explanation engine with sample data."""
    print("🔄 Initializing Explanation Engine Demo")
    
    explainer = RecommendationExplainer()
    
    # Sample dataset
    sample_dataset = {
        'title': 'HDB Resale Flat Prices',
        'description': 'Comprehensive data on HDB resale flat transactions including prices, locations, and flat characteristics in Singapore.',
        'category': 'housing',
        'source': 'Housing Development Board Singapore',
        'quality_score': 0.92,
        'tags': 'housing, property, resale, HDB, singapore'
    }
    
    # Sample user context
    user_context = {
        'preferred_query_terms': [('housing', 5), ('singapore', 4), ('property', 3)],
        'engagement_score': 0.75
    }
    
    # Test explanation
    query = "singapore housing market data"
    explanation = explainer.explain_recommendation(
        query=query,
        recommended_dataset=sample_dataset,
        all_datasets=pd.DataFrame(),  # Empty for demo
        similarity_score=0.89,
        user_context=user_context
    )
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📊 Dataset: {sample_dataset['title']}")
    print(f"⭐ Similarity Score: {explanation['similarity_score']:.2f}")
    print(f"🎯 Confidence: {explanation['confidence_level']}")
    print(f"\n📝 Explanation:")
    print(f"{explanation['explanation_text']}")
    
    print(f"\n🔍 Primary Reasons:")
    for reason in explanation['primary_reasons']:
        print(f"  • {reason}")
    
    print(f"\n🔍 Secondary Reasons:")
    for reason in explanation['secondary_reasons']:
        print(f"  • {reason}")
    
    print(f"\n📋 Technical Details:")
    for key, value in explanation['technical_details'].items():
        print(f"  {key}: {value}")
    
    print("\n✅ Explanation demo complete!")


if __name__ == "__main__":
    demo_explanation_engine()