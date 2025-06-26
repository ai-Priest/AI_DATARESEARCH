"""
Progressive Search System
Provides intelligent search suggestions and real-time query assistance.
"""

import logging
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ProgressiveSearchEngine:
    """Intelligent progressive search with autocomplete and suggestions."""
    
    def __init__(self):
        self.vocabulary = set()
        self.popular_queries = Counter()
        self.query_suggestions = defaultdict(list)
        self.category_terms = {}
        self.autocomplete_trie = {}
        self.singapore_terms = set()
        self.abbreviation_map = {}
        
    def initialize_from_datasets(self, datasets_df: pd.DataFrame):
        """Initialize search engine with dataset vocabulary."""
        logger.info("🔄 Initializing progressive search from datasets")
        
        # Build vocabulary from all text fields
        all_text = []
        
        for _, row in datasets_df.iterrows():
            text_fields = [
                str(row.get('title', '')),
                str(row.get('description', '')),
                str(row.get('tags', '')),
                str(row.get('category', ''))
            ]
            all_text.extend([field for field in text_fields if field and field != 'nan'])
        
        # Extract vocabulary
        for text in all_text:
            words = re.findall(r'\b\w{2,}\b', text.lower())
            self.vocabulary.update(words)
        
        # Build category-specific terms
        self._build_category_terms(datasets_df)
        
        # Build Singapore-specific terms
        self._build_singapore_terms()
        
        # Build autocomplete trie
        self._build_autocomplete_trie()
        
        logger.info(f"✅ Initialized with {len(self.vocabulary)} terms")
    
    def _build_category_terms(self, datasets_df: pd.DataFrame):
        """Build category-specific term mappings."""
        
        for category in datasets_df['category'].dropna().unique():
            category_datasets = datasets_df[datasets_df['category'] == category]
            
            # Extract terms specific to this category
            category_text = []
            for _, row in category_datasets.iterrows():
                text = f"{row.get('title', '')} {row.get('description', '')} {row.get('tags', '')}"
                category_text.append(text)
            
            # Find characteristic terms
            all_text = ' '.join(category_text).lower()
            terms = re.findall(r'\b\w{3,}\b', all_text)
            term_counts = Counter(terms)
            
            # Keep most common terms for this category
            self.category_terms[category.lower()] = [
                term for term, count in term_counts.most_common(20)
                if len(term) > 2
            ]
    
    def _build_singapore_terms(self):
        """Build Singapore-specific vocabulary."""
        
        singapore_vocabulary = {
            # Government agencies
            'hdb', 'lta', 'ura', 'nea', 'mom', 'moh', 'moe', 'psd',
            
            # Common terms
            'singapore', 'government', 'ministry', 'authority', 'board',
            'resale', 'flat', 'housing', 'transport', 'traffic',
            'mrt', 'bus', 'taxi', 'population', 'demographics',
            'environment', 'weather', 'planning', 'development',
            
            # Abbreviations
            'sg', 'gov', 'cpf', 'gst', 'coe', 'bto'
        }
        
        self.singapore_terms = singapore_vocabulary
        
        # Build abbreviation map
        abbreviations = {
            'hdb': 'housing development board',
            'lta': 'land transport authority', 
            'ura': 'urban redevelopment authority',
            'nea': 'national environment agency',
            'mom': 'ministry of manpower',
            'moh': 'ministry of health',
            'moe': 'ministry of education',
            'mrt': 'mass rapid transit',
            'cpf': 'central provident fund',
            'gst': 'goods and services tax',
            'coe': 'certificate of entitlement',
            'bto': 'build to order',
            'sg': 'singapore'
        }
        
        self.abbreviation_map = abbreviations
    
    def _build_autocomplete_trie(self):
        """Build trie structure for fast autocomplete."""
        
        self.autocomplete_trie = {}
        
        for word in self.vocabulary:
            if len(word) < 3:  # Skip very short words
                continue
                
            current = self.autocomplete_trie
            for char in word.lower():
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_end'] = word
    
    def get_autocomplete_suggestions(self, partial_query: str, max_suggestions: int = 8) -> List[Dict]:
        """
        Get autocomplete suggestions for partial query.
        
        Args:
            partial_query: Partial user input
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries
        """
        if len(partial_query) < 2:
            return self._get_popular_starter_suggestions()
        
        suggestions = []
        words = partial_query.lower().split()
        
        if not words:
            return []
        
        last_word = words[-1]
        completed_words = words[:-1]
        
        # Get suggestions for the last (incomplete) word
        word_suggestions = self._get_word_completions(last_word, max_suggestions * 2)
        
        # Build full query suggestions
        for suggestion in word_suggestions:
            full_suggestion = ' '.join(completed_words + [suggestion])
            
            suggestion_obj = {
                'text': full_suggestion,
                'type': 'autocomplete',
                'confidence': self._calculate_suggestion_confidence(suggestion, partial_query),
                'preview': self._get_suggestion_preview(full_suggestion)
            }
            
            suggestions.append(suggestion_obj)
        
        # Add semantic suggestions
        semantic_suggestions = self._get_semantic_suggestions(partial_query, max_suggestions // 2)
        suggestions.extend(semantic_suggestions)
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions[:max_suggestions]
    
    def _get_word_completions(self, partial_word: str, max_completions: int) -> List[str]:
        """Get word completions using trie structure."""
        
        if len(partial_word) < 2:
            return []
        
        # Navigate to the partial word in trie
        current = self.autocomplete_trie
        for char in partial_word.lower():
            if char in current:
                current = current[char]
            else:
                return []  # No completions found
        
        # Collect all completions from this point
        completions = []
        self._collect_completions(current, completions, max_completions)
        
        return completions[:max_completions]
    
    def _collect_completions(self, trie_node: Dict, completions: List[str], max_completions: int):
        """Recursively collect completions from trie node."""
        
        if len(completions) >= max_completions:
            return
        
        if '_end' in trie_node:
            completions.append(trie_node['_end'])
        
        for char, child_node in trie_node.items():
            if char != '_end':
                self._collect_completions(child_node, completions, max_completions)
    
    def _get_semantic_suggestions(self, query: str, max_suggestions: int) -> List[Dict]:
        """Get semantically related suggestions."""
        
        suggestions = []
        query_lower = query.lower()
        
        # Singapore-specific suggestions
        if any(term in query_lower for term in ['sg', 'singapore', 'government']):
            sg_suggestions = [
                'singapore government data',
                'singapore housing statistics', 
                'singapore transport information',
                'singapore population data',
                'singapore economic indicators'
            ]
            
            for sugg in sg_suggestions[:max_suggestions//2]:
                if query_lower not in sugg.lower():
                    suggestions.append({
                        'text': sugg,
                        'type': 'semantic_singapore',
                        'confidence': 0.8,
                        'preview': f'Singapore government datasets about {sugg.split()[-1]}'
                    })
        
        # Category-based suggestions
        for category, terms in self.category_terms.items():
            if any(term in query_lower for term in terms[:5]):  # Check top terms
                category_suggestion = f"{query} {category}"
                if category_suggestion not in [s['text'] for s in suggestions]:
                    suggestions.append({
                        'text': category_suggestion,
                        'type': f'category_{category}',
                        'confidence': 0.7,
                        'preview': f'Related {category} datasets'
                    })
                    break  # Only add one category suggestion
        
        return suggestions[:max_suggestions]
    
    def _get_popular_starter_suggestions(self) -> List[Dict]:
        """Get popular starting suggestions when user hasn't typed much."""
        
        popular_starters = [
            ('singapore housing data', 'housing', 'Housing and property datasets'),
            ('transport statistics', 'transport', 'Transportation and traffic data'),
            ('population demographics', 'population', 'Population and census data'),
            ('government budget', 'government', 'Government financial data'),
            ('environment data', 'environment', 'Environmental monitoring data'),
            ('economic indicators', 'economy', 'Economic and trade statistics')
        ]
        
        suggestions = []
        for text, category, preview in popular_starters:
            suggestions.append({
                'text': text,
                'type': f'popular_{category}',
                'confidence': 0.9,
                'preview': preview
            })
        
        return suggestions
    
    def _calculate_suggestion_confidence(self, suggestion: str, partial_query: str) -> float:
        """Calculate confidence score for a suggestion."""
        
        # Base confidence on string similarity
        similarity = SequenceMatcher(None, suggestion.lower(), partial_query.lower()).ratio()
        
        # Boost for Singapore-specific terms
        if any(term in suggestion.lower() for term in self.singapore_terms):
            similarity += 0.2
        
        # Boost for complete words vs partial matches
        if partial_query.lower() in suggestion.lower():
            similarity += 0.1
        
        return min(1.0, similarity)
    
    def _get_suggestion_preview(self, suggestion: str) -> str:
        """Generate preview text for suggestion."""
        
        # Check for category matches
        suggestion_lower = suggestion.lower()
        
        category_previews = {
            'housing': 'Housing, property, and residential data',
            'transport': 'Transportation, traffic, and mobility data', 
            'population': 'Demographics and population statistics',
            'environment': 'Environmental and climate data',
            'government': 'Government operations and policy data',
            'economic': 'Economic indicators and financial data'
        }
        
        for category, preview in category_previews.items():
            if category in suggestion_lower:
                return preview
        
        # Default preview
        return f'Datasets related to {suggestion}'
    
    def get_query_refinement_suggestions(self, query: str, num_results: int) -> List[Dict]:
        """
        Suggest query refinements when search returns too few/many results.
        
        Args:
            query: Current search query
            num_results: Number of results returned
            
        Returns:
            List of refinement suggestions
        """
        
        suggestions = []
        
        if num_results == 0:
            # Too few results - suggest broader queries
            suggestions.extend(self._suggest_broader_queries(query))
        elif num_results > 50:
            # Too many results - suggest narrower queries  
            suggestions.extend(self._suggest_narrower_queries(query))
        elif num_results < 5:
            # Very few results - suggest alternatives
            suggestions.extend(self._suggest_alternative_queries(query))
        
        return suggestions
    
    def _suggest_broader_queries(self, query: str) -> List[Dict]:
        """Suggest broader queries when no results found."""
        
        suggestions = []
        query_words = query.lower().split()
        
        # Remove specific terms to broaden
        if len(query_words) > 1:
            for i in range(len(query_words)):
                broader_query = ' '.join(query_words[:i] + query_words[i+1:])
                suggestions.append({
                    'text': broader_query,
                    'type': 'broader',
                    'reason': f'Try removing "{query_words[i]}" to find more results'
                })
        
        # Add general categories
        general_terms = ['data', 'statistics', 'information', 'singapore']
        for term in general_terms:
            if term not in query.lower():
                broader_query = f"{query} {term}"
                suggestions.append({
                    'text': broader_query,
                    'type': 'broader_category',
                    'reason': f'Add "{term}" to expand search scope'
                })
        
        return suggestions[:3]
    
    def _suggest_narrower_queries(self, query: str) -> List[Dict]:
        """Suggest narrower queries when too many results."""
        
        suggestions = []
        
        # Add specific terms to narrow down
        specific_terms = {
            'housing': ['resale', 'rental', 'public', 'private'],
            'transport': ['traffic', 'ridership', 'routes', 'statistics'],
            'population': ['age', 'income', 'education', 'employment'],
            'government': ['budget', 'expenditure', 'revenue', 'departments']
        }
        
        query_lower = query.lower()
        for category, terms in specific_terms.items():
            if category in query_lower:
                for term in terms:
                    if term not in query_lower:
                        narrower_query = f"{query} {term}"
                        suggestions.append({
                            'text': narrower_query,
                            'type': 'narrower',
                            'reason': f'Add "{term}" to focus on specific aspect'
                        })
                        break  # Only suggest one term per category
        
        # Add time-based narrowing
        time_terms = ['2023', '2024', 'latest', 'recent', 'annual']
        for term in time_terms:
            if term not in query_lower:
                narrower_query = f"{query} {term}"
                suggestions.append({
                    'text': narrower_query,
                    'type': 'temporal',
                    'reason': f'Add "{term}" to get more recent data'
                })
                break
        
        return suggestions[:3]
    
    def _suggest_alternative_queries(self, query: str) -> List[Dict]:
        """Suggest alternative queries when few results."""
        
        suggestions = []
        
        # Expand abbreviations
        query_words = query.lower().split()
        for i, word in enumerate(query_words):
            if word in self.abbreviation_map:
                expanded = self.abbreviation_map[word]
                new_query_words = query_words.copy()
                new_query_words[i] = expanded
                alternative = ' '.join(new_query_words)
                
                suggestions.append({
                    'text': alternative,
                    'type': 'abbreviation_expansion',
                    'reason': f'Expand "{word}" to "{expanded}"'
                })
        
        # Suggest synonyms
        synonym_map = {
            'housing': ['property', 'residential', 'accommodation'],
            'transport': ['transportation', 'mobility', 'traffic'],
            'population': ['demographics', 'residents', 'people'],
            'government': ['public sector', 'administration', 'official'],
            'data': ['statistics', 'information', 'records']
        }
        
        for word in query_words:
            if word in synonym_map:
                for synonym in synonym_map[word]:
                    alternative = query.replace(word, synonym)
                    suggestions.append({
                        'text': alternative,
                        'type': 'synonym',
                        'reason': f'Try "{synonym}" instead of "{word}"'
                    })
                    break  # Only one synonym per word
        
        return suggestions[:3]


def demo_progressive_search():
    """Demonstrate progressive search capabilities."""
    print("🔄 Initializing Progressive Search Demo")
    
    # Create sample dataset
    sample_data = {
        'title': [
            'HDB Resale Flat Prices',
            'LTA Traffic Volume Data',
            'Population Census 2020',
            'Government Budget 2024',
            'Environment Air Quality Index'
        ],
        'description': [
            'Housing resale transaction data from HDB',
            'Traffic statistics from Land Transport Authority', 
            'Singapore population demographics',
            'Annual government budget allocation',
            'Air quality monitoring data'
        ],
        'category': ['housing', 'transport', 'population', 'government', 'environment'],
        'tags': [
            'housing property resale HDB',
            'transport traffic LTA statistics',
            'population demographics census',
            'government budget finance',
            'environment air quality NEA'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize search engine
    search_engine = ProgressiveSearchEngine()
    search_engine.initialize_from_datasets(df)
    
    # Test autocomplete
    print("\n🔍 Testing Autocomplete:")
    test_queries = ['hou', 'trans', 'gov', 'sing']
    
    for query in test_queries:
        suggestions = search_engine.get_autocomplete_suggestions(query, max_suggestions=5)
        print(f"\nQuery: '{query}'")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['text']} ({suggestion['type']}, confidence: {suggestion['confidence']:.2f})")
            print(f"     Preview: {suggestion['preview']}")
    
    # Test query refinement
    print("\n🔍 Testing Query Refinement:")
    
    # Test with zero results
    refinements = search_engine.get_query_refinement_suggestions("nonexistent term", num_results=0)
    print(f"\nZero results for 'nonexistent term':")
    for ref in refinements:
        print(f"  • {ref['text']} - {ref['reason']}")
    
    # Test with too many results
    refinements = search_engine.get_query_refinement_suggestions("data", num_results=100)
    print(f"\nToo many results for 'data':")
    for ref in refinements:
        print(f"  • {ref['text']} - {ref['reason']}")
    
    print("\n✅ Progressive search demo complete!")


if __name__ == "__main__":
    demo_progressive_search()