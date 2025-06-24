"""
Intelligent Query Expansion for Dataset Search
Enhances user queries with related terms, synonyms, and domain-specific expansions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
import json
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class QueryExpander:
    """Intelligent query expansion for dataset search."""
    
    def __init__(self):
        self.domain_vocabulary = {}
        self.keyword_associations = {}
        self.category_keywords = {}
        self.abbreviation_map = {}
        self.singapore_terms = {}
        self.vectorizer = None
        self.keyword_vectors = None
        
    def load_datasets(self, singapore_path: str, global_path: str) -> pd.DataFrame:
        """Load datasets and build expansion vocabulary."""
        logger.info("📊 Loading datasets for query expansion")
        
        singapore_df = pd.read_csv(singapore_path)
        global_df = pd.read_csv(global_path)
        combined_df = pd.concat([singapore_df, global_df], ignore_index=True)
        
        logger.info(f"✅ Loaded {len(combined_df)} datasets")
        return combined_df
    
    def build_domain_vocabulary(self, df: pd.DataFrame):
        """Build domain-specific vocabulary from dataset content."""
        logger.info("🔄 Building domain vocabulary")
        
        # Extract all text content
        all_text = []
        for _, row in df.iterrows():
            text_parts = [
                str(row.get('title', '')),
                str(row.get('description', '')),
                str(row.get('tags', '')),
                str(row.get('category', ''))
            ]
            all_text.extend([part for part in text_parts if part and part != 'nan'])
        
        # Build TF-IDF vocabulary - each text part separately
        documents = [text for text in all_text if len(text.strip()) > 5]
        
        if len(documents) < 5:
            logger.warning("Not enough documents for TF-IDF, using simple vocabulary")
            # Fallback: simple word extraction
            words = set()
            for doc in documents:
                words.update(re.findall(r'\b\w{3,}\b', doc.lower()))
            self.domain_vocabulary = {word: idx for idx, word in enumerate(list(words)[:1000])}
            return
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.vectorizer.fit(documents[:100])  # Use first 100 documents
        
        # Get feature names (vocabulary)
        feature_names = self.vectorizer.get_feature_names_out()
        self.domain_vocabulary = {term: idx for idx, term in enumerate(feature_names)}
        
        logger.info(f"✅ Built vocabulary with {len(self.domain_vocabulary)} terms")
    
    def build_keyword_associations(self, df: pd.DataFrame):
        """Build keyword co-occurrence associations."""
        logger.info("🔄 Building keyword associations")
        
        # Track keyword co-occurrences
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            # Extract keywords from tags and description
            keywords = set()
            
            # From tags
            if pd.notna(row.get('tags')):
                tags = str(row['tags']).lower().split()
                keywords.update([tag.strip() for tag in tags if len(tag.strip()) > 2])
            
            # From title (important words)
            if pd.notna(row.get('title')):
                title_words = re.findall(r'\b\w{3,}\b', str(row['title']).lower())
                keywords.update(title_words)
            
            # From category
            if pd.notna(row.get('category')):
                category_words = re.findall(r'\b\w{3,}\b', str(row['category']).lower())
                keywords.update(category_words)
            
            # Build co-occurrence matrix
            keyword_list = list(keywords)
            for i, kw1 in enumerate(keyword_list):
                for kw2 in keyword_list[i+1:]:
                    cooccurrence[kw1][kw2] += 1
                    cooccurrence[kw2][kw1] += 1
        
        # Convert to association scores
        for kw1 in cooccurrence:
            total_occurrences = sum(cooccurrence[kw1].values())
            if total_occurrences > 0:
                self.keyword_associations[kw1] = {
                    kw2: count / total_occurrences 
                    for kw2, count in cooccurrence[kw1].items()
                    if count > 1  # Minimum co-occurrence threshold
                }
        
        logger.info(f"✅ Built associations for {len(self.keyword_associations)} keywords")
    
    def build_category_keywords(self, df: pd.DataFrame):
        """Build category-specific keyword mappings."""
        logger.info("🔄 Building category keyword mappings")
        
        category_text = defaultdict(list)
        
        for _, row in df.iterrows():
            category = str(row.get('category', 'unknown')).lower()
            
            # Collect text for each category
            text_parts = [
                str(row.get('title', '')),
                str(row.get('description', '')),
                str(row.get('tags', ''))
            ]
            
            text = ' '.join([part for part in text_parts if part and part != 'nan'])
            if text:
                category_text[category].append(text)
        
        # Extract top keywords for each category
        for category, texts in category_text.items():
            if len(texts) < 2:
                continue
                
            # Use TF-IDF to find characteristic terms
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores
                avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top keywords for this category
                top_indices = np.argsort(avg_scores)[-10:]  # Top 10
                top_keywords = [feature_names[i] for i in top_indices]
                
                self.category_keywords[category] = top_keywords
                
            except Exception as e:
                logger.warning(f"Could not process category {category}: {e}")
        
        logger.info(f"✅ Built keyword mappings for {len(self.category_keywords)} categories")
    
    def build_singapore_terms(self):
        """Build Singapore-specific term mappings and expansions."""
        logger.info("🔄 Building Singapore-specific terms")
        
        # Singapore government agencies and terms
        self.singapore_terms = {
            # Agencies
            'hdb': ['housing development board', 'public housing', 'resale flat', 'bto'],
            'lta': ['land transport authority', 'transport', 'traffic', 'mrt', 'bus'],
            'ura': ['urban redevelopment authority', 'planning', 'development', 'zoning'],
            'nea': ['national environment agency', 'environment', 'weather', 'pollution'],
            'mom': ['ministry of manpower', 'employment', 'workforce', 'salary'],
            'moh': ['ministry of health', 'healthcare', 'hospital', 'medical'],
            'moe': ['ministry of education', 'school', 'education', 'student'],
            'psd': ['public service division', 'government', 'civil service'],
            
            # Singapore-specific terms
            'singapore': ['sg', 'republic of singapore', 'singapore government'],
            'government': ['gov', 'public sector', 'ministry', 'statutory board'],
            'planning': ['urban planning', 'development', 'master plan', 'land use'],
            'transport': ['transportation', 'mobility', 'public transport', 'traffic'],
            'housing': ['residential', 'property', 'real estate', 'accommodation'],
            'population': ['demographics', 'residents', 'citizens', 'inhabitants'],
            'economy': ['economic', 'gdp', 'trade', 'business', 'finance'],
            'environment': ['environmental', 'green', 'sustainability', 'climate'],
            
            # Common abbreviations
            'mrt': ['mass rapid transit', 'train', 'subway', 'metro'],
            'gst': ['goods and services tax', 'tax', 'taxation'],
            'cpf': ['central provident fund', 'retirement', 'savings'],
            'coe': ['certificate of entitlement', 'vehicle quota', 'car ownership']
        }
        
        # Build abbreviation map
        for term, expansions in self.singapore_terms.items():
            for expansion in expansions:
                self.abbreviation_map[expansion] = term
        
        logger.info(f"✅ Built Singapore terms with {len(self.singapore_terms)} mappings")
    
    def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, any]:
        """
        Expand a user query with related terms and context.
        
        Args:
            query: Original user query
            max_expansions: Maximum number of expansion terms
            
        Returns:
            Dictionary with expanded query components
        """
        logger.info(f"🔍 Expanding query: '{query}'")
        
        original_query = query.lower().strip()
        expanded_terms = set()
        expansion_sources = []
        
        # 1. Singapore-specific expansions
        singapore_expansions = self._expand_singapore_terms(original_query)
        expanded_terms.update(singapore_expansions['terms'])
        if singapore_expansions['terms']:
            expansion_sources.append('singapore_domain')
        
        # 2. Keyword association expansions
        association_expansions = self._expand_by_associations(original_query, max_expansions//2)
        expanded_terms.update(association_expansions)
        if association_expansions:
            expansion_sources.append('keyword_associations')
        
        # 3. Category-based expansions
        category_expansions = self._expand_by_category(original_query, max_expansions//3)
        expanded_terms.update(category_expansions)
        if category_expansions:
            expansion_sources.append('category_context')
        
        # 4. Abbreviation expansions
        abbrev_expansions = self._expand_abbreviations(original_query)
        expanded_terms.update(abbrev_expansions)
        if abbrev_expansions:
            expansion_sources.append('abbreviations')
        
        # Build final expanded query
        final_terms = [original_query] + list(expanded_terms)[:max_expansions]
        expanded_query = ' '.join(final_terms)
        
        result = {
            'original_query': query,
            'expanded_query': expanded_query,
            'expansion_terms': list(expanded_terms)[:max_expansions],
            'expansion_sources': expansion_sources,
            'total_expansions': len(expanded_terms)
        }
        
        logger.info(f"✅ Expanded to {len(expanded_terms)} terms from {len(expansion_sources)} sources")
        return result
    
    def _expand_singapore_terms(self, query: str) -> Dict[str, List[str]]:
        """Expand Singapore-specific terms and abbreviations."""
        expansions = []
        
        for term, related_terms in self.singapore_terms.items():
            if term in query or any(rt in query for rt in related_terms):
                # Add related terms that aren't already in query
                for rt in related_terms:
                    if rt not in query and len(rt) > 2:
                        expansions.append(rt)
        
        return {'terms': expansions[:3], 'source': 'singapore_domain'}
    
    def _expand_by_associations(self, query: str, max_terms: int) -> List[str]:
        """Expand using keyword associations."""
        expansions = []
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        for word in query_words:
            if word in self.keyword_associations:
                # Get most associated terms
                associations = self.keyword_associations[word]
                sorted_associations = sorted(associations.items(), key=lambda x: x[1], reverse=True)
                
                for assoc_word, score in sorted_associations[:2]:  # Top 2 per word
                    if assoc_word not in query and score > 0.1:
                        expansions.append(assoc_word)
        
        return expansions[:max_terms]
    
    def _expand_by_category(self, query: str, max_terms: int) -> List[str]:
        """Expand using category-specific keywords."""
        expansions = []
        
        # Find the most relevant category
        best_category = None
        best_overlap = 0
        
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        for category, keywords in self.category_keywords.items():
            overlap = len(query_words.intersection(set(keywords)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_category = category
        
        # Add keywords from best matching category
        if best_category and best_overlap > 0:
            category_keywords = self.category_keywords[best_category]
            for keyword in category_keywords[:max_terms]:
                if keyword not in query:
                    expansions.append(keyword)
        
        return expansions[:max_terms]
    
    def _expand_abbreviations(self, query: str) -> List[str]:
        """Expand abbreviations and acronyms."""
        expansions = []
        
        # Check for abbreviations to expand
        for abbrev, full_form in self.abbreviation_map.items():
            if abbrev in query.lower() and full_form not in query.lower():
                expansions.append(full_form)
        
        # Check for full forms to abbreviate
        for full_form, abbrev in self.singapore_terms.items():
            if full_form in query.lower():
                expansions.extend([a for a in abbrev if a not in query.lower()][:2])
        
        return expansions[:3]
    
    def get_expansion_explanation(self, expansion_result: Dict) -> str:
        """Generate human-readable explanation of query expansion."""
        explanations = []
        
        if 'singapore_domain' in expansion_result['expansion_sources']:
            explanations.append("Added Singapore government context")
        
        if 'keyword_associations' in expansion_result['expansion_sources']:
            explanations.append("Included related terms from dataset relationships")
        
        if 'category_context' in expansion_result['expansion_sources']:
            explanations.append("Added category-specific keywords")
        
        if 'abbreviations' in expansion_result['expansion_sources']:
            explanations.append("Expanded abbreviations and acronyms")
        
        if not explanations:
            return "No expansions applied - query was specific enough"
        
        return "Query enhanced by: " + ", ".join(explanations)


def initialize_query_expander():
    """Initialize and train the query expander."""
    logger.info("🚀 Initializing Query Expander")
    
    expander = QueryExpander()
    
    # Load datasets
    df = expander.load_datasets(
        "data/processed/singapore_datasets.csv",
        "data/processed/global_datasets.csv"
    )
    
    # Build vocabularies
    expander.build_domain_vocabulary(df)
    expander.build_keyword_associations(df)
    expander.build_category_keywords(df)
    expander.build_singapore_terms()
    
    logger.info("✅ Query Expander initialized successfully")
    return expander


def test_query_expansion():
    """Test the query expansion system."""
    expander = initialize_query_expander()
    
    test_queries = [
        "housing data",
        "LTA traffic information", 
        "government budget",
        "transport planning",
        "HDB resale prices",
        "population statistics",
        "environment data"
    ]
    
    print("🧪 Testing Query Expansion:")
    print("=" * 50)
    
    for query in test_queries:
        result = expander.expand_query(query, max_expansions=4)
        explanation = expander.get_expansion_explanation(result)
        
        print(f"\nOriginal: '{query}'")
        print(f"Expanded: '{result['expanded_query']}'")
        print(f"Added terms: {result['expansion_terms']}")
        print(f"Explanation: {explanation}")
        print("-" * 30)


if __name__ == "__main__":
    test_query_expansion()