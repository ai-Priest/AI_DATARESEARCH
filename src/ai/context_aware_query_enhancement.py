"""
Context-Aware Query Enhancement
Implements intelligent query expansion, refinement suggestions, and geographic context enhancement
based on successful mappings from training_mappings.md and domain-specific terminology.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class QueryEnhancement:
    """Result of query enhancement with expansion and refinement suggestions"""
    original_query: str
    enhanced_query: str
    expansion_terms: List[str]
    refinement_suggestions: List[str]
    geographic_context: Dict[str, any]
    domain_context: Dict[str, any]
    confidence_score: float
    enhancement_sources: List[str]
    explanation: str


@dataclass
class GeographicContext:
    """Geographic context information for queries"""
    detected_location: Optional[str]
    scope: str  # local, singapore, regional, global
    suggested_sources: List[str]
    location_terms: List[str]
    confidence: float


@dataclass
class DomainContext:
    """Domain-specific context information"""
    primary_domain: str
    secondary_domains: List[str]
    domain_terms: List[str]
    specialized_vocabulary: List[str]
    confidence: float


class ContextAwareQueryEnhancer:
    """Context-aware query enhancement using training mappings and domain knowledge"""
    
    def __init__(self, training_mappings_path: str = "training_mappings.md"):
        self.training_mappings_path = training_mappings_path
        
        # Load successful mappings from training data
        self.successful_mappings = {}
        self.domain_terminology = {}
        self.geographic_patterns = {}
        self.query_refinement_patterns = {}
        
        # Initialize enhancement components
        self._load_training_mappings()
        self._build_domain_terminology()
        self._build_geographic_patterns()
        self._build_refinement_patterns()
        
        logger.info(f"🎯 ContextAwareQueryEnhancer initialized")
        logger.info(f"  Training mappings: {len(self.successful_mappings)}")
        logger.info(f"  Domain terminologies: {len(self.domain_terminology)}")
        logger.info(f"  Geographic patterns: {len(self.geographic_patterns)}")
    
    def _load_training_mappings(self):
        """Load successful mappings from training_mappings.md"""
        try:
            if not Path(self.training_mappings_path).exists():
                logger.warning(f"Training mappings file not found: {self.training_mappings_path}")
                return
            
            with open(self.training_mappings_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse mappings by domain sections
            current_domain = None
            mappings = defaultdict(list)
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Detect domain headers
                if line.startswith('## ') and 'Queries' in line:
                    current_domain = line.replace('## ', '').replace(' Queries', '').lower()
                    continue
                
                # Parse mapping lines
                if line.startswith('- ') and '→' in line and current_domain:
                    try:
                        # Parse: query → source (score) - explanation
                        parts = line[2:].split('→')
                        if len(parts) == 2:
                            query = parts[0].strip()
                            rest = parts[1].strip()
                            
                            # Extract source and score
                            if '(' in rest and ')' in rest:
                                source_part = rest.split('(')[0].strip()
                                score_part = rest.split('(')[1].split(')')[0]
                                explanation = rest.split(')', 1)[1].strip(' -')
                                
                                try:
                                    score = float(score_part)
                                    mappings[current_domain].append({
                                        'query': query,
                                        'source': source_part,
                                        'score': score,
                                        'explanation': explanation,
                                        'domain': current_domain
                                    })
                                except ValueError:
                                    continue
                    except Exception as e:
                        logger.debug(f"Could not parse mapping line: {line} - {e}")
            
            self.successful_mappings = dict(mappings)
            logger.info(f"✅ Loaded {sum(len(v) for v in mappings.values())} training mappings")
            
        except Exception as e:
            logger.error(f"Error loading training mappings: {e}")
    
    def _build_domain_terminology(self):
        """Build domain-specific terminology from successful mappings"""
        domain_terms = defaultdict(set)
        
        for domain, mappings in self.successful_mappings.items():
            for mapping in mappings:
                query = mapping['query'].lower()
                
                # Extract key terms from successful queries
                terms = re.findall(r'\b\w{3,}\b', query)
                domain_terms[domain].update(terms)
                
                # Add explanation terms as domain vocabulary
                explanation = mapping['explanation'].lower()
                explanation_terms = re.findall(r'\b\w{4,}\b', explanation)
                domain_terms[domain].update(explanation_terms[:3])  # Top 3 explanation terms
        
        # Build specialized terminology mappings
        self.domain_terminology = {
            'psychology': {
                'core_terms': ['psychology', 'mental', 'health', 'behavioral', 'cognitive', 'psychological'],
                'related_terms': ['behavior', 'mind', 'brain', 'therapy', 'clinical', 'research', 'study'],
                'synonyms': {'psychology': ['psychological', 'mental health', 'behavioral science'],
                           'mental health': ['psychological wellbeing', 'mental wellness'],
                           'behavioral': ['behaviour', 'conduct', 'actions']},
                'expansion_terms': list(domain_terms.get('psychology', set()))
            },
            'machine learning': {
                'core_terms': ['machine', 'learning', 'ml', 'ai', 'artificial', 'intelligence', 'neural'],
                'related_terms': ['algorithm', 'model', 'training', 'prediction', 'classification', 'regression'],
                'synonyms': {'machine learning': ['ml', 'artificial intelligence', 'ai'],
                           'artificial intelligence': ['ai', 'machine intelligence'],
                           'neural networks': ['neural nets', 'deep learning']},
                'expansion_terms': list(domain_terms.get('machine learning', set()))
            },
            'climate': {
                'core_terms': ['climate', 'weather', 'environmental', 'temperature', 'change'],
                'related_terms': ['global warming', 'greenhouse', 'carbon', 'emissions', 'sustainability'],
                'synonyms': {'climate': ['weather patterns', 'atmospheric conditions'],
                           'environmental': ['ecological', 'green', 'sustainability'],
                           'climate change': ['global warming', 'climate crisis']},
                'expansion_terms': list(domain_terms.get('climate', set()))
            },
            'economics': {
                'core_terms': ['economic', 'economy', 'gdp', 'financial', 'trade', 'poverty'],
                'related_terms': ['finance', 'business', 'market', 'investment', 'growth', 'development'],
                'synonyms': {'economic': ['financial', 'monetary', 'fiscal'],
                           'gdp': ['gross domestic product', 'economic output'],
                           'trade': ['commerce', 'business', 'export', 'import']},
                'expansion_terms': list(domain_terms.get('economics', set()))
            },
            'singapore': {
                'core_terms': ['singapore', 'sg', 'hdb', 'mrt', 'lta', 'singstat'],
                'related_terms': ['government', 'public', 'transport', 'housing', 'statistics'],
                'synonyms': {'singapore': ['sg', 'republic of singapore'],
                           'hdb': ['housing development board', 'public housing'],
                           'lta': ['land transport authority', 'transport authority'],
                           'mrt': ['mass rapid transit', 'train', 'metro']},
                'expansion_terms': list(domain_terms.get('singapore-specific', set()))
            },
            'health': {
                'core_terms': ['health', 'medical', 'healthcare', 'disease', 'hospital'],
                'related_terms': ['medicine', 'treatment', 'patient', 'clinical', 'public health'],
                'synonyms': {'health': ['healthcare', 'medical', 'wellness'],
                           'medical': ['healthcare', 'clinical', 'therapeutic'],
                           'disease': ['illness', 'condition', 'disorder']},
                'expansion_terms': list(domain_terms.get('health', set()))
            },
            'education': {
                'core_terms': ['education', 'student', 'university', 'school', 'learning'],
                'related_terms': ['academic', 'curriculum', 'teaching', 'research', 'knowledge'],
                'synonyms': {'education': ['learning', 'schooling', 'academic'],
                           'student': ['learner', 'pupil', 'scholar'],
                           'university': ['college', 'higher education', 'tertiary']},
                'expansion_terms': list(domain_terms.get('education', set()))
            }
        }
        
        logger.info(f"✅ Built domain terminology for {len(self.domain_terminology)} domains")
    
    def _build_geographic_patterns(self):
        """Build geographic context patterns from successful mappings"""
        self.geographic_patterns = {
            'singapore_indicators': {
                'explicit': ['singapore', 'sg', 'republic of singapore'],
                'agencies': ['hdb', 'lta', 'ura', 'nea', 'mom', 'moh', 'moe', 'singstat'],
                'locations': ['orchard', 'marina bay', 'sentosa', 'changi', 'jurong'],
                'terms': ['government', 'public', 'national', 'ministry', 'authority', 'board'],
                'sources': ['data_gov_sg', 'singstat', 'lta_datamall']
            },
            'global_indicators': {
                'explicit': ['global', 'international', 'worldwide', 'world', 'countries'],
                'organizations': ['world bank', 'un', 'who', 'unesco', 'imf'],
                'terms': ['cross-country', 'comparative', 'international comparison'],
                'sources': ['world_bank', 'data_un', 'zenodo']
            },
            'regional_indicators': {
                'explicit': ['asia', 'asean', 'southeast asia', 'regional'],
                'terms': ['regional', 'neighboring', 'asia-pacific'],
                'sources': ['world_bank', 'zenodo']
            }
        }
        
        logger.info("✅ Built geographic context patterns")
    
    def _build_refinement_patterns(self):
        """Build query refinement patterns from successful mappings"""
        refinement_patterns = defaultdict(list)
        
        for domain, mappings in self.successful_mappings.items():
            high_score_mappings = [m for m in mappings if m['score'] >= 0.9]
            
            for mapping in high_score_mappings:
                query = mapping['query']
                source = mapping['source']
                
                # Build refinement suggestions based on successful patterns
                if 'research' not in query and source in ['zenodo', 'kaggle']:
                    refinement_patterns[domain].append(f"Add 'research' for academic datasets")
                
                if 'data' not in query:
                    refinement_patterns[domain].append(f"Add 'data' to find datasets")
                
                if domain == 'singapore' and not any(sg in query for sg in ['singapore', 'sg']):
                    refinement_patterns[domain].append(f"Add 'singapore' for local context")
        
        self.query_refinement_patterns = dict(refinement_patterns)
        logger.info(f"✅ Built refinement patterns for {len(refinement_patterns)} domains")
    
    def enhance_query(self, query: str, max_expansions: int = 5) -> QueryEnhancement:
        """
        Enhance query with context-aware expansion and refinement suggestions
        
        Args:
            query: Original user query
            max_expansions: Maximum number of expansion terms
            
        Returns:
            QueryEnhancement with expanded query and suggestions
        """
        logger.info(f"🎯 Enhancing query: '{query}'")
        
        original_query = query.strip()
        query_lower = original_query.lower()
        
        # 1. Detect domain context
        domain_context = self._detect_domain_context(query_lower)
        
        # 2. Detect geographic context
        geographic_context = self._detect_geographic_context(query_lower)
        
        # 3. Generate domain-specific expansions
        expansion_terms = self._generate_domain_expansions(
            query_lower, domain_context, max_expansions
        )
        
        # 4. Add geographic expansions
        geo_expansions = self._generate_geographic_expansions(
            query_lower, geographic_context
        )
        expansion_terms.extend(geo_expansions)
        
        # 5. Generate refinement suggestions
        refinement_suggestions = self._generate_refinement_suggestions(
            query_lower, domain_context, geographic_context
        )
        
        # 6. Build enhanced query
        enhanced_query = self._build_enhanced_query(
            original_query, expansion_terms[:max_expansions]
        )
        
        # 7. Calculate confidence and sources
        confidence_score = self._calculate_enhancement_confidence(
            domain_context, geographic_context, expansion_terms
        )
        
        enhancement_sources = []
        if domain_context.confidence > 0.5:
            enhancement_sources.append('domain_terminology')
        if geographic_context.confidence > 0.5:
            enhancement_sources.append('geographic_context')
        if expansion_terms:
            enhancement_sources.append('training_mappings')
        
        # 8. Generate explanation
        explanation = self._generate_enhancement_explanation(
            domain_context, geographic_context, expansion_terms, refinement_suggestions
        )
        
        enhancement = QueryEnhancement(
            original_query=original_query,
            enhanced_query=enhanced_query,
            expansion_terms=expansion_terms[:max_expansions],
            refinement_suggestions=refinement_suggestions,
            geographic_context=geographic_context.__dict__,
            domain_context=domain_context.__dict__,
            confidence_score=confidence_score,
            enhancement_sources=enhancement_sources,
            explanation=explanation
        )
        
        logger.info(f"✅ Enhanced query with {len(expansion_terms)} expansions, "
                   f"{len(refinement_suggestions)} suggestions, confidence: {confidence_score:.2f}")
        
        return enhancement
    
    def _detect_domain_context(self, query: str) -> DomainContext:
        """Detect domain context from query"""
        domain_scores = {}
        
        for domain, terminology in self.domain_terminology.items():
            score = 0.0
            matched_terms = []
            
            # Check core terms (higher weight)
            for term in terminology['core_terms']:
                if term in query:
                    score += 2.0
                    matched_terms.append(term)
            
            # Check related terms
            for term in terminology['related_terms']:
                if term in query:
                    score += 1.0
                    matched_terms.append(term)
            
            # Check synonyms
            for main_term, synonyms in terminology['synonyms'].items():
                if main_term in query:
                    score += 1.5
                    matched_terms.append(main_term)
                for synonym in synonyms:
                    if synonym in query:
                        score += 1.0
                        matched_terms.append(synonym)
            
            if score > 0:
                domain_scores[domain] = {
                    'score': score,
                    'matched_terms': matched_terms
                }
        
        # Find primary domain
        if domain_scores:
            primary_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d]['score'])
            primary_score = domain_scores[primary_domain]['score']
            
            # Find secondary domains
            secondary_domains = [
                domain for domain, data in domain_scores.items()
                if domain != primary_domain and data['score'] >= primary_score * 0.5
            ]
            
            confidence = min(1.0, primary_score / 3.0)  # Normalize confidence
            
            return DomainContext(
                primary_domain=primary_domain,
                secondary_domains=secondary_domains,
                domain_terms=domain_scores[primary_domain]['matched_terms'],
                specialized_vocabulary=self.domain_terminology[primary_domain]['expansion_terms'][:5],
                confidence=confidence
            )
        
        return DomainContext(
            primary_domain='general',
            secondary_domains=[],
            domain_terms=[],
            specialized_vocabulary=[],
            confidence=0.0
        )
    
    def _detect_geographic_context(self, query: str) -> GeographicContext:
        """Detect geographic context from query"""
        detected_location = None
        scope = 'general'
        suggested_sources = []
        location_terms = []
        confidence = 0.0
        
        # Check Singapore indicators
        singapore_score = 0.0
        for indicator_type, indicators in self.geographic_patterns['singapore_indicators'].items():
            for indicator in indicators:
                if indicator in query:
                    singapore_score += 2.0 if indicator_type == 'explicit' else 1.0
                    location_terms.append(indicator)
        
        # Check global indicators
        global_score = 0.0
        for indicator_type, indicators in self.geographic_patterns['global_indicators'].items():
            for indicator in indicators:
                if indicator in query:
                    global_score += 2.0 if indicator_type == 'explicit' else 1.0
                    location_terms.append(indicator)
        
        # Determine geographic context
        if singapore_score > global_score and singapore_score > 0:
            detected_location = 'singapore'
            scope = 'singapore'
            suggested_sources = self.geographic_patterns['singapore_indicators']['sources']
            confidence = min(1.0, singapore_score / 3.0)
        elif global_score > 0:
            detected_location = 'global'
            scope = 'global'
            suggested_sources = self.geographic_patterns['global_indicators']['sources']
            confidence = min(1.0, global_score / 3.0)
        
        return GeographicContext(
            detected_location=detected_location,
            scope=scope,
            suggested_sources=suggested_sources,
            location_terms=location_terms,
            confidence=confidence
        )
    
    def _generate_domain_expansions(self, query: str, domain_context: DomainContext, 
                                  max_expansions: int) -> List[str]:
        """Generate domain-specific expansion terms"""
        expansions = []
        
        if domain_context.primary_domain == 'general':
            return expansions
        
        terminology = self.domain_terminology.get(domain_context.primary_domain, {})
        
        # Add related terms not already in query
        for term in terminology.get('related_terms', []):
            if term not in query and len(expansions) < max_expansions:
                expansions.append(term)
        
        # Add synonyms for terms in query
        for main_term, synonyms in terminology.get('synonyms', {}).items():
            if main_term in query:
                for synonym in synonyms:
                    if synonym not in query and len(expansions) < max_expansions:
                        expansions.append(synonym)
        
        # Add specialized vocabulary
        for term in terminology.get('expansion_terms', []):
            if term not in query and len(expansions) < max_expansions and len(term) > 3:
                expansions.append(term)
        
        return expansions[:max_expansions]
    
    def _generate_geographic_expansions(self, query: str, 
                                      geographic_context: GeographicContext) -> List[str]:
        """Generate geographic expansion terms"""
        expansions = []
        
        if geographic_context.detected_location == 'singapore':
            # Add Singapore-specific terms
            singapore_terms = ['government', 'public', 'national', 'official']
            for term in singapore_terms:
                if term not in query:
                    expansions.append(term)
                    if len(expansions) >= 2:
                        break
        
        elif geographic_context.detected_location == 'global':
            # Add global context terms
            global_terms = ['international', 'comparative', 'cross-country']
            for term in global_terms:
                if term not in query:
                    expansions.append(term)
                    if len(expansions) >= 2:
                        break
        
        return expansions
    
    def _generate_refinement_suggestions(self, query: str, domain_context: DomainContext,
                                       geographic_context: GeographicContext) -> List[str]:
        """Generate query refinement suggestions based on successful patterns"""
        suggestions = []
        
        # Domain-specific suggestions
        if domain_context.primary_domain in self.query_refinement_patterns:
            domain_suggestions = self.query_refinement_patterns[domain_context.primary_domain]
            suggestions.extend(domain_suggestions[:2])
        
        # Geographic suggestions
        if geographic_context.detected_location == 'singapore':
            if 'singapore' not in query and 'sg' not in query:
                suggestions.append("Add 'Singapore' to focus on local government data")
        
        # General suggestions based on successful mappings
        if 'data' not in query and 'dataset' not in query:
            suggestions.append("Add 'data' or 'dataset' to find relevant datasets")
        
        if domain_context.primary_domain == 'psychology' and 'research' not in query:
            suggestions.append("Add 'research' to find academic psychology datasets")
        
        if domain_context.primary_domain == 'climate' and 'indicators' not in query:
            suggestions.append("Add 'indicators' to find climate measurement data")
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:4]
    
    def _build_enhanced_query(self, original_query: str, expansion_terms: List[str]) -> str:
        """Build enhanced query with expansion terms"""
        if not expansion_terms:
            return original_query
        
        # Add expansion terms that provide the most value
        valuable_terms = []
        for term in expansion_terms:
            if len(term) > 2 and term not in original_query.lower():
                valuable_terms.append(term)
        
        if valuable_terms:
            return f"{original_query} {' '.join(valuable_terms[:3])}"
        
        return original_query
    
    def _calculate_enhancement_confidence(self, domain_context: DomainContext,
                                        geographic_context: GeographicContext,
                                        expansion_terms: List[str]) -> float:
        """Calculate confidence score for the enhancement"""
        confidence = 0.0
        
        # Domain confidence contribution
        confidence += domain_context.confidence * 0.4
        
        # Geographic confidence contribution
        confidence += geographic_context.confidence * 0.3
        
        # Expansion terms contribution
        if expansion_terms:
            expansion_confidence = min(1.0, len(expansion_terms) / 5.0)
            confidence += expansion_confidence * 0.3
        
        return min(1.0, confidence)
    
    def _generate_enhancement_explanation(self, domain_context: DomainContext,
                                        geographic_context: GeographicContext,
                                        expansion_terms: List[str],
                                        refinement_suggestions: List[str]) -> str:
        """Generate explanation for the enhancement"""
        explanations = []
        
        if domain_context.primary_domain != 'general':
            explanations.append(f"Detected {domain_context.primary_domain} domain")
        
        if geographic_context.detected_location:
            explanations.append(f"Geographic context: {geographic_context.detected_location}")
        
        if expansion_terms:
            explanations.append(f"Added {len(expansion_terms)} domain-specific terms")
        
        if refinement_suggestions:
            explanations.append(f"Generated {len(refinement_suggestions)} refinement suggestions")
        
        if not explanations:
            return "Query was specific enough - no enhancements needed"
        
        return "Enhanced by: " + "; ".join(explanations)
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get query completion suggestions based on successful mappings"""
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        # Find matching queries from successful mappings
        for domain, mappings in self.successful_mappings.items():
            for mapping in mappings:
                query = mapping['query']
                if query.lower().startswith(partial_lower) and mapping['score'] >= 0.8:
                    suggestions.append(query)
        
        # Remove duplicates and sort by length (shorter first)
        unique_suggestions = list(set(suggestions))
        unique_suggestions.sort(key=len)
        
        return unique_suggestions[:max_suggestions]
    
    def test_enhancement_examples(self):
        """Test the enhancement system with example queries"""
        test_queries = [
            "psychology data",
            "singapore housing",
            "climate change",
            "machine learning datasets",
            "transport information",
            "health statistics",
            "education research"
        ]
        
        logger.info("🧪 Testing Context-Aware Query Enhancement:")
        
        for query in test_queries:
            enhancement = self.enhance_query(query, max_expansions=3)
            
            logger.info(f"  Query: '{query}'")
            logger.info(f"    Enhanced: '{enhancement.enhanced_query}'")
            logger.info(f"    Domain: {enhancement.domain_context['primary_domain']}")
            logger.info(f"    Geographic: {enhancement.geographic_context['scope']}")
            logger.info(f"    Expansions: {enhancement.expansion_terms}")
            logger.info(f"    Suggestions: {enhancement.refinement_suggestions}")
            logger.info(f"    Confidence: {enhancement.confidence_score:.2f}")
            logger.info(f"    Explanation: {enhancement.explanation}")
            logger.info("")


def create_context_aware_query_enhancer(training_mappings_path: str = "training_mappings.md") -> ContextAwareQueryEnhancer:
    """Factory function to create context-aware query enhancer"""
    return ContextAwareQueryEnhancer(training_mappings_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the context-aware query enhancer
    enhancer = create_context_aware_query_enhancer()
    
    # Test with example queries
    enhancer.test_enhancement_examples()
    
    print("✅ Context-Aware Query Enhancement testing completed!")