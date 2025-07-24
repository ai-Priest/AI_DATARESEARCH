"""
Integrated Query Processor
Combines Context-Aware Query Enhancement with Enhanced Query Router
for complete query processing pipeline
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .context_aware_query_enhancement import create_context_aware_query_enhancer, QueryEnhancement
from .enhanced_query_router import create_enhanced_query_router, QueryClassification

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Complete query processing result"""
    original_query: str
    enhanced_query: str
    enhancement: QueryEnhancement
    classification: QueryClassification
    recommended_sources: List[Dict[str, any]]
    processing_confidence: float
    explanation: str


class IntegratedQueryProcessor:
    """Integrated query processor combining enhancement and routing"""
    
    def __init__(self, training_mappings_path: str = "training_mappings.md"):
        self.enhancer = create_context_aware_query_enhancer(training_mappings_path)
        self.router = create_enhanced_query_router()
        
        logger.info("🔗 IntegratedQueryProcessor initialized")
        logger.info("  Enhancement and routing components ready")
    
    def process_query(self, query: str, max_expansions: int = 3) -> ProcessedQuery:
        """
        Process query through complete enhancement and routing pipeline
        
        Args:
            query: Original user query
            max_expansions: Maximum number of expansion terms
            
        Returns:
            ProcessedQuery with complete processing results
        """
        logger.info(f"🔄 Processing query: '{query}'")
        
        # Step 1: Enhance the query
        enhancement = self.enhancer.enhance_query(query, max_expansions)
        
        # Step 2: Route the enhanced query
        classification = self.router.classify_query(enhancement.enhanced_query)
        
        # Step 3: Get recommended sources
        recommended_sources = self.router.route_to_sources(classification)
        
        # Step 4: Calculate overall processing confidence
        processing_confidence = self._calculate_processing_confidence(
            enhancement, classification
        )
        
        # Step 5: Generate comprehensive explanation
        explanation = self._generate_processing_explanation(
            enhancement, classification, recommended_sources
        )
        
        result = ProcessedQuery(
            original_query=query,
            enhanced_query=enhancement.enhanced_query,
            enhancement=enhancement,
            classification=classification,
            recommended_sources=recommended_sources,
            processing_confidence=processing_confidence,
            explanation=explanation
        )
        
        logger.info(f"✅ Query processed with confidence: {processing_confidence:.2f}")
        logger.info(f"   Domain: {classification.domain}")
        logger.info(f"   Singapore-first: {classification.singapore_first_applicable}")
        logger.info(f"   Top sources: {[s['name'] for s in recommended_sources[:3]]}")
        
        return result
    
    def _calculate_processing_confidence(self, enhancement: QueryEnhancement, 
                                       classification: QueryClassification) -> float:
        """Calculate overall processing confidence"""
        # Combine enhancement and classification confidence
        enhancement_weight = 0.4
        classification_weight = 0.6
        
        overall_confidence = (
            enhancement.confidence_score * enhancement_weight +
            classification.confidence * classification_weight
        )
        
        return min(1.0, overall_confidence)
    
    def _generate_processing_explanation(self, enhancement: QueryEnhancement,
                                       classification: QueryClassification,
                                       sources: List[Dict[str, any]]) -> str:
        """Generate comprehensive processing explanation"""
        explanations = []
        
        # Enhancement explanation
        if enhancement.expansion_terms:
            explanations.append(f"Enhanced with {len(enhancement.expansion_terms)} domain terms")
        
        # Classification explanation
        explanations.append(f"Classified as {classification.domain}")
        
        # Geographic context
        if classification.singapore_first_applicable:
            explanations.append("Singapore-first strategy applied")
        
        # Source recommendations
        if sources:
            top_source = sources[0]['name']
            explanations.append(f"Recommending {top_source} as primary source")
        
        # Refinement suggestions
        if enhancement.refinement_suggestions:
            explanations.append(f"{len(enhancement.refinement_suggestions)} refinement suggestions available")
        
        return "; ".join(explanations)
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get query completion suggestions"""
        return self.enhancer.get_query_suggestions(partial_query, max_suggestions)
    
    def analyze_query_quality(self, query: str) -> Dict[str, any]:
        """Analyze query quality and provide improvement suggestions"""
        enhancement = self.enhancer.enhance_query(query)
        classification = self.router.classify_query(query)
        
        quality_analysis = {
            'original_query': query,
            'domain_clarity': classification.confidence,
            'enhancement_potential': len(enhancement.expansion_terms),
            'geographic_context': enhancement.geographic_context['confidence'],
            'refinement_suggestions': enhancement.refinement_suggestions,
            'overall_quality': min(1.0, (classification.confidence + 
                                        enhancement.confidence_score) / 2),
            'improvement_areas': []
        }
        
        # Identify improvement areas
        if classification.confidence < 0.7:
            quality_analysis['improvement_areas'].append('Domain clarity could be improved')
        
        if not enhancement.expansion_terms:
            quality_analysis['improvement_areas'].append('Query could benefit from more specific terms')
        
        if enhancement.geographic_context['confidence'] < 0.5 and 'singapore' not in query.lower():
            quality_analysis['improvement_areas'].append('Consider adding geographic context')
        
        return quality_analysis
    
    def test_processing_examples(self):
        """Test the integrated processor with example queries"""
        test_queries = [
            "psychology research",
            "singapore housing data",
            "climate change indicators",
            "machine learning datasets",
            "transport statistics",
            "health data",
            "education research"
        ]
        
        logger.info("🧪 Testing Integrated Query Processing:")
        
        for query in test_queries:
            result = self.process_query(query)
            
            logger.info(f"  Query: '{query}'")
            logger.info(f"    Enhanced: '{result.enhanced_query}'")
            logger.info(f"    Domain: {result.classification.domain}")
            logger.info(f"    Singapore-first: {result.classification.singapore_first_applicable}")
            logger.info(f"    Top sources: {[s['name'] for s in result.recommended_sources[:3]]}")
            logger.info(f"    Confidence: {result.processing_confidence:.2f}")
            logger.info(f"    Explanation: {result.explanation}")
            logger.info("")


def create_integrated_query_processor(training_mappings_path: str = "training_mappings.md") -> IntegratedQueryProcessor:
    """Factory function to create integrated query processor"""
    return IntegratedQueryProcessor(training_mappings_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the integrated query processor
    processor = create_integrated_query_processor()
    
    # Test with example queries
    processor.test_processing_examples()
    
    # Test query quality analysis
    print("\n📊 Query Quality Analysis Examples:")
    print("-" * 40)
    
    test_queries = ["psychology", "singapore housing data", "data"]
    
    for query in test_queries:
        analysis = processor.analyze_query_quality(query)
        print(f"\nQuery: '{query}'")
        print(f"  Domain clarity: {analysis['domain_clarity']:.2f}")
        print(f"  Enhancement potential: {analysis['enhancement_potential']}")
        print(f"  Overall quality: {analysis['overall_quality']:.2f}")
        print(f"  Improvement areas: {analysis['improvement_areas']}")
    
    print("\n✅ Integrated Query Processing testing completed!")