"""
Enhanced Training Data Parser - Quality-first training data integration
Implements comprehensive parsing of training_mappings.md with domain classification,
Singapore-first strategy detection, and negative example generation.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import random

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingExample:
    """Enhanced training example with domain classification and quality metrics"""
    query: str
    source: str
    relevance_score: float  # 0.0 - 1.0 from training_mappings.md
    domain: str  # psychology, climate, singapore, etc.
    explanation: str  # Why this source is/isn't relevant
    geographic_scope: str  # singapore, global, international
    query_intent: str  # research, analysis, comparison
    negative_examples: List[str]  # Sources that should rank lower
    singapore_first_applicable: bool  # Whether Singapore-first strategy applies
    
    def to_neural_training_format(self) -> Dict:
        """Convert to format suitable for neural training"""
        return {
            "query": self.query,
            "source": self.source,  # Keep consistent field name
            "positive_source": self.source,
            "relevance_score": self.relevance_score,
            "domain": self.domain,
            "geographic_scope": self.geographic_scope,
            "query_intent": self.query_intent,
            "negative_sources": self.negative_examples,
            "singapore_first": self.singapore_first_applicable,
            "explanation": self.explanation,
            "training_metadata": {
                "created_at": datetime.now().isoformat(),
                "source_type": "manual_mapping",
                "quality_validated": True
            }
        }
    
    def generate_hard_negatives(self, all_sources: Set[str]) -> List['EnhancedTrainingExample']:
        """Generate hard negative examples for better ranking"""
        hard_negatives = []
        
        # Get sources that should rank lower than current source
        for neg_source in self.negative_examples:
            if neg_source in all_sources:
                # Ensure negative relevance is between 0.3 and (original_score - 0.2)
                negative_relevance = max(0.3, min(0.6, self.relevance_score - 0.2))
                
                neg_example = EnhancedTrainingExample(
                    query=self.query,
                    source=neg_source,
                    relevance_score=negative_relevance,
                    domain=self.domain,
                    explanation=f"Lower relevance than {self.source} for {self.query}",
                    geographic_scope=self.geographic_scope,
                    query_intent=self.query_intent,
                    negative_examples=[],
                    singapore_first_applicable=self.singapore_first_applicable
                )
                hard_negatives.append(neg_example)
        
        return hard_negatives


@dataclass
class QueryClassification:
    """Classification of query characteristics"""
    original_query: str
    domain: str  # psychology, climate, technology, singapore, etc.
    geographic_scope: str  # local, singapore, global, international
    intent: str  # research, analysis, comparison, exploration
    confidence: float
    singapore_first_applicable: bool
    recommended_sources: List[str]
    
    def get_source_priority_order(self) -> List[str]:
        """Get prioritized list of sources for this query"""
        if self.singapore_first_applicable:
            # Prioritize Singapore sources
            singapore_sources = ["data_gov_sg", "singstat", "lta_datamall"]
            other_sources = [s for s in self.recommended_sources if s not in singapore_sources]
            return singapore_sources + other_sources
        return self.recommended_sources


class TrainingDataIntegrator:
    """
    Enhanced training data integrator that parses training_mappings.md
    and converts to neural training examples with domain classification
    """
    
    def __init__(self, mappings_file: str = "training_mappings.md"):
        self.mappings_file = Path(mappings_file)
        self.domain_patterns = self._initialize_domain_patterns()
        self.singapore_patterns = self._initialize_singapore_patterns()
        self.intent_patterns = self._initialize_intent_patterns()
        self.source_categories = self._initialize_source_categories()
        
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for domain classification"""
        return {
            "psychology": [
                "psychology", "mental health", "behavioral", "cognitive", 
                "psychological", "behavior", "mind", "brain"
            ],
            "machine_learning": [
                "machine learning", "ml", "artificial intelligence", "ai",
                "deep learning", "neural network", "algorithm"
            ],
            "climate": [
                "climate", "weather", "environmental", "temperature",
                "climate change", "environment", "carbon", "emission"
            ],
            "economics": [
                "economic", "economy", "gdp", "financial", "finance",
                "trade", "poverty", "income", "market"
            ],
            "singapore": [
                "singapore", "sg", "singstat", "hdb", "lta", "moh",
                "singapore government", "data.gov.sg"
            ],
            "health": [
                "health", "medical", "healthcare", "disease", "hospital",
                "medicine", "patient", "treatment"
            ],
            "education": [
                "education", "school", "university", "student", "learning",
                "academic", "research", "study"
            ]
        }
    
    def _initialize_singapore_patterns(self) -> List[str]:
        """Initialize patterns for Singapore-first detection"""
        return [
            "singapore", "sg", "local", "singstat", "data.gov.sg",
            "hdb", "lta", "moh", "singapore government", "singapore data"
        ]
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for intent classification"""
        return {
            "research": [
                "research", "study", "analysis", "investigation",
                "academic", "scientific", "paper"
            ],
            "analysis": [
                "data", "statistics", "metrics", "indicators",
                "trends", "patterns", "insights"
            ],
            "comparison": [
                "compare", "comparison", "versus", "vs", "difference",
                "benchmark", "evaluate"
            ],
            "exploration": [
                "explore", "discover", "find", "search", "look for",
                "available", "datasets"
            ]
        }
    
    def _initialize_source_categories(self) -> Dict[str, Dict[str, any]]:
        """Initialize source categorization and characteristics"""
        return {
            "kaggle": {
                "type": "commercial_platform",
                "focus": ["machine_learning", "data_science", "competitions"],
                "geographic_scope": "global",
                "academic": False,
                "government": False
            },
            "zenodo": {
                "type": "academic_repository", 
                "focus": ["research", "academic", "scientific"],
                "geographic_scope": "global",
                "academic": True,
                "government": False
            },
            "world_bank": {
                "type": "international_organization",
                "focus": ["economics", "development", "climate", "health"],
                "geographic_scope": "global",
                "academic": False,
                "government": True
            },
            "data_gov_sg": {
                "type": "government_portal",
                "focus": ["singapore", "government", "official"],
                "geographic_scope": "singapore",
                "academic": False,
                "government": True
            },
            "singstat": {
                "type": "statistical_office",
                "focus": ["singapore", "statistics", "demographics"],
                "geographic_scope": "singapore", 
                "academic": False,
                "government": True
            },
            "lta_datamall": {
                "type": "transport_authority",
                "focus": ["singapore", "transport", "traffic"],
                "geographic_scope": "singapore",
                "academic": False,
                "government": True
            }
        }
    
    def parse_training_mappings(self, mappings_file: Optional[str] = None) -> List[EnhancedTrainingExample]:
        """Parse manual feedback mappings into enhanced training examples"""
        file_path = Path(mappings_file) if mappings_file else self.mappings_file
        
        if not file_path.exists():
            logger.error(f"Training mappings file not found: {file_path}")
            return []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        examples = []
        current_domain_section = None
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect domain sections
            if line.startswith('## ') and 'Queries' in line:
                current_domain_section = self._extract_domain_from_section(line)
                continue
            
            # Parse mapping lines: "- query → source (score) - reason"
            if line.startswith('- ') and '→' in line and '(' in line and ')' in line:
                try:
                    example = self._parse_mapping_line(line, current_domain_section)
                    if example:
                        examples.append(example)
                except Exception as e:
                    logger.warning(f"Could not parse mapping line: {line} - {e}")
                    continue
        
        logger.info(f"✅ Parsed {len(examples)} enhanced training examples")
        return examples
    
    def _extract_domain_from_section(self, section_line: str) -> str:
        """Extract domain from section header"""
        # "## Psychology Queries" -> "psychology"
        section = section_line.replace('##', '').replace('Queries', '').strip().lower()
        
        # Map section names to standardized domains
        domain_mapping = {
            "psychology": "psychology",
            "machine learning": "machine_learning", 
            "climate & environment": "climate",
            "economics & finance": "economics",
            "singapore-specific": "singapore",
            "health & medical": "health",
            "education": "education"
        }
        
        return domain_mapping.get(section, section.replace(' ', '_'))
    
    def _parse_mapping_line(self, line: str, domain_section: Optional[str]) -> Optional[EnhancedTrainingExample]:
        """Parse individual mapping line into EnhancedTrainingExample"""
        try:
            # Parse: "- psychology → kaggle (0.95) - Best platform"
            parts = line[2:].split('→')  # Remove "- " prefix
            query = parts[0].strip()
            
            rest = parts[1].strip()
            source = rest.split('(')[0].strip()
            score_part = rest.split('(')[1].split(')')[0]
            reason_part = rest.split(')')[1].strip().lstrip('- ').strip()
            
            relevance_score = float(score_part)
            
            # Classify the query
            classification = self.classify_query(query)
            
            # Determine domain (use section if available, otherwise classify)
            domain = domain_section if domain_section else classification.domain
            
            # Generate negative examples based on low-scoring sources for same query
            negative_examples = self._generate_negative_examples(query, source, relevance_score)
            
            example = EnhancedTrainingExample(
                query=query,
                source=source,
                relevance_score=relevance_score,
                domain=domain,
                explanation=reason_part,
                geographic_scope=classification.geographic_scope,
                query_intent=classification.intent,
                negative_examples=negative_examples,
                singapore_first_applicable=classification.singapore_first_applicable
            )
            
            return example
            
        except Exception as e:
            logger.warning(f"Error parsing mapping line '{line}': {e}")
            return None
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query characteristics for enhanced training"""
        query_lower = query.lower()
        
        # Detect domain
        domain = self._detect_domain(query_lower)
        
        # Detect geographic scope
        geographic_scope = self._detect_geographic_scope(query_lower)
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Determine if Singapore-first strategy applies
        singapore_first = self._detect_singapore_first_applicable(query_lower)
        
        # Get recommended sources based on classification
        recommended_sources = self._get_recommended_sources(domain, geographic_scope)
        
        return QueryClassification(
            original_query=query,
            domain=domain,
            geographic_scope=geographic_scope,
            intent=intent,
            confidence=0.8,  # Default confidence
            singapore_first_applicable=singapore_first,
            recommended_sources=recommended_sources
        )
    
    def _detect_domain(self, query_lower: str) -> str:
        """Detect domain from query text"""
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"
    
    def _detect_geographic_scope(self, query_lower: str) -> str:
        """Detect geographic scope from query"""
        if any(pattern in query_lower for pattern in self.singapore_patterns):
            return "singapore"
        elif any(word in query_lower for word in ["global", "international", "world"]):
            return "global"
        else:
            return "general"
    
    def _detect_intent(self, query_lower: str) -> str:
        """Detect query intent"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return "exploration"
    
    def _detect_singapore_first_applicable(self, query_lower: str) -> bool:
        """Determine if Singapore-first strategy should be applied"""
        return any(pattern in query_lower for pattern in self.singapore_patterns)
    
    def _get_recommended_sources(self, domain: str, geographic_scope: str) -> List[str]:
        """Get recommended sources based on domain and geographic scope"""
        sources = []
        
        # Singapore-specific sources for Singapore queries
        if geographic_scope == "singapore":
            sources.extend(["data_gov_sg", "singstat", "lta_datamall"])
        
        # Domain-specific source recommendations
        domain_source_mapping = {
            "psychology": ["kaggle", "zenodo"],
            "machine_learning": ["kaggle", "zenodo"],
            "climate": ["world_bank", "zenodo"],
            "economics": ["world_bank", "kaggle"],
            "health": ["world_bank", "zenodo"],
            "education": ["world_bank", "zenodo"]
        }
        
        if domain in domain_source_mapping:
            sources.extend(domain_source_mapping[domain])
        
        # Add general sources if none found
        if not sources:
            sources = ["kaggle", "zenodo", "world_bank"]
        
        return list(set(sources))  # Remove duplicates
    
    def _generate_negative_examples(self, query: str, positive_source: str, relevance_score: float) -> List[str]:
        """Generate negative examples for better ranking discrimination"""
        all_sources = set(self.source_categories.keys())
        negative_sources = []
        
        # Sources that are clearly not good for this query type
        query_lower = query.lower()
        
        # If it's a Singapore query, non-Singapore sources are negatives
        if any(pattern in query_lower for pattern in self.singapore_patterns):
            non_singapore_sources = [s for s in all_sources 
                                   if self.source_categories[s]["geographic_scope"] != "singapore"]
            negative_sources.extend(non_singapore_sources[:2])
        
        # Domain-specific negatives
        domain = self._detect_domain(query_lower)
        if domain == "psychology":
            # World Bank is not great for psychology
            if "world_bank" in all_sources and "world_bank" != positive_source:
                negative_sources.append("world_bank")
        elif domain == "economics":
            # Kaggle is less ideal for economics than World Bank
            if "kaggle" in all_sources and positive_source == "world_bank":
                negative_sources.append("kaggle")
        
        return list(set(negative_sources))[:3]  # Limit to 3 negatives
    
    def augment_training_data(self, base_examples: List[EnhancedTrainingExample]) -> List[EnhancedTrainingExample]:
        """Augment training data with synthetic examples and paraphrases"""
        augmented_examples = base_examples.copy()
        
        # Generate paraphrases for Singapore-specific queries
        singapore_examples = [ex for ex in base_examples if ex.singapore_first_applicable]
        for example in singapore_examples:
            paraphrases = self._generate_query_paraphrases(example.query, "singapore")
            for paraphrase in paraphrases:
                aug_example = EnhancedTrainingExample(
                    query=paraphrase,
                    source=example.source,
                    relevance_score=example.relevance_score,
                    domain=example.domain,
                    explanation=f"Paraphrase of: {example.explanation}",
                    geographic_scope=example.geographic_scope,
                    query_intent=example.query_intent,
                    negative_examples=example.negative_examples,
                    singapore_first_applicable=example.singapore_first_applicable
                )
                augmented_examples.append(aug_example)
        
        # Generate domain-specific synthetic examples
        domain_examples = self._generate_domain_specific_examples()
        augmented_examples.extend(domain_examples)
        
        # Generate hard negatives
        for example in base_examples:
            hard_negatives = example.generate_hard_negatives(set(self.source_categories.keys()))
            augmented_examples.extend(hard_negatives)
        
        logger.info(f"✅ Augmented training data: {len(base_examples)} → {len(augmented_examples)} examples")
        return augmented_examples
    
    def _generate_query_paraphrases(self, query: str, domain: str) -> List[str]:
        """Generate paraphrases for queries to increase training diversity"""
        paraphrases = []
        
        if domain == "singapore":
            singapore_paraphrases = {
                "singapore data": ["sg data", "singapore statistics", "singapore information"],
                "singapore housing": ["hdb data", "singapore property", "housing statistics singapore"],
                "singapore transport": ["lta data", "singapore traffic", "transport data sg"],
                "singapore demographics": ["singapore population", "sg demographics", "singapore census"]
            }
            
            for original, variants in singapore_paraphrases.items():
                if original in query.lower():
                    paraphrases.extend(variants)
        
        # General paraphrases
        general_paraphrases = {
            "data": ["dataset", "information", "statistics"],
            "research": ["study", "analysis", "investigation"],
            "statistics": ["data", "metrics", "indicators"]
        }
        
        query_words = query.lower().split()
        for word in query_words:
            if word in general_paraphrases:
                for variant in general_paraphrases[word]:
                    new_query = query.lower().replace(word, variant)
                    if new_query != query.lower():
                        paraphrases.append(new_query)
        
        return paraphrases[:3]  # Limit to 3 paraphrases per query
    
    def _generate_domain_specific_examples(self) -> List[EnhancedTrainingExample]:
        """Generate synthetic training examples for domain-specific routing"""
        synthetic_examples = []
        
        # Psychology → Kaggle/Zenodo routing examples
        psychology_queries = [
            "behavioral psychology datasets", "cognitive research data", 
            "mental health analysis", "psychology experiments data"
        ]
        
        for query in psychology_queries:
            # High relevance for Kaggle
            synthetic_examples.append(EnhancedTrainingExample(
                query=query,
                source="kaggle",
                relevance_score=0.92,
                domain="psychology",
                explanation="Synthetic: Psychology datasets available on Kaggle",
                geographic_scope="global",
                query_intent="research",
                negative_examples=["world_bank"],
                singapore_first_applicable=False
            ))
            
            # Medium relevance for Zenodo
            synthetic_examples.append(EnhancedTrainingExample(
                query=query,
                source="zenodo",
                relevance_score=0.88,
                domain="psychology",
                explanation="Synthetic: Academic psychology research on Zenodo",
                geographic_scope="global", 
                query_intent="research",
                negative_examples=["world_bank"],
                singapore_first_applicable=False
            ))
        
        # Climate → World Bank routing examples
        climate_queries = [
            "global climate indicators", "climate change statistics",
            "environmental development data", "carbon emission data"
        ]
        
        for query in climate_queries:
            synthetic_examples.append(EnhancedTrainingExample(
                query=query,
                source="world_bank",
                relevance_score=0.94,
                domain="climate",
                explanation="Synthetic: World Bank climate and environment data",
                geographic_scope="global",
                query_intent="analysis",
                negative_examples=["kaggle"],
                singapore_first_applicable=False
            ))
        
        logger.info(f"✅ Generated {len(synthetic_examples)} synthetic domain-specific examples")
        return synthetic_examples
    
    def create_domain_specific_splits(self, examples: List[EnhancedTrainingExample]) -> Dict[str, List[EnhancedTrainingExample]]:
        """Create domain-specific training splits"""
        domain_splits = {}
        
        for example in examples:
            domain = example.domain
            if domain not in domain_splits:
                domain_splits[domain] = []
            domain_splits[domain].append(example)
        
        # Ensure balanced representation
        min_examples_per_domain = 5
        for domain, domain_examples in domain_splits.items():
            if len(domain_examples) < min_examples_per_domain:
                # Generate additional synthetic examples for underrepresented domains
                additional_examples = self._generate_additional_domain_examples(domain, min_examples_per_domain - len(domain_examples))
                domain_splits[domain].extend(additional_examples)
        
        logger.info(f"✅ Created domain splits: {[(d, len(examples)) for d, examples in domain_splits.items()]}")
        return domain_splits
    
    def _generate_additional_domain_examples(self, domain: str, count: int) -> List[EnhancedTrainingExample]:
        """Generate additional examples for underrepresented domains"""
        additional_examples = []
        
        domain_templates = {
            "health": {
                "queries": ["health indicators", "medical statistics", "healthcare data"],
                "sources": [("world_bank", 0.88), ("zenodo", 0.85)]
            },
            "education": {
                "queries": ["education statistics", "school data", "university research"],
                "sources": [("world_bank", 0.90), ("zenodo", 0.87)]
            }
        }
        
        if domain in domain_templates:
            template = domain_templates[domain]
            for i in range(count):
                query = random.choice(template["queries"])
                source, relevance = random.choice(template["sources"])
                
                additional_examples.append(EnhancedTrainingExample(
                    query=f"{query} {i+1}",  # Add variation
                    source=source,
                    relevance_score=relevance,
                    domain=domain,
                    explanation=f"Synthetic example for {domain} domain",
                    geographic_scope="global",
                    query_intent="analysis",
                    negative_examples=[],
                    singapore_first_applicable=False
                ))
        
        return additional_examples
    
    def validate_training_quality(self, examples: List[EnhancedTrainingExample]) -> Dict[str, any]:
        """Validate training data quality and coverage"""
        validation_report = {
            "total_examples": len(examples),
            "domain_coverage": {},
            "geographic_coverage": {},
            "singapore_first_coverage": 0,
            "relevance_score_distribution": {},
            "quality_issues": []
        }
        
        # Domain coverage
        domain_counts = {}
        for example in examples:
            domain_counts[example.domain] = domain_counts.get(example.domain, 0) + 1
        validation_report["domain_coverage"] = domain_counts
        
        # Geographic coverage
        geo_counts = {}
        for example in examples:
            geo_counts[example.geographic_scope] = geo_counts.get(example.geographic_scope, 0) + 1
        validation_report["geographic_coverage"] = geo_counts
        
        # Singapore-first strategy coverage
        singapore_first_count = sum(1 for ex in examples if ex.singapore_first_applicable)
        validation_report["singapore_first_coverage"] = singapore_first_count
        
        # Relevance score distribution
        score_ranges = {"high": 0, "medium": 0, "low": 0}
        for example in examples:
            if example.relevance_score >= 0.8:
                score_ranges["high"] += 1
            elif example.relevance_score >= 0.5:
                score_ranges["medium"] += 1
            else:
                score_ranges["low"] += 1
        validation_report["relevance_score_distribution"] = score_ranges
        
        # Quality issues detection
        quality_issues = []
        
        # Check for domain balance
        min_domain_examples = 3
        for domain, count in domain_counts.items():
            if count < min_domain_examples:
                quality_issues.append(f"Domain '{domain}' has only {count} examples (minimum: {min_domain_examples})")
        
        # Check for Singapore representation
        if singapore_first_count < 5:
            quality_issues.append(f"Singapore-first examples: {singapore_first_count} (recommended: 5+)")
        
        validation_report["quality_issues"] = quality_issues
        
        logger.info(f"✅ Training data validation complete: {len(examples)} examples, {len(quality_issues)} issues")
        return validation_report


# Convenience functions for easy usage
def parse_training_mappings(mappings_file: str = "training_mappings.md") -> List[EnhancedTrainingExample]:
    """Quick function to parse training mappings"""
    integrator = TrainingDataIntegrator(mappings_file)
    return integrator.parse_training_mappings()


def create_enhanced_training_dataset(mappings_file: str = "training_mappings.md", 
                                   output_file: str = "data/processed/enhanced_training_data.json") -> int:
    """Create complete enhanced training dataset"""
    integrator = TrainingDataIntegrator(mappings_file)
    
    # Parse base examples
    base_examples = integrator.parse_training_mappings()
    
    # Augment with synthetic data
    augmented_examples = integrator.augment_training_data(base_examples)
    
    # Create domain splits
    domain_splits = integrator.create_domain_specific_splits(augmented_examples)
    
    # Validate quality
    validation_report = integrator.validate_training_quality(augmented_examples)
    
    # Save enhanced training data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    training_data = {
        "examples": [example.to_neural_training_format() for example in augmented_examples],
        "domain_splits": {domain: [ex.to_neural_training_format() for ex in examples] 
                         for domain, examples in domain_splits.items()},
        "validation_report": validation_report,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source_file": mappings_file,
            "total_examples": len(augmented_examples),
            "base_examples": len(base_examples)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"✅ Enhanced training dataset saved: {output_file}")
    logger.info(f"📊 Validation report: {validation_report}")
    
    return len(augmented_examples)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced training dataset
    count = create_enhanced_training_dataset()
    print(f"✅ Created enhanced training dataset with {count} examples!")