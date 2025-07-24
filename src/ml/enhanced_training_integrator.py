"""
Enhanced Training Data Integrator - Quality-First Approach
Integrates manual training mappings into neural training pipeline with domain-specific routing
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingExample:
    """Enhanced training example with quality-first features"""
    query: str
    source: str
    relevance_score: float  # 0.0 - 1.0 from training_mappings.md
    domain: str  # psychology, climate, singapore, etc.
    explanation: str  # Why this source is/isn't relevant
    geographic_scope: str  # singapore, global, international
    query_intent: str  # research, analysis, comparison
    negative_examples: List[str]  # Sources that should rank lower
    singapore_first_applicable: bool  # Whether Singapore-first strategy applies
    quality_tier: str  # high, medium, low based on relevance score
    
    def to_neural_training_format(self) -> Dict:
        """Convert to format suitable for neural training"""
        return {
            'query': self.query,
            'positive_source': self.source,
            'relevance_score': self.relevance_score,
            'domain': self.domain,
            'explanation': self.explanation,
            'geographic_scope': self.geographic_scope,
            'query_intent': self.query_intent,
            'negative_sources': self.negative_examples,
            'singapore_first': self.singapore_first_applicable,
            'quality_tier': self.quality_tier,
            'training_type': 'manual_mapping'
        }
    
    def generate_hard_negatives(self, all_sources: Set[str]) -> List[str]:
        """Generate hard negative examples for better ranking"""
        # Sources that should rank lower than this one
        negatives = []
        
        # If this is a high-relevance example, add medium/low relevance sources as negatives
        if self.relevance_score >= 0.8:
            # Add sources from different domains as hard negatives
            domain_mismatches = {
                'psychology': ['world_bank', 'data_gov_sg', 'lta_datamall'],
                'climate': ['kaggle', 'data_gov_sg', 'lta_datamall'],
                'singapore': ['world_bank', 'zenodo', 'kaggle'],
                'economics': ['kaggle', 'data_gov_sg', 'lta_datamall'],
                'machine_learning': ['world_bank', 'data_gov_sg', 'singstat']
            }
            
            if self.domain in domain_mismatches:
                negatives.extend(domain_mismatches[self.domain])
        
        # Remove the positive source from negatives
        negatives = [neg for neg in negatives if neg != self.source]
        
        return negatives[:3]  # Limit to top 3 hard negatives


class EnhancedTrainingDataIntegrator:
    """Integrates manual training mappings with quality-first neural training"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.training_examples: List[EnhancedTrainingExample] = []
        self.domain_stats = {}
        self.source_coverage = {}
        
        # Define domain categories and their characteristics
        self.domain_definitions = {
            'psychology': {
                'keywords': ['psychology', 'mental health', 'behavioral', 'cognitive'],
                'preferred_sources': ['kaggle', 'zenodo'],
                'singapore_first': False
            },
            'climate': {
                'keywords': ['climate', 'weather', 'environmental', 'temperature'],
                'preferred_sources': ['world_bank', 'zenodo'],
                'singapore_first': False
            },
            'singapore': {
                'keywords': ['singapore', 'sg', 'hdb', 'mrt', 'lta'],
                'preferred_sources': ['data_gov_sg', 'singstat', 'lta_datamall'],
                'singapore_first': True
            },
            'economics': {
                'keywords': ['economic', 'gdp', 'financial', 'trade', 'poverty'],
                'preferred_sources': ['world_bank', 'singstat'],
                'singapore_first': False
            },
            'machine_learning': {
                'keywords': ['machine learning', 'ml', 'ai', 'neural', 'deep learning'],
                'preferred_sources': ['kaggle', 'zenodo'],
                'singapore_first': False
            },
            'health': {
                'keywords': ['health', 'medical', 'healthcare', 'disease'],
                'preferred_sources': ['world_bank', 'zenodo'],
                'singapore_first': False
            },
            'education': {
                'keywords': ['education', 'student', 'university', 'school'],
                'preferred_sources': ['world_bank', 'zenodo'],
                'singapore_first': False
            }
        }
    
    def parse_training_mappings(self, mappings_file: str) -> List[EnhancedTrainingExample]:
        """Parse manual feedback mappings into enhanced training examples"""
        logger.info(f"📖 Parsing training mappings from {mappings_file}")
        
        mappings_path = Path(mappings_file)
        if not mappings_path.exists():
            logger.error(f"Training mappings file not found: {mappings_file}")
            return []
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        examples = []
        current_domain = 'general'
        
        # Parse markdown content
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Detect domain sections
            if line.startswith('##') and 'Queries' in line:
                current_domain = self._extract_domain_from_header(line)
                logger.debug(f"Found domain section: {current_domain}")
                continue
            
            # Look for mapping lines: "- query → source (score) - reason"
            if line.startswith('- ') and '→' in line and '(' in line and ')' in line:
                try:
                    example = self._parse_mapping_line(line, current_domain, line_num)
                    if example:
                        examples.append(example)
                except Exception as e:
                    logger.warning(f"Could not parse line {line_num}: {line} - {e}")
                    continue
        
        logger.info(f"✅ Parsed {len(examples)} training examples from {mappings_file}")
        self._log_parsing_stats(examples)
        
        return examples
    
    def _extract_domain_from_header(self, header: str) -> str:
        """Extract domain name from markdown header"""
        # "## Psychology Queries" -> "psychology"
        domain = header.replace('##', '').replace('Queries', '').strip().lower()
        
        # Map common variations to standard domain names
        domain_mapping = {
            'psychology': 'psychology',
            'machine learning': 'machine_learning',
            'climate & environment': 'climate',
            'economics & finance': 'economics',
            'singapore-specific': 'singapore',
            'health & medical': 'health',
            'education': 'education'
        }
        
        return domain_mapping.get(domain, domain.replace(' ', '_'))
    
    def _parse_mapping_line(self, line: str, domain: str, line_num: int) -> Optional[EnhancedTrainingExample]:
        """Parse a single mapping line into an EnhancedTrainingExample"""
        try:
            # Parse: "- psychology → kaggle (0.95) - Best platform for psychology datasets"
            parts = line[2:].split('→')  # Remove "- " prefix
            if len(parts) != 2:
                return None
                
            query = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract source, score, and reason
            if '(' not in rest or ')' not in rest:
                return None
                
            source_part = rest.split('(')[0].strip()
            score_part = rest.split('(')[1].split(')')[0].strip()
            reason_part = rest.split(')', 1)[1].strip() if ')' in rest else ""
            
            source = source_part
            relevance_score = float(score_part)
            explanation = reason_part.lstrip('- ').strip()
            
            # Determine geographic scope and Singapore-first applicability
            geographic_scope = self._determine_geographic_scope(query, domain)
            singapore_first = self._should_apply_singapore_first(query, domain)
            query_intent = self._determine_query_intent(query)
            quality_tier = self._determine_quality_tier(relevance_score)
            
            # Get all sources for negative example generation
            all_sources = {'kaggle', 'zenodo', 'world_bank', 'data_gov_sg', 'singstat', 'lta_datamall', 'aws_opendata', 'data_un'}
            
            example = EnhancedTrainingExample(
                query=query,
                source=source,
                relevance_score=relevance_score,
                domain=domain,
                explanation=explanation,
                geographic_scope=geographic_scope,
                query_intent=query_intent,
                negative_examples=[],  # Will be populated later
                singapore_first_applicable=singapore_first,
                quality_tier=quality_tier
            )
            
            # Generate hard negatives
            example.negative_examples = example.generate_hard_negatives(all_sources)
            
            return example
            
        except Exception as e:
            logger.warning(f"Error parsing line {line_num}: {e}")
            return None
    
    def _determine_geographic_scope(self, query: str, domain: str) -> str:
        """Determine geographic scope of the query"""
        query_lower = query.lower()
        
        if 'singapore' in query_lower or domain == 'singapore':
            return 'singapore'
        elif any(word in query_lower for word in ['global', 'international', 'worldwide']):
            return 'global'
        elif domain in ['climate', 'economics', 'health']:
            return 'global'  # These domains are typically global
        else:
            return 'general'
    
    def _should_apply_singapore_first(self, query: str, domain: str) -> bool:
        """Determine if Singapore-first strategy should be applied"""
        query_lower = query.lower()
        
        # Explicit Singapore queries
        if 'singapore' in query_lower or domain == 'singapore':
            return True
        
        # Generic queries that should prioritize Singapore sources for local users
        generic_singapore_terms = ['housing', 'transport', 'population', 'economy', 'education', 'health']
        if any(term in query_lower for term in generic_singapore_terms):
            # Only if not explicitly global
            if not any(word in query_lower for word in ['global', 'international', 'worldwide']):
                return True
        
        return False
    
    def _determine_query_intent(self, query: str) -> str:
        """Determine the intent behind the query"""
        query_lower = query.lower()
        
        if 'research' in query_lower or 'study' in query_lower:
            return 'research'
        elif 'analysis' in query_lower or 'analyze' in query_lower:
            return 'analysis'
        elif 'compare' in query_lower or 'comparison' in query_lower:
            return 'comparison'
        elif 'data' in query_lower or 'dataset' in query_lower:
            return 'data_access'
        else:
            return 'exploration'
    
    def _determine_quality_tier(self, relevance_score: float) -> str:
        """Determine quality tier based on relevance score"""
        if relevance_score >= 0.8:
            return 'high'
        elif relevance_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _log_parsing_stats(self, examples: List[EnhancedTrainingExample]):
        """Log statistics about parsed examples"""
        domain_counts = {}
        source_counts = {}
        quality_counts = {}
        
        for example in examples:
            domain_counts[example.domain] = domain_counts.get(example.domain, 0) + 1
            source_counts[example.source] = source_counts.get(example.source, 0) + 1
            quality_counts[example.quality_tier] = quality_counts.get(example.quality_tier, 0) + 1
        
        logger.info("📊 Training mappings statistics:")
        logger.info(f"  Domains: {dict(sorted(domain_counts.items()))}")
        logger.info(f"  Sources: {dict(sorted(source_counts.items()))}")
        logger.info(f"  Quality tiers: {dict(sorted(quality_counts.items()))}")
    
    def augment_training_data(self, base_examples: List[EnhancedTrainingExample]) -> List[EnhancedTrainingExample]:
        """Augment training data with synthetic examples and variations"""
        logger.info(f"🔄 Augmenting {len(base_examples)} base examples")
        
        augmented = base_examples.copy()
        
        # Generate query variations for high-quality examples
        for example in base_examples:
            if example.quality_tier == 'high':
                variations = self._generate_query_variations(example)
                augmented.extend(variations)
        
        # Generate domain-specific synthetic examples
        synthetic_examples = self._generate_synthetic_examples()
        augmented.extend(synthetic_examples)
        
        logger.info(f"✅ Augmented to {len(augmented)} total examples")
        return augmented
    
    def _generate_query_variations(self, example: EnhancedTrainingExample) -> List[EnhancedTrainingExample]:
        """Generate variations of high-quality queries"""
        variations = []
        
        # Query variation patterns
        if example.domain == 'psychology':
            patterns = [
                f"{example.query} datasets",
                f"{example.query} research data",
                f"find {example.query} data"
            ]
        elif example.domain == 'singapore':
            patterns = [
                f"{example.query} statistics",
                f"{example.query} information",
                f"singapore {example.query.replace('singapore', '').strip()}"
            ]
        else:
            patterns = [
                f"{example.query} datasets",
                f"{example.query} statistics"
            ]
        
        for pattern in patterns:
            if pattern != example.query:  # Avoid duplicates
                variation = EnhancedTrainingExample(
                    query=pattern,
                    source=example.source,
                    relevance_score=example.relevance_score * 0.95,  # Slightly lower for variations
                    domain=example.domain,
                    explanation=f"Variation of: {example.explanation}",
                    geographic_scope=example.geographic_scope,
                    query_intent=example.query_intent,
                    negative_examples=example.negative_examples,
                    singapore_first_applicable=example.singapore_first_applicable,
                    quality_tier=example.quality_tier
                )
                variations.append(variation)
        
        return variations[:2]  # Limit to 2 variations per example
    
    def _generate_synthetic_examples(self) -> List[EnhancedTrainingExample]:
        """Generate synthetic examples for better coverage"""
        synthetic = []
        
        # Generate examples for underrepresented domains
        synthetic_patterns = {
            'singapore': [
                ('singapore government data', 'data_gov_sg', 0.95),
                ('singapore statistics', 'singstat', 0.94),
                ('singapore transport', 'lta_datamall', 0.93)
            ],
            'psychology': [
                ('psychology datasets', 'kaggle', 0.92),
                ('mental health research', 'zenodo', 0.90)
            ],
            'climate': [
                ('climate indicators', 'world_bank', 0.91),
                ('environmental statistics', 'world_bank', 0.89)
            ]
        }
        
        for domain, patterns in synthetic_patterns.items():
            for query, source, score in patterns:
                example = EnhancedTrainingExample(
                    query=query,
                    source=source,
                    relevance_score=score,
                    domain=domain,
                    explanation=f"Synthetic example for {domain} domain",
                    geographic_scope='singapore' if domain == 'singapore' else 'global',
                    query_intent='data_access',
                    negative_examples=[],
                    singapore_first_applicable=(domain == 'singapore'),
                    quality_tier='high' if score >= 0.8 else 'medium'
                )
                example.negative_examples = example.generate_hard_negatives({'kaggle', 'zenodo', 'world_bank', 'data_gov_sg', 'singstat', 'lta_datamall'})
                synthetic.append(example)
        
        return synthetic
    
    def create_domain_specific_splits(self, examples: List[EnhancedTrainingExample]) -> Dict[str, List[EnhancedTrainingExample]]:
        """Create domain-specific training splits"""
        logger.info("📂 Creating domain-specific training splits")
        
        domain_splits = {}
        
        for example in examples:
            if example.domain not in domain_splits:
                domain_splits[example.domain] = []
            domain_splits[example.domain].append(example)
        
        # Log split statistics
        for domain, domain_examples in domain_splits.items():
            logger.info(f"  {domain}: {len(domain_examples)} examples")
        
        return domain_splits
    
    def validate_training_quality(self, examples: List[EnhancedTrainingExample]) -> Dict:
        """Validate training data quality and coverage"""
        logger.info("🔍 Validating training data quality")
        
        validation_report = {
            'total_examples': len(examples),
            'domain_coverage': {},
            'source_coverage': {},
            'quality_distribution': {},
            'singapore_first_coverage': 0,
            'issues': []
        }
        
        # Analyze coverage
        for example in examples:
            # Domain coverage
            domain = example.domain
            if domain not in validation_report['domain_coverage']:
                validation_report['domain_coverage'][domain] = 0
            validation_report['domain_coverage'][domain] += 1
            
            # Source coverage
            source = example.source
            if source not in validation_report['source_coverage']:
                validation_report['source_coverage'][source] = 0
            validation_report['source_coverage'][source] += 1
            
            # Quality distribution
            quality = example.quality_tier
            if quality not in validation_report['quality_distribution']:
                validation_report['quality_distribution'][quality] = 0
            validation_report['quality_distribution'][quality] += 1
            
            # Singapore-first coverage
            if example.singapore_first_applicable:
                validation_report['singapore_first_coverage'] += 1
        
        # Check for issues
        if validation_report['total_examples'] < 50:
            validation_report['issues'].append("Low total example count")
        
        if len(validation_report['domain_coverage']) < 3:
            validation_report['issues'].append("Limited domain coverage")
        
        if validation_report['quality_distribution'].get('high', 0) < 10:
            validation_report['issues'].append("Insufficient high-quality examples")
        
        logger.info(f"✅ Validation complete: {len(validation_report['issues'])} issues found")
        
        return validation_report
    
    def export_to_neural_format(self, examples: List[EnhancedTrainingExample], output_path: str):
        """Export training examples to neural training format"""
        logger.info(f"💾 Exporting {len(examples)} examples to neural format: {output_path}")
        
        # Convert to neural training format
        neural_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_examples': len(examples),
                'source': 'enhanced_training_integrator',
                'quality_first': True
            },
            'examples': [example.to_neural_training_format() for example in examples]
        }
        
        # Create train/validation/test splits
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(examples))
        
        train_size = int(0.7 * len(examples))
        val_size = int(0.15 * len(examples))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            'train': [neural_data['examples'][i] for i in train_indices],
            'validation': [neural_data['examples'][i] for i in val_indices],
            'test': [neural_data['examples'][i] for i in test_indices]
        }
        
        # Add split information to metadata
        neural_data['metadata']['splits'] = {
            'train': len(splits['train']),
            'validation': len(splits['validation']),
            'test': len(splits['test'])
        }
        
        neural_data.update(splits)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(neural_data, f, indent=2)
        
        logger.info(f"✅ Neural training data exported to {output_path}")
        logger.info(f"  Train: {len(splits['train'])} examples")
        logger.info(f"  Validation: {len(splits['validation'])} examples")
        logger.info(f"  Test: {len(splits['test'])} examples")
        
        return output_path


def integrate_training_mappings(mappings_file: str = "training_mappings.md", 
                              output_file: str = "data/processed/enhanced_training_mappings.json") -> Dict:
    """Main function to integrate training mappings into neural training pipeline"""
    logger.info("🚀 Starting enhanced training data integration")
    
    integrator = EnhancedTrainingDataIntegrator()
    
    # Parse training mappings
    examples = integrator.parse_training_mappings(mappings_file)
    if not examples:
        logger.error("No training examples parsed - aborting integration")
        return {}
    
    # Augment training data
    augmented_examples = integrator.augment_training_data(examples)
    
    # Create domain-specific splits
    domain_splits = integrator.create_domain_specific_splits(augmented_examples)
    
    # Validate training quality
    validation_report = integrator.validate_training_quality(augmented_examples)
    
    # Export to neural format
    output_path = integrator.export_to_neural_format(augmented_examples, output_file)
    
    # Return integration summary
    summary = {
        'success': True,
        'total_examples': len(augmented_examples),
        'original_examples': len(examples),
        'augmented_examples': len(augmented_examples) - len(examples),
        'domain_splits': {domain: len(examples) for domain, examples in domain_splits.items()},
        'validation_report': validation_report,
        'output_file': str(output_path),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info("✅ Enhanced training data integration complete")
    logger.info(f"  Total examples: {summary['total_examples']}")
    logger.info(f"  Domains covered: {len(domain_splits)}")
    logger.info(f"  Output file: {output_path}")
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run integration
    result = integrate_training_mappings()
    
    if result.get('success'):
        print(f"✅ Successfully integrated {result['total_examples']} training examples!")
        print(f"📁 Output: {result['output_file']}")
    else:
        print("❌ Integration failed")