"""
Domain-Specific Data Augmentation for Neural Training
Creates synthetic training examples, query paraphrasing, and hard negatives
to improve domain-specific routing (psychology→Kaggle, climate→World Bank, etc.)
"""

import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)

class DomainSpecificDataAugmentation:
    """Generate domain-specific training data augmentation for neural model improvement."""
    
    def __init__(self, training_mappings_path: str = "training_mappings.md"):
        self.training_mappings_path = training_mappings_path
        self.domain_patterns = {}
        self.source_preferences = {}
        self.singapore_terms = {}
        self.query_templates = {}
        self.hard_negatives = {}
        
        # Load existing training mappings
        self._load_training_mappings()
        self._build_domain_patterns()
        self._build_singapore_query_templates()
        self._build_hard_negative_patterns()
    
    def _load_training_mappings(self):
        """Load and parse existing training mappings from markdown file."""
        logger.info(f"📖 Loading training mappings from {self.training_mappings_path}")
        
        try:
            with open(self.training_mappings_path, 'r') as f:
                content = f.read()
            
            # Parse mappings by domain sections
            current_domain = None
            mappings = []
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Detect domain sections
                if line.startswith('## ') and 'Queries' in line:
                    current_domain = line.replace('## ', '').replace(' Queries', '').lower()
                    continue
                
                # Parse mapping lines: "- query → source (score) - reason"
                if line.startswith('- ') and '→' in line and '(' in line:
                    try:
                        parts = line[2:].split('→')
                        query = parts[0].strip()
                        
                        rest = parts[1].strip()
                        source = rest.split('(')[0].strip()
                        score_part = rest.split('(')[1].split(')')[0]
                        reason = rest.split(')')[1].strip().lstrip('- ')
                        
                        relevance = float(score_part)
                        
                        mappings.append({
                            'domain': current_domain,
                            'query': query,
                            'source': source,
                            'relevance': relevance,
                            'reason': reason
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not parse mapping line: {line} - {e}")
            
            # Organize by domain
            for mapping in mappings:
                domain = mapping['domain']
                if domain not in self.domain_patterns:
                    self.domain_patterns[domain] = []
                self.domain_patterns[domain].append(mapping)
            
            logger.info(f"✅ Loaded {len(mappings)} mappings across {len(self.domain_patterns)} domains")
            
        except FileNotFoundError:
            logger.error(f"Training mappings file not found: {self.training_mappings_path}")
            self.domain_patterns = {}
    
    def _build_domain_patterns(self):
        """Build domain-specific source preference patterns."""
        logger.info("🔄 Building domain-specific source patterns")
        
        for domain, mappings in self.domain_patterns.items():
            # Calculate source preferences for this domain
            source_scores = {}
            for mapping in mappings:
                source = mapping['source']
                relevance = mapping['relevance']
                
                if source not in source_scores:
                    source_scores[source] = []
                source_scores[source].append(relevance)
            
            # Average scores per source
            self.source_preferences[domain] = {
                source: np.mean(scores) 
                for source, scores in source_scores.items()
            }
        
        logger.info(f"✅ Built source preferences for {len(self.source_preferences)} domains")
    
    def _build_singapore_query_templates(self):
        """Build Singapore-specific query paraphrasing templates."""
        logger.info("🔄 Building Singapore query templates")
        
        self.singapore_terms = {
            # Government agencies and their variations
            'data_gov_sg': [
                'singapore government data', 'sg gov data', 'singapore official data',
                'government singapore', 'singapore public data', 'sg government statistics'
            ],
            'singstat': [
                'singapore statistics', 'singapore demographic data', 'sg population data',
                'singapore census', 'singapore statistical data', 'sg demographics'
            ],
            'lta_datamall': [
                'singapore transport data', 'sg traffic data', 'singapore mobility',
                'lta singapore', 'singapore public transport', 'sg transportation'
            ],
            'hdb': [
                'singapore housing data', 'sg property data', 'singapore real estate',
                'hdb singapore', 'singapore residential data', 'sg housing statistics'
            ],
            'ura': [
                'singapore urban planning', 'sg development data', 'singapore city planning',
                'urban singapore', 'singapore land use', 'sg planning data'
            ]
        }
        
        # Query paraphrasing templates for Singapore
        self.query_templates = {
            'singapore_general': [
                'singapore {topic}', 'sg {topic}', '{topic} singapore',
                'singapore government {topic}', '{topic} data singapore',
                'singapore official {topic}', 'sg {topic} statistics'
            ],
            'government_focus': [
                'government {topic} singapore', 'official {topic} sg',
                'public {topic} singapore', 'singapore ministry {topic}',
                'statutory board {topic} singapore'
            ],
            'research_focus': [
                'singapore {topic} research', '{topic} analysis singapore',
                'singapore {topic} study', '{topic} trends singapore',
                'singapore {topic} indicators'
            ]
        }
        
        logger.info("✅ Built Singapore query templates")
    
    def _build_hard_negative_patterns(self):
        """Build patterns for generating hard negative examples."""
        logger.info("🔄 Building hard negative patterns")
        
        # Sources that should NOT be recommended for specific domains
        self.hard_negatives = {
            'psychology': {
                'avoid_sources': ['world_bank', 'data_un', 'aws_opendata'],
                'reasons': [
                    'Limited psychology-specific datasets',
                    'No behavioral research data available',
                    'Focus on economic/development data, not psychology'
                ]
            },
            'machine learning': {
                'avoid_sources': ['world_bank', 'data_gov_sg', 'singstat'],
                'reasons': [
                    'Government data not suitable for ML competitions',
                    'Limited ML-specific datasets',
                    'Focus on statistics, not ML training data'
                ]
            },
            'climate & environment': {
                'avoid_sources': ['kaggle', 'zenodo'],
                'reasons': [
                    'Competition datasets may lack global coverage',
                    'Academic papers not comprehensive data sources',
                    'World Bank has better climate indicators'
                ]
            },
            'singapore-specific': {
                'avoid_sources': ['kaggle', 'world_bank', 'data_un'],
                'reasons': [
                    'International sources lack Singapore-specific context',
                    'Global datasets may not include Singapore data',
                    'Government sources more authoritative for local data'
                ]
            }
        }
        
        logger.info("✅ Built hard negative patterns")
    
    def generate_synthetic_domain_examples(self, domain: str, num_examples: int = 20) -> List[Dict]:
        """Generate synthetic training examples for specific domain routing."""
        logger.info(f"🎯 Generating {num_examples} synthetic examples for {domain}")
        
        if domain not in self.domain_patterns:
            logger.warning(f"No patterns found for domain: {domain}")
            return []
        
        synthetic_examples = []
        base_mappings = self.domain_patterns[domain]
        
        for i in range(num_examples):
            # Select a base mapping to augment
            base_mapping = random.choice(base_mappings)
            base_query = base_mapping['query']
            
            # Generate query variations
            variations = self._generate_query_variations(base_query, domain)
            
            for variation in variations[:3]:  # Limit variations per base query
                # Create positive example
                positive_example = {
                    'query': variation,
                    'source': base_mapping['source'],
                    'relevance': base_mapping['relevance'] + random.uniform(-0.05, 0.05),
                    'domain': domain,
                    'synthetic': True,
                    'generation_method': 'domain_specific_augmentation',
                    'base_query': base_query,
                    'reason': f"Synthetic variation: {base_mapping['reason']}"
                }
                synthetic_examples.append(positive_example)
                
                # Generate hard negative examples
                hard_negatives = self._generate_hard_negatives(variation, domain)
                synthetic_examples.extend(hard_negatives)
        
        logger.info(f"✅ Generated {len(synthetic_examples)} synthetic examples for {domain}")
        return synthetic_examples
    
    def _generate_query_variations(self, base_query: str, domain: str) -> List[str]:
        """Generate variations of a base query using domain-specific patterns."""
        variations = []
        
        # Synonym replacement
        synonyms = {
            'data': ['dataset', 'information', 'statistics', 'records'],
            'research': ['study', 'analysis', 'investigation', 'examination'],
            'statistics': ['data', 'metrics', 'indicators', 'figures'],
            'analysis': ['research', 'study', 'evaluation', 'assessment']
        }
        
        # Generate synonym variations
        words = base_query.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                for synonym in synonyms[word.lower()]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    variations.append(' '.join(new_words))
        
        # Add domain-specific modifiers
        domain_modifiers = {
            'psychology': ['behavioral', 'cognitive', 'mental health', 'psychological'],
            'machine learning': ['ml', 'artificial intelligence', 'deep learning', 'neural'],
            'climate & environment': ['environmental', 'climate change', 'weather', 'sustainability'],
            'singapore-specific': ['singapore', 'sg', 'local', 'government']
        }
        
        if domain in domain_modifiers:
            for modifier in domain_modifiers[domain]:
                if modifier.lower() not in base_query.lower():
                    variations.extend([
                        f"{modifier} {base_query}",
                        f"{base_query} {modifier}",
                        base_query.replace(base_query.split()[0], f"{modifier} {base_query.split()[0]}")
                    ])
        
        # Remove duplicates and return
        return list(set(variations))[:5]
    
    def _generate_hard_negatives(self, query: str, domain: str) -> List[Dict]:
        """Generate hard negative examples for better ranking discrimination."""
        hard_negatives = []
        
        if domain not in self.hard_negatives:
            return hard_negatives
        
        negative_config = self.hard_negatives[domain]
        avoid_sources = negative_config['avoid_sources']
        reasons = negative_config['reasons']
        
        for source in avoid_sources:
            # Generate low relevance score (0.1 - 0.4)
            low_relevance = random.uniform(0.1, 0.4)
            reason = random.choice(reasons)
            
            hard_negative = {
                'query': query,
                'source': source,
                'relevance': low_relevance,
                'domain': domain,
                'synthetic': True,
                'hard_negative': True,
                'generation_method': 'hard_negative_generation',
                'reason': f"Hard negative: {reason}"
            }
            hard_negatives.append(hard_negative)
        
        return hard_negatives
    
    def generate_singapore_query_paraphrases(self, base_queries: List[str], num_paraphrases: int = 5) -> List[Dict]:
        """Generate paraphrased Singapore-specific queries."""
        logger.info(f"🇸🇬 Generating Singapore query paraphrases for {len(base_queries)} base queries")
        
        paraphrases = []
        
        for base_query in base_queries:
            # Extract topic from query
            topic = self._extract_topic_from_query(base_query)
            
            # Generate paraphrases using templates
            for template_type, templates in self.query_templates.items():
                for template in templates[:2]:  # Limit templates per type
                    try:
                        paraphrase = template.format(topic=topic)
                        if paraphrase != base_query and len(paraphrase) > 5:
                            paraphrase_data = {
                                'original_query': base_query,
                                'paraphrased_query': paraphrase,
                                'template_type': template_type,
                                'topic': topic,
                                'singapore_specific': True,
                                'generation_method': 'singapore_paraphrasing'
                            }
                            paraphrases.append(paraphrase_data)
                    except KeyError:
                        continue
        
        # Remove duplicates
        seen_paraphrases = set()
        unique_paraphrases = []
        for p in paraphrases:
            if p['paraphrased_query'] not in seen_paraphrases:
                seen_paraphrases.add(p['paraphrased_query'])
                unique_paraphrases.append(p)
        
        logger.info(f"✅ Generated {len(unique_paraphrases)} unique Singapore paraphrases")
        return unique_paraphrases[:num_paraphrases * len(base_queries)]
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from a query for paraphrasing."""
        # Remove common stop words and Singapore-specific terms
        stop_words = {'singapore', 'sg', 'data', 'dataset', 'information', 'statistics'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        topic_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if topic_words:
            return ' '.join(topic_words[:2])  # Take first 2 meaningful words
        else:
            return query.split()[0] if query.split() else 'data'
    
    def create_comprehensive_training_augmentation(self, output_path: str = None) -> Dict:
        """Create comprehensive training data augmentation for all domains."""
        logger.info("🚀 Creating comprehensive domain-specific training augmentation")
        
        augmentation_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0',
                'source_mappings_file': self.training_mappings_path,
                'domains_processed': list(self.domain_patterns.keys())
            },
            'synthetic_examples': [],
            'singapore_paraphrases': [],
            'hard_negatives': [],
            'domain_statistics': {}
        }
        
        # Generate synthetic examples for each domain
        for domain in self.domain_patterns.keys():
            domain_examples = self.generate_synthetic_domain_examples(domain, num_examples=15)
            
            # Separate positive examples and hard negatives
            positive_examples = [ex for ex in domain_examples if not ex.get('hard_negative', False)]
            hard_negatives = [ex for ex in domain_examples if ex.get('hard_negative', False)]
            
            augmentation_data['synthetic_examples'].extend(positive_examples)
            augmentation_data['hard_negatives'].extend(hard_negatives)
            
            # Track statistics
            augmentation_data['domain_statistics'][domain] = {
                'positive_examples': len(positive_examples),
                'hard_negatives': len(hard_negatives),
                'total_examples': len(domain_examples)
            }
        
        # Generate Singapore-specific paraphrases
        singapore_queries = [
            'singapore housing data', 'singapore transport statistics', 
            'singapore population data', 'singapore economic indicators',
            'singapore government data', 'singapore urban planning'
        ]
        
        singapore_paraphrases = self.generate_singapore_query_paraphrases(singapore_queries, num_paraphrases=3)
        augmentation_data['singapore_paraphrases'] = singapore_paraphrases
        
        # Summary statistics
        total_examples = len(augmentation_data['synthetic_examples'])
        total_negatives = len(augmentation_data['hard_negatives'])
        total_paraphrases = len(augmentation_data['singapore_paraphrases'])
        
        augmentation_data['metadata']['summary'] = {
            'total_synthetic_examples': total_examples,
            'total_hard_negatives': total_negatives,
            'total_singapore_paraphrases': total_paraphrases,
            'total_augmented_data_points': total_examples + total_negatives + total_paraphrases
        }
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(augmentation_data, f, indent=2)
            
            logger.info(f"💾 Saved augmentation data to: {output_path}")
        
        logger.info(f"✅ Generated comprehensive augmentation:")
        logger.info(f"   📊 {total_examples} synthetic examples")
        logger.info(f"   ❌ {total_negatives} hard negatives")
        logger.info(f"   🇸🇬 {total_paraphrases} Singapore paraphrases")
        logger.info(f"   📈 {total_examples + total_negatives + total_paraphrases} total data points")
        
        return augmentation_data
    
    def integrate_with_training_system(self, augmentation_data: Dict) -> int:
        """Integrate augmented data with existing training system."""
        logger.info("🔗 Integrating augmented data with training system")
        
        # Import training data injector
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from ml.training_data_injector import TrainingDataInjector
            injector = TrainingDataInjector()
            
            # Prepare mappings for injection
            mappings = []
            
            # Add synthetic examples
            for example in augmentation_data['synthetic_examples']:
                mappings.append((
                    example['query'],
                    example['source'],
                    example['relevance'],
                    example['reason']
                ))
            
            # Add hard negatives
            for negative in augmentation_data['hard_negatives']:
                mappings.append((
                    negative['query'],
                    negative['source'],
                    negative['relevance'],
                    negative['reason']
                ))
            
            # Inject into training system
            injected_count = injector.inject_query_source_mappings(mappings)
            
            logger.info(f"✅ Injected {injected_count} augmented training examples")
            return injected_count
            
        except ImportError as e:
            logger.warning(f"Could not import training data injector: {e}")
            logger.info("💾 Saving augmented data for manual integration")
            
            # Save as training mappings format for manual integration
            self._save_as_training_mappings(augmentation_data)
            return len(augmentation_data['synthetic_examples']) + len(augmentation_data['hard_negatives'])
    
    def _save_as_training_mappings(self, augmentation_data: Dict):
        """Save augmented data in training mappings format for manual integration."""
        output_path = "data/processed/augmented_training_mappings.md"
        
        with open(output_path, 'w') as f:
            f.write("# Augmented Training Data Mappings\n\n")
            f.write("Generated by Domain-Specific Data Augmentation\n\n")
            
            # Group by domain
            domain_groups = {}
            
            # Process synthetic examples
            for example in augmentation_data['synthetic_examples']:
                domain = example['domain']
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(example)
            
            # Process hard negatives
            for negative in augmentation_data['hard_negatives']:
                domain = negative['domain']
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(negative)
            
            # Write by domain
            for domain, examples in domain_groups.items():
                f.write(f"## {domain.title()} Queries (Augmented)\n\n")
                
                for example in examples:
                    f.write(f"- {example['query']} → {example['source']} ({example['relevance']:.2f}) - {example['reason']}\n")
                
                f.write("\n")
        
        logger.info(f"💾 Saved augmented mappings to: {output_path}")


def create_domain_specific_augmentation():
    """Main function to create domain-specific data augmentation."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize augmentation system
    augmenter = DomainSpecificDataAugmentation()
    
    # Create comprehensive augmentation
    augmentation_data = augmenter.create_comprehensive_training_augmentation(
        output_path="data/processed/domain_specific_augmentation.json"
    )
    
    # Integrate with training system
    injected_count = augmenter.integrate_with_training_system(augmentation_data)
    
    print(f"🎯 Domain-Specific Data Augmentation Complete!")
    print(f"📊 Generated {augmentation_data['metadata']['summary']['total_augmented_data_points']} augmented data points")
    print(f"💉 Injected {injected_count} examples into training system")
    
    return augmentation_data


if __name__ == "__main__":
    create_domain_specific_augmentation()