"""
Domain-Enhanced Training Data Generation
Phase 2.1: Generate enhanced training data with domain diversity and improved quality
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dl.graded_relevance import GradedRelevanceScorer, create_graded_relevance_config

logger = logging.getLogger(__name__)


class DomainEnhancedTrainingGenerator:
    """
    Generate enhanced training data with domain diversity and improved quality.
    """
    
    def __init__(self):
        self.config = create_graded_relevance_config()
        self.scorer = GradedRelevanceScorer(self.config)
        
        # Enhanced domain specifications
        self.domains = {
            'housing': {
                'weight': 0.25,
                'query_templates': [
                    "HDB resale prices {area} {year}",
                    "housing affordability {income_group} {year}",
                    "property market trends {property_type}",
                    "rental prices {area} {room_type}",
                    "housing supply new launches {year}",
                    "public housing waiting time {year}",
                    "private property transactions {area}",
                    "housing price index singapore {year}"
                ],
                'keywords': ['hdb', 'housing', 'property', 'resale', 'rental', 'bto', 'condo', 'prices', 'affordability'],
                'areas': ['singapore', 'central', 'north', 'south', 'east', 'west', 'jurong', 'tampines', 'woodlands', 'bishan'],
                'years': ['2024', '2023', '2022', '2021', '2020'],
                'room_types': ['1-room', '2-room', '3-room', '4-room', '5-room', 'executive'],
                'property_types': ['hdb', 'condo', 'landed', 'private'],
                'income_groups': ['low-income', 'middle-income', 'high-income']
            },
            'transportation': {
                'weight': 0.20,
                'query_templates': [
                    "MRT ridership data {line} {year}",
                    "bus routes and schedules {area}",
                    "traffic congestion {area} {time}",
                    "transport statistics {mode} {year}",
                    "public transport usage trends {year}",
                    "taxi and ride-hailing data {year}",
                    "cycling infrastructure {area}",
                    "COE prices {vehicle_type} {year}"
                ],
                'keywords': ['mrt', 'bus', 'lrt', 'transport', 'traffic', 'taxi', 'grab', 'coe', 'cycling'],
                'lines': ['north-south', 'east-west', 'circle', 'downtown', 'thomson-east-coast'],
                'modes': ['mrt', 'bus', 'taxi', 'private-hire', 'cycling', 'walking'],
                'vehicle_types': ['car', 'motorcycle', 'commercial'],
                'times': ['peak-hours', 'off-peak', 'weekend', 'holiday']
            },
            'healthcare': {
                'weight': 0.15,
                'query_templates': [
                    "hospital statistics {specialty} {year}",
                    "healthcare utilization data {age_group}",
                    "disease prevalence {disease} singapore {year}",
                    "medical services {area} accessibility",
                    "health outcomes {indicator} {year}",
                    "healthcare spending {category} {year}",
                    "polyclinic services {area} {year}",
                    "mental health statistics singapore {year}"
                ],
                'keywords': ['hospital', 'clinic', 'health', 'medical', 'doctor', 'patient', 'disease', 'polyclinic'],
                'specialties': ['cardiology', 'oncology', 'pediatrics', 'geriatrics', 'emergency'],
                'age_groups': ['children', 'adults', 'elderly', 'working-age'],
                'diseases': ['diabetes', 'hypertension', 'cancer', 'mental-health', 'infectious'],
                'indicators': ['life-expectancy', 'mortality', 'morbidity', 'quality-of-life'],
                'categories': ['public', 'private', 'subsidized', 'total']
            },
            'economics': {
                'weight': 0.15,
                'query_templates': [
                    "GDP growth {sector} {year}",
                    "employment statistics {industry} {year}",
                    "inflation rates {category} {year}",
                    "trade data singapore {partner} {year}",
                    "economic indicators {indicator} {year}",
                    "wage levels {occupation} {year}",
                    "business formation {sector} {year}",
                    "foreign investment {country} {year}"
                ],
                'keywords': ['gdp', 'inflation', 'employment', 'wage', 'economy', 'trade', 'finance', 'investment'],
                'sectors': ['manufacturing', 'services', 'construction', 'finance', 'technology'],
                'industries': ['manufacturing', 'finance', 'technology', 'retail', 'healthcare'],
                'indicators': ['productivity', 'competitiveness', 'innovation', 'growth'],
                'partners': ['china', 'malaysia', 'usa', 'eu', 'asean'],
                'occupations': ['professional', 'technical', 'service', 'administrative']
            },
            'education': {
                'weight': 0.10,
                'query_templates': [
                    "school performance {level} {year}",
                    "university enrollment {field} {year}",
                    "education outcomes {metric} {year}",
                    "teacher statistics {level} {year}",
                    "student demographics {characteristic} {year}",
                    "education spending {category} {year}",
                    "skills training programs {industry} {year}",
                    "international student data {year}"
                ],
                'keywords': ['school', 'university', 'student', 'education', 'exam', 'curriculum', 'teacher'],
                'levels': ['primary', 'secondary', 'junior-college', 'polytechnic', 'university'],
                'fields': ['engineering', 'business', 'medicine', 'arts', 'science'],
                'metrics': ['performance', 'graduation-rate', 'literacy', 'numeracy'],
                'characteristics': ['nationality', 'socioeconomic', 'special-needs']
            },
            'demographics': {
                'weight': 0.15,
                'query_templates': [
                    "population statistics {age_group} {year}",
                    "birth rates singapore {year}",
                    "marriage and divorce data {year}",
                    "migration patterns {type} {year}",
                    "household composition {area} {year}",
                    "life expectancy {gender} {year}",
                    "ethnic distribution singapore {year}",
                    "aging population trends {year}"
                ],
                'keywords': ['population', 'age', 'birth', 'death', 'marriage', 'citizen', 'resident', 'demographics'],
                'age_groups': ['children', 'working-age', 'elderly', 'youth', 'seniors'],
                'types': ['immigration', 'emigration', 'internal-migration'],
                'genders': ['male', 'female', 'overall']
            }
        }
        
        logger.info(f"🎯 Domain-enhanced training generator initialized with {len(self.domains)} domains")
    
    def generate_enhanced_training_data(self, 
                                      num_samples: int = 3000,
                                      output_path: str = "data/processed/domain_enhanced_training.json") -> str:
        """
        Generate enhanced training data with domain diversity.
        
        Args:
            num_samples: Total number of samples to generate
            output_path: Output file path
            
        Returns:
            Path to generated training data
        """
        logger.info(f"🔄 Generating {num_samples} domain-enhanced training samples...")
        
        # Load datasets
        datasets_path = Path("data/processed")
        singapore_datasets = pd.read_csv(datasets_path / "singapore_datasets.csv")
        global_datasets = pd.read_csv(datasets_path / "global_datasets.csv")
        all_datasets = pd.concat([singapore_datasets, global_datasets], ignore_index=True)
        
        # Generate samples by domain
        all_samples = []
        
        for domain, domain_config in self.domains.items():
            domain_samples = int(num_samples * domain_config['weight'])
            logger.info(f"  Generating {domain_samples} samples for {domain}")
            
            domain_data = self._generate_domain_samples(
                domain=domain,
                domain_config=domain_config,
                datasets=all_datasets,
                num_samples=domain_samples
            )
            
            all_samples.extend(domain_data)
        
        # Add existing high-quality scenarios
        existing_samples = self._load_existing_scenarios(all_datasets)
        all_samples.extend(existing_samples)
        
        # Calculate final statistics
        final_count = len(all_samples)
        score_distribution = self._calculate_score_distribution(all_samples)
        domain_distribution = self._calculate_domain_distribution(all_samples)
        
        # Create enhanced training data structure
        enhanced_data = {
            'metadata': {
                'version': '2.1',
                'created_date': datetime.now().isoformat(),
                'total_samples': final_count,
                'generation_method': 'domain_enhanced',
                'domain_weights': {domain: config['weight'] for domain, config in self.domains.items()},
                'score_distribution': score_distribution,
                'domain_distribution': domain_distribution,
                'graded_levels': [0.0, 0.3, 0.7, 1.0]
            },
            'training_samples': all_samples
        }
        
        # Save enhanced data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        logger.info(f"✅ Enhanced training data saved to {output_path}")
        logger.info(f"📊 Score distribution: {score_distribution}")
        logger.info(f"🏷️  Domain distribution: {domain_distribution}")
        
        return str(output_path)
    
    def _generate_domain_samples(self,
                               domain: str,
                               domain_config: Dict,
                               datasets: pd.DataFrame,
                               num_samples: int) -> List[Dict]:
        """Generate samples for a specific domain."""
        samples = []
        templates = domain_config['query_templates']
        
        for i in range(num_samples):
            # Select random template
            template = np.random.choice(templates)
            
            # Fill template with domain-specific values
            query = self._fill_template(template, domain_config)
            
            # Select relevant dataset (bias towards domain-relevant datasets)
            dataset = self._select_relevant_dataset(datasets, domain, domain_config)
            
            # Score relevance
            graded_score = self.scorer.score_relevance(query, dataset.to_dict())
            
            samples.append({
                'query': query,
                'dataset_id': dataset['dataset_id'],
                'relevance_score': graded_score,
                'query_type': 'domain_generated',
                'domain': domain,
                'template_used': template
            })
        
        return samples
    
    def _fill_template(self, template: str, domain_config: Dict) -> str:
        """Fill query template with random domain-specific values."""
        query = template
        
        # Replace placeholders with random values
        replacements = {
            '{area}': domain_config.get('areas', ['singapore']),
            '{year}': domain_config.get('years', ['2024', '2023', '2022']),
            '{line}': domain_config.get('lines', ['north-south', 'east-west']),
            '{mode}': domain_config.get('modes', ['mrt', 'bus']),
            '{specialty}': domain_config.get('specialties', ['general']),
            '{age_group}': domain_config.get('age_groups', ['adults']),
            '{disease}': domain_config.get('diseases', ['diabetes']),
            '{sector}': domain_config.get('sectors', ['services']),
            '{industry}': domain_config.get('industries', ['manufacturing']),
            '{level}': domain_config.get('levels', ['secondary']),
            '{field}': domain_config.get('fields', ['engineering']),
            '{room_type}': domain_config.get('room_types', ['3-room']),
            '{property_type}': domain_config.get('property_types', ['hdb']),
            '{income_group}': domain_config.get('income_groups', ['middle-income']),
            '{vehicle_type}': domain_config.get('vehicle_types', ['car']),
            '{time}': domain_config.get('times', ['peak-hours']),
            '{indicator}': domain_config.get('indicators', ['growth']),
            '{partner}': domain_config.get('partners', ['malaysia']),
            '{occupation}': domain_config.get('occupations', ['professional']),
            '{metric}': domain_config.get('metrics', ['performance']),
            '{characteristic}': domain_config.get('characteristics', ['nationality']),
            '{category}': domain_config.get('categories', ['public']),
            '{type}': domain_config.get('types', ['immigration']),
            '{gender}': domain_config.get('genders', ['overall'])
        }
        
        for placeholder, options in replacements.items():
            if placeholder in query:
                query = query.replace(placeholder, np.random.choice(options))
        
        return query
    
    def _select_relevant_dataset(self, 
                               datasets: pd.DataFrame, 
                               domain: str, 
                               domain_config: Dict) -> pd.Series:
        """Select dataset with bias towards domain relevance."""
        
        # Filter datasets that might be relevant to domain
        domain_keywords = domain_config.get('keywords', [])
        relevant_mask = datasets['title'].str.lower().str.contains('|'.join(domain_keywords), na=False)
        
        if relevant_mask.any() and np.random.random() < 0.7:  # 70% chance to select relevant
            relevant_datasets = datasets[relevant_mask]
            return relevant_datasets.sample(1).iloc[0]
        else:
            # Select random dataset (30% chance for diversity)
            return datasets.sample(1).iloc[0]
    
    def _load_existing_scenarios(self, datasets: pd.DataFrame) -> List[Dict]:
        """Load and enhance existing high-quality scenarios."""
        try:
            existing_path = Path("data/processed/intelligent_ground_truth.json")
            if not existing_path.exists():
                return []
                
            with open(existing_path, 'r') as f:
                existing_data = json.load(f)
            
            enhanced_scenarios = []
            for scenario in existing_data.get('ground_truth_scenarios', []):
                query = scenario['search_query']
                
                for dataset_id in scenario['relevant_datasets']:
                    dataset_row = datasets[datasets['dataset_id'] == dataset_id]
                    if not dataset_row.empty:
                        dataset = dataset_row.iloc[0].to_dict()
                        graded_score = self.scorer.score_relevance(query, dataset)
                        
                        enhanced_scenarios.append({
                            'query': query,
                            'dataset_id': dataset_id,
                            'relevance_score': graded_score,
                            'query_type': 'existing_scenario',
                            'domain': scenario.get('domain', 'general'),
                            'original_confidence': scenario.get('confidence_score', 0.8)
                        })
            
            logger.info(f"📚 Enhanced {len(enhanced_scenarios)} existing scenarios")
            return enhanced_scenarios
            
        except Exception as e:
            logger.warning(f"Could not load existing scenarios: {e}")
            return []
    
    def _calculate_score_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of relevance scores."""
        distribution = {}
        for sample in samples:
            score = sample['relevance_score']
            distribution[str(score)] = distribution.get(str(score), 0) + 1
        return distribution
    
    def _calculate_domain_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of domains."""
        distribution = {}
        for sample in samples:
            domain = sample['domain']
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    
    generator = DomainEnhancedTrainingGenerator()
    
    # Generate enhanced training data
    output_path = generator.generate_enhanced_training_data(
        num_samples=3000,
        output_path="data/processed/domain_enhanced_training_20250622.json"
    )
    
    logger.info(f"🎉 Domain-enhanced training data generation completed!")
    logger.info(f"📂 Output: {output_path}")


if __name__ == "__main__":
    main()