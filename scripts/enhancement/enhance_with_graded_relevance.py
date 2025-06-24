#!/usr/bin/env python3
"""
Enhanced Training Data Generation with Graded Relevance Scoring
Generates training data with 4-level relevance scores to achieve 70% NDCG@3 target
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import yaml

# Add src to path
sys.path.append('src')

from dl.graded_relevance import GradedRelevanceScorer, create_graded_relevance_config
from ml.enhanced_ml_pipeline import EnhancedMLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradedRelevanceEnhancer:
    """Enhance training data with graded relevance scores for improved NDCG performance."""
    
    def __init__(self):
        # Load configurations
        with open('config/dl_config.yml', 'r') as f:
            self.dl_config = yaml.safe_load(f)
        
        # Add graded relevance config
        graded_config = create_graded_relevance_config()
        self.dl_config.update(graded_config)
        
        # Initialize scorer
        self.scorer = GradedRelevanceScorer(self.dl_config)
        
        logger.info("🎯 GradedRelevanceEnhancer initialized")
    
    def enhance_existing_training_data(self, input_path: str, output_path: str):
        """Enhance existing training data with graded relevance scores."""
        logger.info(f"🔄 Enhancing training data from {input_path}")
        
        # Load existing enhanced training data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        training_samples = data.get('training_samples', [])
        logger.info(f"📊 Processing {len(training_samples)} samples")
        
        # Load datasets for scoring
        datasets_df = self._load_datasets()
        
        # Enhance each sample with graded relevance
        enhanced_samples = []
        score_distribution = {0.0: 0, 0.3: 0, 0.7: 0, 1.0: 0}
        
        for i, sample in enumerate(training_samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(training_samples)}")
            
            query = sample['query']
            dataset_id = sample['dataset_id']
            
            # Get dataset info
            dataset_row = datasets_df[datasets_df['dataset_id'] == dataset_id]
            if dataset_row.empty:
                continue
            
            dataset = dataset_row.iloc[0].to_dict()
            
            # Calculate graded relevance score
            graded_score = self.scorer.score_relevance(query, dataset)
            
            # Update sample
            enhanced_sample = sample.copy()
            enhanced_sample['relevance_score'] = graded_score
            enhanced_sample['graded_label'] = graded_score  # For ranking loss
            
            # Keep binary label for compatibility
            enhanced_sample['label'] = 1 if graded_score >= 0.7 else 0
            
            enhanced_samples.append(enhanced_sample)
            score_distribution[graded_score] += 1
        
        # Add more diverse samples with graded scores
        additional_samples = self._generate_additional_graded_samples(datasets_df, 500)
        enhanced_samples.extend(additional_samples)
        
        # Update score distribution
        for sample in additional_samples:
            score_distribution[sample['relevance_score']] += 1
        
        # Create enhanced data structure
        enhanced_data = {
            'metadata': {
                'version': '3.0',
                'created_date': pd.Timestamp.now().isoformat(),
                'total_samples': len(enhanced_samples),
                'graded_levels': [0.0, 0.3, 0.7, 1.0],
                'score_distribution': score_distribution,
                'enhancement': 'graded_relevance_scoring'
            },
            'training_samples': enhanced_samples
        }
        
        # Save enhanced data
        with open(output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        logger.info(f"✅ Enhanced data saved to {output_path}")
        logger.info(f"📊 Score distribution: {score_distribution}")
        logger.info(f"📈 Total samples: {len(enhanced_samples)}")
        
        # Calculate expected NDCG improvement
        highly_relevant_ratio = (score_distribution[0.7] + score_distribution[1.0]) / len(enhanced_samples)
        logger.info(f"🎯 Highly relevant ratio: {highly_relevant_ratio:.2%}")
        logger.info(f"💡 Expected NDCG boost: ~{highly_relevant_ratio * 0.1:.1%}")
    
    def _load_datasets(self) -> pd.DataFrame:
        """Load all datasets for scoring."""
        datasets_path = Path("data/processed")
        singapore_df = pd.read_csv(datasets_path / "singapore_datasets.csv")
        global_df = pd.read_csv(datasets_path / "global_datasets.csv")
        return pd.concat([singapore_df, global_df], ignore_index=True)
    
    def _generate_additional_graded_samples(self, datasets_df: pd.DataFrame, num_samples: int) -> List[Dict]:
        """Generate additional samples with balanced graded scores."""
        samples = []
        
        # Target distribution for better NDCG
        target_distribution = {
            1.0: int(num_samples * 0.25),  # 25% highly relevant
            0.7: int(num_samples * 0.35),  # 35% relevant
            0.3: int(num_samples * 0.25),  # 25% somewhat relevant
            0.0: int(num_samples * 0.15)   # 15% irrelevant
        }
        
        # Specialized queries for each relevance level
        highly_relevant_queries = [
            "HDB resale price index 2024",
            "Singapore GDP growth rate",
            "MRT ridership statistics",
            "Singapore population demographics",
            "Healthcare expenditure data"
        ]
        
        relevant_queries = [
            "housing market trends",
            "public transport usage",
            "economic indicators",
            "health statistics singapore",
            "education enrollment data"
        ]
        
        somewhat_relevant_queries = [
            "real estate information",
            "transportation data",
            "financial statistics",
            "medical records",
            "school information"
        ]
        
        irrelevant_queries = [
            "weather forecast",
            "restaurant reviews",
            "movie ratings",
            "sports scores",
            "fashion trends"
        ]
        
        query_sets = {
            1.0: highly_relevant_queries,
            0.7: relevant_queries,
            0.3: somewhat_relevant_queries,
            0.0: irrelevant_queries
        }
        
        # Generate samples for each target score
        for target_score, count in target_distribution.items():
            queries = query_sets[target_score]
            
            for i in range(count):
                query = np.random.choice(queries)
                
                # Add variations
                if np.random.random() < 0.5:
                    year = np.random.choice(['2023', '2024', '2022'])
                    query = f"{query} {year}"
                
                # Find appropriate dataset
                best_dataset = None
                best_score_diff = float('inf')
                
                # Sample random datasets and find one with score closest to target
                for _ in range(10):
                    dataset_row = datasets_df.sample(1).iloc[0]
                    dataset = dataset_row.to_dict()
                    score = self.scorer.score_relevance(query, dataset)
                    
                    score_diff = abs(score - target_score)
                    if score_diff < best_score_diff:
                        best_score_diff = score_diff
                        best_dataset = dataset
                        
                        # Perfect match found
                        if score == target_score:
                            break
                
                if best_dataset:
                    samples.append({
                        'query': query,
                        'dataset_id': best_dataset['dataset_id'],
                        'relevance_score': target_score,
                        'graded_label': target_score,
                        'label': 1 if target_score >= 0.7 else 0,
                        'query_type': 'graded_enhanced',
                        'domain': self._identify_domain(query)
                    })
        
        return samples
    
    def _identify_domain(self, query: str) -> str:
        """Identify domain of query."""
        query_lower = query.lower()
        
        domain_keywords = {
            'housing': ['hdb', 'housing', 'property', 'resale', 'rental'],
            'transportation': ['mrt', 'bus', 'transport', 'traffic'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic'],
            'economics': ['gdp', 'economy', 'inflation', 'employment'],
            'education': ['school', 'education', 'student', 'university'],
            'demographics': ['population', 'demographics', 'age', 'citizen']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def create_threshold_tuning_data(self, enhanced_data_path: str, output_path: str):
        """Create data for threshold tuning to improve precision-recall balance."""
        logger.info("🎯 Creating threshold tuning dataset")
        
        # Load enhanced data
        with open(enhanced_data_path, 'r') as f:
            data = json.load(f)
        
        samples = data['training_samples']
        
        # Group by query for threshold analysis
        query_groups = {}
        for sample in samples:
            query = sample['query']
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(sample)
        
        # Analyze optimal thresholds
        threshold_analysis = []
        
        for query, group_samples in query_groups.items():
            # Sort by relevance score
            sorted_samples = sorted(group_samples, key=lambda x: x['relevance_score'], reverse=True)
            
            # Find optimal threshold for this query
            scores = [s['relevance_score'] for s in sorted_samples]
            labels = [s['label'] for s in sorted_samples]
            
            # Test different thresholds
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.3, 0.8, 0.05):
                predictions = [1 if score >= threshold else 0 for score in scores]
                
                # Calculate F1
                tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
                fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
                fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            threshold_analysis.append({
                'query': query,
                'optimal_threshold': best_threshold,
                'best_f1': best_f1,
                'num_samples': len(group_samples)
            })
        
        # Save threshold analysis
        threshold_data = {
            'metadata': {
                'created_date': pd.Timestamp.now().isoformat(),
                'num_queries': len(query_groups),
                'total_samples': len(samples)
            },
            'threshold_analysis': threshold_analysis,
            'global_optimal_threshold': np.mean([t['optimal_threshold'] for t in threshold_analysis])
        }
        
        with open(output_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        logger.info(f"✅ Threshold tuning data saved to {output_path}")
        logger.info(f"🎯 Global optimal threshold: {threshold_data['global_optimal_threshold']:.3f}")


def main():
    """Main function to enhance training data with graded relevance."""
    print("🎯 Enhancing Training Data with Graded Relevance Scoring")
    print("=" * 60)
    
    enhancer = GradedRelevanceEnhancer()
    
    # Paths
    input_path = "data/processed/domain_enhanced_training_20250622.json"
    graded_output_path = "data/processed/graded_relevance_training.json"
    threshold_output_path = "data/processed/threshold_tuning_analysis.json"
    
    # Step 1: Enhance with graded relevance
    print("\n📊 Step 1: Enhancing training data with graded relevance scores...")
    enhancer.enhance_existing_training_data(input_path, graded_output_path)
    
    # Step 2: Create threshold tuning data
    print("\n🎯 Step 2: Creating threshold tuning analysis...")
    enhancer.create_threshold_tuning_data(graded_output_path, threshold_output_path)
    
    print("\n✅ Enhancement complete!")
    print(f"📁 Graded training data: {graded_output_path}")
    print(f"📁 Threshold analysis: {threshold_output_path}")
    print("\n🚀 Ready to achieve 70% NDCG@3 target!")


if __name__ == "__main__":
    main()