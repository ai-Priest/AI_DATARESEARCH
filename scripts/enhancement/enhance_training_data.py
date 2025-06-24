#!/usr/bin/env python3
"""
Enhanced Training Data Generator
Fixes the critical data shortage problem (100 → 1000+ samples)
"""

import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import itertools
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataEnhancer:
    """Generates high-quality training data from existing ground truth and datasets."""
    
    def __init__(self):
        self.ground_truth_path = Path("data/processed/intelligent_ground_truth.json")
        self.datasets_path = Path("data/processed/singapore_datasets.csv")
        self.output_path = Path("data/processed/enhanced_training_data.json")
        
        # Load existing data
        self.ground_truth = self._load_ground_truth()
        self.datasets = self._load_datasets()
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth scenarios")
        logger.info(f"Loaded {len(self.datasets)} datasets")
    
    def _load_ground_truth(self) -> Dict:
        """Load the current ground truth."""
        if not self.ground_truth_path.exists():
            logger.error(f"Ground truth not found at {self.ground_truth_path}")
            return {}
        
        with open(self.ground_truth_path, 'r') as f:
            return json.load(f)
    
    def _load_datasets(self) -> pd.DataFrame:
        """Load the datasets CSV."""
        if not self.datasets_path.exists():
            logger.error(f"Datasets not found at {self.datasets_path}")
            return pd.DataFrame()
        
        return pd.read_csv(self.datasets_path)
    
    def generate_query_variations(self, base_query: str) -> List[str]:
        """Generate variations of a query through paraphrasing and augmentation."""
        variations = [base_query]
        
        # Simple paraphrasing patterns
        paraphrase_patterns = [
            # Original query variations
            base_query,
            f"{base_query} data",
            f"{base_query} information",
            f"{base_query} statistics", 
            f"{base_query} analysis",
            f"{base_query} trends",
            f"{base_query} report",
            f"{base_query} dataset",
            
            # Question forms
            f"what is {base_query}",
            f"show me {base_query}",
            f"find {base_query}",
            f"get {base_query}",
            f"search {base_query}",
            
            # More specific forms
            f"{base_query} singapore",
            f"singapore {base_query}",
            f"{base_query} sg",
            f"latest {base_query}",
            f"recent {base_query}",
            f"current {base_query}",
        ]
        
        # Word substitutions
        substitutions = {
            'housing': ['property', 'residential', 'home', 'apartment', 'hdb'],
            'prices': ['cost', 'pricing', 'rates', 'value', 'expense'],
            'traffic': ['transport', 'transportation', 'road', 'vehicle'],
            'data': ['information', 'statistics', 'numbers', 'figures'],
            'economic': ['economy', 'financial', 'business', 'trade'],
            'population': ['demographic', 'people', 'residents', 'citizens'],
            'employment': ['jobs', 'work', 'labor', 'career'],
            'education': ['school', 'learning', 'university', 'academic']
        }
        
        # Apply substitutions
        words = base_query.lower().split()
        for i, word in enumerate(words):
            if word in substitutions:
                for substitute in substitutions[word][:2]:  # Take first 2 substitutes
                    new_words = words.copy()
                    new_words[i] = substitute
                    variations.append(' '.join(new_words))
        
        return list(set(variations))  # Remove duplicates
    
    def create_negative_samples(self, positive_query: str, positive_datasets: List[str]) -> List[Dict]:
        """Create negative samples for a positive query-dataset pair."""
        negative_samples = []
        
        # Get all available dataset IDs
        all_dataset_ids = set(self.datasets['dataset_id'].tolist())
        positive_dataset_ids = set(positive_datasets)
        negative_dataset_ids = list(all_dataset_ids - positive_dataset_ids)
        
        # Sample 5 negative datasets for each positive query
        if len(negative_dataset_ids) >= 5:
            sampled_negatives = random.sample(negative_dataset_ids, 5)
        else:
            sampled_negatives = negative_dataset_ids
        
        for neg_dataset_id in sampled_negatives:
            # Get dataset info
            dataset_info = self.datasets[self.datasets['dataset_id'] == neg_dataset_id]
            if not dataset_info.empty:
                dataset_row = dataset_info.iloc[0]
                negative_samples.append({
                    'query': positive_query,
                    'dataset_id': neg_dataset_id,
                    'dataset_title': dataset_row.get('title', ''),
                    'dataset_description': dataset_row.get('description', ''),
                    'relevance_score': 0.0,  # Negative sample
                    'label': 0,  # Not relevant
                    'sample_type': 'negative'
                })
        
        return negative_samples
    
    def create_training_pairs(self) -> List[Dict]:
        """Create training pairs from ground truth with query variations and negative sampling."""
        training_pairs = []
        
        for scenario_id, scenario_data in self.ground_truth.items():
            primary_query = scenario_data.get('primary', '')
            complementary_datasets = scenario_data.get('complementary', [])
            confidence = scenario_data.get('confidence', 0.5)
            category = scenario_data.get('category', 'general')
            
            if not primary_query or not complementary_datasets:
                continue
            
            # Generate query variations
            query_variations = self.generate_query_variations(primary_query)
            
            # Create positive samples for each query variation
            for query_variant in query_variations:
                for dataset_title in complementary_datasets:
                    # Find dataset ID by title
                    matching_datasets = self.datasets[
                        self.datasets['title'].str.contains(dataset_title.split(' - ')[0], 
                                                           case=False, na=False)
                    ]
                    
                    if not matching_datasets.empty:
                        dataset_row = matching_datasets.iloc[0]
                        
                        # Create positive sample
                        positive_sample = {
                            'query': query_variant,
                            'dataset_id': dataset_row['dataset_id'],
                            'dataset_title': dataset_row['title'],
                            'dataset_description': dataset_row.get('description', ''),
                            'relevance_score': confidence,  # Use confidence as relevance
                            'label': 1,  # Relevant
                            'sample_type': 'positive',
                            'original_scenario': scenario_id,
                            'category': category,
                            'query_type': 'variation' if query_variant != primary_query else 'original'
                        }
                        training_pairs.append(positive_sample)
                        
                        # Create negative samples
                        negative_samples = self.create_negative_samples(
                            query_variant, 
                            [dataset_row['dataset_id']]
                        )
                        training_pairs.extend(negative_samples)
        
        return training_pairs
    
    def create_cross_category_samples(self) -> List[Dict]:
        """Create cross-category training samples for better generalization."""
        cross_samples = []
        
        # Group scenarios by category
        categories = defaultdict(list)
        for scenario_id, scenario_data in self.ground_truth.items():
            category = scenario_data.get('category', 'general')
            categories[category].append((scenario_id, scenario_data))
        
        # Create cross-category negative samples
        for cat1, scenarios1 in categories.items():
            for cat2, scenarios2 in categories.items():
                if cat1 != cat2:  # Different categories
                    # Take a few samples from each category
                    sample1 = random.sample(scenarios1, min(2, len(scenarios1)))
                    sample2 = random.sample(scenarios2, min(2, len(scenarios2)))
                    
                    for (_, scenario1), (_, scenario2) in itertools.product(sample1, sample2):
                        query1 = scenario1.get('primary', '')
                        datasets2 = scenario2.get('complementary', [])
                        
                        if query1 and datasets2:
                            for dataset_title in datasets2[:2]:  # Take first 2
                                matching_datasets = self.datasets[
                                    self.datasets['title'].str.contains(
                                        dataset_title.split(' - ')[0], 
                                        case=False, na=False
                                    )
                                ]
                                
                                if not matching_datasets.empty:
                                    dataset_row = matching_datasets.iloc[0]
                                    
                                    cross_sample = {
                                        'query': query1,
                                        'dataset_id': dataset_row['dataset_id'],
                                        'dataset_title': dataset_row['title'],
                                        'dataset_description': dataset_row.get('description', ''),
                                        'relevance_score': 0.1,  # Low relevance
                                        'label': 0,  # Not relevant (cross-category)
                                        'sample_type': 'cross_category_negative',
                                        'query_category': cat1,
                                        'dataset_category': cat2
                                    }
                                    cross_samples.append(cross_sample)
        
        return cross_samples
    
    def create_hard_negatives(self) -> List[Dict]:
        """Create hard negative samples (similar queries with irrelevant datasets)."""
        hard_negatives = []
        
        # Keywords that might cause confusion
        confusing_keywords = {
            'housing': ['construction', 'building', 'infrastructure'],
            'traffic': ['logistics', 'supply', 'movement'],
            'economic': ['statistics', 'numbers', 'data'],
            'population': ['survey', 'census', 'count'],
            'employment': ['business', 'industry', 'sector']
        }
        
        for scenario_id, scenario_data in self.ground_truth.items():
            primary_query = scenario_data.get('primary', '')
            category = scenario_data.get('category', 'general')
            
            if category in confusing_keywords:
                for confusing_keyword in confusing_keywords[category]:
                    # Create confusing query
                    confusing_query = primary_query.replace(
                        category, confusing_keyword
                    ) if category in primary_query else f"{confusing_keyword} {primary_query}"
                    
                    # Find datasets that match the confusing keyword but not the intent
                    confusing_datasets = self.datasets[
                        self.datasets['title'].str.contains(confusing_keyword, case=False, na=False) |
                        self.datasets['description'].str.contains(confusing_keyword, case=False, na=False)
                    ]
                    
                    for _, dataset_row in confusing_datasets.head(3).iterrows():
                        hard_negative = {
                            'query': confusing_query,
                            'dataset_id': dataset_row['dataset_id'],
                            'dataset_title': dataset_row['title'],
                            'dataset_description': dataset_row.get('description', ''),
                            'relevance_score': 0.2,  # Low but not zero
                            'label': 0,  # Not relevant
                            'sample_type': 'hard_negative',
                            'difficulty': 'hard',
                            'original_query': primary_query
                        }
                        hard_negatives.append(hard_negative)
        
        return hard_negatives
    
    def create_enhanced_training_data(self) -> Dict:
        """Create the complete enhanced training dataset."""
        logger.info("🚀 Starting enhanced training data generation...")
        
        # Generate all types of training samples
        basic_pairs = self.create_training_pairs()
        cross_category = self.create_cross_category_samples()
        hard_negatives = self.create_hard_negatives()
        
        # Combine all samples
        all_samples = basic_pairs + cross_category + hard_negatives
        
        # Shuffle and create train/val/test splits
        random.shuffle(all_samples)
        
        total_samples = len(all_samples)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        test_samples = all_samples[train_size + val_size:]
        
        # Create summary statistics
        summary = {
            'total_samples': total_samples,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'positive_samples': len([s for s in all_samples if s.get('label') == 1]),
            'negative_samples': len([s for s in all_samples if s.get('label') == 0]),
            'sample_types': {
                'positive': len([s for s in all_samples if s.get('sample_type') == 'positive']),
                'negative': len([s for s in all_samples if s.get('sample_type') == 'negative']),
                'cross_category_negative': len([s for s in all_samples if s.get('sample_type') == 'cross_category_negative']),
                'hard_negative': len([s for s in all_samples if s.get('sample_type') == 'hard_negative'])
            },
            'categories_covered': len(set(s.get('category', 'unknown') for s in all_samples if 'category' in s))
        }
        
        enhanced_data = {
            'metadata': {
                'version': '1.0',
                'created_from': str(self.ground_truth_path),
                'total_original_scenarios': len(self.ground_truth),
                'enhancement_methods': [
                    'query_variations',
                    'negative_sampling', 
                    'cross_category_sampling',
                    'hard_negative_generation'
                ],
                'summary': summary
            },
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        return enhanced_data
    
    def save_enhanced_data(self, enhanced_data: Dict):
        """Save the enhanced training data."""
        # Save main enhanced data
        with open(self.output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        logger.info(f"✅ Enhanced training data saved to {self.output_path}")
        
        # Save summary report
        summary_path = Path("data/processed/training_data_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(enhanced_data['metadata'], f, indent=2)
        
        # Print summary
        metadata = enhanced_data['metadata']
        summary = metadata['summary']
        
        print("\n" + "="*60)
        print("🎉 ENHANCED TRAINING DATA GENERATION COMPLETE")
        print("="*60)
        print(f"📊 Total Samples Generated: {summary['total_samples']}")
        print(f"   ├─ Training: {summary['train_samples']}")
        print(f"   ├─ Validation: {summary['val_samples']}")
        print(f"   └─ Test: {summary['test_samples']}")
        print(f"\n📈 Sample Distribution:")
        print(f"   ├─ Positive samples: {summary['positive_samples']}")
        print(f"   └─ Negative samples: {summary['negative_samples']}")
        print(f"\n🎯 Sample Types:")
        for sample_type, count in summary['sample_types'].items():
            print(f"   ├─ {sample_type}: {count}")
        print(f"\n📋 Categories covered: {summary['categories_covered']}")
        print(f"\n💾 Files created:")
        print(f"   ├─ {self.output_path}")
        print(f"   └─ {summary_path}")

def main():
    """Main function to enhance training data."""
    enhancer = TrainingDataEnhancer()
    
    if not enhancer.ground_truth:
        logger.error("❌ Cannot proceed without ground truth data")
        return
    
    if enhancer.datasets.empty:
        logger.error("❌ Cannot proceed without datasets")
        return
    
    # Generate enhanced training data
    enhanced_data = enhancer.create_enhanced_training_data()
    
    # Save the data
    enhancer.save_enhanced_data(enhanced_data)
    
    print(f"\n🚀 Ready for training! Use this data with:")
    print(f"   python dl_pipeline.py --use-enhanced-data")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()