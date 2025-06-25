"""
Graded Relevance Enhancement for Neural Training
Converts binary relevance labels to 4-level graded relevance for better NDCG@3 performance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

class GradedRelevanceEnhancer:
    """Enhances training data with graded relevance scoring."""
    
    def __init__(self, thresholds: Dict = None):
        # Enhanced graded relevance thresholds for better NDCG@3 performance
        self.thresholds = thresholds or {
            'highly_relevant': 0.80,      # 1.0 - Perfect match (stricter)
            'relevant': 0.60,             # 0.7 - Good match (raised)
            'somewhat_relevant': 0.35,    # 0.3 - Partial match (raised)
            'irrelevant': 0.0             # 0.0 - No match
        }
        
        # Graded relevance values
        self.relevance_levels = {
            'highly_relevant': 1.0,
            'relevant': 0.7,
            'somewhat_relevant': 0.3,
            'irrelevant': 0.0
        }
    
    def convert_to_graded_relevance(self, relevance_score: float) -> float:
        """Convert continuous relevance score to graded relevance."""
        if relevance_score >= self.thresholds['highly_relevant']:
            return self.relevance_levels['highly_relevant']  # 1.0
        elif relevance_score >= self.thresholds['relevant']:
            return self.relevance_levels['relevant']  # 0.7
        elif relevance_score >= self.thresholds['somewhat_relevant']:
            return self.relevance_levels['somewhat_relevant']  # 0.3
        else:
            return self.relevance_levels['irrelevant']  # 0.0
    
    def create_advanced_graded_data(self, input_file: str, output_file: str, 
                                   semantic_boost: bool = True) -> Dict:
        """Create advanced graded relevance data with semantic enhancement."""
        
        logger.info(f"🚀 Creating advanced graded relevance data...")
        
        # Load existing data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        stats = {
            'total_samples': 0,
            'relevance_distribution': {1.0: 0, 0.7: 0, 0.3: 0, 0.0: 0},
            'semantic_enhancements': 0,
            'quality_improvements': 0
        }
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in data:
                for sample in data[split_name]:
                    stats['total_samples'] += 1
                    
                    # Get relevance score
                    relevance_score = sample.get('relevance_score', 0.0)
                    
                    # Apply semantic boost for partial matches
                    if semantic_boost and 0.2 <= relevance_score < 0.6:
                        # Boost partial semantic matches
                        query = sample.get('query', '').lower()
                        title = sample.get('title', '').lower()
                        
                        # Check for keyword overlap
                        query_words = set(query.split())
                        title_words = set(title.split())
                        overlap = len(query_words.intersection(title_words))
                        
                        if overlap > 0:
                            relevance_score = min(0.65, relevance_score + (overlap * 0.1))
                            sample['relevance_score'] = relevance_score
                            stats['semantic_enhancements'] += 1
                    
                    # Convert to graded relevance
                    graded_relevance = self.convert_to_graded_relevance(relevance_score)
                    
                    # Update sample
                    sample['graded_relevance'] = graded_relevance
                    sample['label'] = graded_relevance
                    
                    # Track distribution
                    stats['relevance_distribution'][graded_relevance] += 1
        
        # Update metadata
        data['metadata']['advanced_graded_relevance'] = True
        data['metadata']['graded_stats'] = stats
        data['metadata']['thresholds'] = self.thresholds
        
        # Save enhanced data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("✅ Advanced graded relevance data created!")
        logger.info(f"📊 Distribution: {stats['relevance_distribution']}")
        logger.info(f"🔧 Semantic enhancements: {stats['semantic_enhancements']}")
        
        return stats
    
    def enhance_training_data(self, input_file: str, output_file: str) -> Dict:
        """Enhance existing training data with graded relevance."""
        
        logger.info(f"🎯 Converting training data to graded relevance...")
        logger.info(f"📂 Input: {input_file}")
        logger.info(f"📂 Output: {output_file}")
        
        # Load existing data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Statistics tracking
        stats = {
            'total_samples': 0,
            'relevance_distribution': {1.0: 0, 0.7: 0, 0.3: 0, 0.0: 0},
            'conversion_stats': {'binary_to_graded': 0, 'unchanged': 0}
        }
        
        # Process each split
        for split_name in ['train', 'validation', 'test']:
            if split_name in data:
                logger.info(f"🔄 Processing {split_name} split...")
                
                for sample in data[split_name]:
                    stats['total_samples'] += 1
                    
                    # Get original relevance score
                    relevance_score = sample.get('relevance_score', 0.0)
                    original_label = sample.get('label', 0)
                    
                    # Convert to graded relevance
                    graded_relevance = self.convert_to_graded_relevance(relevance_score)
                    
                    # Update sample
                    sample['graded_relevance'] = graded_relevance
                    sample['original_binary_label'] = original_label
                    
                    # Update label to use graded relevance for training
                    sample['label'] = graded_relevance
                    
                    # Track statistics
                    stats['relevance_distribution'][graded_relevance] += 1
                    
                    if graded_relevance != original_label:
                        stats['conversion_stats']['binary_to_graded'] += 1
                    else:
                        stats['conversion_stats']['unchanged'] += 1
        
        # Update metadata
        data['metadata']['graded_relevance_enhanced'] = True
        data['metadata']['graded_relevance_stats'] = stats
        data['metadata']['graded_relevance_thresholds'] = self.thresholds
        
        # Save enhanced data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("✅ Graded relevance enhancement complete!")
        logger.info(f"📊 Total samples processed: {stats['total_samples']}")
        logger.info(f"📈 Relevance distribution:")
        for relevance, count in stats['relevance_distribution'].items():
            percentage = (count / stats['total_samples']) * 100
            logger.info(f"  {relevance}: {count} samples ({percentage:.1f}%)")
        
        logger.info(f"🔄 Conversion stats:")
        logger.info(f"  Binary→Graded: {stats['conversion_stats']['binary_to_graded']}")
        logger.info(f"  Unchanged: {stats['conversion_stats']['unchanged']}")
        
        return stats

def enhance_existing_training_data():
    """Enhance the existing training data with graded relevance."""
    
    enhancer = GradedRelevanceEnhancer()
    
    input_file = "data/processed/enhanced_training_data.json"
    output_file = "data/processed/enhanced_training_data_graded.json"
    
    if not Path(input_file).exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return None
    
    try:
        stats = enhancer.enhance_training_data(input_file, output_file)
        logger.info(f"✅ Enhanced training data saved to: {output_file}")
        return stats
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    enhance_existing_training_data()