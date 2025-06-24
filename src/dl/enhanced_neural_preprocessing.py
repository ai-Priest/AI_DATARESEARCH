"""
Enhanced Neural Preprocessing Module - Uses the new 1914-sample training data
Handles the enhanced training data with proper negative sampling and ranking pairs
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RankingDataset(Dataset):
    """PyTorch Dataset for ranking tasks with positive/negative pairs."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Created RankingDataset with {len(samples)} samples")
        
        # Log sample distribution
        positive_count = sum(1 for s in samples if s.get('label') == 1)
        negative_count = sum(1 for s in samples if s.get('label') == 0)
        logger.info(f"  Positive samples: {positive_count}")
        logger.info(f"  Negative samples: {negative_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get text data
        query = sample.get('query', '')
        dataset_title = sample.get('dataset_title', '')
        dataset_description = sample.get('dataset_description', '')
        
        # Combine dataset info
        dataset_text = f"{dataset_title}. {dataset_description}".strip()
        
        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize dataset
        dataset_tokens = self.tokenizer(
            dataset_text,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels and scores
        relevance_score = float(sample.get('relevance_score', 0.0))
        label = int(sample.get('label', 0))
        
        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(),
            'dataset_input_ids': dataset_tokens['input_ids'].squeeze(),
            'dataset_attention_mask': dataset_tokens['attention_mask'].squeeze(),
            'relevance_score': torch.tensor(relevance_score, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_type': sample.get('sample_type', 'unknown'),
            'dataset_id': sample.get('dataset_id', ''),
            'original_query': query,
            'original_dataset_text': dataset_text
        }

class EnhancedNeuralPreprocessor:
    """Enhanced preprocessing using the new 1914-sample training data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data_processing', {})
        self.enhanced_data_path = Path("data/processed/enhanced_training_data.json")
        
        # Initialize tokenizer
        model_name = self.data_config.get('neural_preprocessing', {}).get('text_processing', {}).get('tokenization', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("🚀 Enhanced Neural Preprocessor initialized")
        logger.info(f"Using tokenizer: {model_name}")
    
    def load_enhanced_data(self) -> Dict:
        """Load the enhanced training data."""
        if not self.enhanced_data_path.exists():
            raise FileNotFoundError(f"Enhanced training data not found at {self.enhanced_data_path}")
        
        with open(self.enhanced_data_path, 'r') as f:
            data = json.load(f)
        
        logger.info("✅ Enhanced training data loaded")
        logger.info(f"  Total samples: {data['metadata']['summary']['total_samples']}")
        logger.info(f"  Train: {len(data['train'])}")
        logger.info(f"  Validation: {len(data['validation'])}")
        logger.info(f"  Test: {len(data['test'])}")
        
        return data
    
    def create_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders for train/val/test."""
        
        # Load enhanced data
        enhanced_data = self.load_enhanced_data()
        
        # Create datasets
        train_dataset = RankingDataset(
            enhanced_data['train'], 
            self.tokenizer,
            max_length=self.data_config.get('neural_preprocessing', {}).get('text_processing', {}).get('max_length', 256)
        )
        
        val_dataset = RankingDataset(
            enhanced_data['validation'],
            self.tokenizer,
            max_length=self.data_config.get('neural_preprocessing', {}).get('text_processing', {}).get('max_length', 256)
        )
        
        test_dataset = RankingDataset(
            enhanced_data['test'],
            self.tokenizer,
            max_length=self.data_config.get('neural_preprocessing', {}).get('text_processing', {}).get('max_length', 256)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info("✅ Created enhanced data loaders")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self) -> Dict:
        """Get a sample batch for testing."""
        train_loader, _, _ = self.create_data_loaders(batch_size=4)
        
        for batch in train_loader:
            logger.info("📦 Sample batch created:")
            logger.info(f"  Query input shape: {batch['query_input_ids'].shape}")
            logger.info(f"  Dataset input shape: {batch['dataset_input_ids'].shape}")
            logger.info(f"  Labels: {batch['label']}")
            logger.info(f"  Relevance scores: {batch['relevance_score']}")
            logger.info(f"  Sample types: {batch['sample_type']}")
            return batch
    
    def preprocess_for_ranking(self) -> Dict[str, Any]:
        """Main preprocessing method that returns enhanced data for ranking."""
        
        logger.info("🔄 Starting enhanced neural preprocessing...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            batch_size=self.config.get('training', {}).get('batch_size', 32)
        )
        
        # Get metadata
        enhanced_data = self.load_enhanced_data()
        metadata = enhanced_data['metadata']
        
        # Prepare return data
        processed_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'metadata': metadata,
            'tokenizer': self.tokenizer,
            'config': self.config,
            'preprocessing_type': 'enhanced_ranking',
            'sample_count': {
                'train': len(enhanced_data['train']),
                'validation': len(enhanced_data['validation']),
                'test': len(enhanced_data['test'])
            }
        }
        
        logger.info("✅ Enhanced neural preprocessing complete")
        logger.info(f"  Ready for ranking-based training with {metadata['summary']['total_samples']} samples")
        
        return processed_data

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing."""
    import yaml
    
    # Load config
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test preprocessing
    preprocessor = EnhancedNeuralPreprocessor(config)
    
    # Test data loading
    try:
        processed_data = preprocessor.preprocess_for_ranking()
        print("✅ Enhanced preprocessing test successful!")
        print(f"Train samples: {processed_data['sample_count']['train']}")
        print(f"Val samples: {processed_data['sample_count']['validation']}")
        print(f"Test samples: {processed_data['sample_count']['test']}")
        
        # Test a sample batch
        sample_batch = preprocessor.get_sample_batch()
        print(f"Sample batch keys: {sample_batch.keys()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_preprocessing()