"""
Training Script for Quality-Aware Model
Demonstrates how to train the model with quality-first approach
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dl.quality_aware_trainer import QualityAwareTrainer, TrainingConfig
from src.dl.quality_first_neural_model import QualityAwareRankingModel
from src.ml.enhanced_training_data_parser import TrainingDataIntegrator
import torch
from torch.utils.data import Dataset
import json

class TrainingDataset(Dataset):
    """Dataset for training the quality-aware model"""
    
    def __init__(self, training_data: list):
        self.data = training_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert to training format
        return {
            'query_ids': self._tokenize_query(item['query']),
            'source_ids': self._get_source_id(item['source']),
            'relevance_score': float(item['relevance_score']),
            'domain_id': self._get_domain_id(item.get('domain', 'general')),
            'singapore_label': 1 if item.get('singapore_first', False) else 0
        }
    
    def _tokenize_query(self, query: str) -> list:
        """Simple tokenization (replace with proper tokenizer)"""
        words = query.lower().split()
        # Simple hash-based mapping
        token_ids = []
        for word in words[:20]:  # Limit to 20 tokens
            token_id = hash(word) % 9999 + 1
            token_ids.append(token_id)
        
        # Pad to fixed length
        while len(token_ids) < 20:
            token_ids.append(0)  # Padding token
        
        return token_ids[:20]
    
    def _get_source_id(self, source: str) -> int:
        """Map source name to ID"""
        source_map = {
            'kaggle': 0,
            'zenodo': 1,
            'world_bank': 2,
            'data_gov_sg': 3,
            'singstat': 4,
            'lta_datamall': 5,
            'aws_opendata': 6,
            'data_un': 7,
            'arxiv': 8,
            'github': 9
        }
        return source_map.get(source.lower(), 0)
    
    def _get_domain_id(self, domain: str) -> int:
        """Map domain name to ID"""
        domain_map = {
            'psychology': 0,
            'machine_learning': 1,
            'climate': 2,
            'economics': 3,
            'singapore': 4,
            'health': 5,
            'education': 6,
            'general': 7
        }
        return domain_map.get(domain.lower(), 7)

def load_training_data():
    """Load and prepare training data"""
    logger = logging.getLogger(__name__)
    logger.info("📚 Loading training data...")
    
    # Load training mappings
    integrator = TrainingDataIntegrator()
    training_examples = integrator.parse_training_mappings()
    
    # Convert to simple format for training
    training_data = []
    for example in training_examples:
        training_data.append({
            'query': example.query,
            'source': example.source,
            'relevance_score': example.relevance_score,
            'domain': example.domain,
            'singapore_first': example.singapore_first_applicable
        })
    
    if not training_data:
        logger.warning("⚠️ No training data found, creating synthetic examples")
        # Create some synthetic training examples for demonstration
        training_data = [
            {
                'query': 'psychology research data',
                'source': 'kaggle',
                'relevance_score': 0.95,
                'domain': 'psychology',
                'singapore_first': False
            },
            {
                'query': 'singapore housing statistics',
                'source': 'data_gov_sg',
                'relevance_score': 0.98,
                'domain': 'singapore',
                'singapore_first': True
            },
            {
                'query': 'climate change indicators',
                'source': 'world_bank',
                'relevance_score': 0.90,
                'domain': 'climate',
                'singapore_first': False
            },
            {
                'query': 'machine learning datasets',
                'source': 'kaggle',
                'relevance_score': 0.92,
                'domain': 'machine_learning',
                'singapore_first': False
            },
            {
                'query': 'singapore transportation data',
                'source': 'lta_datamall',
                'relevance_score': 0.96,
                'domain': 'transportation',
                'singapore_first': True
            }
        ]
    
    logger.info(f"✅ Loaded {len(training_data)} training examples")
    
    # Split into train/validation
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Create datasets
    train_dataset = TrainingDataset(train_data)
    val_dataset = TrainingDataset(val_data)
    
    logger.info(f"📊 Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def main():
    """Main training function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Quality-Aware Model Training")
    
    # Load training data
    train_dataset, val_dataset = load_training_data()
    
    # Create model
    model_config = {
        'vocab_size': 10000,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_sources': 10,
        'num_domains': 8,
        'dropout': 0.1
    }
    
    model = QualityAwareRankingModel(model_config)
    logger.info(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=8,  # Small batch for demo
        max_epochs=20,  # Reduced for demo
        patience=5,
        curriculum_stages=2,  # Simplified curriculum
        ndcg_target=0.7,
        validation_frequency=2
    )
    
    # Create trainer
    trainer = QualityAwareTrainer(model, config)
    
    # Train model
    try:
        training_history = trainer.train(train_dataset, val_dataset)
        
        # Print results
        logger.info("📊 Training Results:")
        logger.info(f"  Best NDCG@3: {trainer.best_ndcg:.4f}")
        logger.info(f"  Total epochs: {len(training_history)}")
        logger.info(f"  Target achieved: {trainer.best_ndcg >= config.ndcg_target}")
        
        # Save summary
        summary_path = Path("outputs/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'best_ndcg_3': float(trainer.best_ndcg),
            'total_epochs': len(training_history),
            'target_achieved': bool(trainer.best_ndcg >= config.ndcg_target),
            'final_epoch_metrics': training_history[-1] if training_history else None
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Training summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()