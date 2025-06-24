"""
Enhanced Training Pipeline with Graded Relevance Scoring
Implements 4-level relevance training for improved NDCG performance
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dl.graded_relevance import GradedRelevanceScorer, create_graded_relevance_config
from src.dl.improved_model_architecture import BERTCrossAttentionRanker
from src.dl.neural_preprocessing import NeuralDataPreprocessor
from src.dl.ranking_losses import NDCGLoss
from src.dl.deep_evaluation import DeepEvaluationMetrics

logger = logging.getLogger(__name__)


class GradedRelevanceTrainer:
    """
    Enhanced trainer for graded relevance scoring system.
    """
    
    def __init__(self, config_path: str = "config/dl_config.yml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize components
        self.graded_scorer = GradedRelevanceScorer(self.config)
        self.preprocessor = NeuralDataPreprocessor(self.config)
        
        # Model configuration
        self.model_config = {
            'model_name': 'distilbert-base-uncased',  # Faster than BERT
            'hidden_dim': 768,
            'dropout': 0.3,
            'num_attention_heads': 8
        }
        
        logger.info(f"🚀 GradedRelevanceTrainer initialized on {self.device}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration with graded relevance settings."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = {}
        
        # Add graded relevance configuration
        graded_config = create_graded_relevance_config()
        config.update(graded_config)
        
        return config
    
    def generate_enhanced_training_data(self, output_dir: str = "data/processed") -> str:
        """
        Generate enhanced training data with graded relevance scores.
        
        Returns:
            Path to generated training data file
        """
        logger.info("🔄 Generating enhanced training data with graded relevance...")
        
        output_path = Path(output_dir) / f"graded_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Use existing ground truth as base
        existing_data_path = Path("data/processed/intelligent_ground_truth.json")
        
        self.graded_scorer.generate_graded_training_data(
            existing_data_path=str(existing_data_path),
            output_path=str(output_path),
            num_samples=2500  # Increased from 1914
        )
        
        return str(output_path)
    
    def train_graded_model(self, training_data_path: str) -> Dict:
        """
        Train model with graded relevance scoring.
        
        Args:
            training_data_path: Path to graded training data
            
        Returns:
            Training results dictionary
        """
        logger.info("🎯 Training model with graded relevance scoring...")
        
        # Load training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        samples = training_data['training_samples']
        
        # Prepare datasets
        train_loader, val_loader = self._prepare_data_loaders(samples)
        
        # Initialize model
        model = BERTCrossAttentionRanker(**self.model_config)
        model.to(self.device)
        
        # Initialize loss function (designed for graded relevance)
        loss_function = GradedRelevanceLoss()
        
        # Optimizer with scheduling
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Training loop
        best_ndcg = 0.0
        training_results = {
            'epoch_results': [],
            'best_epoch': 0,
            'best_ndcg': 0.0,
            'training_samples': len(samples)
        }
        
        for epoch in range(30):  # Increased epochs for graded training
            epoch_start = datetime.now()
            
            # Training phase
            model.train()
            train_loss = self._train_epoch(model, train_loader, loss_function, optimizer)
            
            # Validation phase
            model.eval()
            val_metrics = self._validate_epoch(model, val_loader)
            
            # Update learning rate
            scheduler.step()
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            # Log progress
            logger.info(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, NDCG@3={val_metrics['ndcg_at_3']:.4f}, Time={epoch_time:.1f}s")
            
            # Save best model
            if val_metrics['ndcg_at_3'] > best_ndcg:
                best_ndcg = val_metrics['ndcg_at_3']
                training_results['best_epoch'] = epoch + 1
                training_results['best_ndcg'] = best_ndcg
                
                # Save model
                model_save_path = f"models/dl/graded_relevance_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.model_config,
                    'epoch': epoch + 1,
                    'ndcg_at_3': best_ndcg
                }, model_save_path)
                
                logger.info(f"💾 New best model saved: NDCG@3 = {best_ndcg:.4f}")
            
            # Record epoch results
            training_results['epoch_results'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_ndcg_at_3': val_metrics['ndcg_at_3'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': scheduler.get_last_lr()[0],
                'training_time': epoch_time
            })
            
            # Early stopping if no improvement for 5 epochs
            if epoch - (training_results['best_epoch'] - 1) >= 5:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"🎉 Training completed! Best NDCG@3: {best_ndcg:.4f} at epoch {training_results['best_epoch']}")
        
        return training_results
    
    def _prepare_data_loaders(self, samples: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Load datasets for processing
        datasets_path = Path("data/processed")
        singapore_datasets = pd.read_csv(datasets_path / "singapore_datasets.csv")
        global_datasets = pd.read_csv(datasets_path / "global_datasets.csv")
        all_datasets = pd.concat([singapore_datasets, global_datasets], ignore_index=True)
        
        # Prepare samples
        processed_samples = []
        for sample in samples:
            dataset_row = all_datasets[all_datasets['dataset_id'] == sample['dataset_id']]
            if not dataset_row.empty:
                dataset = dataset_row.iloc[0].to_dict()
                processed_samples.append({
                    'query': sample['query'],
                    'dataset': dataset,
                    'relevance_score': sample['relevance_score']
                })
        
        # Split into train/validation
        split_idx = int(0.8 * len(processed_samples))
        train_samples = processed_samples[:split_idx]
        val_samples = processed_samples[split_idx:]
        
        # Create datasets
        train_dataset = GradedRelevanceDataset(train_samples, self.preprocessor)
        val_dataset = GradedRelevanceDataset(val_samples, self.preprocessor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"📊 Data prepared: {len(train_samples)} train, {len(val_samples)} val samples")
        
        return train_loader, val_loader
    
    def _train_epoch(self, model, data_loader, loss_function, optimizer) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            # Forward pass
            query_encoding = batch['query_encoding'].to(self.device)
            dataset_encoding = batch['dataset_encoding'].to(self.device)
            relevance_scores = batch['relevance_score'].to(self.device)
            
            # Model prediction
            predictions = model(query_encoding, dataset_encoding)
            
            # Calculate loss
            loss = loss_function(predictions.squeeze(), relevance_scores)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, model, data_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                query_encoding = batch['query_encoding'].to(self.device)
                dataset_encoding = batch['dataset_encoding'].to(self.device)
                relevance_scores = batch['relevance_score'].to(self.device)
                
                predictions = model(query_encoding, dataset_encoding)
                
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(relevance_scores.cpu().numpy())
        
        # Calculate metrics
        evaluator = DeepEvaluationMetrics({})
        metrics = evaluator.calculate_ranking_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        return metrics


class GradedRelevanceDataset(torch.utils.data.Dataset):
    """Dataset for graded relevance training."""
    
    def __init__(self, samples: List[Dict], preprocessor: NeuralDataPreprocessor):
        self.samples = samples
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode query and dataset
        query_encoding = self.preprocessor.encode_text(sample['query'])
        
        dataset_text = f"{sample['dataset']['title']} {sample['dataset']['description']}"
        dataset_encoding = self.preprocessor.encode_text(dataset_text)
        
        return {
            'query_encoding': torch.tensor(query_encoding, dtype=torch.float32),
            'dataset_encoding': torch.tensor(dataset_encoding, dtype=torch.float32),
            'relevance_score': torch.tensor(sample['relevance_score'], dtype=torch.float32)
        }


class GradedRelevanceLoss(nn.Module):
    """Loss function optimized for graded relevance scoring."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)
    
    def forward(self, predictions, targets):
        # MSE loss for graded scores
        mse = self.mse_loss(predictions, targets)
        
        # Additional ranking loss for relative ordering
        if len(predictions) > 1:
            # Create pairs for ranking loss
            n = len(predictions)
            ranking_loss = 0.0
            num_pairs = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    if targets[i] != targets[j]:
                        # Create ranking target
                        target_ranking = 1.0 if targets[i] > targets[j] else -1.0
                        
                        ranking_loss += self.ranking_loss(
                            predictions[i].unsqueeze(0),
                            predictions[j].unsqueeze(0),
                            torch.tensor(target_ranking, device=predictions.device)
                        )
                        num_pairs += 1
            
            if num_pairs > 0:
                ranking_loss /= num_pairs
                return mse + 0.1 * ranking_loss
        
        return mse


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = GradedRelevanceTrainer()
    
    # Generate enhanced training data
    training_data_path = trainer.generate_enhanced_training_data()
    
    # Train model with graded relevance
    results = trainer.train_graded_model(training_data_path)
    
    # Save results
    results_path = f"outputs/DL/graded_relevance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to {results_path}")
    logger.info(f"🎯 Final NDCG@3: {results['best_ndcg']:.4f}")


if __name__ == "__main__":
    main()