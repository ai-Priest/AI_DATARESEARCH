"""
Enhanced Neural Model Training with Domain-Enhanced Data
Retrain neural model using Phase 2.1 enhanced training data for improved performance
"""

import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dl.improved_model_architecture import BERTCrossAttentionRanker
from src.dl.neural_preprocessing import NeuralDataPreprocessor
from src.dl.deep_evaluation import DeepEvaluator
from src.dl.graded_relevance import GradedRelevanceScorer

logger = logging.getLogger(__name__)


class EnhancedNeuralTrainer:
    """
    Enhanced neural model trainer using domain-enhanced training data.
    """
    
    def __init__(self, config_path: str = "config/dl_config.yml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize components
        self.preprocessor = NeuralDataPreprocessor(self.config)
        self.evaluator = DeepEvaluator(self.config)
        
        # Enhanced model configuration
        self.model_config = {
            'model_name': 'distilbert-base-uncased',
            'hidden_dim': 768,
            'dropout': 0.3,
            'num_attention_heads': 8
        }
        
        logger.info(f"🚀 EnhancedNeuralTrainer initialized on {self.device}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def train_with_enhanced_data(self, 
                               training_data_path: str,
                               output_dir: str = "models/dl") -> Dict[str, Any]:
        """
        Train neural model with enhanced domain-diverse training data.
        
        Args:
            training_data_path: Path to enhanced training data
            output_dir: Directory to save trained models
            
        Returns:
            Training results and performance metrics
        """
        logger.info("🎯 Starting enhanced neural model training...")
        
        # Load enhanced training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        samples = training_data['training_samples']
        metadata = training_data['metadata']
        
        logger.info(f"📊 Loaded {len(samples)} enhanced training samples")
        logger.info(f"📈 Score distribution: {metadata.get('score_distribution', {})}")
        logger.info(f"🏷️  Domain distribution: {metadata.get('domain_distribution', {})}")
        
        # Prepare training data
        train_loader, val_loader, test_loader = self._prepare_enhanced_data_loaders(samples)
        
        # Initialize enhanced model
        model = BERTCrossAttentionRanker(**self.model_config)
        model.to(self.device)
        
        # Enhanced training setup
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=3e-5,  # Slightly higher for enhanced data
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        num_training_steps = len(train_loader) * 20  # 20 epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-5,
            total_steps=num_training_steps,
            pct_start=0.1  # 10% warmup
        )
        
        # Enhanced loss function for graded relevance
        loss_function = EnhancedGradedLoss()
        
        # Training loop
        best_ndcg = 0.0
        training_results = {
            'training_data_path': training_data_path,
            'model_config': self.model_config,
            'training_metadata': metadata,
            'epoch_results': [],
            'best_epoch': 0,
            'best_ndcg': 0.0,
            'final_evaluation': {}
        }
        
        for epoch in range(20):  # Enhanced training epochs
            epoch_start = datetime.now()
            
            # Training phase
            model.train()
            train_loss, train_metrics = self._train_epoch_enhanced(
                model, train_loader, loss_function, optimizer, scheduler
            )
            
            # Validation phase
            model.eval()
            val_metrics = self._validate_epoch_enhanced(model, val_loader)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            # Log progress
            logger.info(f"Epoch {epoch+1:2d}: "
                       f"Loss={train_loss:.4f}, "
                       f"NDCG@3={val_metrics['ndcg_at_3']:.4f}, "
                       f"Accuracy={val_metrics['accuracy']:.4f}, "
                       f"Time={epoch_time:.1f}s")
            
            # Save best model
            if val_metrics['ndcg_at_3'] > best_ndcg:
                best_ndcg = val_metrics['ndcg_at_3']
                training_results['best_epoch'] = epoch + 1
                training_results['best_ndcg'] = best_ndcg
                
                # Save enhanced model
                model_save_path = Path(output_dir) / "enhanced_neural_model_best.pt"
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.model_config,
                    'epoch': epoch + 1,
                    'ndcg_at_3': best_ndcg,
                    'training_metadata': metadata
                }, model_save_path)
                
                logger.info(f"💾 Enhanced model saved: NDCG@3 = {best_ndcg:.4f}")
            
            # Record epoch results
            training_results['epoch_results'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_metrics.get('accuracy', 0.0),
                'val_ndcg_at_3': val_metrics['ndcg_at_3'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision_at_3': val_metrics.get('precision_at_3', 0.0),
                'val_recall_at_3': val_metrics.get('recall_at_3', 0.0),
                'learning_rate': scheduler.get_last_lr()[0],
                'training_time': epoch_time
            })
            
            # Early stopping check
            if epoch - (training_results['best_epoch'] - 1) >= 5:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation on test set
        logger.info("🔍 Performing final evaluation on test set...")
        model.eval()
        test_metrics = self._comprehensive_evaluation(model, test_loader)
        training_results['final_evaluation'] = test_metrics
        
        # Save final results
        results_path = Path("outputs/DL") / f"enhanced_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"🎉 Enhanced training completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Best NDCG@3: {best_ndcg:.4f} (epoch {training_results['best_epoch']})")
        logger.info(f"   Test NDCG@3: {test_metrics.get('ndcg_at_3', 0.0):.4f}")
        logger.info(f"   Test Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"   Results saved: {results_path}")
        
        return training_results
    
    def _prepare_enhanced_data_loaders(self, samples: List[Dict]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare enhanced data loaders with domain balancing."""
        # Load datasets
        datasets_path = Path("data/processed")
        singapore_datasets = pd.read_csv(datasets_path / "singapore_datasets.csv")
        global_datasets = pd.read_csv(datasets_path / "global_datasets.csv")
        all_datasets = pd.concat([singapore_datasets, global_datasets], ignore_index=True)
        
        # Process samples and filter valid ones
        processed_samples = []
        domain_counts = {}
        
        for sample in samples:
            dataset_row = all_datasets[all_datasets['dataset_id'] == sample['dataset_id']]
            if not dataset_row.empty:
                dataset = dataset_row.iloc[0].to_dict()
                processed_samples.append({
                    'query': sample['query'],
                    'dataset': dataset,
                    'relevance_score': sample['relevance_score'],
                    'domain': sample.get('domain', 'general'),
                    'query_type': sample.get('query_type', 'unknown')
                })
                
                # Track domain distribution
                domain = sample.get('domain', 'general')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info(f"📊 Processed samples by domain: {domain_counts}")
        
        # Split data: 70% train, 15% val, 15% test
        train_samples, temp_samples = train_test_split(
            processed_samples, test_size=0.3, random_state=42, 
            stratify=[s['domain'] for s in processed_samples]
        )
        
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42,
            stratify=[s['domain'] for s in temp_samples]
        )
        
        # Create datasets
        train_dataset = EnhancedRelevanceDataset(train_samples, self.preprocessor)
        val_dataset = EnhancedRelevanceDataset(val_samples, self.preprocessor)
        test_dataset = EnhancedRelevanceDataset(test_samples, self.preprocessor)
        
        # Create data loaders with enhanced batching
        train_loader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=0,  # MPS compatibility
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"📚 Enhanced data loaders: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_loader, val_loader, test_loader
    
    def _train_epoch_enhanced(self, 
                            model: nn.Module, 
                            data_loader: DataLoader, 
                            loss_function: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            scheduler: torch.optim.lr_scheduler._LRScheduler) -> Tuple[float, Dict]:
        """Enhanced training epoch with improved metrics."""
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            try:
                # Move data to device
                query_encoding = batch['query_encoding'].to(self.device)
                dataset_encoding = batch['dataset_encoding'].to(self.device)
                relevance_scores = batch['relevance_score'].to(self.device)
                
                # Forward pass
                predictions = model(query_encoding, dataset_encoding)
                predictions = predictions.squeeze()
                
                # Calculate loss
                loss = loss_function(predictions, relevance_scores)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(relevance_scores.detach().cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Batch training error: {e}")
                continue
        
        # Calculate training metrics
        train_metrics = {}
        if all_predictions and all_targets:
            predictions_array = np.array(all_predictions)
            targets_array = np.array(all_targets)
            
            # Calculate accuracy (within tolerance)
            accuracy = np.mean(np.abs(predictions_array - targets_array) < 0.2)
            train_metrics['accuracy'] = accuracy
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, train_metrics
    
    def _validate_epoch_enhanced(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Enhanced validation with comprehensive metrics."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    query_encoding = batch['query_encoding'].to(self.device)
                    dataset_encoding = batch['dataset_encoding'].to(self.device)
                    relevance_scores = batch['relevance_score'].to(self.device)
                    
                    predictions = model(query_encoding, dataset_encoding)
                    predictions = predictions.squeeze()
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(relevance_scores.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Batch validation error: {e}")
                    continue
        
        # Calculate comprehensive metrics
        if not all_predictions or not all_targets:
            return {'ndcg_at_3': 0.0, 'accuracy': 0.0}
        
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        # Enhanced metrics calculation
        metrics = self.evaluator.calculate_ranking_metrics(predictions_array, targets_array)
        
        return metrics
    
    def _comprehensive_evaluation(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation on test set."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        domain_predictions = {}
        domain_targets = {}
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    query_encoding = batch['query_encoding'].to(self.device)
                    dataset_encoding = batch['dataset_encoding'].to(self.device)
                    relevance_scores = batch['relevance_score'].to(self.device)
                    domains = batch.get('domain', ['general'] * len(relevance_scores))
                    
                    predictions = model(query_encoding, dataset_encoding)
                    predictions = predictions.squeeze()
                    
                    batch_predictions = predictions.cpu().numpy()
                    batch_targets = relevance_scores.cpu().numpy()
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
                    # Track by domain
                    for i, domain in enumerate(domains):
                        if domain not in domain_predictions:
                            domain_predictions[domain] = []
                            domain_targets[domain] = []
                        
                        if i < len(batch_predictions):
                            domain_predictions[domain].append(batch_predictions[i])
                            domain_targets[domain].append(batch_targets[i])
                    
                except Exception as e:
                    logger.warning(f"Test batch error: {e}")
                    continue
        
        # Overall metrics
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        overall_metrics = self.evaluator.calculate_ranking_metrics(predictions_array, targets_array)
        
        # Domain-specific metrics
        domain_metrics = {}
        for domain in domain_predictions:
            if len(domain_predictions[domain]) > 10:  # Minimum samples for reliable metrics
                domain_preds = np.array(domain_predictions[domain])
                domain_targs = np.array(domain_targets[domain])
                domain_metrics[domain] = self.evaluator.calculate_ranking_metrics(domain_preds, domain_targs)
        
        return {
            'overall_metrics': overall_metrics,
            'domain_metrics': domain_metrics,
            'total_samples': len(all_predictions),
            'domain_sample_counts': {d: len(domain_predictions[d]) for d in domain_predictions}
        }


class EnhancedRelevanceDataset(torch.utils.data.Dataset):
    """Enhanced dataset for graded relevance training with domain information."""
    
    def __init__(self, samples: List[Dict], preprocessor: NeuralDataPreprocessor):
        self.samples = samples
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Encode query and dataset
            query_encoding = self.preprocessor.encode_text(sample['query'])
            
            dataset_text = f"{sample['dataset']['title']} {sample['dataset']['description']}"
            dataset_encoding = self.preprocessor.encode_text(dataset_text)
            
            return {
                'query_encoding': torch.tensor(query_encoding, dtype=torch.float32),
                'dataset_encoding': torch.tensor(dataset_encoding, dtype=torch.float32),
                'relevance_score': torch.tensor(sample['relevance_score'], dtype=torch.float32),
                'domain': sample.get('domain', 'general'),
                'query_type': sample.get('query_type', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"Dataset item error at index {idx}: {e}")
            # Return zero tensors as fallback
            return {
                'query_encoding': torch.zeros(768, dtype=torch.float32),
                'dataset_encoding': torch.zeros(768, dtype=torch.float32),
                'relevance_score': torch.tensor(0.0, dtype=torch.float32),
                'domain': 'general',
                'query_type': 'unknown'
            }


class EnhancedGradedLoss(nn.Module):
    """Enhanced loss function for graded relevance training."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for ranking loss
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Primary loss: Huber loss (more robust than MSE)
        primary_loss = self.huber_loss(predictions, targets)
        
        # Secondary loss: Ranking preservation
        ranking_loss = self._ranking_loss(predictions, targets)
        
        return self.alpha * primary_loss + self.beta * ranking_loss
    
    def _ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Ranking loss to preserve relative ordering."""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Create pairwise comparisons
        n = len(predictions)
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if targets[i] != targets[j]:  # Only consider different relevance levels
                    # Target ranking: 1 if i should rank higher than j, -1 otherwise
                    target_ranking = 1.0 if targets[i] > targets[j] else -1.0
                    
                    # Predicted ranking difference
                    pred_diff = predictions[i] - predictions[j]
                    
                    # Hinge loss for ranking
                    loss = torch.max(torch.tensor(0.0, device=predictions.device), 
                                   1.0 - target_ranking * pred_diff)
                    
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=predictions.device)


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced trainer
    trainer = EnhancedNeuralTrainer()
    
    # Train with enhanced data
    training_data_path = "data/processed/domain_enhanced_training_20250622.json"
    
    if not Path(training_data_path).exists():
        logger.error(f"Enhanced training data not found: {training_data_path}")
        logger.info("Please run domain_enhanced_training.py first to generate enhanced training data")
        return
    
    # Start enhanced training
    results = trainer.train_with_enhanced_data(training_data_path)
    
    logger.info("🎉 Enhanced neural model training completed successfully!")


if __name__ == "__main__":
    main()