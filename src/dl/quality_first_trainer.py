"""
Quality-First Neural Trainer
Trains neural models with focus on recommendation quality using enhanced training data
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

try:
    from .quality_first_neural_model import (
        QualityAwareRankingModel, 
        QualityFirstLoss,
        EnhancedTrainingDataLoader
    )
    from .ranking_losses import CombinedRankingLoss, create_ranking_loss
except ImportError:
    # Handle running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from quality_first_neural_model import (
        QualityAwareRankingModel, 
        QualityFirstLoss,
        EnhancedTrainingDataLoader
    )
    from ranking_losses import CombinedRankingLoss, create_ranking_loss

logger = logging.getLogger(__name__)


class QualityFirstTrainer:
    """Quality-first trainer for neural ranking models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()
        
        # Training configuration
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.0001)  # Lower LR for stability
        self.batch_size = config.get('batch_size', 32)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('min_delta', 0.001)
        
        # Quality-first settings
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.ranking_focus = config.get('ranking_focus', True)
        self.singapore_first_weight = config.get('singapore_first_weight', 0.2)
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ranking_loss_fn = None
        self.quality_loss_fn = None
        
        # Training state
        self.best_ndcg = 0.0
        self.best_model_state = None
        self.training_history = []
        
        logger.info(f"🎯 QualityFirstTrainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Quality threshold: {self.quality_threshold}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def setup_model_and_training(self, model_config: Dict = None):
        """Setup model, optimizer, and loss functions"""
        logger.info("🔧 Setting up model and training components")
        
        # Create model
        if model_config is None:
            model_config = {
                'embedding_dim': 256,
                'hidden_dim': 128,
                'num_domains': 8,
                'num_sources': 10,
                'vocab_size': 10000
            }
        
        self.model = QualityAwareRankingModel(model_config).to(self.device)
        
        # Setup optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize NDCG
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Setup ranking loss function
        self.ranking_loss_fn = CombinedRankingLoss(
            listmle_weight=0.4,
            ranknet_weight=0.3,
            lambdarank_weight=0.3,
            k=3  # NDCG@3 optimization
        ).to(self.device)
        
        # Setup quality-aware loss
        quality_config = {
            'ranking_weight': 0.6,  # Primary focus on ranking
            'singapore_weight': self.singapore_first_weight,
            'domain_weight': 0.2
        }
        self.quality_loss_fn = QualityFirstLoss(quality_config).to(self.device)
        
        logger.info(f"✅ Model setup complete")
        logger.info(f"  Model parameters: {self.model.count_parameters()/1e6:.1f}M")
        logger.info(f"  Optimizer: AdamW with weight decay")
        logger.info(f"  Scheduler: ReduceLROnPlateau")
    
    def train_with_enhanced_data(self, 
                                data_path: str,
                                output_dir: str = "models/dl/quality_first") -> Dict:
        """Train model using enhanced training data"""
        logger.info(f"🚀 Starting quality-first training")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Output directory: {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load enhanced training data
        data_loader = EnhancedTrainingDataLoader(data_path, self.batch_size)
        train_loader = data_loader.get_data_loader('train')
        val_loader = data_loader.get_data_loader('validation')
        test_loader = data_loader.get_data_loader('test')
        
        logger.info(f"📊 Data loaded:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['ndcg_at_3'])
            
            # Track training history
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['total_loss'],
                'train_ranking_loss': train_metrics['ranking_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_ndcg_at_3': val_metrics['ndcg_at_3'],
                'val_singapore_acc': val_metrics['singapore_accuracy'],
                'val_domain_acc': val_metrics['domain_accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': time.time() - epoch_start
            }
            self.training_history.append(epoch_metrics)
            
            # Log progress
            logger.info(f"Epoch {epoch + 1}/{self.epochs}:")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Val NDCG@3: {val_metrics['ndcg_at_3']:.4f}")
            logger.info(f"  Val Singapore Acc: {val_metrics['singapore_accuracy']:.4f}")
            logger.info(f"  Val Domain Acc: {val_metrics['domain_accuracy']:.4f}")
            logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping and model saving
            if val_metrics['ndcg_at_3'] > self.best_ndcg + self.min_delta:
                self.best_ndcg = val_metrics['ndcg_at_3']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                best_model_path = output_path / "best_quality_model.pt"
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_ndcg': self.best_ndcg,
                    'config': self.config,
                    'training_history': self.training_history
                }, best_model_path)
                
                logger.info(f"💾 New best model saved (NDCG@3: {self.best_ndcg:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered (patience: {self.early_stopping_patience})")
                break
        
        # Final evaluation on test set
        logger.info("🧪 Final evaluation on test set")
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        test_metrics = self._evaluate_model(test_loader)
        
        # Training summary
        total_time = time.time() - start_time
        training_summary = {
            'success': True,
            'total_epochs': len(self.training_history),
            'best_ndcg_at_3': self.best_ndcg,
            'final_test_ndcg': test_metrics['ndcg_at_3'],
            'final_test_singapore_acc': test_metrics['singapore_accuracy'],
            'final_test_domain_acc': test_metrics['domain_accuracy'],
            'training_time_minutes': total_time / 60,
            'model_parameters': self.model.count_parameters(),
            'output_directory': str(output_path),
            'training_history': self.training_history,
            'test_metrics': test_metrics
        }
        
        # Save training summary
        summary_path = output_path / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info("✅ Quality-first training completed!")
        logger.info(f"  Best validation NDCG@3: {self.best_ndcg:.4f}")
        logger.info(f"  Final test NDCG@3: {test_metrics['ndcg_at_3']:.4f}")
        logger.info(f"  Training time: {total_time/60:.1f} minutes")
        logger.info(f"  Model saved to: {output_path}")
        
        return training_summary
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Prepare targets for quality loss
            targets = {
                'relevance_score': batch['relevance_score'],
                'singapore_labels': batch['singapore_labels'],
                'domain_labels': batch['domain_labels']
            }
            
            # Compute quality loss
            quality_losses = self.quality_loss_fn(outputs, targets)
            
            # Compute ranking loss (for pairs/groups)
            # Note: This is simplified - in practice, you'd need to create ranking pairs
            ranking_loss = torch.tensor(0.0, device=self.device)  # Placeholder
            
            # Total loss
            total_batch_loss = quality_losses['total_loss'] + 0.1 * ranking_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"  Batch {batch_idx}/{len(train_loader)}: Loss {total_batch_loss.item():.4f}")
        
        return {
            'total_loss': total_loss / num_batches,
            'ranking_loss': total_ranking_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_relevance_scores = []
        all_singapore_preds = []
        all_singapore_labels = []
        all_domain_preds = []
        all_domain_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Prepare targets
                targets = {
                    'relevance_score': batch['relevance_score'],
                    'singapore_labels': batch['singapore_labels'],
                    'domain_labels': batch['domain_labels']
                }
                
                # Compute loss
                quality_losses = self.quality_loss_fn(outputs, targets)
                total_loss += quality_losses['total_loss'].item()
                
                # Collect predictions for metrics
                all_predictions.append(outputs['relevance_score'].cpu())
                all_relevance_scores.append(batch['relevance_score'].cpu())
                all_singapore_preds.append(outputs['singapore_logits'].argmax(dim=-1).cpu())
                all_singapore_labels.append(batch['singapore_labels'].cpu())
                all_domain_preds.append(outputs['domain_logits'].argmax(dim=-1).cpu())
                all_domain_labels.append(batch['domain_labels'].cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions)
        all_relevance_scores = torch.cat(all_relevance_scores)
        all_singapore_preds = torch.cat(all_singapore_preds)
        all_singapore_labels = torch.cat(all_singapore_labels)
        all_domain_preds = torch.cat(all_domain_preds)
        all_domain_labels = torch.cat(all_domain_labels)
        
        # Compute metrics
        ndcg_at_3 = self._compute_ndcg_at_k(all_predictions, all_relevance_scores, k=3)
        singapore_accuracy = (all_singapore_preds == all_singapore_labels).float().mean().item()
        domain_accuracy = (all_domain_preds == all_domain_labels).float().mean().item()
        
        return {
            'total_loss': total_loss / len(val_loader),
            'ndcg_at_3': ndcg_at_3,
            'singapore_accuracy': singapore_accuracy,
            'domain_accuracy': domain_accuracy
        }
    
    def _evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_relevance_scores = []
        all_singapore_preds = []
        all_singapore_labels = []
        all_domain_preds = []
        all_domain_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                
                all_predictions.append(outputs['relevance_score'].cpu())
                all_relevance_scores.append(batch['relevance_score'].cpu())
                all_singapore_preds.append(outputs['singapore_logits'].argmax(dim=-1).cpu())
                all_singapore_labels.append(batch['singapore_labels'].cpu())
                all_domain_preds.append(outputs['domain_logits'].argmax(dim=-1).cpu())
                all_domain_labels.append(batch['domain_labels'].cpu())
        
        # Concatenate predictions
        all_predictions = torch.cat(all_predictions)
        all_relevance_scores = torch.cat(all_relevance_scores)
        all_singapore_preds = torch.cat(all_singapore_preds)
        all_singapore_labels = torch.cat(all_singapore_labels)
        all_domain_preds = torch.cat(all_domain_preds)
        all_domain_labels = torch.cat(all_domain_labels)
        
        # Compute comprehensive metrics
        metrics = {
            'ndcg_at_1': self._compute_ndcg_at_k(all_predictions, all_relevance_scores, k=1),
            'ndcg_at_3': self._compute_ndcg_at_k(all_predictions, all_relevance_scores, k=3),
            'ndcg_at_5': self._compute_ndcg_at_k(all_predictions, all_relevance_scores, k=5),
            'singapore_accuracy': (all_singapore_preds == all_singapore_labels).float().mean().item(),
            'domain_accuracy': (all_domain_preds == all_domain_labels).float().mean().item(),
            'relevance_mse': nn.MSELoss()(all_predictions, all_relevance_scores).item(),
            'relevance_mae': nn.L1Loss()(all_predictions, all_relevance_scores).item()
        }
        
        return metrics
    
    def _compute_ndcg_at_k(self, predictions: torch.Tensor, relevance_scores: torch.Tensor, k: int) -> float:
        """Compute NDCG@k metric"""
        # For simplicity, compute NDCG assuming each sample is independent
        # In practice, you'd group by query and compute NDCG per query
        
        # Sort by predictions
        _, pred_indices = torch.sort(predictions, descending=True)
        
        # Get top-k relevance scores
        if len(pred_indices) < k:
            k = len(pred_indices)
        
        top_k_relevance = relevance_scores[pred_indices[:k]]
        
        # Compute DCG@k
        positions = torch.arange(1, k + 1, dtype=torch.float)
        dcg_weights = 1.0 / torch.log2(positions + 1)
        dcg = torch.sum(top_k_relevance * dcg_weights)
        
        # Compute IDCG@k
        sorted_relevance, _ = torch.sort(relevance_scores, descending=True)
        ideal_relevance = sorted_relevance[:k]
        idcg = torch.sum(ideal_relevance * dcg_weights)
        
        # NDCG@k
        if idcg > 0:
            ndcg = (dcg / idcg).item()
        else:
            ndcg = 0.0
        
        return ndcg


def train_quality_first_model(
    data_path: str = "data/processed/enhanced_training_mappings.json",
    output_dir: str = "models/dl/quality_first",
    config: Dict = None
) -> Dict:
    """Main function to train quality-first model"""
    
    if config is None:
        config = {
            'epochs': 30,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'early_stopping_patience': 8,
            'quality_threshold': 0.7,
            'singapore_first_weight': 0.2
        }
    
    # Create trainer
    trainer = QualityFirstTrainer(config)
    
    # Setup model and training
    trainer.setup_model_and_training()
    
    # Train model
    results = trainer.train_with_enhanced_data(data_path, output_dir)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train quality-first model
    logger.info("🚀 Starting quality-first neural training")
    
    try:
        results = train_quality_first_model()
        
        if results['success']:
            print(f"✅ Training completed successfully!")
            print(f"  Best NDCG@3: {results['best_ndcg_at_3']:.4f}")
            print(f"  Final test NDCG@3: {results['final_test_ndcg']:.4f}")
            print(f"  Training time: {results['training_time_minutes']:.1f} minutes")
            print(f"  Model saved to: {results['output_directory']}")
        else:
            print("❌ Training failed")
            
    except FileNotFoundError as e:
        print(f"❌ Training data not found: {e}")
        print("💡 Run enhanced_training_integrator.py first to create training data")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        logger.exception("Training error details:")