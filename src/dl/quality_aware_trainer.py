"""
Quality-Aware Model Training
Implements training loop that prioritizes relevance over convergence speed
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import random

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for quality-aware training"""
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.001
    curriculum_stages: int = 3
    quality_threshold: float = 0.7
    ndcg_target: float = 0.7
    save_best_only: bool = True
    validation_frequency: int = 1

class QualityAwareTrainer:
    """Trainer that prioritizes recommendation quality over speed"""
    
    def __init__(self, model, config: TrainingConfig = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_ndcg = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Curriculum learning state
        self.current_stage = 0
        self.stage_epochs = self.config.max_epochs // self.config.curriculum_stages
        
        logger.info(f"🎯 QualityAwareTrainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Target NDCG@3: {self.config.ndcg_target}")
        logger.info(f"  Curriculum stages: {self.config.curriculum_stages}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def train(self, train_dataset, val_dataset, save_path: str = "models/dl/quality_first/"):
        """Train model with quality-first approach"""
        logger.info(f"🚀 Starting quality-aware training")
        
        # Create save directory
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer with conservative learning rate
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01  # Regularization for better generalization
        )
        
        # Learning rate scheduler for quality-focused training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Curriculum learning: start with high-confidence examples
        curriculum_datasets = self._create_curriculum_datasets(train_dataset)
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Determine current curriculum stage
            stage = min(epoch // self.stage_epochs, self.config.curriculum_stages - 1)
            if stage != self.current_stage:
                self.current_stage = stage
                logger.info(f"📚 Advancing to curriculum stage {stage + 1}/{self.config.curriculum_stages}")
            
            # Get current dataset for curriculum learning
            current_dataset = curriculum_datasets[stage]
            train_loader = DataLoader(
                current_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn
            )
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validation phase (every N epochs or at end)
            if epoch % self.config.validation_frequency == 0 or epoch == self.config.max_epochs - 1:
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=self.config.batch_size, 
                    shuffle=False,
                    collate_fn=self._collate_fn
                )
                val_metrics = self._validate_epoch(val_loader)
                
                # Update learning rate based on validation NDCG
                scheduler.step(val_metrics['ndcg_3'])
                
                # Check for improvement
                current_ndcg = val_metrics['ndcg_3']
                if current_ndcg > self.best_ndcg + self.config.min_delta:
                    self.best_ndcg = current_ndcg
                    self.patience_counter = 0
                    
                    # Save best model
                    if self.config.save_best_only:
                        self._save_checkpoint(save_path / "best_quality_model.pt", val_metrics)
                        logger.info(f"💾 Saved new best model (NDCG@3: {current_ndcg:.4f})")
                else:
                    self.patience_counter += 1
                
                # Log progress
                logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
                logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
                logger.info(f"  Val NDCG@3: {val_metrics['ndcg_3']:.4f} (best: {self.best_ndcg:.4f})")
                logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                logger.info(f"  Patience: {self.patience_counter}/{self.config.patience}")
                
                # Store training history
                epoch_history = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_ndcg_3': val_metrics['ndcg_3'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'curriculum_stage': stage + 1
                }
                self.training_history.append(epoch_history)
                
                # Early stopping based on NDCG@3
                if self.patience_counter >= self.config.patience:
                    logger.info(f"🛑 Early stopping triggered (patience: {self.config.patience})")
                    break
                
                # Check if target quality achieved
                if current_ndcg >= self.config.ndcg_target:
                    logger.info(f"🎯 Target NDCG@3 achieved: {current_ndcg:.4f} >= {self.config.ndcg_target}")
                    break
        
        # Save final training history
        self._save_training_history(save_path / "training_history.json")
        
        logger.info(f"✅ Training completed")
        logger.info(f"  Best NDCG@3: {self.best_ndcg:.4f}")
        logger.info(f"  Total epochs: {self.current_epoch + 1}")
        
        return self.training_history
    
    def _create_curriculum_datasets(self, dataset) -> List[Dataset]:
        """Create curriculum learning datasets with increasing difficulty"""
        # Sort examples by confidence/quality score
        examples = list(dataset)
        
        # Sort by relevance score (higher first)
        examples.sort(key=lambda x: x.get('relevance_score', 0.5), reverse=True)
        
        # Create curriculum stages
        curriculum_datasets = []
        total_examples = len(examples)
        
        for stage in range(self.config.curriculum_stages):
            # Gradually increase dataset size and difficulty
            if stage == 0:
                # Stage 1: Top 30% highest confidence examples
                end_idx = int(0.3 * total_examples)
                stage_examples = examples[:end_idx]
            elif stage == 1:
                # Stage 2: Top 70% examples
                end_idx = int(0.7 * total_examples)
                stage_examples = examples[:end_idx]
            else:
                # Stage 3: All examples
                stage_examples = examples
            
            curriculum_datasets.append(CurriculumDataset(stage_examples))
            logger.info(f"📚 Curriculum stage {stage + 1}: {len(stage_examples)} examples")
        
        return curriculum_datasets
    
    def _train_epoch(self, train_loader, optimizer) -> Dict[str, float]:
        """Train for one epoch with quality focus"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate quality-focused loss
            loss = self._calculate_quality_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate model with quality metrics"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Get predictions
                predictions = torch.sigmoid(outputs['relevance_score']).cpu().numpy()
                targets = batch['relevance_score'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Calculate accuracy (threshold at 0.5)
                pred_binary = (predictions > 0.5).astype(int)
                target_binary = (targets > 0.5).astype(int)
                correct_predictions += (pred_binary == target_binary).sum()
                total_predictions += len(pred_binary)
        
        # Calculate NDCG@3
        ndcg_3 = self._calculate_ndcg_at_k(all_predictions, all_targets, k=3)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'ndcg_3': ndcg_3,
            'accuracy': accuracy
        }
    
    def _calculate_quality_loss(self, outputs, batch) -> torch.Tensor:
        """Calculate loss that prioritizes quality over speed"""
        # Primary loss: ranking loss for relevance
        relevance_loss = nn.BCEWithLogitsLoss()(
            outputs['relevance_score'], 
            batch['relevance_score']
        )
        
        # Secondary losses for domain and Singapore classification
        domain_loss = nn.CrossEntropyLoss()(
            outputs.get('domain_logits', torch.zeros(1, 8).to(self.device)), 
            batch.get('domain_id', torch.zeros(1, dtype=torch.long).to(self.device))
        )
        
        singapore_loss = nn.CrossEntropyLoss()(
            outputs.get('singapore_logits', torch.zeros(1, 2).to(self.device)), 
            batch.get('singapore_label', torch.zeros(1, dtype=torch.long).to(self.device))
        )
        
        # Weighted combination prioritizing relevance
        total_loss = 0.7 * relevance_loss + 0.2 * domain_loss + 0.1 * singapore_loss
        
        return total_loss
    
    def _calculate_ndcg_at_k(self, predictions: List[float], targets: List[float], k: int = 3) -> float:
        """Calculate NDCG@k for ranking quality"""
        if len(predictions) == 0:
            return 0.0
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Sort by predictions (descending)
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_targets = targets[sorted_indices]
        
        # Calculate DCG@k
        dcg = 0.0
        for i in range(min(k, len(sorted_targets))):
            relevance = sorted_targets[i]
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate IDCG@k (ideal DCG)
        ideal_targets = np.sort(targets)[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_targets))):
            relevance = ideal_targets[i]
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate NDCG@k
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # This is a simplified collate function
        # In practice, you'd need proper tokenization and padding
        collated = {}
        
        if len(batch) > 0:
            # Get all keys from first item
            keys = batch[0].keys()
            
            for key in keys:
                values = [item[key] for item in batch]
                
                if key in ['query_ids', 'source_ids', 'domain_id', 'singapore_label']:
                    # Convert to tensor
                    collated[key] = torch.tensor(values, dtype=torch.long)
                elif key in ['relevance_score']:
                    # Convert to float tensor
                    collated[key] = torch.tensor(values, dtype=torch.float)
                else:
                    # Keep as list
                    collated[key] = values
        
        return collated
    
    def _save_checkpoint(self, path: Path, metrics: Dict):
        """Save model checkpoint with training state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_ndcg': self.best_ndcg,
            'metrics': metrics,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
    
    def _save_training_history(self, path: Path):
        """Save training history to JSON"""
        history_data = {
            'training_history': self.training_history,
            'final_metrics': {
                'best_ndcg_3': float(self.best_ndcg),
                'total_epochs': int(self.current_epoch + 1),
                'target_achieved': bool(self.best_ndcg >= self.config.ndcg_target)
            },
            'config': self.config.__dict__
        }
        
        with open(path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"📊 Training history saved to {path}")

class CurriculumDataset(Dataset):
    """Dataset wrapper for curriculum learning"""
    
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_quality_aware_model(model, train_dataset, val_dataset, config: TrainingConfig = None):
    """Train a model with quality-aware approach"""
    trainer = QualityAwareTrainer(model, config)
    return trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This would normally load your actual model and datasets
    print("🧪 Quality-Aware Trainer Test")
    print("💡 This module requires a trained model and datasets to run")
    print("   Use this trainer with your QualityAwareRankingModel")
    
    # Example configuration
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,
        max_epochs=50,
        patience=10,
        curriculum_stages=3,
        ndcg_target=0.7
    )
    
    print(f"📋 Training Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  NDCG@3 target: {config.ndcg_target}")
    print(f"  Curriculum stages: {config.curriculum_stages}")
    
    print("✅ Quality-Aware Trainer ready for use!")