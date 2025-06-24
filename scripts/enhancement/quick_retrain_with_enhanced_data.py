#!/usr/bin/env python3
"""
Quick Retrain with Enhanced Data
Uses the new 1914 training samples and ranking losses to achieve 70%+ NDCG@3
"""

import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from dl.enhanced_neural_preprocessing import EnhancedNeuralPreprocessor
from dl.ranking_losses import CombinedRankingLoss, NDCGLoss
from dl.model_architecture import create_neural_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickRankingTrainer:
    """Quick trainer focused on ranking performance."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = EnhancedNeuralPreprocessor(config)
        
        # Initialize ranking loss
        self.ranking_loss = CombinedRankingLoss(
            ndcg_weight=0.5,
            listmle_weight=0.3,
            binary_weight=0.2,
            k=3
        )
        
        self.models = {}
        self.optimizers = {}
        
    def create_simple_ranking_model(self) -> nn.Module:
        """Create a simple ranking model optimized for our task."""
        
        class SimpleRankingModel(nn.Module):
            def __init__(self, vocab_size=30522, hidden_dim=256, dropout=0.3):
                super().__init__()
                
                # Simple embedding layers
                self.query_embedding = nn.Embedding(vocab_size, hidden_dim)
                self.dataset_embedding = nn.Embedding(vocab_size, hidden_dim)
                
                # Interaction layers
                self.interaction = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, query_ids, dataset_ids):
                # Get embeddings (mean pooling)
                query_emb = self.query_embedding(query_ids).mean(dim=1)
                dataset_emb = self.dataset_embedding(dataset_ids).mean(dim=1)
                
                # Concatenate and predict relevance
                combined = torch.cat([query_emb, dataset_emb], dim=1)
                relevance = self.interaction(combined)
                
                return relevance.squeeze()
        
        return SimpleRankingModel().to(self.device)
    
    def train_single_model(self, model_name: str = "simple_ranking") -> dict:
        """Train a single ranking model with enhanced data."""
        
        logger.info(f"🚀 Training {model_name} with enhanced data...")
        
        # Get data loaders
        processed_data = self.preprocessor.preprocess_for_ranking()
        train_loader = processed_data['train_loader']
        val_loader = processed_data['val_loader']
        
        # Create model
        model = self.create_simple_ranking_model()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,  # Higher learning rate for simple model
            weight_decay=0.01
        )
        
        # Training parameters
        num_epochs = 20
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                query_ids = batch['query_input_ids'].to(self.device)
                dataset_ids = batch['dataset_input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                relevance_scores = batch['relevance_score'].to(self.device)
                
                # Forward pass
                predictions = model(query_ids, dataset_ids)
                
                # Calculate loss (use binary loss for simplicity)
                loss = nn.BCELoss()(predictions, labels.float())
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = total_train_loss / train_batches
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_batches = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    query_ids = batch['query_input_ids'].to(self.device)
                    dataset_ids = batch['dataset_input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    predictions = model(query_ids, dataset_ids)
                    loss = nn.BCELoss()(predictions, labels.float())
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # Collect for metrics
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / val_batches
            
            # Calculate accuracy
            val_predictions = np.array(val_predictions)
            val_labels = np.array(val_labels)
            val_accuracy = np.mean((val_predictions > 0.5) == val_labels)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            })
            
            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'models/dl/{model_name}_best.pt')
                logger.info(f"💾 Saved best model with val_loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⏰ Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'model': model,
            'best_loss': best_loss,
            'training_history': training_history,
            'final_accuracy': val_accuracy
        }
    
    def evaluate_model(self, model, test_loader) -> dict:
        """Evaluate model on test set with NDCG calculation."""
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_queries = []
        
        with torch.no_grad():
            for batch in test_loader:
                query_ids = batch['query_input_ids'].to(self.device)
                dataset_ids = batch['dataset_input_ids'].to(self.device)
                labels = batch['label']
                queries = batch['original_query']
                
                predictions = model(query_ids, dataset_ids)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_queries.extend(queries)
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Simple accuracy
        accuracy = np.mean((predictions > 0.5) == labels)
        
        # Calculate NDCG@3 per query
        query_ndcgs = []
        unique_queries = list(set(all_queries))
        
        for query in unique_queries:
            # Get indices for this query
            query_mask = np.array(all_queries) == query
            query_preds = predictions[query_mask]
            query_labels = labels[query_mask]
            
            if len(query_preds) >= 3:
                # Sort by predictions
                sorted_indices = np.argsort(query_preds)[::-1]
                top_3_labels = query_labels[sorted_indices[:3]]
                
                # Calculate NDCG@3
                dcg = sum([label / np.log2(i + 2) for i, label in enumerate(top_3_labels)])
                
                # Ideal DCG (sort by true labels)
                ideal_labels = np.sort(query_labels)[::-1][:3]
                idcg = sum([label / np.log2(i + 2) for i, label in enumerate(ideal_labels)])
                
                ndcg = dcg / idcg if idcg > 0 else 0
                query_ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(query_ndcgs) if query_ndcgs else 0
        
        return {
            'accuracy': accuracy,
            'ndcg_at_3': avg_ndcg,
            'num_queries_evaluated': len(query_ndcgs)
        }
    
    def run_quick_training(self) -> dict:
        """Run the complete quick training pipeline."""
        
        logger.info("🎯 Starting quick training with enhanced data...")
        
        # Train model
        results = self.train_single_model("simple_ranking_v2")
        
        # Evaluate on test set
        processed_data = self.preprocessor.preprocess_for_ranking()
        test_loader = processed_data['test_loader']
        
        evaluation_results = self.evaluate_model(results['model'], test_loader)
        
        # Combine results
        final_results = {
            'training_results': results,
            'evaluation_results': evaluation_results,
            'data_summary': processed_data['metadata']['summary']
        }
        
        # Log final results
        logger.info("🎉 Quick training complete!")
        logger.info(f"📊 Final Results:")
        logger.info(f"  Test Accuracy: {evaluation_results['accuracy']:.3f}")
        logger.info(f"  NDCG@3: {evaluation_results['ndcg_at_3']:.3f} ({evaluation_results['ndcg_at_3']:.1%})")
        logger.info(f"  Queries evaluated: {evaluation_results['num_queries_evaluated']}")
        
        # Check if we hit target
        ndcg_percent = evaluation_results['ndcg_at_3'] * 100
        if ndcg_percent >= 70:
            logger.info(f"🎯 TARGET ACHIEVED! {ndcg_percent:.1f}% >= 70%")
        else:
            logger.info(f"📈 Progress: {ndcg_percent:.1f}% (need {70 - ndcg_percent:.1f}% more)")
        
        return final_results

def main():
    """Main training function."""
    
    # Load config
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = QuickRankingTrainer(config)
    
    # Run training
    results = trainer.run_quick_training()
    
    # Save results
    results_path = Path(f"outputs/DL/quick_retrain_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any torch tensors to lists for JSON serialization
    import json
    
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    main()