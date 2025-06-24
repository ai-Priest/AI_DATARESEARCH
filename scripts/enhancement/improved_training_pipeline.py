#!/usr/bin/env python3
"""
Improved Training Pipeline with BERT Cross-Attention Architecture
Integrates the sophisticated BERT-based ranking model with enhanced training data
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
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append('src')

from dl.enhanced_neural_preprocessing import EnhancedNeuralPreprocessor
from dl.ranking_losses import CombinedRankingLoss, NDCGLoss
from dl.improved_model_architecture import create_improved_ranking_model, BERTCrossAttentionRanker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTrainingPipeline:
    """Advanced training pipeline with BERT cross-attention architecture."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"🚀 Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = EnhancedNeuralPreprocessor(config)
        
        # Initialize loss functions
        self.ranking_loss = CombinedRankingLoss(
            ndcg_weight=0.4,
            listmle_weight=0.3,
            binary_weight=0.3,
            k=3
        )
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Model storage
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
    def create_tokenizer_based_model(self, model_type: str = "lightweight") -> nn.Module:
        """Create a model that uses the same tokenizer as preprocessing."""
        
        if model_type == "bert_cross_attention":
            # Use DistilBERT for memory efficiency
            model = BERTCrossAttentionRanker(
                model_name="distilbert-base-uncased",
                hidden_dim=768,
                dropout=0.3,
                num_attention_heads=8
            )
        elif model_type == "lightweight":
            model = create_improved_ranking_model("lightweight")
        else:
            model = create_improved_ranking_model("deep_interaction")
            
        return model.to(self.device)
    
    def train_bert_model(self, num_epochs: int = 15) -> Dict:
        """Train BERT cross-attention model for ranking."""
        
        logger.info("🏗️ Training Lightweight Cross-Attention Ranking Model...")
        
        # Get data loaders
        processed_data = self.preprocessor.preprocess_for_ranking()
        train_loader = processed_data['train_loader']
        val_loader = processed_data['val_loader']
        
        # Create lightweight model (memory efficient)
        model = self.create_tokenizer_based_model("lightweight")
        
        # Create optimizer with appropriate learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,  # Higher LR for lightweight model
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=3, 
            factor=0.5,
            min_lr=1e-6
        )
        
        # Training setup
        best_loss = float('inf')
        patience = 7
        patience_counter = 0
        training_history = []
        
        logger.info(f"🎯 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                query_ids = batch['query_input_ids'].to(self.device)
                query_mask = batch['query_attention_mask'].to(self.device)
                dataset_ids = batch['dataset_input_ids'].to(self.device)
                dataset_mask = batch['dataset_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = model(query_ids, query_mask, dataset_ids, dataset_mask)
                
                # Calculate loss
                loss = self.bce_loss(predictions, labels.float())
                
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
                    query_mask = batch['query_attention_mask'].to(self.device)
                    dataset_ids = batch['dataset_input_ids'].to(self.device)
                    dataset_mask = batch['dataset_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    predictions = model(query_ids, query_mask, dataset_ids, dataset_mask)
                    loss = self.bce_loss(predictions, labels.float())
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # Collect predictions
                    val_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / val_batches
            
            # Calculate validation accuracy
            val_predictions = np.array(val_predictions)
            val_labels = np.array(val_labels)
            val_accuracy = np.mean((val_predictions > 0.5) == val_labels)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping and model saving
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = Path('models/dl/lightweight_cross_attention_best.pt')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                }, model_path)
                
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
    
    def evaluate_model_comprehensive(self, model, test_loader) -> Dict:
        """Comprehensive evaluation including NDCG@3 calculation."""
        
        logger.info("📊 Starting comprehensive model evaluation...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_queries = []
        all_relevance_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                query_ids = batch['query_input_ids'].to(self.device)
                query_mask = batch['query_attention_mask'].to(self.device)
                dataset_ids = batch['dataset_input_ids'].to(self.device)
                dataset_mask = batch['dataset_attention_mask'].to(self.device)
                labels = batch['label']
                queries = batch['original_query']
                relevance_scores = batch['relevance_score']
                
                # Get predictions
                predictions = model(query_ids, query_mask, dataset_ids, dataset_mask)
                predictions = torch.sigmoid(predictions)  # Apply sigmoid for probabilities
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_queries.extend(queries)
                all_relevance_scores.extend(relevance_scores.numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        relevance_scores = np.array(all_relevance_scores)
        
        # Calculate basic metrics
        accuracy = np.mean((predictions > 0.5) == labels)
        
        # Calculate precision, recall, F1
        tp = np.sum((predictions > 0.5) & (labels == 1))
        fp = np.sum((predictions > 0.5) & (labels == 0))
        fn = np.sum((predictions <= 0.5) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate NDCG@3 per query
        query_ndcgs = []
        unique_queries = list(set(all_queries))
        
        logger.info(f"Calculating NDCG@3 for {len(unique_queries)} unique queries...")
        
        for query in unique_queries:
            # Get indices for this query
            query_mask = np.array(all_queries) == query
            query_preds = predictions[query_mask]
            query_labels = labels[query_mask]
            query_rel_scores = relevance_scores[query_mask]
            
            if len(query_preds) >= 3:
                # Sort by predictions (descending)
                sorted_indices = np.argsort(query_preds)[::-1]
                
                # Get top 3 predictions
                top_3_indices = sorted_indices[:3]
                top_3_relevance = query_rel_scores[top_3_indices]
                
                # Calculate DCG@3
                dcg = 0
                for i, rel_score in enumerate(top_3_relevance):
                    dcg += rel_score / np.log2(i + 2)
                
                # Calculate IDCG@3 (ideal DCG)
                ideal_relevance = np.sort(query_rel_scores)[::-1][:3]
                idcg = 0
                for i, rel_score in enumerate(ideal_relevance):
                    idcg += rel_score / np.log2(i + 2)
                
                # Calculate NDCG@3
                ndcg = dcg / idcg if idcg > 0 else 0
                query_ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(query_ndcgs) if query_ndcgs else 0
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ndcg_at_3': avg_ndcg,
            'num_queries_evaluated': len(query_ndcgs),
            'total_test_samples': len(predictions),
            'positive_samples': np.sum(labels == 1),
            'negative_samples': np.sum(labels == 0)
        }
        
        logger.info("📈 Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  NDCG@3: {avg_ndcg:.4f} ({avg_ndcg:.1%})")
        logger.info(f"  Queries evaluated: {len(query_ndcgs)}")
        
        return evaluation_results
    
    def run_improved_pipeline(self) -> Dict:
        """Run the complete improved training and evaluation pipeline."""
        
        logger.info("🎯 Starting Improved Training Pipeline with BERT Cross-Attention...")
        
        # Train lightweight model
        training_results = self.train_bert_model(num_epochs=15)
        
        # Evaluate on test set
        processed_data = self.preprocessor.preprocess_for_ranking()
        test_loader = processed_data['test_loader']
        
        evaluation_results = self.evaluate_model_comprehensive(training_results['model'], test_loader)
        
        # Combine results
        final_results = {
            'model_architecture': 'Lightweight Cross-Attention Ranker',
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_summary': processed_data['metadata']['summary'],
            'training_config': {
                'epochs': 15,
                'learning_rate': 2e-5,
                'batch_size': self.config.get('training', {}).get('batch_size', 32),
                'device': str(self.device)
            }
        }
        
        # Check if target achieved
        ndcg_percent = evaluation_results['ndcg_at_3'] * 100
        logger.info(f"\n🎯 FINAL RESULTS:")
        logger.info(f"  Model: Lightweight Cross-Attention Ranker")
        logger.info(f"  NDCG@3: {ndcg_percent:.1f}%")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.1%}")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.3f}")
        
        if ndcg_percent >= 70:
            logger.info(f"🎉 TARGET ACHIEVED! {ndcg_percent:.1f}% >= 70%")
        else:
            logger.info(f"📈 Progress: {ndcg_percent:.1f}% (need {70 - ndcg_percent:.1f}% more)")
            
        return final_results

def main():
    """Main function to run the improved training pipeline."""
    
    # Load configuration
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run pipeline
    pipeline = ImprovedTrainingPipeline(config)
    results = pipeline.run_improved_pipeline()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path(f"outputs/DL/improved_training_results_{timestamp}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    # Remove the model object before saving
    if 'model' in json_results.get('training_results', {}):
        del json_results['training_results']['model']
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    main()