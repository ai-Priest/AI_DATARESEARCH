#!/usr/bin/env python3
"""
Deep Learning Pipeline with Graded Relevance - Target 70% NDCG@3
Enhanced pipeline with graded relevance scoring and optimized thresholds
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
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Add src to path
sys.path.append('src')

from dl.enhanced_neural_preprocessing import EnhancedNeuralPreprocessor, RankingDataset
from dl.ranking_losses import CombinedRankingLoss, NDCGLoss
from dl.improved_model_architecture import create_improved_ranking_model, BERTCrossAttentionRanker
from dl.deep_evaluation import DeepEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradedRelevancePipeline:
    """Enhanced pipeline with graded relevance scoring for 70% NDCG@3 target."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"🚀 Using device: {self.device}")
        
        # Load graded relevance training data path
        self.graded_data_path = "data/processed/graded_relevance_training.json"
        
        # Load threshold tuning data
        with open("data/processed/threshold_tuning_analysis.json", 'r') as f:
            threshold_data = json.load(f)
        self.optimal_threshold = threshold_data['global_optimal_threshold']
        logger.info(f"🎯 Using optimized threshold: {self.optimal_threshold:.3f}")
        
        # Update config for graded relevance
        if 'preprocessing' not in self.config:
            self.config['preprocessing'] = {}
        self.config['preprocessing']['enhanced_training_data_path'] = self.graded_data_path
        
        # Initialize preprocessor with graded data
        self.preprocessor = self._create_graded_preprocessor(config)
        
        # Initialize enhanced loss for graded relevance
        self.graded_ranking_loss = self._create_graded_ranking_loss()
        
        # Model storage
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
    
    def _create_graded_preprocessor(self, config):
        """Create a preprocessor that uses graded relevance data."""
        
        class GradedNeuralPreprocessor:
            def __init__(self, config, graded_data_path):
                self.config = config
                self.graded_data_path = graded_data_path
                
                # Initialize tokenizer
                model_name = "bert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                logger.info("🚀 Graded Neural Preprocessor initialized")
                logger.info(f"Using graded data: {graded_data_path}")
            
            def preprocess_for_ranking(self):
                """Preprocess graded relevance data for ranking."""
                # Load graded data
                with open(self.graded_data_path, 'r') as f:
                    data = json.load(f)
                
                samples = data['training_samples']
                logger.info(f"📊 Loading {len(samples)} graded samples")
                
                # Load datasets for additional info
                datasets_df = self._load_datasets()
                
                # Enhance samples with dataset text
                enhanced_samples = []
                for sample in samples:
                    dataset_id = sample['dataset_id']
                    dataset_row = datasets_df[datasets_df['dataset_id'] == dataset_id]
                    
                    if not dataset_row.empty:
                        dataset = dataset_row.iloc[0]
                        enhanced_sample = sample.copy()
                        enhanced_sample['dataset_title'] = dataset.get('title', '')
                        enhanced_sample['dataset_description'] = dataset.get('description', '')
                        enhanced_samples.append(enhanced_sample)
                
                # Split data
                np.random.seed(42)
                np.random.shuffle(enhanced_samples)
                
                n_samples = len(enhanced_samples)
                n_train = int(0.7 * n_samples)
                n_val = int(0.15 * n_samples)
                
                train_samples = enhanced_samples[:n_train]
                val_samples = enhanced_samples[n_train:n_train+n_val]
                test_samples = enhanced_samples[n_train+n_val:]
                
                # Create datasets
                train_dataset = RankingDataset(train_samples, self.tokenizer)
                val_dataset = RankingDataset(val_samples, self.tokenizer)
                test_dataset = RankingDataset(test_samples, self.tokenizer)
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                logger.info("✅ Created graded data loaders")
                logger.info(f"  Train batches: {len(train_loader)}")
                logger.info(f"  Val batches: {len(val_loader)}")
                logger.info(f"  Test batches: {len(test_loader)}")
                
                return {
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader,
                    'metadata': {
                        'train_samples': len(train_samples),
                        'val_samples': len(val_samples),
                        'test_samples': len(test_samples),
                        'summary': data.get('metadata', {})
                    }
                }
            
            def _load_datasets(self):
                """Load datasets for text enrichment."""
                datasets_path = Path("data/processed")
                singapore_df = pd.read_csv(datasets_path / "singapore_datasets.csv")
                global_df = pd.read_csv(datasets_path / "global_datasets.csv")
                return pd.concat([singapore_df, global_df], ignore_index=True)
        
        return GradedNeuralPreprocessor(config, self.graded_data_path)
        
    def _create_graded_ranking_loss(self):
        """Create loss function that handles graded relevance scores."""
        class GradedRankingLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse_loss = nn.MSELoss()
                self.bce_loss = nn.BCEWithLogitsLoss()
                
            def forward(self, predictions, labels, relevance_scores=None):
                # If we have graded relevance scores, use MSE loss
                if relevance_scores is not None:
                    # Convert predictions to [0, 1] range
                    pred_probs = torch.sigmoid(predictions)
                    # MSE loss with graded relevance
                    return self.mse_loss(pred_probs, relevance_scores)
                else:
                    # Fallback to binary cross-entropy
                    return self.bce_loss(predictions, labels.float())
        
        return GradedRankingLoss()
    
    def create_enhanced_model(self, model_type: str = "lightweight_graded") -> nn.Module:
        """Create model optimized for graded relevance."""
        
        class GradedRankingModel(nn.Module):
            def __init__(self, vocab_size=50000, embedding_dim=256, hidden_dim=512, dropout=0.3):
                super().__init__()
                self.query_encoder = nn.Sequential(
                    nn.Embedding(vocab_size, embedding_dim),
                    nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True),
                    nn.Dropout(dropout)
                )
                
                self.doc_encoder = nn.Sequential(
                    nn.Embedding(vocab_size, embedding_dim),
                    nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True),
                    nn.Dropout(dropout)
                )
                
                # Cross-attention with 8 heads
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, 
                    num_heads=8, 
                    dropout=dropout,
                    batch_first=True
                )
                
                # Enhanced ranking head for graded scores
                self.ranking_head = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                logger.info(f"🚀 GradedRankingModel initialized for 4-level relevance scoring")
                
            def forward(self, query_ids, query_mask, doc_ids, doc_mask):
                # Encode query
                query_emb = self.query_encoder[0](query_ids)
                query_encoded, _ = self.query_encoder[1](query_emb)
                query_encoded = self.query_encoder[2](query_encoded)
                
                # Encode document
                doc_emb = self.doc_encoder[0](doc_ids)
                doc_encoded, _ = self.doc_encoder[1](doc_emb)
                doc_encoded = self.doc_encoder[2](doc_encoded)
                
                # Cross-attention
                attended, _ = self.cross_attention(
                    query_encoded, doc_encoded, doc_encoded,
                    key_padding_mask=~doc_mask.bool()
                )
                
                # Pool representations
                query_pooled = query_encoded.mean(dim=1)
                doc_pooled = doc_encoded.mean(dim=1)
                attended_pooled = attended.mean(dim=1)
                
                # Combine features
                combined = torch.cat([query_pooled, doc_pooled, attended_pooled], dim=-1)
                
                # Get ranking score
                score = self.ranking_head(combined)
                
                return score.squeeze(-1)
        
        model = GradedRankingModel()
        return model.to(self.device)
    
    def train_graded_model(self, num_epochs: int = 20) -> Dict:
        """Train model with graded relevance scoring."""
        
        logger.info("🏗️ Training Graded Relevance Ranking Model...")
        
        # Get data loaders with graded scores
        processed_data = self.preprocessor.preprocess_for_ranking()
        train_loader = processed_data['train_loader']
        val_loader = processed_data['val_loader']
        
        # Create model
        model = self.create_enhanced_model()
        
        # Optimizer with adjusted learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,  # Lower learning rate for graded learning
            weight_decay=0.02,
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training setup
        best_loss = float('inf')
        best_ndcg = 0.0
        patience = 10
        patience_counter = 0
        training_history = []
        
        logger.info(f"🎯 Starting graded relevance training for {num_epochs} epochs...")
        
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
                relevance_scores = batch['relevance_score'].to(self.device)
                
                # Forward pass
                predictions = model(query_ids, query_mask, dataset_ids, dataset_mask)
                
                # Graded loss
                loss = self.graded_ranking_loss(predictions, labels, relevance_scores)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = total_train_loss / train_batches
            
            # Validation phase with NDCG calculation
            model.eval()
            total_val_loss = 0
            val_batches = 0
            val_predictions = []
            val_labels = []
            val_relevance = []
            val_queries = []
            
            with torch.no_grad():
                for batch in val_loader:
                    query_ids = batch['query_input_ids'].to(self.device)
                    query_mask = batch['query_attention_mask'].to(self.device)
                    dataset_ids = batch['dataset_input_ids'].to(self.device)
                    dataset_mask = batch['dataset_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    relevance_scores = batch['relevance_score'].to(self.device)
                    
                    predictions = model(query_ids, query_mask, dataset_ids, dataset_mask)
                    loss = self.graded_ranking_loss(predictions, labels, relevance_scores)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # Collect for NDCG
                    val_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_relevance.extend(relevance_scores.cpu().numpy())
                    val_queries.extend(batch['original_query'])
            
            avg_val_loss = total_val_loss / val_batches
            
            # Calculate NDCG@3 with graded relevance
            val_ndcg = self._calculate_graded_ndcg(val_queries, val_predictions, val_relevance, k=3)
            
            # Update scheduler
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val NDCG@3: {val_ndcg:.4f} ({val_ndcg:.1%}), "
                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_ndcg': val_ndcg,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Save best model based on NDCG
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_loss = avg_val_loss
                patience_counter = 0
                
                # Save model
                model_path = Path('models/dl/graded_relevance_best.pt')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                    'val_ndcg': val_ndcg
                }, model_path)
                
                logger.info(f"💾 Saved best model with NDCG@3: {best_ndcg:.4f} ({best_ndcg:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⏰ Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'model': model,
            'best_loss': best_loss,
            'best_ndcg': best_ndcg,
            'training_history': training_history
        }
    
    def _calculate_graded_ndcg(self, queries: List[str], predictions: np.ndarray, 
                              relevance_scores: np.ndarray, k: int = 3) -> float:
        """Calculate NDCG@k with graded relevance scores."""
        
        query_ndcgs = []
        unique_queries = list(set(queries))
        
        for query in unique_queries:
            # Get indices for this query
            query_indices = [i for i, q in enumerate(queries) if q == query]
            if not query_indices:
                continue
            
            query_preds = np.array([predictions[i] for i in query_indices])
            query_relevance = np.array([relevance_scores[i] for i in query_indices])
            
            if len(query_preds) >= k:
                # Sort by predictions (descending)
                sorted_indices = np.argsort(query_preds)[::-1]
                
                # Get top k relevance scores
                top_k_relevance = query_relevance[sorted_indices[:k]]
                
                # Calculate DCG@k
                dcg = 0
                for i, rel_score in enumerate(top_k_relevance):
                    dcg += rel_score / np.log2(i + 2)
                
                # Calculate IDCG@k (ideal DCG)
                ideal_relevance = np.sort(query_relevance)[::-1][:k]
                idcg = 0
                for i, rel_score in enumerate(ideal_relevance):
                    idcg += rel_score / np.log2(i + 2)
                
                # Calculate NDCG@k
                ndcg = dcg / idcg if idcg > 0 else 0
                query_ndcgs.append(ndcg)
        
        return np.mean(query_ndcgs) if query_ndcgs else 0
    
    def evaluate_with_threshold_tuning(self, model, test_loader, threshold: float = None) -> Dict:
        """Evaluate model with optimized threshold."""
        
        if threshold is None:
            threshold = self.optimal_threshold
            
        logger.info(f"📊 Evaluating model with threshold: {threshold:.3f}")
        
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
                predictions = torch.sigmoid(predictions)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_queries.extend(queries)
                all_relevance_scores.extend(relevance_scores.numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        relevance_scores = np.array(all_relevance_scores)
        
        # Apply optimized threshold
        binary_predictions = (predictions > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((binary_predictions == 1) & (labels == 1))
        fp = np.sum((binary_predictions == 1) & (labels == 0))
        fn = np.sum((binary_predictions == 0) & (labels == 1))
        tn = np.sum((binary_predictions == 0) & (labels == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate NDCG@3 with graded relevance
        ndcg_at_3 = self._calculate_graded_ndcg(all_queries, predictions.tolist(), relevance_scores.tolist(), k=3)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ndcg_at_3': ndcg_at_3,
            'threshold_used': threshold,
            'num_queries_evaluated': len(set(all_queries)),
            'total_test_samples': len(predictions),
            'positive_samples': np.sum(labels == 1),
            'negative_samples': np.sum(labels == 0)
        }
        
        logger.info("📈 Evaluation Results with Optimized Threshold:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        logger.info(f"  Precision: {precision:.4f} (improved)")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  NDCG@3: {ndcg_at_3:.4f} ({ndcg_at_3:.1%})")
        
        return evaluation_results
    
    def run_graded_pipeline(self) -> Dict:
        """Run the complete graded relevance pipeline."""
        
        logger.info("🎯 Starting Graded Relevance Pipeline for 70% NDCG@3 Target...")
        
        # Train model with graded relevance
        training_results = self.train_graded_model(num_epochs=20)
        
        # Evaluate with optimized threshold
        processed_data = self.preprocessor.preprocess_for_ranking()
        test_loader = processed_data['test_loader']
        
        evaluation_results = self.evaluate_with_threshold_tuning(
            training_results['model'], 
            test_loader
        )
        
        # Combine results
        final_results = {
            'model_architecture': 'Graded Relevance Cross-Attention Ranker',
            'training_results': {
                'best_loss': training_results['best_loss'],
                'best_ndcg': training_results['best_ndcg'],
                'training_history': training_results['training_history']
            },
            'evaluation_results': evaluation_results,
            'improvements': {
                'graded_relevance': 'Implemented 4-level scoring (0.0, 0.3, 0.7, 1.0)',
                'threshold_tuning': f'Optimized to {self.optimal_threshold:.3f}',
                'expected_improvement': 'Target 70% NDCG@3'
            }
        }
        
        # Check target achievement
        ndcg_percent = evaluation_results['ndcg_at_3'] * 100
        logger.info(f"\n🎯 FINAL RESULTS - GRADED RELEVANCE:")
        logger.info(f"  Model: Graded Relevance Cross-Attention Ranker")
        logger.info(f"  NDCG@3: {ndcg_percent:.1f}%")
        logger.info(f"  Precision: {evaluation_results['precision']:.3f}")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.3f}")
        
        if ndcg_percent >= 70:
            logger.info(f"🎉 TARGET ACHIEVED! {ndcg_percent:.1f}% >= 70%")
        else:
            logger.info(f"📈 Progress: {ndcg_percent:.1f}% (need {70 - ndcg_percent:.1f}% more)")
            
        return final_results


def main():
    """Main function for graded relevance DL pipeline."""
    
    print("🎯 AI Data Research Assistant - Graded Relevance DL Pipeline")
    print("🚀 Target: 70% NDCG@3 Achievement")
    print("=" * 60)
    
    # Load configuration
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run pipeline
    pipeline = GradedRelevancePipeline(config)
    results = pipeline.run_graded_pipeline()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path(f"outputs/DL/graded_relevance_results_{timestamp}.json")
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
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_path}")
    
    # Generate visualizations
    from dl_pipeline import ImprovedTrainingPipeline
    temp_pipeline = ImprovedTrainingPipeline(config)
    temp_pipeline.generate_comprehensive_visualizations(results, Path("outputs/DL"))
    
    return results


if __name__ == "__main__":
    main()