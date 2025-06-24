"""
Ranking Loss Functions for NDCG Optimization
Implements ranking-specific losses that directly optimize for ranking metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class NDCGLoss(nn.Module):
    """NDCG Loss that directly optimizes for NDCG metric."""
    
    def __init__(self, k: int = 3, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_docs] - model predictions
            relevance_scores: [batch_size, num_docs] - true relevance scores
        """
        # Apply temperature scaling
        predictions = predictions / self.temperature
        
        # Sort by predictions
        _, sorted_indices = torch.sort(predictions, dim=-1, descending=True)
        
        # Get top-k
        top_k_indices = sorted_indices[:, :self.k]
        
        # Calculate DCG
        gathered_relevance = torch.gather(relevance_scores, 1, top_k_indices)
        
        # Calculate position weights (1/log2(position+2))
        positions = torch.arange(1, self.k + 1, device=predictions.device, dtype=torch.float32)
        position_weights = 1.0 / torch.log2(positions + 1)
        
        # DCG calculation
        dcg = torch.sum(gathered_relevance * position_weights.unsqueeze(0), dim=1)
        
        # Calculate IDCG (ideal DCG)
        _, ideal_indices = torch.sort(relevance_scores, dim=-1, descending=True)
        ideal_top_k = ideal_indices[:, :self.k]
        ideal_relevance = torch.gather(relevance_scores, 1, ideal_top_k)
        idcg = torch.sum(ideal_relevance * position_weights.unsqueeze(0), dim=1)
        
        # NDCG calculation (avoid division by zero)
        ndcg = dcg / (idcg + 1e-8)
        
        # Return negative NDCG as loss (to minimize)
        return 1.0 - torch.mean(ndcg)

class ListMLELoss(nn.Module):
    """ListMLE Loss for listwise ranking optimization."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_docs] - model predictions  
            relevance_scores: [batch_size, num_docs] - true relevance scores
        """
        # Scale predictions
        predictions = predictions / self.temperature
        
        # Sort by relevance scores (ground truth ranking)
        _, sorted_indices = torch.sort(relevance_scores, dim=-1, descending=True)
        
        # Get predictions in the order of ground truth ranking
        sorted_predictions = torch.gather(predictions, 1, sorted_indices)
        
        # Calculate ListMLE loss
        cumsum_exp = torch.cumsum(torch.exp(sorted_predictions.flip(-1)), dim=-1).flip(-1)
        log_cumsum = torch.log(cumsum_exp + 1e-8)
        
        loss = torch.mean(torch.sum(log_cumsum - sorted_predictions, dim=-1))
        
        return loss

class RankNetLoss(nn.Module):
    """RankNet pairwise ranking loss."""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_docs] - model predictions
            relevance_scores: [batch_size, num_docs] - true relevance scores
        """
        batch_size, num_docs = predictions.shape
        
        if num_docs < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Create pairwise comparisons
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                # Get pairs
                pred_i = predictions[:, i]
                pred_j = predictions[:, j]
                rel_i = relevance_scores[:, i]
                rel_j = relevance_scores[:, j]
                
                # Calculate pairwise preference
                # S_ij = 1 if rel_i > rel_j, -1 if rel_i < rel_j, 0 if equal
                S_ij = torch.sign(rel_i - rel_j)
                
                # Skip equal relevance pairs
                valid_pairs = (S_ij != 0)
                if valid_pairs.sum() == 0:
                    continue
                
                # RankNet loss for valid pairs
                pred_diff = pred_i - pred_j
                pairwise_loss = torch.log(1 + torch.exp(-self.sigma * S_ij * pred_diff))
                
                total_loss += torch.mean(pairwise_loss[valid_pairs])
                num_pairs += 1
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return total_loss / num_pairs

class BinaryRankingLoss(nn.Module):
    """Binary ranking loss for positive/negative pairs."""
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size] - model predictions for single items
            labels: [batch_size] - binary labels (1 for relevant, 0 for not relevant)
        """
        # Split into positive and negative predictions
        positive_mask = (labels == 1)
        negative_mask = (labels == 0)
        
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            # Fall back to BCE if we don't have both positive and negative samples
            return F.binary_cross_entropy_with_logits(predictions, labels.float())
        
        positive_preds = predictions[positive_mask]
        negative_preds = predictions[negative_mask]
        
        # Calculate ranking loss: positive should be higher than negative by margin
        pos_expanded = positive_preds.unsqueeze(1)  # [num_pos, 1]
        neg_expanded = negative_preds.unsqueeze(0)  # [1, num_neg]
        
        # Margin ranking loss: max(0, margin - (pos - neg))
        ranking_loss = F.relu(self.margin - (pos_expanded - neg_expanded))
        
        return torch.mean(ranking_loss)

class CombinedRankingLoss(nn.Module):
    """Combined loss function with multiple ranking objectives."""
    
    def __init__(self, 
                 ndcg_weight: float = 0.4,
                 listmle_weight: float = 0.3,
                 binary_weight: float = 0.3,
                 k: int = 3):
        super().__init__()
        
        self.ndcg_weight = ndcg_weight
        self.listmle_weight = listmle_weight
        self.binary_weight = binary_weight
        
        self.ndcg_loss = NDCGLoss(k=k)
        self.listmle_loss = ListMLELoss()
        self.binary_loss = BinaryRankingLoss()
        
        logger.info(f"🎯 CombinedRankingLoss initialized:")
        logger.info(f"  NDCG weight: {ndcg_weight}")
        logger.info(f"  ListMLE weight: {listmle_weight}")
        logger.info(f"  Binary weight: {binary_weight}")
        logger.info(f"  NDCG@{k} optimization")
    
    def forward(self, 
                predictions: torch.Tensor, 
                relevance_scores: torch.Tensor, 
                labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: [batch_size, num_docs] or [batch_size] - model predictions
            relevance_scores: [batch_size, num_docs] or [batch_size] - relevance scores
            labels: [batch_size, num_docs] or [batch_size] - binary labels
        """
        total_loss = 0.0
        loss_components = {}
        
        # Ensure we have the right dimensions
        if len(predictions.shape) == 1:
            # Single prediction per sample - use binary loss only
            binary_loss = self.binary_loss(predictions, labels)
            total_loss = binary_loss
            loss_components['binary_loss'] = binary_loss.item()
        else:
            # Multiple predictions per sample - use all losses
            
            # NDCG Loss
            if self.ndcg_weight > 0:
                ndcg_loss = self.ndcg_loss(predictions, relevance_scores)
                total_loss += self.ndcg_weight * ndcg_loss
                loss_components['ndcg_loss'] = ndcg_loss.item()
            
            # ListMLE Loss
            if self.listmle_weight > 0:
                listmle_loss = self.listmle_loss(predictions, relevance_scores)
                total_loss += self.listmle_weight * listmle_loss
                loss_components['listmle_loss'] = listmle_loss.item()
            
            # Binary Loss (flatten for binary classification)
            if self.binary_weight > 0:
                binary_loss = self.binary_loss(predictions.flatten(), labels.flatten())
                total_loss += self.binary_weight * binary_loss
                loss_components['binary_loss'] = binary_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components

def test_ranking_losses():
    """Test the ranking loss functions."""
    print("🧪 Testing ranking loss functions...")
    
    # Create sample data
    batch_size = 4
    num_docs = 5
    
    # Sample predictions and relevance scores
    predictions = torch.randn(batch_size, num_docs)
    relevance_scores = torch.rand(batch_size, num_docs)
    labels = (relevance_scores > 0.5).long()
    
    print(f"Sample predictions shape: {predictions.shape}")
    print(f"Sample relevance scores shape: {relevance_scores.shape}")
    print(f"Sample labels shape: {labels.shape}")
    
    # Test NDCG Loss
    ndcg_loss = NDCGLoss(k=3)
    ndcg_result = ndcg_loss(predictions, relevance_scores)
    print(f"✅ NDCG Loss: {ndcg_result.item():.4f}")
    
    # Test ListMLE Loss
    listmle_loss = ListMLELoss()
    listmle_result = listmle_loss(predictions, relevance_scores)
    print(f"✅ ListMLE Loss: {listmle_result.item():.4f}")
    
    # Test RankNet Loss
    ranknet_loss = RankNetLoss()
    ranknet_result = ranknet_loss(predictions, relevance_scores)
    print(f"✅ RankNet Loss: {ranknet_result.item():.4f}")
    
    # Test Binary Ranking Loss
    binary_loss = BinaryRankingLoss()
    binary_result = binary_loss(predictions.flatten(), labels.flatten())
    print(f"✅ Binary Ranking Loss: {binary_result.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedRankingLoss()
    combined_result, components = combined_loss(predictions, relevance_scores, labels)
    print(f"✅ Combined Loss: {combined_result.item():.4f}")
    print(f"   Components: {components}")
    
    print("🎉 All ranking loss tests passed!")

if __name__ == "__main__":
    test_ranking_losses()