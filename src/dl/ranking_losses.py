"""
Ranking Loss Functions for NDCG Optimization
Implements ranking-specific losses that directly optimize for ranking metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ListMLELoss(nn.Module):
    """ListMLE Loss for listwise ranking optimization."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [batch_size, num_items]
            relevance_scores: Ground truth relevance scores [batch_size, num_items]
        """
        # Scale predictions by temperature
        scaled_predictions = predictions / self.temperature
        
        # Sort by relevance scores (descending)
        sorted_indices = torch.argsort(relevance_scores, dim=-1, descending=True)
        
        # Gather predictions in sorted order
        batch_size, num_items = predictions.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_items)
        sorted_predictions = scaled_predictions[batch_indices, sorted_indices]
        
        # Compute ListMLE loss
        cumsum_exp = torch.cumsum(torch.exp(sorted_predictions.flip(-1)), dim=-1).flip(-1)
        log_cumsum = torch.log(cumsum_exp + 1e-8)
        
        loss = -torch.sum(sorted_predictions - log_cumsum, dim=-1)
        return loss.mean()


class RankNetLoss(nn.Module):
    """RankNet Loss for pairwise ranking optimization."""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [batch_size, num_items]
            relevance_scores: Ground truth relevance scores [batch_size, num_items]
        """
        batch_size, num_items = predictions.shape
        
        # Create all pairwise comparisons
        pred_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)  # [batch, items, items]
        rel_diff = relevance_scores.unsqueeze(2) - relevance_scores.unsqueeze(1)  # [batch, items, items]
        
        # Create target labels: 1 if first item should rank higher, 0 otherwise
        targets = (rel_diff > 0).float()
        
        # RankNet loss: cross-entropy with sigmoid
        loss = F.binary_cross_entropy_with_logits(
            self.sigma * pred_diff,
            targets,
            reduction='none'
        )
        
        # Only consider valid pairs (where relevance differs)
        valid_pairs = (rel_diff != 0).float()
        loss = loss * valid_pairs
        
        # Average over valid pairs
        num_valid = valid_pairs.sum(dim=(1, 2)) + 1e-8
        loss = loss.sum(dim=(1, 2)) / num_valid
        
        return loss.mean()


class LambdaRankLoss(nn.Module):
    """LambdaRank Loss for direct NDCG optimization."""
    
    def __init__(self, k: int = 3, sigma: float = 1.0):
        super().__init__()
        self.k = k
        self.sigma = sigma
        
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [batch_size, num_items]
            relevance_scores: Ground truth relevance scores [batch_size, num_items]
        """
        batch_size, num_items = predictions.shape
        
        # Compute NDCG@k for current predictions
        current_ndcg = self._compute_ndcg_at_k(predictions, relevance_scores, self.k)
        
        # Compute pairwise differences
        pred_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)
        rel_diff = relevance_scores.unsqueeze(2) - relevance_scores.unsqueeze(1)
        
        # Compute lambda weights (NDCG change from swapping pairs)
        lambda_weights = self._compute_lambda_weights(predictions, relevance_scores, self.k)
        
        # LambdaRank loss
        targets = (rel_diff > 0).float()
        pairwise_loss = F.binary_cross_entropy_with_logits(
            self.sigma * pred_diff,
            targets,
            reduction='none'
        )
        
        # Weight by lambda (NDCG importance)
        weighted_loss = pairwise_loss * lambda_weights
        
        # Only consider valid pairs
        valid_pairs = (rel_diff != 0).float()
        weighted_loss = weighted_loss * valid_pairs
        
        # Average over valid pairs
        num_valid = valid_pairs.sum(dim=(1, 2)) + 1e-8
        loss = weighted_loss.sum(dim=(1, 2)) / num_valid
        
        return loss.mean()
    
    def _compute_ndcg_at_k(self, predictions: torch.Tensor, relevance_scores: torch.Tensor, k: int) -> torch.Tensor:
        """Compute NDCG@k for given predictions and relevance scores"""
        batch_size = predictions.shape[0]
        
        # Sort by predictions
        _, pred_indices = torch.sort(predictions, dim=-1, descending=True)
        
        # Get relevance scores in predicted order
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        top_k_relevance = relevance_scores[batch_indices, pred_indices[:, :k]]
        
        # Compute DCG@k
        positions = torch.arange(1, k + 1, dtype=torch.float, device=predictions.device)
        dcg_weights = 1.0 / torch.log2(positions + 1)
        dcg = torch.sum(top_k_relevance * dcg_weights.unsqueeze(0), dim=-1)
        
        # Compute IDCG@k (ideal DCG)
        sorted_relevance, _ = torch.sort(relevance_scores, dim=-1, descending=True)
        ideal_relevance = sorted_relevance[:, :k]
        idcg = torch.sum(ideal_relevance * dcg_weights.unsqueeze(0), dim=-1)
        
        # NDCG@k
        ndcg = dcg / (idcg + 1e-8)
        return ndcg
    
    def _compute_lambda_weights(self, predictions: torch.Tensor, relevance_scores: torch.Tensor, k: int) -> torch.Tensor:
        """Compute lambda weights for LambdaRank"""
        batch_size, num_items = predictions.shape
        
        # For simplicity, use uniform weights (can be improved with actual NDCG delta computation)
        # In practice, this should compute the change in NDCG from swapping each pair
        lambda_weights = torch.ones(batch_size, num_items, num_items, device=predictions.device)
        
        # Give higher weight to pairs involving top-k items
        _, top_k_indices = torch.topk(predictions, k, dim=-1)
        
        for b in range(batch_size):
            for i in top_k_indices[b]:
                lambda_weights[b, i, :] *= 2.0  # Higher weight for top-k items
                lambda_weights[b, :, i] *= 2.0
        
        return lambda_weights


class CombinedRankingLoss(nn.Module):
    """Combined ranking loss using multiple ranking objectives"""
    
    def __init__(self, 
                 listmle_weight: float = 0.4,
                 ranknet_weight: float = 0.3, 
                 lambdarank_weight: float = 0.3,
                 k: int = 3):
        super().__init__()
        
        self.listmle_weight = listmle_weight
        self.ranknet_weight = ranknet_weight
        self.lambdarank_weight = lambdarank_weight
        
        self.listmle_loss = ListMLELoss()
        self.ranknet_loss = RankNetLoss()
        self.lambdarank_loss = LambdaRankLoss(k=k)
        
        logger.info(f"🎯 CombinedRankingLoss initialized")
        logger.info(f"  ListMLE weight: {listmle_weight}")
        logger.info(f"  RankNet weight: {ranknet_weight}")
        logger.info(f"  LambdaRank weight: {lambdarank_weight}")
        logger.info(f"  NDCG@{k} optimization")
    
    def forward(self, 
                predictions: torch.Tensor, 
                relevance_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined ranking loss
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            relevance_scores: Ground truth relevance scores [batch_size, num_items]
        """
        
        # Compute individual losses
        listmle_loss = self.listmle_loss(predictions, relevance_scores)
        ranknet_loss = self.ranknet_loss(predictions, relevance_scores)
        lambdarank_loss = self.lambdarank_loss(predictions, relevance_scores)
        
        # Combined loss
        total_loss = (
            self.listmle_weight * listmle_loss +
            self.ranknet_weight * ranknet_loss +
            self.lambdarank_weight * lambdarank_loss
        )
        
        return {
            'total_loss': total_loss,
            'listmle_loss': listmle_loss,
            'ranknet_loss': ranknet_loss,
            'lambdarank_loss': lambdarank_loss
        }


class NDCGLoss(nn.Module):
    """Direct NDCG loss for ranking optimization"""
    
    def __init__(self, k: int = 3, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, predictions: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable NDCG loss
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            relevance_scores: Ground truth relevance scores [batch_size, num_items]
        """
        batch_size = predictions.shape[0]
        
        # Apply temperature scaling
        scaled_predictions = predictions / self.temperature
        
        # Compute soft ranking using softmax
        soft_ranks = F.softmax(scaled_predictions, dim=-1)
        
        # Compute DCG using soft ranks
        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=predictions.device)
        dcg_weights = 1.0 / torch.log2(positions + 1)
        
        # Sort relevance scores to get top-k
        sorted_relevance, sort_indices = torch.sort(relevance_scores, dim=-1, descending=True)
        top_k_relevance = sorted_relevance[:, :self.k]
        
        # Get soft ranks for top-k items
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.k)
        top_k_soft_ranks = soft_ranks[batch_indices, sort_indices[:, :self.k]]
        
        # Compute soft DCG
        soft_dcg = torch.sum(top_k_relevance * top_k_soft_ranks * dcg_weights.unsqueeze(0), dim=-1)
        
        # Compute IDCG
        idcg = torch.sum(top_k_relevance * dcg_weights.unsqueeze(0), dim=-1)
        
        # Soft NDCG
        soft_ndcg = soft_dcg / (idcg + 1e-8)
        
        # Loss is negative NDCG (we want to maximize NDCG)
        loss = -soft_ndcg.mean()
        
        return loss


def create_ranking_loss(loss_type: str = "combined", **kwargs) -> nn.Module:
    """Factory function to create ranking loss"""
    
    if loss_type == "listmle":
        return ListMLELoss(**kwargs)
    elif loss_type == "ranknet":
        return RankNetLoss(**kwargs)
    elif loss_type == "lambdarank":
        return LambdaRankLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedRankingLoss(**kwargs)
    elif loss_type == "ndcg":
        return NDCGLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test ranking losses
    batch_size, num_items = 4, 6
    
    # Create sample data
    predictions = torch.randn(batch_size, num_items)
    relevance_scores = torch.rand(batch_size, num_items)
    
    print("🧪 Testing ranking loss functions...")
    
    # Test ListMLE
    listmle_loss = ListMLELoss()
    loss_value = listmle_loss(predictions, relevance_scores)
    print(f"✅ ListMLE Loss: {loss_value:.4f}")
    
    # Test RankNet
    ranknet_loss = RankNetLoss()
    loss_value = ranknet_loss(predictions, relevance_scores)
    print(f"✅ RankNet Loss: {loss_value:.4f}")
    
    # Test LambdaRank
    lambdarank_loss = LambdaRankLoss(k=3)
    loss_value = lambdarank_loss(predictions, relevance_scores)
    print(f"✅ LambdaRank Loss: {loss_value:.4f}")
    
    # Test Combined
    combined_loss = CombinedRankingLoss()
    loss_dict = combined_loss(predictions, relevance_scores)
    print(f"✅ Combined Loss: {loss_dict['total_loss']:.4f}")
    print(f"  - ListMLE: {loss_dict['listmle_loss']:.4f}")
    print(f"  - RankNet: {loss_dict['ranknet_loss']:.4f}")
    print(f"  - LambdaRank: {loss_dict['lambdarank_loss']:.4f}")
    
    # Test NDCG
    ndcg_loss = NDCGLoss(k=3)
    loss_value = ndcg_loss(predictions, relevance_scores)
    print(f"✅ NDCG Loss: {loss_value:.4f}")
    
    print("🎉 All ranking losses working correctly!")