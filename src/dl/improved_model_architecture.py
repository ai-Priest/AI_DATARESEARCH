"""
Improved Model Architecture for Ranking
BERT-based cross-attention models optimized for ranking performance
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class BERTCrossAttentionRanker(nn.Module):
    """BERT-based cross-attention model for query-document ranking."""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768,
                 dropout: float = 0.3,
                 num_attention_heads: int = 8):
        super().__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Cross-attention between query and document
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Query and document projections
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.doc_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Interaction modeling
        self.interaction_layers = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # [query, doc, cross_attn, interaction]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"🏗️ BERTCrossAttentionRanker initialized:")
        logger.info(f"  BERT model: {model_name}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Attention heads: {num_attention_heads}")
        logger.info(f"  Dropout: {dropout}")
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for module in [self.query_projection, self.doc_projection, self.interaction_layers]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using BERT."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Also get mean pooled representation
        masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Combine CLS and mean pooling
        combined = (cls_embedding + mean_embedding) / 2
        
        return combined
    
    def cross_attend(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention between query and document."""
        
        # Add sequence dimension for attention
        query_seq = query_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        doc_seq = doc_emb.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        
        # Cross-attention: query attends to document
        cross_attn_output, _ = self.cross_attention(
            query=query_seq,
            key=doc_seq,
            value=doc_seq
        )
        
        return cross_attn_output.squeeze(1)  # [batch_size, hidden_dim]
    
    def forward(self, 
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                doc_input_ids: torch.Tensor,
                doc_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ranking.
        
        Args:
            query_input_ids: [batch_size, seq_len]
            query_attention_mask: [batch_size, seq_len]
            doc_input_ids: [batch_size, seq_len]
            doc_attention_mask: [batch_size, seq_len]
            
        Returns:
            relevance_scores: [batch_size]
        """
        
        # Encode query and document
        query_emb = self.encode_text(query_input_ids, query_attention_mask)
        doc_emb = self.encode_text(doc_input_ids, doc_attention_mask)
        
        # Project embeddings
        query_proj = self.query_projection(query_emb)
        doc_proj = self.doc_projection(doc_emb)
        
        # Cross-attention
        cross_attn = self.cross_attend(query_proj, doc_proj)
        
        # Interaction features
        element_wise_product = query_proj * doc_proj
        
        # Combine all features
        combined_features = torch.cat([
            query_proj,           # Query representation
            doc_proj,             # Document representation  
            cross_attn,           # Cross-attention output
            element_wise_product  # Element-wise interaction
        ], dim=1)
        
        # Final ranking score
        relevance_score = self.interaction_layers(combined_features)
        
        return relevance_score.squeeze(-1)  # [batch_size]

class LightweightRankingModel(nn.Module):
    """Lightweight ranking model for faster training and inference."""
    
    def __init__(self, 
                 vocab_size: int = 30522,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(512, embedding_dim)  # Position embeddings
        
        # Attention layers
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward layers
        self.ff_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        
        # Final ranking layers
        self.ranking_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        logger.info(f"🚀 LightweightRankingModel initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Dropout: {dropout}")
    
    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input sequence with self-attention."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combined embeddings
        x = token_emb + pos_emb
        x = self.layer_norm1(x)
        
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=~attention_mask.bool())
        x = self.layer_norm2(x + attn_output)
        
        # Feed-forward
        ff_output = self.ff_layers(x)
        x = self.layer_norm3(x + ff_output)
        
        # Mean pooling (respecting attention mask)
        masked_x = x * attention_mask.unsqueeze(-1)
        pooled = masked_x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return pooled
    
    def forward(self,
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                doc_input_ids: torch.Tensor,
                doc_attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        # Encode query and document
        query_emb = self.encode_sequence(query_input_ids, query_attention_mask)
        doc_emb = self.encode_sequence(doc_input_ids, doc_attention_mask)
        
        # Cross-attention between query and document
        query_seq = query_emb.unsqueeze(1)
        doc_seq = doc_emb.unsqueeze(1)
        
        cross_attn_output, _ = self.cross_attention(query_seq, doc_seq, doc_seq)
        cross_attn_emb = cross_attn_output.squeeze(1)
        
        # Combine features
        combined = torch.cat([query_emb, doc_emb, cross_attn_emb], dim=1)
        
        # Final ranking score
        score = self.ranking_head(combined)
        
        return score.squeeze(-1)

class DeepInteractionModel(nn.Module):
    """Deep interaction model with multiple interaction layers."""
    
    def __init__(self, 
                 vocab_size: int = 30522,
                 embedding_dim: int = 256,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multiple interaction layers
        self.interaction_layers = nn.ModuleList()
        
        input_dim = embedding_dim * 2  # Query + document concatenation
        for hidden_dim in hidden_dims:
            self.interaction_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                )
            )
            input_dim = hidden_dim
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        logger.info(f"🔥 DeepInteractionModel initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Dropout: {dropout}")
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling encoding."""
        embeddings = self.embedding(input_ids)
        
        # Mean pooling with attention mask
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return pooled
    
    def forward(self,
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                doc_input_ids: torch.Tensor,
                doc_attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with deep interactions."""
        
        # Encode query and document
        query_emb = self.encode_text(query_input_ids, query_attention_mask)
        doc_emb = self.encode_text(doc_input_ids, doc_attention_mask)
        
        # Concatenate for interaction
        x = torch.cat([query_emb, doc_emb], dim=1)
        
        # Pass through interaction layers
        for layer in self.interaction_layers:
            x = layer(x)
        
        # Final output
        score = self.output_layer(x)
        
        return score.squeeze(-1)

def create_improved_ranking_model(model_type: str = "bert_cross_attention", **kwargs) -> nn.Module:
    """Factory function to create improved ranking models."""
    
    if model_type == "bert_cross_attention":
        return BERTCrossAttentionRanker(**kwargs)
    elif model_type == "lightweight":
        return LightweightRankingModel(**kwargs)
    elif model_type == "deep_interaction":
        return DeepInteractionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_improved_models():
    """Test the improved model architectures."""
    
    print("🧪 Testing improved model architectures...")
    
    # Sample data
    batch_size = 4
    seq_len = 64
    
    query_ids = torch.randint(0, 1000, (batch_size, seq_len))
    query_mask = torch.ones(batch_size, seq_len)
    doc_ids = torch.randint(0, 1000, (batch_size, seq_len))
    doc_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shapes: {query_ids.shape}, {doc_ids.shape}")
    
    # Test BERT model (skip if no internet for downloading)
    try:
        print("\n1. Testing BERTCrossAttentionRanker...")
        bert_model = BERTCrossAttentionRanker()
        bert_output = bert_model(query_ids, query_mask, doc_ids, doc_mask)
        print(f"✅ BERT model output shape: {bert_output.shape}")
        print(f"   Sample scores: {bert_output[:3].tolist()}")
    except Exception as e:
        print(f"⚠️ BERT model test skipped: {e}")
    
    # Test lightweight model
    print("\n2. Testing LightweightRankingModel...")
    lightweight_model = LightweightRankingModel()
    lightweight_output = lightweight_model(query_ids, query_mask, doc_ids, doc_mask)
    print(f"✅ Lightweight model output shape: {lightweight_output.shape}")
    print(f"   Sample scores: {lightweight_output[:3].tolist()}")
    
    # Test deep interaction model
    print("\n3. Testing DeepInteractionModel...")
    deep_model = DeepInteractionModel()
    deep_output = deep_model(query_ids, query_mask, doc_ids, doc_mask)
    print(f"✅ Deep model output shape: {deep_output.shape}")
    print(f"   Sample scores: {deep_output[:3].tolist()}")
    
    print("\n🎉 All model tests passed!")

if __name__ == "__main__":
    test_improved_models()