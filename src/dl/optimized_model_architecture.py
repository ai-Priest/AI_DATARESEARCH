"""
Optimized Model Architecture - Quality-First with Reduced Parameters
Implements lightweight neural model optimized for ranking quality with reduced parameters
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for query-document matching"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, hidden_dim]
            key: [batch_size, seq_len_k, hidden_dim]  
            value: [batch_size, seq_len_v, hidden_dim]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # [batch_size, seq_len_q, hidden_dim]
        K = self.key_proj(key)      # [batch_size, seq_len_k, hidden_dim]
        V = self.value_proj(value)  # [batch_size, seq_len_v, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )
        
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output

class DomainSpecificHead(nn.Module):
    """Domain-specific head for specialized routing"""
    
    def __init__(self, input_dim: int, domain_name: str, num_sources: int = 10):
        super().__init__()
        self.domain_name = domain_name
        self.num_sources = num_sources
        
        # Lightweight domain-specific layers
        self.domain_proj = nn.Linear(input_dim, input_dim // 2)
        self.source_scorer = nn.Linear(input_dim // 2, num_sources)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            source_scores: [batch_size, num_sources]
        """
        # Domain-specific projection
        domain_features = F.relu(self.domain_proj(x))
        domain_features = self.dropout(domain_features)
        
        # Source scoring
        source_scores = self.source_scorer(domain_features)
        
        return source_scores

class OptimizedQualityModel(nn.Module):
    """Optimized model with reduced parameters while maintaining quality"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions (reduced from original)
        self.vocab_size = config.get('vocab_size', 10000)
        self.embedding_dim = config.get('embedding_dim', 64)  # Reduced from 128
        self.hidden_dim = config.get('hidden_dim', 128)      # Reduced from 256
        self.num_sources = config.get('num_sources', 10)
        self.num_domains = config.get('num_domains', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Shared embeddings (parameter sharing)
        self.query_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.source_embedding = nn.Embedding(self.num_sources, self.embedding_dim)
        
        # Lightweight encoder layers
        self.query_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.source_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Cross-attention for query-source matching
        self.cross_attention = CrossAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=2,  # Reduced from 4
            dropout=self.dropout
        )
        
        # Domain-specific heads for specialized routing
        self.domain_heads = nn.ModuleDict({
            'psychology': DomainSpecificHead(self.hidden_dim, 'psychology', self.num_sources),
            'climate': DomainSpecificHead(self.hidden_dim, 'climate', self.num_sources),
            'singapore': DomainSpecificHead(self.hidden_dim, 'singapore', self.num_sources),
            'economics': DomainSpecificHead(self.hidden_dim, 'economics', self.num_sources),
            'machine_learning': DomainSpecificHead(self.hidden_dim, 'machine_learning', self.num_sources),
            'general': DomainSpecificHead(self.hidden_dim, 'general', self.num_sources)
        })
        
        # Shared classification heads (parameter sharing)
        self.relevance_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_domains)
        )
        
        self.singapore_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, 2)  # Singapore vs Global
        )
        
        # Initialize weights
        self._init_weights()
        
        # Calculate and log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🧠 OptimizedQualityModel initialized")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Target: <5M parameters (Current: {total_params/1e6:.1f}M)")
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optimized architecture
        
        Args:
            batch: Dictionary containing:
                - query_ids: [batch_size, seq_len]
                - source_ids: [batch_size]
                
        Returns:
            Dictionary containing predictions
        """
        query_ids = batch['query_ids']  # [batch_size, seq_len]
        source_ids = batch['source_ids']  # [batch_size]
        
        batch_size = query_ids.shape[0]
        
        # Embed queries and sources
        query_embeds = self.query_embedding(query_ids)  # [batch_size, seq_len, embedding_dim]
        source_embeds = self.source_embedding(source_ids)  # [batch_size, embedding_dim]
        
        # Encode queries and sources
        query_encoded = self.query_encoder(query_embeds)  # [batch_size, seq_len, hidden_dim]
        source_encoded = self.source_encoder(source_embeds)  # [batch_size, hidden_dim]
        
        # Add sequence dimension to source for cross-attention
        source_encoded = source_encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention between query and source
        attended_query = self.cross_attention(
            query=query_encoded,
            key=source_encoded,
            value=source_encoded
        )  # [batch_size, seq_len, hidden_dim]
        
        # Pool query representation (mean pooling)
        query_pooled = attended_query.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Combine query and source representations
        combined = query_pooled + source_encoded.squeeze(1)  # [batch_size, hidden_dim]
        
        # Main predictions
        relevance_score = self.relevance_head(combined).squeeze(-1)  # [batch_size]
        domain_logits = self.domain_classifier(combined)  # [batch_size, num_domains]
        singapore_logits = self.singapore_classifier(combined)  # [batch_size, 2]
        
        # Domain-specific predictions (optional, for specialized routing)
        domain_specific_scores = {}
        for domain_name, head in self.domain_heads.items():
            domain_specific_scores[domain_name] = head(combined)
        
        return {
            'relevance_score': relevance_score,
            'domain_logits': domain_logits,
            'singapore_logits': singapore_logits,
            'domain_specific_scores': domain_specific_scores,
            'query_representation': query_pooled,
            'source_representation': source_encoded.squeeze(1)
        }
    
    def predict_relevance(self, query: str, source: str) -> float:
        """Predict relevance score for a query-source pair"""
        # This would need proper tokenization in practice
        query_tokens = self._simple_tokenize(query)
        source_id = self._get_source_id(source)
        
        # Create batch
        batch = {
            'query_ids': torch.tensor([query_tokens], dtype=torch.long),
            'source_ids': torch.tensor([source_id], dtype=torch.long)
        }
        
        with torch.no_grad():
            outputs = self.forward(batch)
            relevance = torch.sigmoid(outputs['relevance_score']).item()
        
        return relevance
    
    def predict_domain_and_singapore(self, query: str) -> Tuple[str, bool]:
        """Predict domain and Singapore-first applicability"""
        query_tokens = self._simple_tokenize(query)
        
        # Use dummy source for domain prediction
        batch = {
            'query_ids': torch.tensor([query_tokens], dtype=torch.long),
            'source_ids': torch.tensor([0], dtype=torch.long)
        }
        
        with torch.no_grad():
            outputs = self.forward(batch)
            
            # Domain prediction
            domain_probs = F.softmax(outputs['domain_logits'], dim=-1)
            domain_id = domain_probs.argmax().item()
            domain_map = {
                0: 'psychology', 1: 'machine_learning', 2: 'climate',
                3: 'economics', 4: 'singapore', 5: 'health',
                6: 'education', 7: 'general'
            }
            domain = domain_map.get(domain_id, 'general')
            
            # Singapore prediction
            singapore_probs = F.softmax(outputs['singapore_logits'], dim=-1)
            singapore_first = singapore_probs[0, 1].item() > 0.5
        
        return domain, singapore_first
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization (replace with proper tokenizer)"""
        words = text.lower().split()
        token_ids = []
        for word in words[:20]:  # Limit to 20 tokens
            token_id = hash(word) % (self.vocab_size - 1) + 1
            token_ids.append(token_id)
        
        # Pad to fixed length
        while len(token_ids) < 20:
            token_ids.append(0)  # Padding token
        
        return token_ids[:20]
    
    def _get_source_id(self, source: str) -> int:
        """Map source name to ID"""
        source_map = {
            'kaggle': 0, 'zenodo': 1, 'world_bank': 2, 'data_gov_sg': 3,
            'singstat': 4, 'lta_datamall': 5, 'aws_opendata': 6,
            'data_un': 7, 'arxiv': 8, 'github': 9
        }
        return source_map.get(source.lower(), 0)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count breakdown"""
        param_counts = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    param_counts[name] = params
        
        total = sum(param_counts.values())
        param_counts['total'] = total
        
        return param_counts
    
    def compress_model(self) -> 'OptimizedQualityModel':
        """Apply model compression techniques"""
        # Quantization (simplified)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Apply weight pruning (zero out small weights)
                with torch.no_grad():
                    weight = module.weight
                    threshold = torch.quantile(torch.abs(weight), 0.1)  # Remove bottom 10%
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask.float()
        
        logger.info("🗜️ Model compression applied (weight pruning)")
        return self

def create_optimized_model(config: Dict = None) -> OptimizedQualityModel:
    """Create optimized model with default configuration"""
    default_config = {
        'vocab_size': 10000,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_sources': 10,
        'num_domains': 8,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return OptimizedQualityModel(default_config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the optimized model
    print("🧪 Testing Optimized Quality Model")
    
    # Create model
    model = create_optimized_model()
    
    # Test forward pass
    batch = {
        'query_ids': torch.randint(1, 1000, (2, 20)),  # 2 queries, 20 tokens each
        'source_ids': torch.randint(0, 10, (2,))       # 2 sources
    }
    
    outputs = model(batch)
    
    print(f"✅ Forward pass successful")
    print(f"  Relevance scores shape: {outputs['relevance_score'].shape}")
    print(f"  Domain logits shape: {outputs['domain_logits'].shape}")
    print(f"  Singapore logits shape: {outputs['singapore_logits'].shape}")
    
    # Test parameter count
    param_counts = model.get_parameter_count()
    print(f"\n📊 Parameter breakdown:")
    for name, count in param_counts.items():
        if name != 'total':
            print(f"  {name}: {count:,}")
    print(f"  Total: {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    
    # Test compression
    compressed_model = model.compress_model()
    
    # Test prediction methods
    relevance = model.predict_relevance("psychology research data", "kaggle")
    domain, singapore_first = model.predict_domain_and_singapore("singapore housing data")
    
    print(f"\n🔍 Test predictions:")
    print(f"  Relevance (psychology → kaggle): {relevance:.3f}")
    print(f"  Domain (singapore housing): {domain}")
    print(f"  Singapore-first: {singapore_first}")
    
    print("\n✅ Optimized Quality Model test complete!")