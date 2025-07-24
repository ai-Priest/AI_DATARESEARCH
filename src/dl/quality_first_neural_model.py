"""
Quality-First Neural Model - Optimized for Recommendation Relevance
Uses enhanced training data with domain-specific routing and Singapore-first strategy
"""

import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class QualityAwareRankingModel(nn.Module):
    """Quality-first neural model optimized for recommendation relevance"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions - optimized for quality over size
        self.embedding_dim = config.get('embedding_dim', 256)  # Reduced from 512
        self.hidden_dim = config.get('hidden_dim', 128)       # Reduced from 256
        self.num_domains = config.get('num_domains', 8)       # psychology, climate, etc.
        self.num_sources = config.get('num_sources', 10)      # kaggle, zenodo, etc.
        
        # Vocabulary and embeddings
        self.vocab_size = config.get('vocab_size', 10000)
        self.query_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.source_embedding = nn.Embedding(self.num_sources, self.embedding_dim)
        
        # Domain-specific encoders
        self.domain_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.num_domains)
        )
        
        # Query encoder with attention
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=0.3,
                batch_first=True
            ),
            num_layers=2  # Lightweight architecture
        )
        
        # Cross-attention for query-source matching
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,  # Reduced from 8
            dropout=0.3,
            batch_first=True
        )
        
        # Quality-aware ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.hidden_dim),  # query + source + attention
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 1)  # Single relevance score
        )
        
        # Singapore-first routing head
        self.singapore_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 2)  # singapore vs global
        )
        
        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.num_domains)
        )
        
        # Source mapping
        self.source_to_id = {
            'kaggle': 0, 'zenodo': 1, 'world_bank': 2, 'data_gov_sg': 3,
            'singstat': 4, 'lta_datamall': 5, 'aws_opendata': 6, 'data_un': 7,
            'arxiv': 8, 'github': 9
        }
        
        # Domain mapping
        self.domain_to_id = {
            'psychology': 0, 'machine_learning': 1, 'climate': 2, 'economics': 3,
            'singapore': 4, 'health': 5, 'education': 6, 'general': 7
        }
        
        logger.info(f"🧠 QualityAwareRankingModel initialized")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Parameters: ~{self.count_parameters()/1e6:.1f}M")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with quality-first approach"""
        
        # Extract inputs
        query_ids = batch['query_ids']  # [batch_size, seq_len]
        source_ids = batch['source_ids']  # [batch_size]
        
        batch_size = query_ids.size(0)
        
        # Query encoding
        query_emb = self.query_embedding(query_ids)  # [batch_size, seq_len, embedding_dim]
        query_encoded = self.query_encoder(query_emb)  # [batch_size, seq_len, embedding_dim]
        query_pooled = query_encoded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Source encoding
        source_emb = self.source_embedding(source_ids)  # [batch_size, embedding_dim]
        
        # Cross-attention between query and source
        attended_query, attention_weights = self.cross_attention(
            query_encoded, source_emb.unsqueeze(1), source_emb.unsqueeze(1)
        )
        attended_query = attended_query.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Combine representations
        combined = torch.cat([query_pooled, source_emb, attended_query], dim=-1)
        
        # Quality-aware ranking score
        relevance_score = self.ranking_head(combined).squeeze(-1)  # [batch_size]
        
        # Singapore-first classification
        singapore_logits = self.singapore_classifier(query_pooled)  # [batch_size, 2]
        
        # Domain classification
        domain_logits = self.domain_classifier(query_pooled)  # [batch_size, num_domains]
        
        return {
            'relevance_score': relevance_score,
            'singapore_logits': singapore_logits,
            'domain_logits': domain_logits,
            'attention_weights': attention_weights,
            'query_embedding': query_pooled,
            'source_embedding': source_emb
        }
    
    def predict_relevance(self, query_text: str, source: str) -> float:
        """Predict relevance score for a query-source pair"""
        self.eval()
        
        with torch.no_grad():
            # Tokenize query (simplified)
            query_tokens = self._tokenize_query(query_text)
            query_ids = torch.tensor([query_tokens], dtype=torch.long)
            
            # Get source ID
            source_id = self.source_to_id.get(source, 0)
            source_ids = torch.tensor([source_id], dtype=torch.long)
            
            # Forward pass
            batch = {'query_ids': query_ids, 'source_ids': source_ids}
            outputs = self.forward(batch)
            
            # Apply sigmoid to get probability
            relevance_prob = torch.sigmoid(outputs['relevance_score']).item()
            
            return relevance_prob
    
    def predict_domain_and_singapore(self, query_text: str) -> Tuple[str, bool]:
        """Predict domain and Singapore-first applicability"""
        self.eval()
        
        with torch.no_grad():
            # Tokenize query
            query_tokens = self._tokenize_query(query_text)
            query_ids = torch.tensor([query_tokens], dtype=torch.long)
            source_ids = torch.tensor([0], dtype=torch.long)  # Dummy source
            
            # Forward pass
            batch = {'query_ids': query_ids, 'source_ids': source_ids}
            outputs = self.forward(batch)
            
            # Get domain prediction
            domain_probs = F.softmax(outputs['domain_logits'], dim=-1)
            domain_id = domain_probs.argmax().item()
            domain = list(self.domain_to_id.keys())[domain_id]
            
            # Get Singapore-first prediction
            singapore_probs = F.softmax(outputs['singapore_logits'], dim=-1)
            singapore_first = singapore_probs[0, 1].item() > 0.5  # Index 1 is Singapore
            
            return domain, singapore_first
    
    def _tokenize_query(self, query: str) -> List[int]:
        """Simple tokenization (should be replaced with proper tokenizer)"""
        # This is a simplified tokenizer - in practice, use a proper tokenizer
        words = query.lower().split()
        # Map words to IDs (simplified)
        token_ids = []
        for word in words[:20]:  # Limit to 20 tokens
            # Simple hash-based mapping (not ideal for production)
            token_id = hash(word) % (self.vocab_size - 1) + 1
            token_ids.append(token_id)
        
        # Pad to fixed length
        while len(token_ids) < 20:
            token_ids.append(0)  # Padding token
        
        return token_ids[:20]


class QualityFirstLoss(nn.Module):
    """Quality-first loss function combining ranking and classification losses"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.ranking_weight = config.get('ranking_weight', 0.6)
        self.singapore_weight = config.get('singapore_weight', 0.2)
        self.domain_weight = config.get('domain_weight', 0.2)
        
        # Loss functions
        self.ranking_loss = nn.MSELoss()  # For relevance scores
        self.classification_loss = nn.CrossEntropyLoss()
        
        logger.info(f"🎯 QualityFirstLoss initialized")
        logger.info(f"  Ranking weight: {self.ranking_weight}")
        logger.info(f"  Singapore weight: {self.singapore_weight}")
        logger.info(f"  Domain weight: {self.domain_weight}")
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute quality-first loss"""
        
        # Ranking loss (primary)
        ranking_loss = self.ranking_loss(
            outputs['relevance_score'],
            targets['relevance_score']
        )
        
        # Singapore classification loss
        singapore_loss = self.classification_loss(
            outputs['singapore_logits'],
            targets['singapore_labels']
        )
        
        # Domain classification loss
        domain_loss = self.classification_loss(
            outputs['domain_logits'],
            targets['domain_labels']
        )
        
        # Combined loss
        total_loss = (
            self.ranking_weight * ranking_loss +
            self.singapore_weight * singapore_loss +
            self.domain_weight * domain_loss
        )
        
        return {
            'total_loss': total_loss,
            'ranking_loss': ranking_loss,
            'singapore_loss': singapore_loss,
            'domain_loss': domain_loss
        }


class EnhancedTrainingDataLoader:
    """Data loader for enhanced training data with quality-first approach"""
    
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.data = self._load_data()
        
        # Create mappings
        self.source_to_id = {
            'kaggle': 0, 'zenodo': 1, 'world_bank': 2, 'data_gov_sg': 3,
            'singstat': 4, 'lta_datamall': 5, 'aws_opendata': 6, 'data_un': 7,
            'arxiv': 8, 'github': 9
        }
        
        self.domain_to_id = {
            'psychology': 0, 'machine_learning': 1, 'climate': 2, 'economics': 3,
            'singapore': 4, 'health': 5, 'education': 6, 'general': 7
        }
        
        logger.info(f"📊 EnhancedTrainingDataLoader initialized")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Total examples: {len(self.data['examples'])}")
        logger.info(f"  Batch size: {batch_size}")
    
    def _load_data(self) -> Dict:
        """Load enhanced training data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def get_data_loader(self, split: str = 'train') -> torch.utils.data.DataLoader:
        """Get PyTorch data loader for specified split"""
        
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in data")
        
        examples = self.data[split]
        dataset = EnhancedDataset(examples, self.source_to_id, self.domain_to_id)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        queries = [item['query'] for item in batch]
        sources = [item['source'] for item in batch]
        relevance_scores = [item['relevance_score'] for item in batch]
        singapore_labels = [item['singapore_label'] for item in batch]
        domain_labels = [item['domain_label'] for item in batch]
        
        # Tokenize queries (simplified)
        query_ids = []
        for query in queries:
            tokens = self._tokenize_query(query)
            query_ids.append(tokens)
        
        return {
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'source_ids': torch.tensor(sources, dtype=torch.long),
            'relevance_score': torch.tensor(relevance_scores, dtype=torch.float),
            'singapore_labels': torch.tensor(singapore_labels, dtype=torch.long),
            'domain_labels': torch.tensor(domain_labels, dtype=torch.long)
        }
    
    def _tokenize_query(self, query: str) -> List[int]:
        """Simple tokenization"""
        words = query.lower().split()
        token_ids = []
        for word in words[:20]:
            token_id = hash(word) % 9999 + 1  # Simple hash-based mapping
            token_ids.append(token_id)
        
        while len(token_ids) < 20:
            token_ids.append(0)  # Padding
        
        return token_ids[:20]


class EnhancedDataset(torch.utils.data.Dataset):
    """PyTorch dataset for enhanced training examples"""
    
    def __init__(self, examples: List[Dict], source_to_id: Dict, domain_to_id: Dict):
        self.examples = examples
        self.source_to_id = source_to_id
        self.domain_to_id = domain_to_id
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        return {
            'query': example['query'],
            'source': self.source_to_id.get(example['positive_source'], 0),
            'relevance_score': example['relevance_score'],
            'singapore_label': 1 if example['singapore_first'] else 0,
            'domain_label': self.domain_to_id.get(example['domain'], 7)  # 7 is 'general'
        }


def create_quality_first_model(config: Dict = None) -> QualityAwareRankingModel:
    """Factory function to create quality-first neural model"""
    if config is None:
        config = {
            'embedding_dim': 256,
            'hidden_dim': 128,
            'num_domains': 8,
            'num_sources': 10,
            'vocab_size': 10000
        }
    
    return QualityAwareRankingModel(config)


def create_quality_first_loss(config: Dict = None) -> QualityFirstLoss:
    """Factory function to create quality-first loss"""
    if config is None:
        config = {
            'ranking_weight': 0.6,
            'singapore_weight': 0.2,
            'domain_weight': 0.2
        }
    
    return QualityFirstLoss(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    model = create_quality_first_model()
    loss_fn = create_quality_first_loss()
    
    print(f"✅ Quality-first model created with {model.count_parameters()/1e6:.1f}M parameters")
    
    # Test data loader
    try:
        data_loader = EnhancedTrainingDataLoader("data/processed/enhanced_training_mappings.json")
        train_loader = data_loader.get_data_loader('train')
        print(f"✅ Data loader created with {len(train_loader)} batches")
        
        # Test forward pass
        batch = next(iter(train_loader))
        outputs = model(batch)
        print(f"✅ Forward pass successful")
        print(f"  Relevance scores shape: {outputs['relevance_score'].shape}")
        print(f"  Singapore logits shape: {outputs['singapore_logits'].shape}")
        print(f"  Domain logits shape: {outputs['domain_logits'].shape}")
        
    except FileNotFoundError:
        print("⚠️ Enhanced training data not found - run enhanced_training_integrator.py first")