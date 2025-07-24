"""
Quality-Aware Cache Manager
Implements caching system that preserves recommendation quality with validation against training mappings
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass

from .intelligent_cache import IntelligentCache

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for cached recommendations"""
    ndcg_at_3: float
    relevance_accuracy: float
    domain_routing_accuracy: float
    singapore_first_accuracy: float
    user_satisfaction_score: float
    recommendation_diversity: float
    
    def meets_quality_threshold(self, threshold: float = 0.7) -> bool:
        """Check if quality meets minimum threshold"""
        return self.ndcg_at_3 >= threshold and self.relevance_accuracy >= threshold


@dataclass
class CachedRecommendation:
    """Cached recommendation with quality metadata"""
    source: str
    relevance_score: float
    domain: str
    explanation: str
    geographic_scope: str
    query_intent: str
    quality_score: float
    cached_at: float
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if recommendation meets quality threshold"""
        return self.quality_score >= threshold and self.relevance_score >= threshold


class QualityAwareCacheManager:
    """
    Cache manager that preserves recommendation quality through:
    - Quality-based TTL (higher quality = longer cache time)
    - Cache validation against training mappings
    - Quality metrics tracking for cache hits and misses
    - Automatic invalidation of low-quality results
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/quality_aware",
                 max_memory_size: int = 1000,
                 default_ttl: int = 3600,
                 quality_threshold: float = 0.7,
                 training_mappings_path: str = "training_mappings.md"):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.default_ttl = default_ttl
        self.quality_threshold = quality_threshold
        self.training_mappings_path = training_mappings_path
        
        # Initialize base intelligent cache
        self.base_cache = IntelligentCache(
            cache_dir=str(cache_dir),
            max_memory_size=max_memory_size,
            default_ttl=default_ttl
        )
        
        # Quality-specific storage
        self.quality_cache: Dict[str, List[CachedRecommendation]] = {}
        self.quality_metrics: Dict[str, QualityMetrics] = {}
        self.training_mappings: Dict[str, Dict[str, float]] = {}
        
        # Thread safety
        self.lock = RLock()
        
        # Quality tracking database
        self.quality_db_path = self.cache_dir / "quality_metrics.db"
        self._initialize_quality_database()
        
        # Load training mappings for validation
        self._load_training_mappings()
        
        logger.info(f"🎯 QualityAwareCacheManager initialized: threshold={quality_threshold}")
    
    def _initialize_quality_database(self):
        """Initialize database for quality metrics tracking"""
        try:
            with sqlite3.connect(self.quality_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quality_cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        query_text TEXT,
                        recommendations TEXT,
                        quality_score REAL,
                        ndcg_at_3 REAL,
                        relevance_accuracy REAL,
                        domain_routing_accuracy REAL,
                        singapore_first_accuracy REAL,
                        cached_at REAL,
                        last_validated REAL,
                        validation_count INTEGER,
                        hit_count INTEGER,
                        quality_ttl INTEGER
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT,
                        query_text TEXT,
                        quality_score REAL,
                        validation_result TEXT,
                        timestamp REAL,
                        cache_hit BOOLEAN
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_quality_score ON quality_cache_entries(quality_score)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_cached_at ON quality_cache_entries(cached_at)
                ''')
                
            logger.info("📊 Quality cache database initialized")
            
        except Exception as e:
            logger.error(f"Quality database initialization failed: {e}")
    
    def _load_training_mappings(self):
        """Load training mappings for quality validation"""
        try:
            if not Path(self.training_mappings_path).exists():
                logger.warning(f"Training mappings file not found: {self.training_mappings_path}")
                return
            
            with open(self.training_mappings_path, 'r') as f:
                content = f.read()
            
            # Parse training mappings
            lines = content.split('\n')
            for line in lines:
                if '→' in line and '(' in line and ')' in line:
                    try:
                        # Parse format: query → source (relevance_score) - reason
                        parts = line.split('→')
                        if len(parts) >= 2:
                            query = parts[0].strip().lstrip('- ')
                            rest = parts[1].strip()
                            
                            # Extract source and relevance score
                            source_part = rest.split('(')[0].strip()
                            score_part = rest.split('(')[1].split(')')[0].strip()
                            
                            relevance_score = float(score_part)
                            
                            if query not in self.training_mappings:
                                self.training_mappings[query] = {}
                            
                            self.training_mappings[query][source_part] = relevance_score
                            
                    except Exception as e:
                        logger.debug(f"Failed to parse mapping line: {line} - {e}")
            
            logger.info(f"📚 Loaded {len(self.training_mappings)} training mappings for quality validation")
            
        except Exception as e:
            logger.error(f"Failed to load training mappings: {e}")
    
    def cache_recommendations(self, 
                            query: str, 
                            recommendations: List[CachedRecommendation],
                            quality_metrics: Optional[QualityMetrics] = None) -> str:
        """
        Cache recommendations with quality validation and metrics
        
        Args:
            query: Search query
            recommendations: List of recommendations with quality scores
            quality_metrics: Overall quality metrics for the recommendation set
            
        Returns:
            Cache key for stored recommendations
        """
        with self.lock:
            cache_key = self._generate_cache_key(query)
            
            # Calculate quality metrics if not provided
            if quality_metrics is None:
                quality_metrics = self._calculate_quality_metrics(query, recommendations)
            
            # Only cache if quality meets threshold
            if not quality_metrics.meets_quality_threshold(self.quality_threshold):
                logger.debug(f"🚫 Not caching low-quality results for: {query[:50]}...")
                return ""
            
            # Calculate quality-based TTL
            quality_ttl = self._calculate_quality_based_ttl(quality_metrics)
            
            # Store in quality cache
            self.quality_cache[cache_key] = recommendations
            self.quality_metrics[cache_key] = quality_metrics
            
            # Store in base cache with quality-based TTL
            cache_data = {
                'recommendations': [self._recommendation_to_dict(r) for r in recommendations],
                'quality_metrics': self._quality_metrics_to_dict(quality_metrics),
                'quality_score': quality_metrics.ndcg_at_3
            }
            
            self.base_cache.set(
                query, 
                cache_data, 
                cache_type='quality_recommendations',
                ttl=quality_ttl,
                tags=['high_quality', f'domain_{recommendations[0].domain if recommendations else "unknown"}']
            )
            
            # Store quality metrics in database
            self._store_quality_metrics(cache_key, query, recommendations, quality_metrics, quality_ttl)
            
            logger.debug(f"💎 Cached high-quality results: {cache_key[:12]}... TTL={quality_ttl}s")
            return cache_key
    
    def get_cached_recommendations(self, 
                                 query: str, 
                                 min_quality_threshold: Optional[float] = None) -> Optional[Tuple[List[CachedRecommendation], QualityMetrics]]:
        """
        Get cached recommendations meeting quality threshold
        
        Args:
            query: Search query
            min_quality_threshold: Minimum quality threshold (uses default if None)
            
        Returns:
            Tuple of (recommendations, quality_metrics) or None if not found/low quality
        """
        with self.lock:
            cache_key = self._generate_cache_key(query)
            threshold = min_quality_threshold or self.quality_threshold
            
            # Check quality cache first
            if cache_key in self.quality_cache and cache_key in self.quality_metrics:
                quality_metrics = self.quality_metrics[cache_key]
                
                if quality_metrics.meets_quality_threshold(threshold):
                    recommendations = self.quality_cache[cache_key]
                    
                    # Validate against training mappings
                    if self._validate_against_training_mappings(query, recommendations):
                        self._record_quality_cache_hit(cache_key, query, quality_metrics, True)
                        return recommendations, quality_metrics
                    else:
                        # Remove invalid cache entry
                        self._invalidate_cache_entry(cache_key)
            
            # Check base cache
            cached_data = self.base_cache.get(query, 'quality_recommendations')
            if cached_data:
                try:
                    recommendations = [self._dict_to_recommendation(r) for r in cached_data['recommendations']]
                    quality_metrics = self._dict_to_quality_metrics(cached_data['quality_metrics'])
                    
                    if quality_metrics.meets_quality_threshold(threshold):
                        # Validate against training mappings
                        if self._validate_against_training_mappings(query, recommendations):
                            # Restore to quality cache
                            self.quality_cache[cache_key] = recommendations
                            self.quality_metrics[cache_key] = quality_metrics
                            
                            self._record_quality_cache_hit(cache_key, query, quality_metrics, True)
                            return recommendations, quality_metrics
                        else:
                            # Invalidate low-quality cache
                            self.base_cache.invalidate(pattern=query)
                
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached recommendations: {e}")
            
            # No valid cache found
            self._record_quality_cache_hit(cache_key, query, None, False)
            return None
    
    def _calculate_quality_metrics(self, query: str, recommendations: List[CachedRecommendation]) -> QualityMetrics:
        """Calculate quality metrics for recommendations"""
        if not recommendations:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate NDCG@3
        ndcg_at_3 = self._calculate_ndcg_at_3(query, recommendations)
        
        # Calculate relevance accuracy against training mappings
        relevance_accuracy = self._calculate_relevance_accuracy(query, recommendations)
        
        # Calculate domain routing accuracy
        domain_routing_accuracy = self._calculate_domain_routing_accuracy(query, recommendations)
        
        # Calculate Singapore-first accuracy
        singapore_first_accuracy = self._calculate_singapore_first_accuracy(query, recommendations)
        
        # Calculate recommendation diversity
        recommendation_diversity = self._calculate_recommendation_diversity(recommendations)
        
        # User satisfaction score (placeholder - would come from actual user feedback)
        user_satisfaction_score = min(ndcg_at_3 + 0.1, 1.0)
        
        return QualityMetrics(
            ndcg_at_3=ndcg_at_3,
            relevance_accuracy=relevance_accuracy,
            domain_routing_accuracy=domain_routing_accuracy,
            singapore_first_accuracy=singapore_first_accuracy,
            user_satisfaction_score=user_satisfaction_score,
            recommendation_diversity=recommendation_diversity
        )
    
    def _calculate_ndcg_at_3(self, query: str, recommendations: List[CachedRecommendation]) -> float:
        """Calculate NDCG@3 score for recommendations"""
        if not recommendations:
            return 0.0
        
        # Get relevance scores from training mappings
        query_mappings = self.training_mappings.get(query.lower(), {})
        
        if not query_mappings:
            # Fallback to recommendation quality scores
            relevance_scores = [r.relevance_score for r in recommendations[:3]]
        else:
            relevance_scores = []
            for rec in recommendations[:3]:
                # Find best matching source in training mappings
                best_score = 0.0
                for source, score in query_mappings.items():
                    if source.lower() in rec.source.lower() or rec.source.lower() in source.lower():
                        best_score = max(best_score, score)
                
                relevance_scores.append(best_score if best_score > 0 else rec.relevance_score)
        
        # Calculate DCG@3
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@3 (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Calculate NDCG@3
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_relevance_accuracy(self, query: str, recommendations: List[CachedRecommendation]) -> float:
        """Calculate relevance accuracy against training mappings"""
        query_mappings = self.training_mappings.get(query.lower(), {})
        
        if not query_mappings or not recommendations:
            return 0.0
        
        correct_predictions = 0
        total_predictions = min(len(recommendations), 3)  # Check top 3
        
        for rec in recommendations[:total_predictions]:
            # Find if this source is in training mappings
            expected_score = 0.0
            for source, score in query_mappings.items():
                if source.lower() in rec.source.lower() or rec.source.lower() in source.lower():
                    expected_score = score
                    break
            
            # Consider correct if within 0.2 of expected score
            if abs(rec.relevance_score - expected_score) <= 0.2:
                correct_predictions += 1
        
        return correct_predictions / total_predictions
    
    def _calculate_domain_routing_accuracy(self, query: str, recommendations: List[CachedRecommendation]) -> float:
        """Calculate domain routing accuracy"""
        if not recommendations:
            return 0.0
        
        # Check if domain-specific routing is correct based on query
        query_lower = query.lower()
        expected_domains = []
        
        if 'psychology' in query_lower or 'mental health' in query_lower:
            expected_domains = ['kaggle', 'zenodo']
        elif 'climate' in query_lower or 'environment' in query_lower:
            expected_domains = ['world_bank', 'kaggle']
        elif 'singapore' in query_lower:
            expected_domains = ['data_gov_sg', 'singstat', 'lta_datamall']
        elif 'economic' in query_lower or 'gdp' in query_lower:
            expected_domains = ['world_bank']
        elif 'machine learning' in query_lower or 'ml' in query_lower:
            expected_domains = ['kaggle', 'zenodo']
        
        if not expected_domains:
            return 1.0  # No specific domain expectation
        
        # Check if top recommendation matches expected domain
        top_rec = recommendations[0]
        for domain in expected_domains:
            if domain.lower() in top_rec.source.lower():
                return 1.0
        
        return 0.0
    
    def _calculate_singapore_first_accuracy(self, query: str, recommendations: List[CachedRecommendation]) -> float:
        """Calculate Singapore-first strategy accuracy"""
        if not recommendations:
            return 0.0
        
        query_lower = query.lower()
        
        # Check if Singapore-first should apply
        if 'singapore' in query_lower:
            singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
            
            # Check if top recommendation is Singapore source
            top_rec = recommendations[0]
            for source in singapore_sources:
                if source.lower() in top_rec.source.lower():
                    return 1.0
            
            return 0.0
        
        return 1.0  # Singapore-first not applicable
    
    def _calculate_recommendation_diversity(self, recommendations: List[CachedRecommendation]) -> float:
        """Calculate diversity of recommendations"""
        if len(recommendations) <= 1:
            return 0.0
        
        unique_sources = set(rec.source for rec in recommendations)
        unique_domains = set(rec.domain for rec in recommendations)
        
        source_diversity = len(unique_sources) / len(recommendations)
        domain_diversity = len(unique_domains) / len(recommendations)
        
        return (source_diversity + domain_diversity) / 2
    
    def _validate_against_training_mappings(self, query: str, recommendations: List[CachedRecommendation]) -> bool:
        """Validate recommendations against training mappings"""
        query_mappings = self.training_mappings.get(query.lower(), {})
        
        if not query_mappings:
            return True  # No validation data available
        
        # Check if top recommendation aligns with training mappings
        if recommendations:
            top_rec = recommendations[0]
            
            for source, expected_score in query_mappings.items():
                if source.lower() in top_rec.source.lower() or top_rec.source.lower() in source.lower():
                    # Validate that high-scoring sources are ranked highly
                    if expected_score >= 0.8 and top_rec.relevance_score >= 0.7:
                        return True
                    elif expected_score <= 0.3 and top_rec.relevance_score <= 0.5:
                        return False  # Low-quality source ranked too highly
        
        return True
    
    def _calculate_quality_based_ttl(self, quality_metrics: QualityMetrics) -> int:
        """Calculate TTL based on quality metrics"""
        base_ttl = self.default_ttl
        
        # Higher quality = longer cache time
        quality_multiplier = 1.0 + (quality_metrics.ndcg_at_3 - 0.5) * 2  # 0.5-1.0 -> 1.0-2.0
        quality_multiplier = max(0.5, min(3.0, quality_multiplier))  # Clamp between 0.5x and 3x
        
        # Bonus for high relevance accuracy
        if quality_metrics.relevance_accuracy >= 0.9:
            quality_multiplier *= 1.5
        
        return int(base_ttl * quality_multiplier)
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(f"quality_recommendations:{query}".encode()).hexdigest()
    
    def _store_quality_metrics(self, cache_key: str, query: str, recommendations: List[CachedRecommendation], 
                              quality_metrics: QualityMetrics, ttl: int):
        """Store quality metrics in database"""
        try:
            with sqlite3.connect(self.quality_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO quality_cache_entries 
                    (cache_key, query_text, recommendations, quality_score, ndcg_at_3, 
                     relevance_accuracy, domain_routing_accuracy, singapore_first_accuracy,
                     cached_at, last_validated, validation_count, hit_count, quality_ttl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key, query, 
                    json.dumps([self._recommendation_to_dict(r) for r in recommendations]),
                    quality_metrics.ndcg_at_3, quality_metrics.ndcg_at_3,
                    quality_metrics.relevance_accuracy, quality_metrics.domain_routing_accuracy,
                    quality_metrics.singapore_first_accuracy, time.time(), time.time(),
                    0, 0, ttl
                ))
                
        except Exception as e:
            logger.warning(f"Failed to store quality metrics: {e}")
    
    def _record_quality_cache_hit(self, cache_key: str, query: str, quality_metrics: Optional[QualityMetrics], hit: bool):
        """Record cache hit/miss with quality information"""
        try:
            with sqlite3.connect(self.quality_db_path) as conn:
                conn.execute('''
                    INSERT INTO quality_metrics_history 
                    (cache_key, query_text, quality_score, validation_result, timestamp, cache_hit)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key, query,
                    quality_metrics.ndcg_at_3 if quality_metrics else 0.0,
                    "valid" if hit else "miss",
                    time.time(), hit
                ))
                
                # Update hit count
                if hit:
                    conn.execute('''
                        UPDATE quality_cache_entries 
                        SET hit_count = hit_count + 1, last_validated = ?
                        WHERE cache_key = ?
                    ''', (time.time(), cache_key))
                    
        except Exception as e:
            logger.warning(f"Failed to record cache hit: {e}")
    
    def _invalidate_cache_entry(self, cache_key: str):
        """Invalidate a specific cache entry"""
        with self.lock:
            # Remove from quality cache
            if cache_key in self.quality_cache:
                del self.quality_cache[cache_key]
            if cache_key in self.quality_metrics:
                del self.quality_metrics[cache_key]
            
            # Remove from base cache (find by cache key)
            # Note: This is a simplified approach - in practice, we'd need better key mapping
            logger.debug(f"🗑️ Invalidated cache entry: {cache_key[:12]}...")
    
    def invalidate_low_quality_cache(self, quality_threshold: Optional[float] = None):
        """Remove cached results below quality threshold"""
        threshold = quality_threshold or self.quality_threshold
        
        with self.lock:
            keys_to_remove = []
            
            for cache_key, quality_metrics in self.quality_metrics.items():
                if not quality_metrics.meets_quality_threshold(threshold):
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                self._invalidate_cache_entry(key)
            
            logger.info(f"🧹 Invalidated {len(keys_to_remove)} low-quality cache entries")
    
    def get_quality_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality cache statistics"""
        with self.lock:
            try:
                with sqlite3.connect(self.quality_db_path) as conn:
                    # Calculate hit rate
                    cursor = conn.execute('SELECT COUNT(*) FROM quality_metrics_history WHERE cache_hit = 1')
                    hits = cursor.fetchone()[0]
                    
                    cursor = conn.execute('SELECT COUNT(*) FROM quality_metrics_history')
                    total = cursor.fetchone()[0]
                    
                    hit_rate = hits / total if total > 0 else 0.0
                    
                    # Average quality scores
                    cursor = conn.execute('SELECT AVG(quality_score) FROM quality_cache_entries')
                    avg_quality = cursor.fetchone()[0] or 0.0
                    
                    cursor = conn.execute('SELECT AVG(ndcg_at_3) FROM quality_cache_entries')
                    avg_ndcg = cursor.fetchone()[0] or 0.0
                    
                    cursor = conn.execute('SELECT AVG(relevance_accuracy) FROM quality_cache_entries')
                    avg_relevance = cursor.fetchone()[0] or 0.0
                    
                    # Quality distribution
                    cursor = conn.execute('''
                        SELECT 
                            COUNT(CASE WHEN quality_score >= 0.9 THEN 1 END) as excellent,
                            COUNT(CASE WHEN quality_score >= 0.7 AND quality_score < 0.9 THEN 1 END) as good,
                            COUNT(CASE WHEN quality_score >= 0.5 AND quality_score < 0.7 THEN 1 END) as fair,
                            COUNT(CASE WHEN quality_score < 0.5 THEN 1 END) as poor,
                            COUNT(*) as total
                        FROM quality_cache_entries
                    ''')
                    quality_dist = cursor.fetchone()
                    
                return {
                    'hit_rate': hit_rate,
                    'total_entries': len(self.quality_cache),
                    'memory_entries': len(self.quality_cache),
                    'avg_quality_score': avg_quality,
                    'avg_ndcg_at_3': avg_ndcg,
                    'avg_relevance_accuracy': avg_relevance,
                    'quality_threshold': self.quality_threshold,
                    'quality_distribution': {
                        'excellent': quality_dist[0] if quality_dist else 0,
                        'good': quality_dist[1] if quality_dist else 0,
                        'fair': quality_dist[2] if quality_dist else 0,
                        'poor': quality_dist[3] if quality_dist else 0,
                        'total': quality_dist[4] if quality_dist else 0
                    },
                    'training_mappings_loaded': len(self.training_mappings)
                }
                
            except Exception as e:
                logger.error(f"Failed to get quality cache statistics: {e}")
                return {
                    'hit_rate': 0.0,
                    'total_entries': len(self.quality_cache),
                    'error': str(e)
                }
    
    def _recommendation_to_dict(self, rec: CachedRecommendation) -> Dict:
        """Convert recommendation to dictionary for serialization"""
        return {
            'source': rec.source,
            'relevance_score': rec.relevance_score,
            'domain': rec.domain,
            'explanation': rec.explanation,
            'geographic_scope': rec.geographic_scope,
            'query_intent': rec.query_intent,
            'quality_score': rec.quality_score,
            'cached_at': rec.cached_at
        }
    
    def _dict_to_recommendation(self, data: Dict) -> CachedRecommendation:
        """Convert dictionary to recommendation object"""
        return CachedRecommendation(
            source=data['source'],
            relevance_score=data['relevance_score'],
            domain=data['domain'],
            explanation=data['explanation'],
            geographic_scope=data['geographic_scope'],
            query_intent=data['query_intent'],
            quality_score=data['quality_score'],
            cached_at=data['cached_at']
        )
    
    def _quality_metrics_to_dict(self, metrics: QualityMetrics) -> Dict:
        """Convert quality metrics to dictionary"""
        return {
            'ndcg_at_3': metrics.ndcg_at_3,
            'relevance_accuracy': metrics.relevance_accuracy,
            'domain_routing_accuracy': metrics.domain_routing_accuracy,
            'singapore_first_accuracy': metrics.singapore_first_accuracy,
            'user_satisfaction_score': metrics.user_satisfaction_score,
            'recommendation_diversity': metrics.recommendation_diversity
        }
    
    def _dict_to_quality_metrics(self, data: Dict) -> QualityMetrics:
        """Convert dictionary to quality metrics object"""
        return QualityMetrics(
            ndcg_at_3=data['ndcg_at_3'],
            relevance_accuracy=data['relevance_accuracy'],
            domain_routing_accuracy=data['domain_routing_accuracy'],
            singapore_first_accuracy=data['singapore_first_accuracy'],
            user_satisfaction_score=data['user_satisfaction_score'],
            recommendation_diversity=data['recommendation_diversity']
        )