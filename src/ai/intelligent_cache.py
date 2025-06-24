"""
Intelligent Caching Layer
Phase 2.3: Advanced caching system with Redis-style storage, query similarity, and smart invalidation
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sqlite3
from threading import RLock
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class IntelligentCache:
    """
    Intelligent caching system with:
    - Query similarity detection
    - Adaptive TTL based on usage patterns
    - Smart invalidation strategies
    - Memory and disk persistence
    - Query recommendation based on cache patterns
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 max_memory_size: int = 1000,
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.85):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        
        # Memory cache for fast access
        self.memory_cache: Dict[str, Dict] = {}
        self.cache_metadata: Dict[str, Dict] = {}
        self.access_patterns: Dict[str, List[float]] = {}
        
        # Thread safety
        self.lock = RLock()
        
        # Query similarity components
        self.semantic_model = None
        self.query_embeddings: Dict[str, np.ndarray] = {}
        
        # Persistent storage
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._initialize_database()
        self._initialize_semantic_model()
        
        # Load existing cache
        self._load_cache_metadata()
        
        logger.info(f"🗄️ IntelligentCache initialized: {self.cache_dir}, max_size={max_memory_size}")
    
    def _initialize_database(self):
        """Initialize SQLite database for cache metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        cache_key TEXT PRIMARY KEY,
                        query_text TEXT,
                        query_hash TEXT,
                        created_at REAL,
                        last_accessed REAL,
                        access_count INTEGER,
                        ttl INTEGER,
                        data_size INTEGER,
                        content_type TEXT,
                        tags TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS query_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT,
                        query_embedding BLOB,
                        cache_key TEXT,
                        timestamp REAL,
                        response_time REAL,
                        cache_hit BOOLEAN
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_query_hash ON cache_metadata(query_hash)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_query_text ON query_patterns(query_text)
                ''')
                
            logger.info("📊 Cache database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _initialize_semantic_model(self):
        """Initialize semantic model for query similarity."""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("🧠 Semantic model loaded for query similarity")
        except Exception as e:
            logger.warning(f"Semantic model initialization failed: {e}")
            self.semantic_model = None
    
    def _load_cache_metadata(self):
        """Load cache metadata from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM cache_metadata')
                for row in cursor.fetchall():
                    cache_key = row[0]
                    self.cache_metadata[cache_key] = {
                        'query_text': row[1],
                        'query_hash': row[2],
                        'created_at': row[3],
                        'last_accessed': row[4],
                        'access_count': row[5],
                        'ttl': row[6],
                        'data_size': row[7],
                        'content_type': row[8],
                        'tags': json.loads(row[9]) if row[9] else []
                    }
            
            logger.info(f"📚 Loaded metadata for {len(self.cache_metadata)} cache entries")
            
        except Exception as e:
            logger.warning(f"Cache metadata loading failed: {e}")
    
    def get(self, 
            query: str, 
            cache_type: str = 'search_results',
            use_similarity: bool = True) -> Optional[Any]:
        """
        Get cached result with intelligent similarity matching.
        
        Args:
            query: Query string
            cache_type: Type of cached content
            use_similarity: Whether to use semantic similarity for cache hits
            
        Returns:
            Cached result or None if not found
        """
        start_time = time.time()
        
        with self.lock:
            # Generate cache key
            cache_key = self._generate_cache_key(query, cache_type)
            
            # Check exact match first
            result = self._get_exact_match(cache_key)
            if result is not None:
                self._record_cache_hit(query, cache_key, time.time() - start_time, True)
                return result
            
            # Check similarity-based matches if enabled
            if use_similarity and self.semantic_model:
                similar_result = self._get_similar_match(query, cache_type)
                if similar_result is not None:
                    self._record_cache_hit(query, cache_key, time.time() - start_time, True)
                    return similar_result
            
            # No cache hit
            self._record_cache_hit(query, cache_key, time.time() - start_time, False)
            return None
    
    def set(self, 
            query: str, 
            data: Any, 
            cache_type: str = 'search_results',
            ttl: Optional[int] = None,
            tags: Optional[List[str]] = None) -> str:
        """
        Store data in cache with intelligent TTL and metadata.
        
        Args:
            query: Query string
            data: Data to cache
            cache_type: Type of content being cached
            ttl: Time to live in seconds
            tags: Optional tags for categorization
            
        Returns:
            Cache key of stored data
        """
        with self.lock:
            cache_key = self._generate_cache_key(query, cache_type)
            
            # Calculate adaptive TTL
            if ttl is None:
                ttl = self._calculate_adaptive_ttl(query, cache_type)
            
            # Prepare cache entry
            cache_entry = {
                'data': data,
                'query': query,
                'cache_type': cache_type,
                'created_at': time.time(),
                'ttl': ttl,
                'tags': tags or []
            }
            
            # Store in memory if space available
            if len(self.memory_cache) < self.max_memory_size:
                self.memory_cache[cache_key] = cache_entry
            
            # Store on disk
            self._store_to_disk(cache_key, cache_entry)
            
            # Update metadata
            self._update_cache_metadata(cache_key, query, cache_entry)
            
            # Store query embedding for similarity matching
            if self.semantic_model:
                self._store_query_embedding(query, cache_key)
            
            # Cleanup old entries if needed
            self._cleanup_expired_entries()
            
            logger.debug(f"💾 Cached: {cache_key[:12]}... TTL={ttl}s")
            return cache_key
    
    def _get_exact_match(self, cache_key: str) -> Optional[Any]:
        """Get exact cache match."""
        # Check memory first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if self._is_valid_entry(entry):
                self._update_access_pattern(cache_key)
                return entry['data']
            else:
                del self.memory_cache[cache_key]
        
        # Check disk
        disk_entry = self._load_from_disk(cache_key)
        if disk_entry and self._is_valid_entry(disk_entry):
            # Load back to memory if space available
            if len(self.memory_cache) < self.max_memory_size:
                self.memory_cache[cache_key] = disk_entry
            
            self._update_access_pattern(cache_key)
            return disk_entry['data']
        
        return None
    
    def _get_similar_match(self, query: str, cache_type: str) -> Optional[Any]:
        """Get similar query match using semantic similarity."""
        if not self.semantic_model:
            return None
        
        try:
            # Encode current query
            query_embedding = self.semantic_model.encode([query])
            
            # Find similar cached queries
            best_similarity = 0.0
            best_cache_key = None
            
            for stored_query, embedding in self.query_embeddings.items():
                if embedding is not None:
                    similarity = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
                    
                    if similarity >= self.similarity_threshold and similarity > best_similarity:
                        # Check if corresponding cache entry exists and is valid
                        potential_key = self._generate_cache_key(stored_query, cache_type)
                        if self._cache_key_exists_and_valid(potential_key):
                            best_similarity = similarity
                            best_cache_key = potential_key
            
            # Return best similar match
            if best_cache_key:
                logger.debug(f"🔍 Similar cache hit: {best_similarity:.3f} similarity")
                return self._get_exact_match(best_cache_key)
            
        except Exception as e:
            logger.warning(f"Similarity matching failed: {e}")
        
        return None
    
    def _generate_cache_key(self, query: str, cache_type: str) -> str:
        """Generate unique cache key."""
        content = f"{cache_type}:{query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_adaptive_ttl(self, query: str, cache_type: str) -> int:
        """Calculate adaptive TTL based on usage patterns."""
        cache_key = self._generate_cache_key(query, cache_type)
        
        # Check historical access patterns
        if cache_key in self.access_patterns:
            access_times = self.access_patterns[cache_key]
            if len(access_times) >= 2:
                # Calculate access frequency
                time_diffs = np.diff(access_times)
                avg_interval = np.mean(time_diffs)
                
                # Adaptive TTL based on access frequency
                if avg_interval < 300:  # Very frequent (< 5 min)
                    return self.default_ttl * 4  # 4x longer
                elif avg_interval < 1800:  # Frequent (< 30 min)
                    return self.default_ttl * 2  # 2x longer
                elif avg_interval < 7200:  # Regular (< 2 hours)
                    return self.default_ttl
                else:  # Infrequent
                    return self.default_ttl // 2  # Half duration
        
        # Default TTL with content-type adjustments
        ttl_multipliers = {
            'search_results': 1.0,
            'neural_inference': 2.0,  # Neural results are more expensive
            'llm_response': 0.5,  # LLM responses change more often
            'dataset_metadata': 3.0,  # Dataset metadata is more stable
            'relationships': 2.0
        }
        
        multiplier = ttl_multipliers.get(cache_type, 1.0)
        return int(self.default_ttl * multiplier)
    
    def _is_valid_entry(self, entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if not entry:
            return False
        
        created_at = entry.get('created_at', 0)
        ttl = entry.get('ttl', self.default_ttl)
        
        return (time.time() - created_at) < ttl
    
    def _store_to_disk(self, cache_key: str, entry: Dict):
        """Store cache entry to disk."""
        try:
            file_path = self.cache_dir / f"{cache_key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Disk storage failed for {cache_key}: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Dict]:
        """Load cache entry from disk."""
        try:
            file_path = self.cache_dir / f"{cache_key}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Disk loading failed for {cache_key}: {e}")
        
        return None
    
    def _update_cache_metadata(self, cache_key: str, query: str, entry: Dict):
        """Update cache metadata in database."""
        try:
            metadata = {
                'query_text': query,
                'query_hash': hashlib.md5(query.encode()).hexdigest(),
                'created_at': entry['created_at'],
                'last_accessed': time.time(),
                'access_count': 1,
                'ttl': entry['ttl'],
                'data_size': len(str(entry['data'])),
                'content_type': entry['cache_type'],
                'tags': json.dumps(entry['tags'])
            }
            
            self.cache_metadata[cache_key] = metadata
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_metadata 
                    (cache_key, query_text, query_hash, created_at, last_accessed, 
                     access_count, ttl, data_size, content_type, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (cache_key, metadata['query_text'], metadata['query_hash'],
                      metadata['created_at'], metadata['last_accessed'],
                      metadata['access_count'], metadata['ttl'],
                      metadata['data_size'], metadata['content_type'], metadata['tags']))
                
        except Exception as e:
            logger.warning(f"Metadata update failed: {e}")
    
    def _store_query_embedding(self, query: str, cache_key: str):
        """Store query embedding for similarity matching."""
        if not self.semantic_model:
            return
        
        try:
            embedding = self.semantic_model.encode([query])[0]
            self.query_embeddings[query] = embedding
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO query_patterns 
                    (query_text, query_embedding, cache_key, timestamp, response_time, cache_hit)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (query, pickle.dumps(embedding), cache_key, time.time(), 0.0, False))
                
        except Exception as e:
            logger.warning(f"Query embedding storage failed: {e}")
    
    def _update_access_pattern(self, cache_key: str):
        """Update access pattern for adaptive TTL."""
        current_time = time.time()
        
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        
        self.access_patterns[cache_key].append(current_time)
        
        # Keep only recent access times (last 10)
        if len(self.access_patterns[cache_key]) > 10:
            self.access_patterns[cache_key] = self.access_patterns[cache_key][-10:]
        
        # Update metadata
        if cache_key in self.cache_metadata:
            self.cache_metadata[cache_key]['last_accessed'] = current_time
            self.cache_metadata[cache_key]['access_count'] += 1
    
    def _cache_key_exists_and_valid(self, cache_key: str) -> bool:
        """Check if cache key exists and is valid."""
        # Check memory
        if cache_key in self.memory_cache:
            return self._is_valid_entry(self.memory_cache[cache_key])
        
        # Check disk
        entry = self._load_from_disk(cache_key)
        return entry is not None and self._is_valid_entry(entry)
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            current_time = time.time()
            expired_keys = []
            
            # Check memory cache
            for cache_key, entry in self.memory_cache.items():
                if not self._is_valid_entry(entry):
                    expired_keys.append(cache_key)
            
            # Remove expired entries
            for key in expired_keys:
                del self.memory_cache[key]
                
                # Remove disk file
                file_path = self.cache_dir / f"{key}.pkl"
                if file_path.exists():
                    file_path.unlink()
                
                # Remove metadata
                if key in self.cache_metadata:
                    del self.cache_metadata[key]
            
            if expired_keys:
                logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired entries")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _record_cache_hit(self, query: str, cache_key: str, response_time: float, hit: bool):
        """Record cache hit/miss for analytics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO query_patterns 
                    (query_text, cache_key, timestamp, response_time, cache_hit)
                    VALUES (?, ?, ?, ?, ?)
                ''', (query, cache_key, time.time(), response_time, hit))
        except Exception as e:
            logger.warning(f"Cache hit recording failed: {e}")
    
    def invalidate(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None):
        """Invalidate cache entries by pattern or tags."""
        with self.lock:
            keys_to_remove = []
            
            for cache_key in self.cache_metadata.keys():
                should_remove = False
                
                # Check pattern match
                if pattern and pattern in self.cache_metadata[cache_key].get('query_text', ''):
                    should_remove = True
                
                # Check tag match
                if tags:
                    entry_tags = self.cache_metadata[cache_key].get('tags', [])
                    if any(tag in entry_tags for tag in tags):
                        should_remove = True
                
                if should_remove:
                    keys_to_remove.append(cache_key)
            
            # Remove entries
            for key in keys_to_remove:
                self._remove_cache_entry(key)
            
            logger.info(f"🗑️ Invalidated {len(keys_to_remove)} cache entries")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a specific cache entry."""
        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from disk
        file_path = self.cache_dir / f"{cache_key}.pkl"
        if file_path.exists():
            file_path.unlink()
        
        # Remove metadata
        if cache_key in self.cache_metadata:
            del self.cache_metadata[cache_key]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            # Calculate hit rate
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('SELECT COUNT(*) FROM query_patterns WHERE cache_hit = 1')
                    hits = cursor.fetchone()[0]
                    
                    cursor = conn.execute('SELECT COUNT(*) FROM query_patterns')
                    total = cursor.fetchone()[0]
                    
                    hit_rate = hits / total if total > 0 else 0.0
            except:
                hit_rate = 0.0
            
            # Memory usage
            memory_entries = len(self.memory_cache)
            total_entries = len(self.cache_metadata)
            
            # Cache sizes by type
            type_distribution = {}
            for metadata in self.cache_metadata.values():
                content_type = metadata.get('content_type', 'unknown')
                type_distribution[content_type] = type_distribution.get(content_type, 0) + 1
            
            return {
                'hit_rate': hit_rate,
                'total_entries': total_entries,
                'memory_entries': memory_entries,
                'memory_utilization': memory_entries / self.max_memory_size,
                'type_distribution': type_distribution,
                'similarity_threshold': self.similarity_threshold,
                'default_ttl': self.default_ttl,
                'avg_query_embedding_size': len(self.query_embeddings)
            }
    
    def get_query_recommendations(self, partial_query: str, max_recommendations: int = 5) -> List[str]:
        """Get query recommendations based on cache patterns."""
        recommendations = []
        
        if not self.semantic_model or not partial_query.strip():
            return recommendations
        
        try:
            partial_embedding = self.semantic_model.encode([partial_query])
            similarities = []
            
            for cached_query, embedding in self.query_embeddings.items():
                if embedding is not None and len(cached_query) > len(partial_query):
                    similarity = cosine_similarity(partial_embedding, embedding.reshape(1, -1))[0][0]
                    similarities.append((cached_query, similarity))
            
            # Sort by similarity and take top recommendations
            similarities.sort(key=lambda x: x[1], reverse=True)
            recommendations = [query for query, _ in similarities[:max_recommendations]]
            
        except Exception as e:
            logger.warning(f"Query recommendations failed: {e}")
        
        return recommendations


class CacheManager:
    """
    High-level cache manager that orchestrates different cache types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cache_config = config.get('cache', {})
        
        # Initialize different cache instances
        self.search_cache = IntelligentCache(
            cache_dir="cache/search",
            max_memory_size=cache_config.get('search_max_size', 500),
            default_ttl=cache_config.get('search_ttl', 1800)  # 30 minutes
        )
        
        self.neural_cache = IntelligentCache(
            cache_dir="cache/neural",
            max_memory_size=cache_config.get('neural_max_size', 300),
            default_ttl=cache_config.get('neural_ttl', 7200)  # 2 hours
        )
        
        self.llm_cache = IntelligentCache(
            cache_dir="cache/llm",
            max_memory_size=cache_config.get('llm_max_size', 200),
            default_ttl=cache_config.get('llm_ttl', 1800)  # 30 minutes
        )
        
        logger.info("🗄️ CacheManager initialized with specialized caches")
    
    def get_search_result(self, query: str) -> Optional[Any]:
        """Get cached search result."""
        return self.search_cache.get(query, 'search_results')
    
    def cache_search_result(self, query: str, results: Any, ttl: Optional[int] = None) -> str:
        """Cache search results."""
        return self.search_cache.set(query, results, 'search_results', ttl)
    
    def get_neural_result(self, query: str) -> Optional[Any]:
        """Get cached neural inference result."""
        return self.neural_cache.get(query, 'neural_inference')
    
    def cache_neural_result(self, query: str, results: Any, ttl: Optional[int] = None) -> str:
        """Cache neural inference results."""
        return self.neural_cache.set(query, results, 'neural_inference', ttl)
    
    def get_llm_response(self, prompt: str) -> Optional[Any]:
        """Get cached LLM response."""
        return self.llm_cache.get(prompt, 'llm_response', use_similarity=True)
    
    def cache_llm_response(self, prompt: str, response: Any, ttl: Optional[int] = None) -> str:
        """Cache LLM response."""
        return self.llm_cache.set(prompt, response, 'llm_response', ttl)
    
    def invalidate_all(self):
        """Invalidate all caches."""
        self.search_cache.invalidate()
        self.neural_cache.invalidate()
        self.llm_cache.invalidate()
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'search_cache': self.search_cache.get_cache_statistics(),
            'neural_cache': self.neural_cache.get_cache_statistics(),
            'llm_cache': self.llm_cache.get_cache_statistics()
        }