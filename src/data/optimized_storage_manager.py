"""
Optimized Storage Manager
Database and storage optimization for faster quality-aware retrieval
with indexing strategy for domain-specific queries and embedding compression
"""

import gzip
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Optimized dataset metadata structure"""
    id: str
    title: str
    description: str
    source: str
    domain: str
    geographic_scope: str
    quality_score: float
    relevance_keywords: List[str]
    last_updated: float
    access_count: int = 0
    embedding_hash: Optional[str] = None


@dataclass
class QueryIndex:
    """Query indexing structure for fast retrieval"""
    query_hash: str
    query_text: str
    domain: str
    geographic_scope: str
    keywords: List[str]
    embedding_hash: str
    quality_threshold: float
    created_at: float


class OptimizedStorageManager:
    """Optimized storage manager with quality-aware indexing and compression"""
    
    def __init__(self, 
                 storage_dir: str = "data/optimized",
                 enable_compression: bool = True,
                 index_batch_size: int = 1000):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_compression = enable_compression
        self.index_batch_size = index_batch_size
        
        # Database paths
        self.metadata_db_path = self.storage_dir / "metadata.db"
        self.embeddings_db_path = self.storage_dir / "embeddings.db"
        self.query_index_db_path = self.storage_dir / "query_index.db"
        
        # Compressed storage paths
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self.lock = Lock()
        
        # Initialize databases
        self._initialize_databases()
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'compression_ratio': 0.0
        }
        
        logger.info(f"🗄️ OptimizedStorageManager initialized")
        logger.info(f"  Storage directory: {storage_dir}")
        logger.info(f"  Compression enabled: {enable_compression}")
        logger.info(f"  Index batch size: {index_batch_size}")
    
    def _initialize_databases(self):
        """Initialize optimized database schemas"""
        
        # Metadata database
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    source TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    geographic_scope TEXT,
                    quality_score REAL NOT NULL,
                    relevance_keywords TEXT,  -- JSON array
                    last_updated REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    embedding_hash TEXT
                )
            ''')
            
            # Create indexes for fast domain-specific queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_domain ON dataset_metadata(domain)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON dataset_metadata(source)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_geographic_scope ON dataset_metadata(geographic_scope)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_score ON dataset_metadata(quality_score DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_domain_quality ON dataset_metadata(domain, quality_score DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_geographic_quality ON dataset_metadata(geographic_scope, quality_score DESC)')
            
            # Full-text search index for titles and descriptions
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS dataset_fts USING fts5(
                    id, title, description, relevance_keywords,
                    content='dataset_metadata',
                    content_rowid='rowid'
                )
            ''')
        
        # Embeddings database (for metadata only, actual embeddings compressed separately)
        with sqlite3.connect(self.embeddings_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    hash TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    embedding_type TEXT,  -- 'query', 'dataset', 'domain'
                    dimensions INTEGER,
                    compression_method TEXT,
                    file_path TEXT,
                    created_at REAL,
                    access_count INTEGER DEFAULT 0,
                    FOREIGN KEY (dataset_id) REFERENCES dataset_metadata(id)
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_embedding_type ON embedding_metadata(embedding_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_dataset_id ON embedding_metadata(dataset_id)')
        
        # Query index database
        with sqlite3.connect(self.query_index_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_index (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    domain TEXT,
                    geographic_scope TEXT,
                    keywords TEXT,  -- JSON array
                    embedding_hash TEXT,
                    quality_threshold REAL,
                    created_at REAL,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_query_domain ON query_index(domain)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_query_geographic ON query_index(geographic_scope)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_query_quality ON query_index(quality_threshold)')
            
            # Full-text search for queries
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5(
                    query_hash, query_text, keywords,
                    content='query_index',
                    content_rowid='rowid'
                )
            ''')
        
        logger.info("📊 Optimized databases initialized with indexes")
    
    def store_dataset_metadata(self, metadata: DatasetMetadata) -> bool:
        """Store dataset metadata with optimized indexing"""
        try:
            with self.lock:
                with sqlite3.connect(self.metadata_db_path) as conn:
                    # Insert or update metadata
                    conn.execute('''
                        INSERT OR REPLACE INTO dataset_metadata 
                        (id, title, description, source, domain, geographic_scope, 
                         quality_score, relevance_keywords, last_updated, access_count, embedding_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.id, metadata.title, metadata.description, metadata.source,
                        metadata.domain, metadata.geographic_scope, metadata.quality_score,
                        json.dumps(metadata.relevance_keywords), metadata.last_updated,
                        metadata.access_count, metadata.embedding_hash
                    ))
                    
                    # Update FTS index
                    conn.execute('''
                        INSERT OR REPLACE INTO dataset_fts (id, title, description, relevance_keywords)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        metadata.id, metadata.title, metadata.description,
                        ' '.join(metadata.relevance_keywords)
                    ))
            
            logger.debug(f"📝 Stored metadata for dataset: {metadata.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store dataset metadata: {e}")
            return False
    
    def store_embedding(self, 
                       embedding: np.ndarray, 
                       embedding_id: str,
                       embedding_type: str = 'dataset',
                       dataset_id: Optional[str] = None) -> Optional[str]:
        """Store embedding with compression"""
        try:
            embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            # Compress and store embedding
            if self.enable_compression:
                compressed_data = gzip.compress(pickle.dumps(embedding))
                file_path = self.embeddings_dir / f"{embedding_hash}.pkl.gz"
                compression_method = 'gzip_pickle'
            else:
                compressed_data = pickle.dumps(embedding)
                file_path = self.embeddings_dir / f"{embedding_hash}.pkl"
                compression_method = 'pickle'
            
            # Write compressed embedding
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Store metadata
            with self.lock:
                with sqlite3.connect(self.embeddings_db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO embedding_metadata
                        (hash, dataset_id, embedding_type, dimensions, compression_method, 
                         file_path, created_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        embedding_hash, dataset_id, embedding_type, embedding.shape[0],
                        compression_method, str(file_path), time.time(), 0
                    ))
            
            # Update compression ratio stats
            original_size = embedding.nbytes
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size
            
            self.performance_stats['compression_ratio'] = (
                (self.performance_stats['compression_ratio'] + compression_ratio) / 2
            )
            
            logger.debug(f"💾 Stored embedding: {embedding_hash} (compression: {compression_ratio:.3f})")
            return embedding_hash
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return None
    
    def retrieve_embedding(self, embedding_hash: str) -> Optional[np.ndarray]:
        """Retrieve and decompress embedding"""
        try:
            with self.lock:
                with sqlite3.connect(self.embeddings_db_path) as conn:
                    cursor = conn.execute('''
                        SELECT file_path, compression_method FROM embedding_metadata 
                        WHERE hash = ?
                    ''', (embedding_hash,))
                    
                    result = cursor.fetchone()
                    if not result:
                        return None
                    
                    file_path, compression_method = result
                    
                    # Update access count
                    conn.execute('''
                        UPDATE embedding_metadata SET access_count = access_count + 1 
                        WHERE hash = ?
                    ''', (embedding_hash,))
            
            # Load and decompress embedding
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            if compression_method == 'gzip_pickle':
                embedding = pickle.loads(gzip.decompress(compressed_data))
            else:
                embedding = pickle.loads(compressed_data)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding: {e}")
            return None
    
    def query_datasets_by_domain(self, 
                                domain: str, 
                                quality_threshold: float = 0.7,
                                limit: int = 10) -> List[DatasetMetadata]:
        """Optimized domain-specific dataset query"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, title, description, source, domain, geographic_scope,
                           quality_score, relevance_keywords, last_updated, access_count, embedding_hash
                    FROM dataset_metadata 
                    WHERE domain = ? AND quality_score >= ?
                    ORDER BY quality_score DESC, access_count DESC
                    LIMIT ?
                ''', (domain, quality_threshold, limit))
                
                results = []
                for row in cursor.fetchall():
                    metadata = DatasetMetadata(
                        id=row[0], title=row[1], description=row[2], source=row[3],
                        domain=row[4], geographic_scope=row[5], quality_score=row[6],
                        relevance_keywords=json.loads(row[7]) if row[7] else [],
                        last_updated=row[8], access_count=row[9], embedding_hash=row[10]
                    )
                    results.append(metadata)
                
                # Update access counts
                if results:
                    dataset_ids = [r.id for r in results]
                    placeholders = ','.join(['?' for _ in dataset_ids])
                    conn.execute(f'''
                        UPDATE dataset_metadata 
                        SET access_count = access_count + 1 
                        WHERE id IN ({placeholders})
                    ''', dataset_ids)
            
            # Update performance stats
            retrieval_time = time.time() - start_time
            self._update_performance_stats(retrieval_time, len(results) > 0)
            
            logger.debug(f"🔍 Domain query '{domain}': {len(results)} results in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Domain query failed: {e}")
            return []
    
    def query_datasets_by_geographic_scope(self, 
                                         geographic_scope: str,
                                         quality_threshold: float = 0.7,
                                         limit: int = 10) -> List[DatasetMetadata]:
        """Optimized geographic-specific dataset query"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, title, description, source, domain, geographic_scope,
                           quality_score, relevance_keywords, last_updated, access_count, embedding_hash
                    FROM dataset_metadata 
                    WHERE geographic_scope = ? AND quality_score >= ?
                    ORDER BY quality_score DESC, access_count DESC
                    LIMIT ?
                ''', (geographic_scope, quality_threshold, limit))
                
                results = []
                for row in cursor.fetchall():
                    metadata = DatasetMetadata(
                        id=row[0], title=row[1], description=row[2], source=row[3],
                        domain=row[4], geographic_scope=row[5], quality_score=row[6],
                        relevance_keywords=json.loads(row[7]) if row[7] else [],
                        last_updated=row[8], access_count=row[9], embedding_hash=row[10]
                    )
                    results.append(metadata)
                
                # Update access counts
                if results:
                    dataset_ids = [r.id for r in results]
                    placeholders = ','.join(['?' for _ in dataset_ids])
                    conn.execute(f'''
                        UPDATE dataset_metadata 
                        SET access_count = access_count + 1 
                        WHERE id IN ({placeholders})
                    ''', dataset_ids)
            
            # Update performance stats
            retrieval_time = time.time() - start_time
            self._update_performance_stats(retrieval_time, len(results) > 0)
            
            logger.debug(f"🌍 Geographic query '{geographic_scope}': {len(results)} results in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Geographic query failed: {e}")
            return []
    
    def full_text_search(self, 
                        query_text: str,
                        quality_threshold: float = 0.7,
                        limit: int = 10) -> List[DatasetMetadata]:
        """Full-text search with quality filtering"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                # Use FTS for text search, then join with metadata for quality filtering
                cursor = conn.execute('''
                    SELECT m.id, m.title, m.description, m.source, m.domain, m.geographic_scope,
                           m.quality_score, m.relevance_keywords, m.last_updated, m.access_count, m.embedding_hash
                    FROM dataset_fts f
                    JOIN dataset_metadata m ON f.id = m.id
                    WHERE dataset_fts MATCH ? AND m.quality_score >= ?
                    ORDER BY m.quality_score DESC, m.access_count DESC
                    LIMIT ?
                ''', (query_text, quality_threshold, limit))
                
                results = []
                for row in cursor.fetchall():
                    metadata = DatasetMetadata(
                        id=row[0], title=row[1], description=row[2], source=row[3],
                        domain=row[4], geographic_scope=row[5], quality_score=row[6],
                        relevance_keywords=json.loads(row[7]) if row[7] else [],
                        last_updated=row[8], access_count=row[9], embedding_hash=row[10]
                    )
                    results.append(metadata)
            
            # Update performance stats
            retrieval_time = time.time() - start_time
            self._update_performance_stats(retrieval_time, len(results) > 0)
            
            logger.debug(f"📝 Full-text search '{query_text}': {len(results)} results in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []
    
    def store_query_index(self, query_index: QueryIndex) -> bool:
        """Store query index for fast retrieval"""
        try:
            with self.lock:
                with sqlite3.connect(self.query_index_db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO query_index
                        (query_hash, query_text, domain, geographic_scope, keywords,
                         embedding_hash, quality_threshold, created_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        query_index.query_hash, query_index.query_text, query_index.domain,
                        query_index.geographic_scope, json.dumps(query_index.keywords),
                        query_index.embedding_hash, query_index.quality_threshold,
                        query_index.created_at, 0
                    ))
                    
                    # Update FTS index
                    conn.execute('''
                        INSERT OR REPLACE INTO query_fts (query_hash, query_text, keywords)
                        VALUES (?, ?, ?)
                    ''', (
                        query_index.query_hash, query_index.query_text,
                        ' '.join(query_index.keywords)
                    ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store query index: {e}")
            return False
    
    def get_similar_queries(self, query_text: str, limit: int = 5) -> List[QueryIndex]:
        """Find similar queries using FTS"""
        try:
            with sqlite3.connect(self.query_index_db_path) as conn:
                cursor = conn.execute('''
                    SELECT q.query_hash, q.query_text, q.domain, q.geographic_scope, q.keywords,
                           q.embedding_hash, q.quality_threshold, q.created_at
                    FROM query_fts f
                    JOIN query_index q ON f.query_hash = q.query_hash
                    WHERE query_fts MATCH ?
                    ORDER BY q.access_count DESC
                    LIMIT ?
                ''', (query_text, limit))
                
                results = []
                for row in cursor.fetchall():
                    query_index = QueryIndex(
                        query_hash=row[0], query_text=row[1], domain=row[2],
                        geographic_scope=row[3], keywords=json.loads(row[4]) if row[4] else [],
                        embedding_hash=row[5], quality_threshold=row[6], created_at=row[7]
                    )
                    results.append(query_index)
                
                return results
                
        except Exception as e:
            logger.error(f"Similar query search failed: {e}")
            return []
    
    def _update_performance_stats(self, retrieval_time: float, cache_hit: bool):
        """Update performance statistics"""
        self.performance_stats['total_queries'] += 1
        
        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        
        # Update average retrieval time
        total = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_retrieval_time']
        self.performance_stats['avg_retrieval_time'] = (
            (current_avg * (total - 1) + retrieval_time) / total
        )
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        try:
            stats = {
                'performance': self.performance_stats.copy(),
                'storage': {},
                'indexes': {}
            }
            
            # Dataset metadata statistics
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM dataset_metadata')
                stats['storage']['total_datasets'] = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT domain, COUNT(*) FROM dataset_metadata GROUP BY domain')
                stats['storage']['datasets_by_domain'] = dict(cursor.fetchall())
                
                cursor = conn.execute('SELECT AVG(quality_score) FROM dataset_metadata')
                stats['storage']['avg_quality_score'] = cursor.fetchone()[0] or 0.0
            
            # Embedding statistics
            with sqlite3.connect(self.embeddings_db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM embedding_metadata')
                stats['storage']['total_embeddings'] = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT embedding_type, COUNT(*) FROM embedding_metadata GROUP BY embedding_type')
                stats['storage']['embeddings_by_type'] = dict(cursor.fetchall())
            
            # Query index statistics
            with sqlite3.connect(self.query_index_db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM query_index')
                stats['indexes']['total_queries'] = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT domain, COUNT(*) FROM query_index GROUP BY domain')
                stats['indexes']['queries_by_domain'] = dict(cursor.fetchall())
            
            # Storage size information
            storage_size = sum(f.stat().st_size for f in self.storage_dir.rglob('*') if f.is_file())
            stats['storage']['total_size_bytes'] = storage_size
            stats['storage']['total_size_mb'] = storage_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return {'error': str(e)}
    
    def optimize_indexes(self):
        """Optimize database indexes for better performance"""
        try:
            databases = [
                (self.metadata_db_path, "metadata"),
                (self.embeddings_db_path, "embeddings"),
                (self.query_index_db_path, "query_index")
            ]
            
            for db_path, db_name in databases:
                with sqlite3.connect(db_path) as conn:
                    # Analyze tables for better query planning
                    conn.execute('ANALYZE')
                    
                    # Vacuum to reclaim space and optimize
                    conn.execute('VACUUM')
                    
                logger.info(f"🔧 Optimized {db_name} database")
            
            logger.info("✅ Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")


# Factory function
def create_optimized_storage_manager(
    storage_dir: str = "data/optimized",
    enable_compression: bool = True,
    index_batch_size: int = 1000
) -> OptimizedStorageManager:
    """Factory function to create optimized storage manager"""
    
    return OptimizedStorageManager(
        storage_dir=storage_dir,
        enable_compression=enable_compression,
        index_batch_size=index_batch_size
    )


if __name__ == "__main__":
    # Test the optimized storage manager
    logging.basicConfig(level=logging.INFO)
    
    storage_manager = create_optimized_storage_manager()
    
    print("🧪 Testing Optimized Storage Manager\n")
    
    # Test dataset metadata storage
    test_metadata = DatasetMetadata(
        id="test_dataset_1",
        title="Test Psychology Dataset",
        description="A test dataset for psychology research",
        source="kaggle",
        domain="psychology",
        geographic_scope="global",
        quality_score=0.85,
        relevance_keywords=["psychology", "research", "behavior"],
        last_updated=time.time()
    )
    
    success = storage_manager.store_dataset_metadata(test_metadata)
    print(f"📝 Metadata storage: {'✅' if success else '❌'}")
    
    # Test embedding storage
    test_embedding = np.random.rand(256).astype(np.float32)
    embedding_hash = storage_manager.store_embedding(
        test_embedding, "test_embedding_1", "dataset", "test_dataset_1"
    )
    print(f"💾 Embedding storage: {'✅' if embedding_hash else '❌'}")
    
    # Test domain query
    domain_results = storage_manager.query_datasets_by_domain("psychology", 0.7, 5)
    print(f"🔍 Domain query: {len(domain_results)} results")
    
    # Test full-text search
    fts_results = storage_manager.full_text_search("psychology research", 0.7, 5)
    print(f"📝 Full-text search: {len(fts_results)} results")
    
    # Test embedding retrieval
    retrieved_embedding = storage_manager.retrieve_embedding(embedding_hash)
    embedding_match = np.allclose(test_embedding, retrieved_embedding) if retrieved_embedding is not None else False
    print(f"🔄 Embedding retrieval: {'✅' if embedding_match else '❌'}")
    
    # Get statistics
    stats = storage_manager.get_storage_statistics()
    print(f"\n📊 Storage Statistics:")
    print(f"  Total datasets: {stats['storage']['total_datasets']}")
    print(f"  Total embeddings: {stats['storage']['total_embeddings']}")
    print(f"  Storage size: {stats['storage']['total_size_mb']:.2f} MB")
    print(f"  Average quality: {stats['storage']['avg_quality_score']:.3f}")
    print(f"  Compression ratio: {stats['performance']['compression_ratio']:.3f}")
    
    # Optimize indexes
    storage_manager.optimize_indexes()
    
    print("\n✅ Optimized Storage Manager testing complete!")