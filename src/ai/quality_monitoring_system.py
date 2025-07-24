"""
Quality Monitoring System for AI Dataset Research Assistant

This module provides comprehensive quality monitoring capabilities including:
- NDCG@3 and relevance accuracy tracking
- Domain routing accuracy monitoring  
- Singapore-first strategy effectiveness tracking
- Real-time quality metrics collection and alerting
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict
import re

@dataclass
class QualityMetrics:
    """Quality metrics for recommendation system"""
    timestamp: str
    ndcg_at_3: float
    relevance_accuracy: float
    domain_routing_accuracy: float
    singapore_first_accuracy: float
    total_queries: int
    successful_queries: int
    average_response_time: float
    cache_hit_rate: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def meets_quality_threshold(self, threshold: float = 0.7) -> bool:
        """Check if quality meets minimum threshold"""
        return (self.ndcg_at_3 >= threshold and 
                self.relevance_accuracy >= threshold and
                self.domain_routing_accuracy >= threshold)

@dataclass
class QueryEvaluation:
    """Individual query evaluation result"""
    query: str
    timestamp: str
    predicted_sources: List[str]
    actual_relevance_scores: Dict[str, float]
    domain_classification: str
    singapore_first_applied: bool
    response_time: float
    cache_hit: bool
    ndcg_score: float
    relevance_match: bool

class TrainingMappingsParser:
    """Parser for training_mappings.md ground truth data"""
    
    def __init__(self, mappings_file: str = "training_mappings.md"):
        self.mappings_file = mappings_file
        self.ground_truth = self._parse_mappings()
        
    def _parse_mappings(self) -> Dict[str, Dict[str, float]]:
        """Parse training mappings into ground truth dictionary"""
        ground_truth = defaultdict(dict)
        
        try:
            with open(self.mappings_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract mappings using regex
            pattern = r'- (.+?) → (.+?) \(([0-9.]+)\) - (.+)'
            matches = re.findall(pattern, content)
            
            for query, source, score, reason in matches:
                query = query.strip().lower()
                source = source.strip()
                score = float(score)
                ground_truth[query][source] = score
                
        except FileNotFoundError:
            logging.warning(f"Training mappings file {self.mappings_file} not found")
        except Exception as e:
            logging.error(f"Error parsing training mappings: {e}")
            
        return dict(ground_truth)
    
    def get_expected_sources(self, query: str) -> Dict[str, float]:
        """Get expected sources and relevance scores for a query"""
        query_lower = query.lower().strip()
        
        # Direct match
        if query_lower in self.ground_truth:
            return self.ground_truth[query_lower]
        
        # Partial match - find queries that contain the search terms
        matches = {}
        query_words = set(query_lower.split())
        
        for gt_query, sources in self.ground_truth.items():
            gt_words = set(gt_query.split())
            if query_words.intersection(gt_words):
                # Merge sources, taking highest relevance scores
                for source, score in sources.items():
                    if source not in matches or score > matches[source]:
                        matches[source] = score
                        
        return matches
    
    def classify_query_domain(self, query: str) -> str:
        """Classify query domain based on training mappings"""
        query_lower = query.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            'psychology': ['psychology', 'mental health', 'behavioral', 'cognitive'],
            'machine_learning': ['machine learning', 'ml datasets', 'artificial intelligence', 'deep learning', 'neural networks'],
            'climate': ['climate', 'environmental', 'weather', 'temperature'],
            'economics': ['economic', 'gdp', 'financial', 'trade', 'poverty'],
            'singapore': ['singapore'],
            'health': ['health', 'medical', 'healthcare', 'disease', 'public health'],
            'education': ['education', 'student', 'university', 'school']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
                
        return 'general'
    
    def should_apply_singapore_first(self, query: str) -> bool:
        """Determine if Singapore-first strategy should be applied"""
        return 'singapore' in query.lower()

class QualityMonitoringSystem:
    """Comprehensive quality monitoring system"""
    
    def __init__(self, db_path: str = "data/quality_monitoring.db"):
        self.db_path = db_path
        self.mappings_parser = TrainingMappingsParser()
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for quality metrics storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ndcg_at_3 REAL,
                    relevance_accuracy REAL,
                    domain_routing_accuracy REAL,
                    singapore_first_accuracy REAL,
                    total_queries INTEGER,
                    successful_queries INTEGER,
                    average_response_time REAL,
                    cache_hit_rate REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    predicted_sources TEXT,
                    actual_relevance_scores TEXT,
                    domain_classification TEXT,
                    singapore_first_applied BOOLEAN,
                    response_time REAL,
                    cache_hit BOOLEAN,
                    ndcg_score REAL,
                    relevance_match BOOLEAN
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
    def calculate_ndcg_at_k(self, predicted_sources: List[str], 
                           relevance_scores: Dict[str, float], k: int = 3) -> float:
        """Calculate NDCG@k score for predicted sources"""
        if not predicted_sources or not relevance_scores:
            return 0.0
            
        # Get relevance scores for predicted sources (top k)
        predicted_k = predicted_sources[:k]
        predicted_relevance = [relevance_scores.get(source, 0.0) for source in predicted_k]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(predicted_relevance):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
                
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            if rel > 0:
                idcg += rel / np.log2(i + 2)
                
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, query: str, predicted_sources: List[str], 
                      domain_classification: str, singapore_first_applied: bool,
                      response_time: float, cache_hit: bool) -> QueryEvaluation:
        """Evaluate a single query against ground truth"""
        timestamp = datetime.now().isoformat()
        
        # Get expected sources and relevance scores
        expected_sources = self.mappings_parser.get_expected_sources(query)
        
        # Calculate NDCG@3
        ndcg_score = self.calculate_ndcg_at_k(predicted_sources, expected_sources, k=3)
        
        # Check relevance match (at least one highly relevant source in top 3)
        top_3_sources = predicted_sources[:3]
        relevance_match = any(
            expected_sources.get(source, 0.0) >= 0.8 
            for source in top_3_sources
        )
        
        evaluation = QueryEvaluation(
            query=query,
            timestamp=timestamp,
            predicted_sources=predicted_sources,
            actual_relevance_scores=expected_sources,
            domain_classification=domain_classification,
            singapore_first_applied=singapore_first_applied,
            response_time=response_time,
            cache_hit=cache_hit,
            ndcg_score=ndcg_score,
            relevance_match=relevance_match
        )
        
        # Store evaluation in database
        self._store_query_evaluation(evaluation)
        
        return evaluation
    
    def _store_query_evaluation(self, evaluation: QueryEvaluation):
        """Store query evaluation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO query_evaluations 
                (query, timestamp, predicted_sources, actual_relevance_scores,
                 domain_classification, singapore_first_applied, response_time,
                 cache_hit, ndcg_score, relevance_match)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.query,
                evaluation.timestamp,
                json.dumps(evaluation.predicted_sources),
                json.dumps(evaluation.actual_relevance_scores),
                evaluation.domain_classification,
                evaluation.singapore_first_applied,
                evaluation.response_time,
                evaluation.cache_hit,
                evaluation.ndcg_score,
                evaluation.relevance_match
            ))
    
    def calculate_quality_metrics(self, time_window_hours: int = 24) -> QualityMetrics:
        """Calculate quality metrics for specified time window"""
        cutoff_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM query_evaluations 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            evaluations = cursor.fetchall()
            
        if not evaluations:
            return QualityMetrics(
                timestamp=datetime.now().isoformat(),
                ndcg_at_3=0.0,
                relevance_accuracy=0.0,
                domain_routing_accuracy=0.0,
                singapore_first_accuracy=0.0,
                total_queries=0,
                successful_queries=0,
                average_response_time=0.0,
                cache_hit_rate=0.0
            )
        
        # Calculate metrics
        total_queries = len(evaluations)
        ndcg_scores = [eval[9] for eval in evaluations]  # ndcg_score column
        relevance_matches = [eval[10] for eval in evaluations]  # relevance_match column
        response_times = [eval[7] for eval in evaluations]  # response_time column
        cache_hits = [eval[8] for eval in evaluations]  # cache_hit column
        
        # Domain routing accuracy
        domain_correct = 0
        singapore_first_correct = 0
        singapore_queries = 0
        
        for eval_row in evaluations:
            query = eval_row[1]
            domain_classification = eval_row[4]
            singapore_first_applied = eval_row[5]
            
            # Check domain classification accuracy
            expected_domain = self.mappings_parser.classify_query_domain(query)
            if domain_classification == expected_domain:
                domain_correct += 1
                
            # Check Singapore-first strategy accuracy
            if 'singapore' in query.lower():
                singapore_queries += 1
                expected_singapore_first = self.mappings_parser.should_apply_singapore_first(query)
                if singapore_first_applied == expected_singapore_first:
                    singapore_first_correct += 1
        
        metrics = QualityMetrics(
            timestamp=datetime.now().isoformat(),
            ndcg_at_3=np.mean(ndcg_scores),
            relevance_accuracy=np.mean(relevance_matches),
            domain_routing_accuracy=domain_correct / total_queries if total_queries > 0 else 0.0,
            singapore_first_accuracy=singapore_first_correct / singapore_queries if singapore_queries > 0 else 1.0,
            total_queries=total_queries,
            successful_queries=sum(relevance_matches),
            average_response_time=np.mean(response_times),
            cache_hit_rate=np.mean(cache_hits)
        )
        
        # Store metrics
        self._store_quality_metrics(metrics)
        
        return metrics
    
    def _store_quality_metrics(self, metrics: QualityMetrics):
        """Store quality metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO quality_metrics 
                (timestamp, ndcg_at_3, relevance_accuracy, domain_routing_accuracy,
                 singapore_first_accuracy, total_queries, successful_queries,
                 average_response_time, cache_hit_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.ndcg_at_3,
                metrics.relevance_accuracy,
                metrics.domain_routing_accuracy,
                metrics.singapore_first_accuracy,
                metrics.total_queries,
                metrics.successful_queries,
                metrics.average_response_time,
                metrics.cache_hit_rate
            ))
    
    def check_quality_thresholds(self, metrics: QualityMetrics, 
                               min_ndcg: float = 0.7,
                               min_relevance: float = 0.7,
                               min_domain_accuracy: float = 0.8) -> List[str]:
        """Check if quality metrics meet minimum thresholds"""
        alerts = []
        
        if metrics.ndcg_at_3 < min_ndcg:
            alerts.append(f"NDCG@3 below threshold: {metrics.ndcg_at_3:.3f} < {min_ndcg}")
            
        if metrics.relevance_accuracy < min_relevance:
            alerts.append(f"Relevance accuracy below threshold: {metrics.relevance_accuracy:.3f} < {min_relevance}")
            
        if metrics.domain_routing_accuracy < min_domain_accuracy:
            alerts.append(f"Domain routing accuracy below threshold: {metrics.domain_routing_accuracy:.3f} < {min_domain_accuracy}")
            
        # Store alerts
        for alert in alerts:
            self._store_alert("QUALITY_THRESHOLD", alert, "HIGH")
            
        return alerts
    
    def _store_alert(self, alert_type: str, message: str, severity: str):
        """Store quality alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO quality_alerts (timestamp, alert_type, message, severity)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), alert_type, message, severity))
    
    def get_quality_trends(self, days: int = 7) -> Dict[str, List[float]]:
        """Get quality trends over specified number of days"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, ndcg_at_3, relevance_accuracy, 
                       domain_routing_accuracy, singapore_first_accuracy
                FROM quality_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp ASC
            """, (cutoff_time,))
            
            results = cursor.fetchall()
            
        trends = {
            'timestamps': [r[0] for r in results],
            'ndcg_at_3': [r[1] for r in results],
            'relevance_accuracy': [r[2] for r in results],
            'domain_routing_accuracy': [r[3] for r in results],
            'singapore_first_accuracy': [r[4] for r in results]
        }
        
        return trends
    
    def get_domain_performance_breakdown(self, time_window_hours: int = 24) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by domain"""
        cutoff_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT domain_classification, ndcg_score, relevance_match
                FROM query_evaluations 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            results = cursor.fetchall()
            
        domain_stats = defaultdict(lambda: {'ndcg_scores': [], 'relevance_matches': []})
        
        for domain, ndcg, relevance in results:
            domain_stats[domain]['ndcg_scores'].append(ndcg)
            domain_stats[domain]['relevance_matches'].append(relevance)
            
        # Calculate averages
        breakdown = {}
        for domain, stats in domain_stats.items():
            breakdown[domain] = {
                'avg_ndcg': np.mean(stats['ndcg_scores']) if stats['ndcg_scores'] else 0.0,
                'relevance_accuracy': np.mean(stats['relevance_matches']) if stats['relevance_matches'] else 0.0,
                'query_count': len(stats['ndcg_scores'])
            }
            
        return breakdown
    
    def monitor_recommendation_quality(self) -> QualityMetrics:
        """Main monitoring function - calculate and check quality metrics"""
        metrics = self.calculate_quality_metrics()
        alerts = self.check_quality_thresholds(metrics)
        
        if alerts:
            self.logger.warning(f"Quality alerts triggered: {alerts}")
            
        return metrics
    
    def track_domain_routing_accuracy(self, time_window_hours: int = 24) -> float:
        """Track accuracy of domain-specific routing"""
        metrics = self.calculate_quality_metrics(time_window_hours)
        return metrics.domain_routing_accuracy
    
    def measure_singapore_first_effectiveness(self, time_window_hours: int = 24) -> float:
        """Track effectiveness of Singapore-first strategy"""
        metrics = self.calculate_quality_metrics(time_window_hours)
        return metrics.singapore_first_accuracy