"""
Quality Reporting and Analytics System

This module provides comprehensive quality reporting, A/B testing framework,
and user feedback integration for continuous quality enhancement.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict
import statistics
import uuid

from .quality_monitoring_system import QualityMonitoringSystem, QualityMetrics
from .automated_quality_validator import AutomatedQualityValidator, ValidationResult

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    timestamp: str
    time_period: str
    overall_score: float
    metrics_summary: Dict[str, float]
    domain_breakdown: Dict[str, Dict[str, float]]
    trends: Dict[str, List[float]]
    alerts: List[str]
    recommendations: List[str]
    validation_results: Dict[str, bool]
    user_feedback_summary: Dict[str, Any]

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    test_name: str
    description: str
    start_time: str
    end_time: Optional[str]
    control_group_size: int
    treatment_group_size: int
    success_metric: str
    minimum_effect_size: float
    confidence_level: float
    status: str  # 'running', 'completed', 'stopped'

@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    timestamp: str
    control_score: float
    treatment_score: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_sizes: Dict[str, int]
    details: Dict[str, Any]

@dataclass
class UserFeedback:
    """User feedback entry"""
    feedback_id: str
    timestamp: str
    user_id: Optional[str]
    query: str
    recommended_sources: List[str]
    user_rating: int  # 1-5 scale
    relevant_sources: List[str]
    irrelevant_sources: List[str]
    comments: Optional[str]
    session_id: Optional[str]

class QualityReportingSystem:
    """Quality reporting and analytics system"""
    
    def __init__(self, 
                 quality_monitor: QualityMonitoringSystem,
                 validator: AutomatedQualityValidator,
                 reporting_db_path: str = "data/quality_reporting.db"):
        self.quality_monitor = quality_monitor
        self.validator = validator
        self.reporting_db_path = reporting_db_path
        self.logger = logging.getLogger(__name__)
        self._init_reporting_database()
        
    def _init_reporting_database(self):
        """Initialize reporting database"""
        Path(self.reporting_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.reporting_db_path) as conn:
            # Quality reports table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    time_period TEXT NOT NULL,
                    overall_score REAL,
                    metrics_summary TEXT,
                    domain_breakdown TEXT,
                    trends TEXT,
                    alerts TEXT,
                    recommendations TEXT,
                    validation_results TEXT,
                    user_feedback_summary TEXT
                )
            """)
            
            # A/B test configurations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_configs (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    description TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    control_group_size INTEGER,
                    treatment_group_size INTEGER,
                    success_metric TEXT,
                    minimum_effect_size REAL,
                    confidence_level REAL,
                    status TEXT
                )
            """)
            
            # A/B test results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    control_score REAL,
                    treatment_score REAL,
                    effect_size REAL,
                    p_value REAL,
                    confidence_interval TEXT,
                    is_significant BOOLEAN,
                    sample_sizes TEXT,
                    details TEXT,
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs (test_id)
                )
            """)
            
            # User feedback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    query TEXT NOT NULL,
                    recommended_sources TEXT,
                    user_rating INTEGER,
                    relevant_sources TEXT,
                    irrelevant_sources TEXT,
                    comments TEXT,
                    session_id TEXT
                )
            """)
            
            # A/B test assignments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    group_assignment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs (test_id)
                )
            """)
    
    def generate_quality_report(self, 
                              time_period: str = "24h",
                              include_trends: bool = True,
                              include_validation: bool = True) -> QualityReport:
        """Generate comprehensive quality report"""
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Parse time period
        if time_period == "24h":
            hours = 24
            days = 1
        elif time_period == "7d":
            hours = 24 * 7
            days = 7
        elif time_period == "30d":
            hours = 24 * 30
            days = 30
        else:
            hours = 24
            days = 1
        
        # Get quality metrics
        current_metrics = self.quality_monitor.calculate_quality_metrics(hours)
        
        # Get domain breakdown
        domain_breakdown = self.quality_monitor.get_domain_performance_breakdown(hours)
        
        # Get trends if requested
        trends = {}
        if include_trends:
            trends = self.quality_monitor.get_quality_trends(days)
        
        # Get validation results if requested
        validation_results = {}
        if include_validation:
            # Get recent validation results
            training_history = self.validator.get_validation_history("training_mappings_validation", days)
            domain_history = self.validator.get_validation_history("domain_routing_validation", days)
            singapore_history = self.validator.get_validation_history("singapore_first_validation", days)
            
            validation_results = {
                'training_mappings': training_history[0].passed if training_history else False,
                'domain_routing': domain_history[0].passed if domain_history else False,
                'singapore_first': singapore_history[0].passed if singapore_history else False
            }
        
        # Get user feedback summary
        user_feedback_summary = self._get_user_feedback_summary(hours)
        
        # Generate alerts
        alerts = self.quality_monitor.check_quality_thresholds(current_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_metrics, domain_breakdown, validation_results)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(current_metrics, validation_results, user_feedback_summary)
        
        # Create report
        report = QualityReport(
            report_id=report_id,
            timestamp=timestamp,
            time_period=time_period,
            overall_score=overall_score,
            metrics_summary={
                'ndcg_at_3': current_metrics.ndcg_at_3,
                'relevance_accuracy': current_metrics.relevance_accuracy,
                'domain_routing_accuracy': current_metrics.domain_routing_accuracy,
                'singapore_first_accuracy': current_metrics.singapore_first_accuracy,
                'cache_hit_rate': current_metrics.cache_hit_rate,
                'average_response_time': current_metrics.average_response_time
            },
            domain_breakdown=domain_breakdown,
            trends=trends,
            alerts=alerts,
            recommendations=recommendations,
            validation_results=validation_results,
            user_feedback_summary=user_feedback_summary
        )
        
        # Store report
        self._store_quality_report(report)
        
        return report
    
    def _calculate_overall_score(self, 
                               metrics: QualityMetrics,
                               validation_results: Dict[str, bool],
                               user_feedback: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        # Weight different components
        weights = {
            'ndcg': 0.25,
            'relevance': 0.20,
            'domain_routing': 0.15,
            'singapore_first': 0.10,
            'validation': 0.20,
            'user_feedback': 0.10
        }
        
        # Calculate component scores
        ndcg_score = min(metrics.ndcg_at_3 / 0.7, 1.0)  # Normalize to 0.7 target
        relevance_score = min(metrics.relevance_accuracy / 0.7, 1.0)
        domain_score = min(metrics.domain_routing_accuracy / 0.8, 1.0)
        singapore_score = metrics.singapore_first_accuracy
        
        # Validation score
        validation_score = sum(validation_results.values()) / len(validation_results) if validation_results else 0.5
        
        # User feedback score
        feedback_score = user_feedback.get('average_rating', 3.0) / 5.0 if user_feedback else 0.6
        
        # Calculate weighted average
        overall_score = (
            weights['ndcg'] * ndcg_score +
            weights['relevance'] * relevance_score +
            weights['domain_routing'] * domain_score +
            weights['singapore_first'] * singapore_score +
            weights['validation'] * validation_score +
            weights['user_feedback'] * feedback_score
        )
        
        return min(overall_score, 1.0)
    
    def _generate_recommendations(self, 
                                metrics: QualityMetrics,
                                domain_breakdown: Dict[str, Dict[str, float]],
                                validation_results: Dict[str, bool]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # NDCG recommendations
        if metrics.ndcg_at_3 < 0.7:
            recommendations.append(
                f"NDCG@3 is {metrics.ndcg_at_3:.3f}, below target of 0.7. "
                "Consider retraining with more relevant examples from training_mappings.md"
            )
        
        # Relevance accuracy recommendations
        if metrics.relevance_accuracy < 0.7:
            recommendations.append(
                f"Relevance accuracy is {metrics.relevance_accuracy:.3f}, below target of 0.7. "
                "Review and update training mappings for better source matching"
            )
        
        # Domain routing recommendations
        if metrics.domain_routing_accuracy < 0.8:
            recommendations.append(
                f"Domain routing accuracy is {metrics.domain_routing_accuracy:.3f}, below target of 0.8. "
                "Improve domain classification model or add more domain-specific training data"
            )
        
        # Performance recommendations
        if metrics.average_response_time > 4.0:
            recommendations.append(
                f"Average response time is {metrics.average_response_time:.2f}s, above target of 4s. "
                "Consider optimizing neural inference or implementing better caching"
            )
        
        # Cache recommendations
        if metrics.cache_hit_rate < 0.8:
            recommendations.append(
                f"Cache hit rate is {metrics.cache_hit_rate:.3f}, below target of 0.8. "
                "Implement cache warming for popular queries or adjust TTL settings"
            )
        
        # Domain-specific recommendations
        for domain, stats in domain_breakdown.items():
            if stats['avg_ndcg'] < 0.6:
                recommendations.append(
                    f"Domain '{domain}' has low NDCG ({stats['avg_ndcg']:.3f}). "
                    f"Add more training examples for {domain} queries"
                )
        
        # Validation recommendations
        if validation_results:
            failed_validations = [test for test, passed in validation_results.items() if not passed]
            if failed_validations:
                recommendations.append(
                    f"Validation tests failing: {', '.join(failed_validations)}. "
                    "Review and fix underlying issues"
                )
        
        return recommendations
    
    def _get_user_feedback_summary(self, time_window_hours: int) -> Dict[str, Any]:
        """Get user feedback summary for time window"""
        cutoff_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()
        
        with sqlite3.connect(self.reporting_db_path) as conn:
            cursor = conn.execute("""
                SELECT user_rating, relevant_sources, irrelevant_sources, comments
                FROM user_feedback 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            feedback_data = cursor.fetchall()
        
        if not feedback_data:
            return {}
        
        ratings = [row[0] for row in feedback_data if row[0] is not None]
        
        summary = {
            'total_feedback': len(feedback_data),
            'average_rating': statistics.mean(ratings) if ratings else 0.0,
            'rating_distribution': {
                str(i): ratings.count(i) for i in range(1, 6)
            },
            'satisfaction_rate': len([r for r in ratings if r >= 4]) / len(ratings) if ratings else 0.0
        }
        
        return summary
    
    def _store_quality_report(self, report: QualityReport):
        """Store quality report in database"""
        with sqlite3.connect(self.reporting_db_path) as conn:
            conn.execute("""
                INSERT INTO quality_reports 
                (report_id, timestamp, time_period, overall_score, metrics_summary,
                 domain_breakdown, trends, alerts, recommendations, validation_results,
                 user_feedback_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.timestamp,
                report.time_period,
                report.overall_score,
                json.dumps(report.metrics_summary),
                json.dumps(report.domain_breakdown),
                json.dumps(report.trends),
                json.dumps(report.alerts),
                json.dumps(report.recommendations),
                json.dumps(report.validation_results),
                json.dumps(report.user_feedback_summary)
            ))

class ABTestingFramework:
    """A/B testing framework for quality improvements"""
    
    def __init__(self, reporting_db_path: str = "data/quality_reporting.db"):
        self.reporting_db_path = reporting_db_path
        self.logger = logging.getLogger(__name__)
    
    def create_ab_test(self, 
                      test_name: str,
                      description: str,
                      success_metric: str,
                      control_group_size: int = 100,
                      treatment_group_size: int = 100,
                      minimum_effect_size: float = 0.05,
                      confidence_level: float = 0.95) -> str:
        """Create new A/B test"""
        test_id = str(uuid.uuid4())
        
        config = ABTestConfig(
            test_id=test_id,
            test_name=test_name,
            description=description,
            start_time=datetime.now().isoformat(),
            end_time=None,
            control_group_size=control_group_size,
            treatment_group_size=treatment_group_size,
            success_metric=success_metric,
            minimum_effect_size=minimum_effect_size,
            confidence_level=confidence_level,
            status='running'
        )
        
        with sqlite3.connect(self.reporting_db_path) as conn:
            conn.execute("""
                INSERT INTO ab_test_configs 
                (test_id, test_name, description, start_time, end_time,
                 control_group_size, treatment_group_size, success_metric,
                 minimum_effect_size, confidence_level, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.test_id,
                config.test_name,
                config.description,
                config.start_time,
                config.end_time,
                config.control_group_size,
                config.treatment_group_size,
                config.success_metric,
                config.minimum_effect_size,
                config.confidence_level,
                config.status
            ))
        
        self.logger.info(f"Created A/B test: {test_name} (ID: {test_id})")
        return test_id
    
    def assign_user_to_group(self, test_id: str, user_id: str = None, session_id: str = None) -> str:
        """Assign user to control or treatment group"""
        if not user_id and not session_id:
            session_id = str(uuid.uuid4())
        
        # Simple random assignment (50/50 split)
        import random
        group = "treatment" if random.random() < 0.5 else "control"
        
        with sqlite3.connect(self.reporting_db_path) as conn:
            conn.execute("""
                INSERT INTO ab_test_assignments 
                (test_id, user_id, session_id, group_assignment, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (test_id, user_id, session_id, group, datetime.now().isoformat()))
        
        return group
    
    def get_user_group(self, test_id: str, user_id: str = None, session_id: str = None) -> Optional[str]:
        """Get user's group assignment"""
        with sqlite3.connect(self.reporting_db_path) as conn:
            if user_id:
                cursor = conn.execute("""
                    SELECT group_assignment FROM ab_test_assignments 
                    WHERE test_id = ? AND user_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (test_id, user_id))
            elif session_id:
                cursor = conn.execute("""
                    SELECT group_assignment FROM ab_test_assignments 
                    WHERE test_id = ? AND session_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (test_id, session_id))
            else:
                return None
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def analyze_ab_test(self, test_id: str, 
                       control_scores: List[float],
                       treatment_scores: List[float]) -> ABTestResult:
        """Analyze A/B test results"""
        timestamp = datetime.now().isoformat()
        
        # Calculate basic statistics
        control_mean = np.mean(control_scores) if control_scores else 0.0
        treatment_mean = np.mean(treatment_scores) if treatment_scores else 0.0
        effect_size = treatment_mean - control_mean
        
        # Perform t-test (simplified)
        if len(control_scores) > 1 and len(treatment_scores) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)
            
            # Calculate confidence interval for effect size
            pooled_std = np.sqrt(
                ((len(control_scores) - 1) * np.var(control_scores, ddof=1) +
                 (len(treatment_scores) - 1) * np.var(treatment_scores, ddof=1)) /
                (len(control_scores) + len(treatment_scores) - 2)
            )
            
            se_diff = pooled_std * np.sqrt(1/len(control_scores) + 1/len(treatment_scores))
            margin_error = stats.t.ppf(0.975, len(control_scores) + len(treatment_scores) - 2) * se_diff
            
            confidence_interval = (effect_size - margin_error, effect_size + margin_error)
            is_significant = p_value < 0.05
        else:
            p_value = 1.0
            confidence_interval = (0.0, 0.0)
            is_significant = False
        
        result = ABTestResult(
            test_id=test_id,
            timestamp=timestamp,
            control_score=control_mean,
            treatment_score=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            sample_sizes={'control': len(control_scores), 'treatment': len(treatment_scores)},
            details={
                'control_std': np.std(control_scores) if control_scores else 0.0,
                'treatment_std': np.std(treatment_scores) if treatment_scores else 0.0
            }
        )
        
        # Store result
        with sqlite3.connect(self.reporting_db_path) as conn:
            conn.execute("""
                INSERT INTO ab_test_results 
                (test_id, timestamp, control_score, treatment_score, effect_size,
                 p_value, confidence_interval, is_significant, sample_sizes, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id,
                result.timestamp,
                result.control_score,
                result.treatment_score,
                result.effect_size,
                result.p_value,
                json.dumps(result.confidence_interval),
                result.is_significant,
                json.dumps(result.sample_sizes),
                json.dumps(result.details)
            ))
        
        return result

class UserFeedbackIntegration:
    """User feedback integration system"""
    
    def __init__(self, reporting_db_path: str = "data/quality_reporting.db"):
        self.reporting_db_path = reporting_db_path
        self.logger = logging.getLogger(__name__)
    
    def collect_feedback(self, 
                        query: str,
                        recommended_sources: List[str],
                        user_rating: int,
                        relevant_sources: List[str] = None,
                        irrelevant_sources: List[str] = None,
                        comments: str = None,
                        user_id: str = None,
                        session_id: str = None) -> str:
        """Collect user feedback"""
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            query=query,
            recommended_sources=recommended_sources,
            user_rating=user_rating,
            relevant_sources=relevant_sources or [],
            irrelevant_sources=irrelevant_sources or [],
            comments=comments,
            session_id=session_id
        )
        
        with sqlite3.connect(self.reporting_db_path) as conn:
            conn.execute("""
                INSERT INTO user_feedback 
                (feedback_id, timestamp, user_id, query, recommended_sources,
                 user_rating, relevant_sources, irrelevant_sources, comments, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.timestamp,
                feedback.user_id,
                feedback.query,
                json.dumps(feedback.recommended_sources),
                feedback.user_rating,
                json.dumps(feedback.relevant_sources),
                json.dumps(feedback.irrelevant_sources),
                feedback.comments,
                feedback.session_id
            ))
        
        self.logger.info(f"Collected feedback for query: {query} (rating: {user_rating})")
        return feedback_id
    
    def get_feedback_for_training(self, min_rating: int = 4) -> List[Dict[str, Any]]:
        """Get high-quality feedback for training data enhancement"""
        with sqlite3.connect(self.reporting_db_path) as conn:
            cursor = conn.execute("""
                SELECT query, recommended_sources, relevant_sources, irrelevant_sources, user_rating
                FROM user_feedback 
                WHERE user_rating >= ?
                ORDER BY timestamp DESC
            """, (min_rating,))
            
            feedback_data = []
            for row in cursor.fetchall():
                feedback_data.append({
                    'query': row[0],
                    'recommended_sources': json.loads(row[1]),
                    'relevant_sources': json.loads(row[2]),
                    'irrelevant_sources': json.loads(row[3]),
                    'user_rating': row[4]
                })
            
            return feedback_data
    
    def generate_training_mappings_from_feedback(self) -> List[str]:
        """Generate training mappings from user feedback"""
        high_quality_feedback = self.get_feedback_for_training(min_rating=4)
        
        mappings = []
        for feedback in high_quality_feedback:
            query = feedback['query']
            
            # Add positive mappings for relevant sources
            for source in feedback['relevant_sources']:
                relevance_score = 0.9 if feedback['user_rating'] == 5 else 0.8
                mapping = f"- {query} → {source} ({relevance_score}) - User confirmed relevance (rating: {feedback['user_rating']})"
                mappings.append(mapping)
            
            # Add negative mappings for irrelevant sources
            for source in feedback['irrelevant_sources']:
                relevance_score = 0.2
                mapping = f"- {query} → {source} ({relevance_score}) - User marked as irrelevant"
                mappings.append(mapping)
        
        return mappings