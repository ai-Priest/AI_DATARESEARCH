"""
Automated Quality Validation System

This module provides automated testing against training_mappings.md ground truth,
regression testing for quality improvements, and continuous validation of 
recommendation relevance.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict
import asyncio
import time

from .quality_monitoring_system import (
    QualityMonitoringSystem, 
    TrainingMappingsParser, 
    QualityMetrics
)

@dataclass
class ValidationResult:
    """Result of a quality validation test"""
    test_name: str
    timestamp: str
    passed: bool
    score: float
    expected_score: float
    threshold: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class RegressionTestResult:
    """Result of a regression test comparing current vs baseline performance"""
    test_name: str
    timestamp: str
    current_score: float
    baseline_score: float
    improvement: float
    passed: bool
    threshold: float
    details: Dict[str, Any]

class AutomatedQualityValidator:
    """Automated quality validation system"""
    
    def __init__(self, 
                 quality_monitor: QualityMonitoringSystem,
                 validation_db_path: str = "data/quality_validation.db"):
        self.quality_monitor = quality_monitor
        self.validation_db_path = validation_db_path
        self.logger = logging.getLogger(__name__)
        self._init_validation_database()
        
    def _init_validation_database(self):
        """Initialize validation database"""
        Path(self.validation_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.validation_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    score REAL,
                    expected_score REAL,
                    threshold REAL,
                    details TEXT,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_score REAL,
                    baseline_score REAL,
                    improvement REAL,
                    passed BOOLEAN NOT NULL,
                    threshold REAL,
                    details TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    version TEXT,
                    description TEXT
                )
            """)
    
    def validate_against_training_mappings(self, 
                                         recommendation_function: Callable[[str], List[str]],
                                         sample_size: int = 50) -> ValidationResult:
        """
        Validate recommendations against training_mappings.md ground truth
        
        Args:
            recommendation_function: Function that takes query and returns list of sources
            sample_size: Number of queries to test
        """
        test_name = "training_mappings_validation"
        timestamp = datetime.now().isoformat()
        
        try:
            # Get sample queries from training mappings
            ground_truth = self.quality_monitor.mappings_parser.ground_truth
            sample_queries = list(ground_truth.keys())[:sample_size]
            
            if not sample_queries:
                return ValidationResult(
                    test_name=test_name,
                    timestamp=timestamp,
                    passed=False,
                    score=0.0,
                    expected_score=0.7,
                    threshold=0.7,
                    details={},
                    error_message="No training mappings found"
                )
            
            total_ndcg = 0.0
            correct_predictions = 0
            validation_details = {
                'tested_queries': len(sample_queries),
                'query_results': []
            }
            
            for query in sample_queries:
                try:
                    # Get predictions from the recommendation function
                    predicted_sources = recommendation_function(query)
                    expected_sources = ground_truth[query]
                    
                    # Calculate NDCG@3 for this query
                    ndcg = self.quality_monitor.calculate_ndcg_at_k(
                        predicted_sources, expected_sources, k=3
                    )
                    total_ndcg += ndcg
                    
                    # Check if at least one highly relevant source is in top 3
                    top_3 = predicted_sources[:3]
                    has_relevant = any(
                        expected_sources.get(source, 0.0) >= 0.8 
                        for source in top_3
                    )
                    
                    if has_relevant:
                        correct_predictions += 1
                    
                    validation_details['query_results'].append({
                        'query': query,
                        'predicted': predicted_sources[:3],
                        'expected_top_sources': [
                            s for s, score in sorted(expected_sources.items(), 
                                                   key=lambda x: x[1], reverse=True)[:3]
                        ],
                        'ndcg': ndcg,
                        'has_relevant': has_relevant
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error validating query '{query}': {e}")
                    continue
            
            # Calculate overall scores
            avg_ndcg = total_ndcg / len(sample_queries) if sample_queries else 0.0
            relevance_accuracy = correct_predictions / len(sample_queries) if sample_queries else 0.0
            
            # Combined score (weighted average)
            combined_score = (avg_ndcg * 0.6) + (relevance_accuracy * 0.4)
            
            validation_details.update({
                'avg_ndcg': avg_ndcg,
                'relevance_accuracy': relevance_accuracy,
                'combined_score': combined_score
            })
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=combined_score >= 0.7,
                score=combined_score,
                expected_score=0.7,
                threshold=0.7,
                details=validation_details
            )
            
            self._store_validation_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=False,
                score=0.0,
                expected_score=0.7,
                threshold=0.7,
                details={},
                error_message=error_msg
            )
            
            self._store_validation_result(result)
            return result
    
    def validate_domain_routing_accuracy(self, 
                                       domain_classifier: Callable[[str], str],
                                       sample_size: int = 30) -> ValidationResult:
        """Validate domain classification accuracy"""
        test_name = "domain_routing_validation"
        timestamp = datetime.now().isoformat()
        
        try:
            # Create test queries for each domain
            test_queries = [
                ("psychology research", "psychology"),
                ("mental health data", "psychology"),
                ("machine learning datasets", "machine_learning"),
                ("artificial intelligence", "machine_learning"),
                ("climate change data", "climate"),
                ("environmental statistics", "climate"),
                ("singapore housing data", "singapore"),
                ("singapore demographics", "singapore"),
                ("economic indicators", "economics"),
                ("gdp statistics", "economics"),
                ("health data analysis", "health"),
                ("medical research", "health"),
                ("education statistics", "education"),
                ("student performance", "education")
            ]
            
            correct_classifications = 0
            total_tests = min(len(test_queries), sample_size)
            classification_details = []
            
            for query, expected_domain in test_queries[:total_tests]:
                try:
                    predicted_domain = domain_classifier(query)
                    is_correct = predicted_domain == expected_domain
                    
                    if is_correct:
                        correct_classifications += 1
                    
                    classification_details.append({
                        'query': query,
                        'expected': expected_domain,
                        'predicted': predicted_domain,
                        'correct': is_correct
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error classifying query '{query}': {e}")
                    continue
            
            accuracy = correct_classifications / total_tests if total_tests > 0 else 0.0
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=accuracy >= 0.8,
                score=accuracy,
                expected_score=0.8,
                threshold=0.8,
                details={
                    'total_tests': total_tests,
                    'correct_classifications': correct_classifications,
                    'accuracy': accuracy,
                    'classification_details': classification_details
                }
            )
            
            self._store_validation_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Domain routing validation failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=False,
                score=0.0,
                expected_score=0.8,
                threshold=0.8,
                details={},
                error_message=error_msg
            )
            
            self._store_validation_result(result)
            return result
    
    def validate_singapore_first_strategy(self, 
                                        singapore_detector: Callable[[str], bool],
                                        source_ranker: Callable[[str, List[str]], List[str]]) -> ValidationResult:
        """Validate Singapore-first strategy effectiveness"""
        test_name = "singapore_first_validation"
        timestamp = datetime.now().isoformat()
        
        try:
            # Test queries for Singapore-first strategy
            singapore_queries = [
                "singapore housing data",
                "singapore demographics",
                "singapore transport statistics",
                "singapore economic indicators",
                "singapore education data"
            ]
            
            non_singapore_queries = [
                "global climate data",
                "international trade statistics",
                "world population data",
                "european economic indicators"
            ]
            
            singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
            other_sources = ['kaggle', 'zenodo', 'world_bank', 'data_un']
            
            correct_detections = 0
            correct_prioritizations = 0
            total_tests = 0
            validation_details = []
            
            # Test Singapore query detection
            for query in singapore_queries:
                is_singapore = singapore_detector(query)
                correct_detections += 1 if is_singapore else 0
                total_tests += 1
                
                # Test source prioritization for Singapore queries
                if is_singapore:
                    all_sources = singapore_sources + other_sources
                    ranked_sources = source_ranker(query, all_sources)
                    
                    # Check if Singapore sources are prioritized
                    singapore_in_top_3 = any(
                        source in singapore_sources 
                        for source in ranked_sources[:3]
                    )
                    
                    if singapore_in_top_3:
                        correct_prioritizations += 1
                    
                    validation_details.append({
                        'query': query,
                        'detected_singapore': is_singapore,
                        'ranked_sources': ranked_sources[:5],
                        'singapore_prioritized': singapore_in_top_3
                    })
            
            # Test non-Singapore queries
            for query in non_singapore_queries:
                is_singapore = singapore_detector(query)
                correct_detections += 1 if not is_singapore else 0
                total_tests += 1
                
                validation_details.append({
                    'query': query,
                    'detected_singapore': is_singapore,
                    'should_be_singapore': False
                })
            
            detection_accuracy = correct_detections / total_tests if total_tests > 0 else 0.0
            prioritization_accuracy = correct_prioritizations / len(singapore_queries) if singapore_queries else 0.0
            
            # Combined score
            combined_score = (detection_accuracy * 0.5) + (prioritization_accuracy * 0.5)
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=combined_score >= 0.8,
                score=combined_score,
                expected_score=0.8,
                threshold=0.8,
                details={
                    'detection_accuracy': detection_accuracy,
                    'prioritization_accuracy': prioritization_accuracy,
                    'combined_score': combined_score,
                    'total_tests': total_tests,
                    'validation_details': validation_details
                }
            )
            
            self._store_validation_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Singapore-first validation failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = ValidationResult(
                test_name=test_name,
                timestamp=timestamp,
                passed=False,
                score=0.0,
                expected_score=0.8,
                threshold=0.8,
                details={},
                error_message=error_msg
            )
            
            self._store_validation_result(result)
            return result
    
    def run_regression_test(self, 
                           test_name: str,
                           current_metric_function: Callable[[], float],
                           baseline_metric_name: str,
                           improvement_threshold: float = 0.05) -> RegressionTestResult:
        """
        Run regression test comparing current performance to baseline
        
        Args:
            test_name: Name of the regression test
            current_metric_function: Function that returns current metric value
            baseline_metric_name: Name of baseline metric to compare against
            improvement_threshold: Minimum improvement required to pass
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Get current metric value
            current_score = current_metric_function()
            
            # Get baseline metric value
            baseline_score = self._get_baseline_metric(baseline_metric_name)
            
            if baseline_score is None:
                # No baseline found, store current as baseline
                self._store_baseline_metric(baseline_metric_name, current_score)
                improvement = 0.0
                passed = True  # First run always passes
            else:
                improvement = current_score - baseline_score
                passed = improvement >= -improvement_threshold  # Allow small degradation
            
            result = RegressionTestResult(
                test_name=test_name,
                timestamp=timestamp,
                current_score=current_score,
                baseline_score=baseline_score or current_score,
                improvement=improvement,
                passed=passed,
                threshold=improvement_threshold,
                details={
                    'baseline_metric_name': baseline_metric_name,
                    'improvement_percentage': (improvement / baseline_score * 100) if baseline_score else 0.0
                }
            )
            
            self._store_regression_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Regression test '{test_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = RegressionTestResult(
                test_name=test_name,
                timestamp=timestamp,
                current_score=0.0,
                baseline_score=0.0,
                improvement=0.0,
                passed=False,
                threshold=improvement_threshold,
                details={'error': error_msg}
            )
            
            self._store_regression_result(result)
            return result
    
    def continuous_validation_suite(self, 
                                  recommendation_function: Callable[[str], List[str]],
                                  domain_classifier: Callable[[str], str],
                                  singapore_detector: Callable[[str], bool],
                                  source_ranker: Callable[[str, List[str]], List[str]]) -> Dict[str, ValidationResult]:
        """Run complete validation suite"""
        results = {}
        
        # Run all validation tests
        results['training_mappings'] = self.validate_against_training_mappings(
            recommendation_function, sample_size=30
        )
        
        results['domain_routing'] = self.validate_domain_routing_accuracy(
            domain_classifier, sample_size=20
        )
        
        results['singapore_first'] = self.validate_singapore_first_strategy(
            singapore_detector, source_ranker
        )
        
        # Log results
        passed_tests = sum(1 for result in results.values() if result.passed)
        total_tests = len(results)
        
        self.logger.info(f"Validation suite completed: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in results.items():
            if result.passed:
                self.logger.info(f"✅ {test_name}: PASSED (score: {result.score:.3f})")
            else:
                self.logger.warning(f"❌ {test_name}: FAILED (score: {result.score:.3f}, threshold: {result.threshold})")
        
        return results
    
    def _store_validation_result(self, result: ValidationResult):
        """Store validation result in database"""
        with sqlite3.connect(self.validation_db_path) as conn:
            conn.execute("""
                INSERT INTO validation_results 
                (test_name, timestamp, passed, score, expected_score, threshold, details, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_name,
                result.timestamp,
                result.passed,
                result.score,
                result.expected_score,
                result.threshold,
                json.dumps(result.details),
                result.error_message
            ))
    
    def _store_regression_result(self, result: RegressionTestResult):
        """Store regression test result in database"""
        with sqlite3.connect(self.validation_db_path) as conn:
            conn.execute("""
                INSERT INTO regression_tests 
                (test_name, timestamp, current_score, baseline_score, improvement, passed, threshold, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_name,
                result.timestamp,
                result.current_score,
                result.baseline_score,
                result.improvement,
                result.passed,
                result.threshold,
                json.dumps(result.details)
            ))
    
    def _store_baseline_metric(self, metric_name: str, value: float, version: str = "1.0"):
        """Store baseline metric value"""
        with sqlite3.connect(self.validation_db_path) as conn:
            conn.execute("""
                INSERT INTO baseline_metrics (metric_name, timestamp, value, version)
                VALUES (?, ?, ?, ?)
            """, (metric_name, datetime.now().isoformat(), value, version))
    
    def _get_baseline_metric(self, metric_name: str) -> Optional[float]:
        """Get latest baseline metric value"""
        with sqlite3.connect(self.validation_db_path) as conn:
            cursor = conn.execute("""
                SELECT value FROM baseline_metrics 
                WHERE metric_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (metric_name,))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_validation_history(self, test_name: str, days: int = 7) -> List[ValidationResult]:
        """Get validation history for a specific test"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.validation_db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, timestamp, passed, score, expected_score, threshold, details, error_message
                FROM validation_results 
                WHERE test_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (test_name, cutoff_time))
            
            results = []
            for row in cursor.fetchall():
                results.append(ValidationResult(
                    test_name=row[0],
                    timestamp=row[1],
                    passed=bool(row[2]),
                    score=row[3],
                    expected_score=row[4],
                    threshold=row[5],
                    details=json.loads(row[6]) if row[6] else {},
                    error_message=row[7]
                ))
            
            return results
    
    def get_regression_history(self, test_name: str, days: int = 7) -> List[RegressionTestResult]:
        """Get regression test history"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.validation_db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, timestamp, current_score, baseline_score, improvement, passed, threshold, details
                FROM regression_tests 
                WHERE test_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (test_name, cutoff_time))
            
            results = []
            for row in cursor.fetchall():
                results.append(RegressionTestResult(
                    test_name=row[0],
                    timestamp=row[1],
                    current_score=row[2],
                    baseline_score=row[3],
                    improvement=row[4],
                    passed=bool(row[5]),
                    threshold=row[6],
                    details=json.loads(row[7]) if row[7] else {}
                ))
            
            return results

class ContinuousValidator:
    """Continuous validation runner"""
    
    def __init__(self, validator: AutomatedQualityValidator):
        self.validator = validator
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def run_continuous_validation(self, 
                                      recommendation_function: Callable[[str], List[str]],
                                      domain_classifier: Callable[[str], str],
                                      singapore_detector: Callable[[str], bool],
                                      source_ranker: Callable[[str, List[str]], List[str]],
                                      interval_minutes: int = 60):
        """Run continuous validation at specified intervals"""
        self.running = True
        self.logger.info(f"Starting continuous validation (interval: {interval_minutes} minutes)")
        
        while self.running:
            try:
                # Run validation suite
                results = self.validator.continuous_validation_suite(
                    recommendation_function,
                    domain_classifier,
                    singapore_detector,
                    source_ranker
                )
                
                # Check for failures
                failed_tests = [name for name, result in results.items() if not result.passed]
                if failed_tests:
                    self.logger.warning(f"Validation failures detected: {failed_tests}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Continuous validation error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop(self):
        """Stop continuous validation"""
        self.running = False
        self.logger.info("Stopping continuous validation")