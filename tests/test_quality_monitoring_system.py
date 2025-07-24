"""
Test suite for Quality Monitoring System

Tests the core functionality of quality metrics collection,
domain routing accuracy monitoring, and Singapore-first strategy tracking.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime
from src.ai.quality_monitoring_system import (
    QualityMonitoringSystem, 
    QualityMetrics, 
    QueryEvaluation,
    TrainingMappingsParser
)

class TestTrainingMappingsParser:
    """Test the training mappings parser"""
    
    def test_parse_mappings(self):
        """Test parsing of training mappings"""
        # Create temporary mappings file
        mappings_content = """
# Training Data Mappings

## Psychology Queries
- psychology → kaggle (0.95) - Best platform for psychology datasets
- psychology → zenodo (0.90) - Academic repository with psychology research
- mental health data → kaggle (0.92) - Mental health datasets for analysis

## Singapore Queries  
- singapore data → data_gov_sg (0.98) - Official Singapore government data
- singapore statistics → singstat (0.97) - Singapore Department of Statistics
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(mappings_content)
            temp_file = f.name
            
        try:
            parser = TrainingMappingsParser(temp_file)
            ground_truth = parser.ground_truth
            
            # Test psychology mappings
            assert 'psychology' in ground_truth
            assert ground_truth['psychology']['kaggle'] == 0.95
            assert ground_truth['psychology']['zenodo'] == 0.90
            
            # Test mental health mappings
            assert 'mental health data' in ground_truth
            assert ground_truth['mental health data']['kaggle'] == 0.92
            
            # Test Singapore mappings
            assert 'singapore data' in ground_truth
            assert ground_truth['singapore data']['data_gov_sg'] == 0.98
            
        finally:
            os.unlink(temp_file)
    
    def test_get_expected_sources(self):
        """Test getting expected sources for queries"""
        mappings_content = """
- psychology → kaggle (0.95) - Best platform
- psychology → zenodo (0.90) - Academic repository
- machine learning → kaggle (0.98) - ML datasets
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(mappings_content)
            temp_file = f.name
            
        try:
            parser = TrainingMappingsParser(temp_file)
            
            # Test exact match
            sources = parser.get_expected_sources("psychology")
            assert sources['kaggle'] == 0.95
            assert sources['zenodo'] == 0.90
            
            # Test partial match
            sources = parser.get_expected_sources("psychology research")
            assert 'kaggle' in sources
            assert 'zenodo' in sources
            
        finally:
            os.unlink(temp_file)
    
    def test_classify_query_domain(self):
        """Test query domain classification"""
        parser = TrainingMappingsParser("nonexistent.md")  # Will use empty ground truth
        
        assert parser.classify_query_domain("psychology research") == "psychology"
        assert parser.classify_query_domain("machine learning datasets") == "machine_learning"
        assert parser.classify_query_domain("climate change data") == "climate"
        assert parser.classify_query_domain("singapore housing") == "singapore"
        assert parser.classify_query_domain("random query") == "general"
    
    def test_should_apply_singapore_first(self):
        """Test Singapore-first strategy detection"""
        parser = TrainingMappingsParser("nonexistent.md")
        
        assert parser.should_apply_singapore_first("singapore data") == True
        assert parser.should_apply_singapore_first("Singapore housing") == True
        assert parser.should_apply_singapore_first("psychology research") == False

class TestQualityMonitoringSystem:
    """Test the quality monitoring system"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create temporary mappings file
        mappings_content = """
- psychology → kaggle (0.95) - Best platform for psychology datasets
- psychology → zenodo (0.90) - Academic repository
- singapore data → data_gov_sg (0.98) - Official Singapore data
- climate data → world_bank (0.95) - Climate indicators
"""
        
        self.temp_mappings = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        self.temp_mappings.write(mappings_content)
        self.temp_mappings.close()
        
        self.monitor = QualityMonitoringSystem(self.db_path)
        self.monitor.mappings_parser = TrainingMappingsParser(self.temp_mappings.name)
        
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.db_path)
        os.unlink(self.temp_mappings.name)
    
    def test_calculate_ndcg_at_k(self):
        """Test NDCG@k calculation"""
        predicted_sources = ['kaggle', 'zenodo', 'world_bank']
        relevance_scores = {'kaggle': 0.95, 'zenodo': 0.90, 'world_bank': 0.30}
        
        ndcg = self.monitor.calculate_ndcg_at_k(predicted_sources, relevance_scores, k=3)
        
        # Should be high since top sources are highly relevant
        assert ndcg > 0.8
        assert ndcg <= 1.0
    
    def test_evaluate_query(self):
        """Test query evaluation"""
        evaluation = self.monitor.evaluate_query(
            query="psychology research",
            predicted_sources=['kaggle', 'zenodo', 'world_bank'],
            domain_classification="psychology",
            singapore_first_applied=False,
            response_time=1.5,
            cache_hit=False
        )
        
        assert evaluation.query == "psychology research"
        assert evaluation.domain_classification == "psychology"
        assert evaluation.singapore_first_applied == False
        assert evaluation.response_time == 1.5
        assert evaluation.cache_hit == False
        assert evaluation.ndcg_score > 0  # Should have some relevance
        
    def test_calculate_quality_metrics(self):
        """Test quality metrics calculation"""
        # Add some test evaluations
        self.monitor.evaluate_query(
            "psychology", ['kaggle', 'zenodo'], "psychology", False, 1.0, False
        )
        self.monitor.evaluate_query(
            "singapore data", ['data_gov_sg', 'kaggle'], "singapore", True, 1.2, True
        )
        
        metrics = self.monitor.calculate_quality_metrics(time_window_hours=1)
        
        assert metrics.total_queries == 2
        assert metrics.ndcg_at_3 > 0
        assert metrics.relevance_accuracy >= 0
        assert metrics.domain_routing_accuracy >= 0
        assert metrics.singapore_first_accuracy >= 0
        assert metrics.cache_hit_rate == 0.5  # One cache hit out of two
    
    def test_check_quality_thresholds(self):
        """Test quality threshold checking"""
        # Create metrics below thresholds
        metrics = QualityMetrics(
            timestamp=datetime.now().isoformat(),
            ndcg_at_3=0.5,  # Below 0.7 threshold
            relevance_accuracy=0.6,  # Below 0.7 threshold
            domain_routing_accuracy=0.7,  # Below 0.8 threshold
            singapore_first_accuracy=0.9,
            total_queries=10,
            successful_queries=6,
            average_response_time=1.5,
            cache_hit_rate=0.8
        )
        
        alerts = self.monitor.check_quality_thresholds(metrics)
        
        assert len(alerts) == 3  # Should trigger 3 alerts
        assert any("NDCG@3 below threshold" in alert for alert in alerts)
        assert any("Relevance accuracy below threshold" in alert for alert in alerts)
        assert any("Domain routing accuracy below threshold" in alert for alert in alerts)
    
    def test_get_domain_performance_breakdown(self):
        """Test domain performance breakdown"""
        # Add evaluations for different domains
        self.monitor.evaluate_query(
            "psychology", ['kaggle', 'zenodo'], "psychology", False, 1.0, False
        )
        self.monitor.evaluate_query(
            "climate data", ['world_bank', 'kaggle'], "climate", False, 1.1, False
        )
        
        breakdown = self.monitor.get_domain_performance_breakdown(time_window_hours=1)
        
        assert 'psychology' in breakdown
        assert 'climate' in breakdown
        assert breakdown['psychology']['query_count'] == 1
        assert breakdown['climate']['query_count'] == 1
        assert 'avg_ndcg' in breakdown['psychology']
        assert 'relevance_accuracy' in breakdown['psychology']
    
    def test_monitor_recommendation_quality(self):
        """Test main monitoring function"""
        # Add some test data
        self.monitor.evaluate_query(
            "psychology", ['kaggle', 'zenodo'], "psychology", False, 1.0, False
        )
        
        metrics = self.monitor.monitor_recommendation_quality()
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.total_queries > 0

def test_quality_metrics_dataclass():
    """Test QualityMetrics dataclass functionality"""
    metrics = QualityMetrics(
        timestamp="2024-01-01T00:00:00",
        ndcg_at_3=0.8,
        relevance_accuracy=0.75,
        domain_routing_accuracy=0.85,
        singapore_first_accuracy=0.9,
        total_queries=100,
        successful_queries=80,
        average_response_time=1.5,
        cache_hit_rate=0.7
    )
    
    # Test meets_quality_threshold
    assert metrics.meets_quality_threshold(0.7) == True
    assert metrics.meets_quality_threshold(0.9) == False
    
    # Test to_dict conversion
    metrics_dict = metrics.to_dict()
    assert metrics_dict['ndcg_at_3'] == 0.8
    assert metrics_dict['total_queries'] == 100

if __name__ == "__main__":
    # Run basic functionality test
    print("Testing Quality Monitoring System...")
    
    # Test TrainingMappingsParser
    print("✓ Testing TrainingMappingsParser...")
    test_parser = TestTrainingMappingsParser()
    test_parser.test_classify_query_domain()
    test_parser.test_should_apply_singapore_first()
    
    # Test QualityMonitoringSystem
    print("✓ Testing QualityMonitoringSystem...")
    test_monitor = TestQualityMonitoringSystem()
    test_monitor.setup_method()
    try:
        test_monitor.test_calculate_ndcg_at_k()
        test_monitor.test_evaluate_query()
        test_monitor.test_calculate_quality_metrics()
        test_monitor.test_check_quality_thresholds()
        test_monitor.test_get_domain_performance_breakdown()
        test_monitor.test_monitor_recommendation_quality()
    finally:
        test_monitor.teardown_method()
    
    # Test QualityMetrics
    print("✓ Testing QualityMetrics...")
    test_quality_metrics_dataclass()
    
    print("All tests passed! ✅")