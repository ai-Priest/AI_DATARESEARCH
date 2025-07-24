"""
Test suite for Quality Reporting and Analytics System

Tests quality reporting, A/B testing framework, and user feedback integration.
"""

import tempfile
import os
import json
from datetime import datetime
from src.ai.quality_reporting_analytics import (
    QualityReportingSystem,
    ABTestingFramework,
    UserFeedbackIntegration,
    QualityReport,
    ABTestResult,
    UserFeedback
)
from src.ai.quality_monitoring_system import QualityMonitoringSystem
from src.ai.automated_quality_validator import AutomatedQualityValidator

class TestQualityReportingSystem:
    """Test the quality reporting system"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary databases
        self.temp_quality_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_quality_db.close()
        
        self.temp_validation_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_validation_db.close()
        
        self.temp_reporting_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_reporting_db.close()
        
        # Create temporary mappings file
        mappings_content = """
- psychology → kaggle (0.95) - Best platform for psychology datasets
- psychology → zenodo (0.90) - Academic repository
- machine learning → kaggle (0.98) - ML datasets
- singapore data → data_gov_sg (0.98) - Official Singapore data
- climate data → world_bank (0.95) - Climate indicators
"""
        
        self.temp_mappings = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        self.temp_mappings.write(mappings_content)
        self.temp_mappings.close()
        
        # Setup components
        self.quality_monitor = QualityMonitoringSystem(self.temp_quality_db.name)
        self.quality_monitor.mappings_parser.mappings_file = self.temp_mappings.name
        self.quality_monitor.mappings_parser.ground_truth = self.quality_monitor.mappings_parser._parse_mappings()
        
        self.validator = AutomatedQualityValidator(
            self.quality_monitor,
            self.temp_validation_db.name
        )
        
        self.reporting_system = QualityReportingSystem(
            self.quality_monitor,
            self.validator,
            self.temp_reporting_db.name
        )
        
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.temp_quality_db.name)
        os.unlink(self.temp_validation_db.name)
        os.unlink(self.temp_reporting_db.name)
        os.unlink(self.temp_mappings.name)
    
    def test_generate_quality_report(self):
        """Test quality report generation"""
        # Add some test data
        self.quality_monitor.evaluate_query(
            "psychology", ['kaggle', 'zenodo'], "psychology", False, 1.0, False
        )
        self.quality_monitor.evaluate_query(
            "singapore data", ['data_gov_sg', 'kaggle'], "singapore", True, 1.2, True
        )
        
        # Generate report
        report = self.reporting_system.generate_quality_report(
            time_period="24h",
            include_trends=True,
            include_validation=True
        )
        
        assert isinstance(report, QualityReport)
        assert report.report_id is not None
        assert report.time_period == "24h"
        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        assert 'ndcg_at_3' in report.metrics_summary
        assert 'relevance_accuracy' in report.metrics_summary
        assert isinstance(report.recommendations, list)
        assert isinstance(report.alerts, list)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        from src.ai.quality_monitoring_system import QualityMetrics
        
        # Create test metrics
        metrics = QualityMetrics(
            timestamp=datetime.now().isoformat(),
            ndcg_at_3=0.8,
            relevance_accuracy=0.75,
            domain_routing_accuracy=0.85,
            singapore_first_accuracy=0.9,
            total_queries=100,
            successful_queries=80,
            average_response_time=1.5,
            cache_hit_rate=0.8
        )
        
        validation_results = {
            'training_mappings': True,
            'domain_routing': True,
            'singapore_first': True
        }
        
        user_feedback = {
            'average_rating': 4.2,
            'total_feedback': 50
        }
        
        score = self.reporting_system._calculate_overall_score(
            metrics, validation_results, user_feedback
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high with good metrics
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        from src.ai.quality_monitoring_system import QualityMetrics
        
        # Create metrics with issues
        metrics = QualityMetrics(
            timestamp=datetime.now().isoformat(),
            ndcg_at_3=0.6,  # Below threshold
            relevance_accuracy=0.65,  # Below threshold
            domain_routing_accuracy=0.75,  # Below threshold
            singapore_first_accuracy=0.9,
            total_queries=100,
            successful_queries=65,
            average_response_time=5.0,  # Above threshold
            cache_hit_rate=0.6  # Below threshold
        )
        
        domain_breakdown = {
            'psychology': {'avg_ndcg': 0.5, 'relevance_accuracy': 0.6, 'query_count': 10}
        }
        
        validation_results = {
            'training_mappings': False,
            'domain_routing': True,
            'singapore_first': False
        }
        
        recommendations = self.reporting_system._generate_recommendations(
            metrics, domain_breakdown, validation_results
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations address the issues
        rec_text = ' '.join(recommendations).lower()
        assert 'ndcg' in rec_text
        assert 'relevance' in rec_text
        assert 'domain' in rec_text
        assert 'response time' in rec_text
        assert 'cache' in rec_text

class TestABTestingFramework:
    """Test the A/B testing framework"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.ab_framework = ABTestingFramework(self.temp_db.name)
        
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.temp_db.name)
    
    def test_create_ab_test(self):
        """Test A/B test creation"""
        test_id = self.ab_framework.create_ab_test(
            test_name="Neural Model Improvement",
            description="Test improved neural model vs baseline",
            success_metric="ndcg_at_3",
            control_group_size=100,
            treatment_group_size=100
        )
        
        assert test_id is not None
        assert len(test_id) > 0
    
    def test_assign_user_to_group(self):
        """Test user group assignment"""
        test_id = self.ab_framework.create_ab_test(
            "Test Assignment",
            "Test user assignment",
            "ndcg_at_3"
        )
        
        # Test assignment with user ID
        group1 = self.ab_framework.assign_user_to_group(test_id, user_id="user123")
        assert group1 in ["control", "treatment"]
        
        # Test assignment with session ID
        group2 = self.ab_framework.assign_user_to_group(test_id, session_id="session456")
        assert group2 in ["control", "treatment"]
        
        # Test retrieval
        retrieved_group1 = self.ab_framework.get_user_group(test_id, user_id="user123")
        assert retrieved_group1 == group1
        
        retrieved_group2 = self.ab_framework.get_user_group(test_id, session_id="session456")
        assert retrieved_group2 == group2
    
    def test_analyze_ab_test(self):
        """Test A/B test analysis"""
        test_id = self.ab_framework.create_ab_test(
            "Analysis Test",
            "Test analysis functionality",
            "ndcg_at_3"
        )
        
        # Create sample data
        control_scores = [0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72, 0.69, 0.71]
        treatment_scores = [0.75, 0.77, 0.73, 0.76, 0.74, 0.78, 0.75, 0.77, 0.74, 0.76]
        
        result = self.ab_framework.analyze_ab_test(test_id, control_scores, treatment_scores)
        
        assert isinstance(result, ABTestResult)
        assert result.test_id == test_id
        assert result.control_score > 0
        assert result.treatment_score > 0
        assert result.effect_size > 0  # Treatment should be better
        assert result.sample_sizes['control'] == len(control_scores)
        assert result.sample_sizes['treatment'] == len(treatment_scores)

class TestUserFeedbackIntegration:
    """Test the user feedback integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.feedback_system = UserFeedbackIntegration(self.temp_db.name)
        
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.temp_db.name)
    
    def test_collect_feedback(self):
        """Test feedback collection"""
        feedback_id = self.feedback_system.collect_feedback(
            query="psychology research",
            recommended_sources=["kaggle", "zenodo", "world_bank"],
            user_rating=5,
            relevant_sources=["kaggle", "zenodo"],
            irrelevant_sources=["world_bank"],
            comments="Great recommendations for psychology data!",
            user_id="user123",
            session_id="session456"
        )
        
        assert feedback_id is not None
        assert len(feedback_id) > 0
    
    def test_get_feedback_for_training(self):
        """Test getting feedback for training"""
        # Add some feedback
        self.feedback_system.collect_feedback(
            "psychology research", ["kaggle", "zenodo"], 5,
            relevant_sources=["kaggle"], user_id="user1"
        )
        
        self.feedback_system.collect_feedback(
            "climate data", ["world_bank", "kaggle"], 4,
            relevant_sources=["world_bank"], user_id="user2"
        )
        
        self.feedback_system.collect_feedback(
            "poor query", ["random_source"], 2,
            irrelevant_sources=["random_source"], user_id="user3"
        )
        
        # Get high-quality feedback
        training_feedback = self.feedback_system.get_feedback_for_training(min_rating=4)
        
        assert len(training_feedback) == 2  # Only ratings 4 and 5
        assert all(fb['user_rating'] >= 4 for fb in training_feedback)
    
    def test_generate_training_mappings_from_feedback(self):
        """Test generating training mappings from feedback"""
        # Add feedback
        self.feedback_system.collect_feedback(
            "psychology research", ["kaggle", "zenodo", "world_bank"], 5,
            relevant_sources=["kaggle", "zenodo"],
            irrelevant_sources=["world_bank"]
        )
        
        mappings = self.feedback_system.generate_training_mappings_from_feedback()
        
        assert len(mappings) > 0
        assert any("psychology research → kaggle" in mapping for mapping in mappings)
        assert any("psychology research → zenodo" in mapping for mapping in mappings)
        assert any("psychology research → world_bank (0.2)" in mapping for mapping in mappings)

def test_quality_report_dataclass():
    """Test QualityReport dataclass"""
    report = QualityReport(
        report_id="test-123",
        timestamp="2024-01-01T00:00:00",
        time_period="24h",
        overall_score=0.85,
        metrics_summary={"ndcg_at_3": 0.8},
        domain_breakdown={"psychology": {"avg_ndcg": 0.9}},
        trends={"ndcg_at_3": [0.7, 0.8, 0.85]},
        alerts=["No alerts"],
        recommendations=["Keep up the good work"],
        validation_results={"training_mappings": True},
        user_feedback_summary={"average_rating": 4.5}
    )
    
    assert report.report_id == "test-123"
    assert report.overall_score == 0.85

def test_user_feedback_dataclass():
    """Test UserFeedback dataclass"""
    feedback = UserFeedback(
        feedback_id="feedback-123",
        timestamp="2024-01-01T00:00:00",
        user_id="user123",
        query="test query",
        recommended_sources=["kaggle", "zenodo"],
        user_rating=5,
        relevant_sources=["kaggle"],
        irrelevant_sources=["zenodo"],
        comments="Good results",
        session_id="session123"
    )
    
    assert feedback.feedback_id == "feedback-123"
    assert feedback.user_rating == 5

if __name__ == "__main__":
    # Run basic functionality test
    print("Testing Quality Reporting and Analytics System...")
    
    # Test dataclasses
    print("✓ Testing QualityReport dataclass...")
    test_quality_report_dataclass()
    
    print("✓ Testing UserFeedback dataclass...")
    test_user_feedback_dataclass()
    
    # Test QualityReportingSystem
    print("✓ Testing QualityReportingSystem...")
    test_reporting = TestQualityReportingSystem()
    test_reporting.setup_method()
    try:
        test_reporting.test_generate_quality_report()
        test_reporting.test_calculate_overall_score()
        test_reporting.test_generate_recommendations()
    finally:
        test_reporting.teardown_method()
    
    # Test ABTestingFramework
    print("✓ Testing ABTestingFramework...")
    test_ab = TestABTestingFramework()
    test_ab.setup_method()
    try:
        test_ab.test_create_ab_test()
        test_ab.test_assign_user_to_group()
        # Skip analyze test due to scipy dependency
        # test_ab.test_analyze_ab_test()
    finally:
        test_ab.teardown_method()
    
    # Test UserFeedbackIntegration
    print("✓ Testing UserFeedbackIntegration...")
    test_feedback = TestUserFeedbackIntegration()
    test_feedback.setup_method()
    try:
        test_feedback.test_collect_feedback()
        test_feedback.test_get_feedback_for_training()
        test_feedback.test_generate_training_mappings_from_feedback()
    finally:
        test_feedback.teardown_method()
    
    print("All tests passed! ✅")