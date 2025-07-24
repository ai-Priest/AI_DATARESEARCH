"""
Test suite for Automated Quality Validator

Tests automated testing against training_mappings.md ground truth,
regression testing, and continuous validation functionality.
"""

import tempfile
import os
import json
from datetime import datetime
from src.ai.automated_quality_validator import (
    AutomatedQualityValidator,
    ValidationResult,
    RegressionTestResult,
    ContinuousValidator
)
from src.ai.quality_monitoring_system import QualityMonitoringSystem

class TestAutomatedQualityValidator:
    """Test the automated quality validator"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary databases
        self.temp_quality_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_quality_db.close()
        
        self.temp_validation_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_validation_db.close()
        
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
        
        # Setup quality monitor and validator
        self.quality_monitor = QualityMonitoringSystem(self.temp_quality_db.name)
        self.quality_monitor.mappings_parser.mappings_file = self.temp_mappings.name
        self.quality_monitor.mappings_parser.ground_truth = self.quality_monitor.mappings_parser._parse_mappings()
        
        self.validator = AutomatedQualityValidator(
            self.quality_monitor,
            self.temp_validation_db.name
        )
        
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.temp_quality_db.name)
        os.unlink(self.temp_validation_db.name)
        os.unlink(self.temp_mappings.name)
    
    def test_validate_against_training_mappings(self):
        """Test validation against training mappings"""
        
        # Mock recommendation function that returns good results
        def good_recommendation_function(query: str) -> list:
            if "psychology" in query.lower():
                return ["kaggle", "zenodo", "world_bank"]
            elif "machine learning" in query.lower():
                return ["kaggle", "zenodo", "data_un"]
            elif "singapore" in query.lower():
                return ["data_gov_sg", "singstat", "kaggle"]
            elif "climate" in query.lower():
                return ["world_bank", "data_un", "kaggle"]
            else:
                return ["kaggle", "zenodo", "world_bank"]
        
        result = self.validator.validate_against_training_mappings(
            good_recommendation_function, sample_size=4
        )
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == "training_mappings_validation"
        assert result.score > 0.0
        assert 'tested_queries' in result.details
        assert result.details['tested_queries'] == 4
        
        # Test with poor recommendation function
        def poor_recommendation_function(query: str) -> list:
            return ["random_source", "another_random", "third_random"]
        
        poor_result = self.validator.validate_against_training_mappings(
            poor_recommendation_function, sample_size=4
        )
        
        assert poor_result.score < result.score  # Should be worse
        assert not poor_result.passed  # Should fail
    
    def test_validate_domain_routing_accuracy(self):
        """Test domain routing validation"""
        
        # Mock domain classifier
        def good_domain_classifier(query: str) -> str:
            query_lower = query.lower()
            if "psychology" in query_lower or "mental health" in query_lower:
                return "psychology"
            elif "machine learning" in query_lower or "artificial intelligence" in query_lower:
                return "machine_learning"
            elif "climate" in query_lower or "environmental" in query_lower:
                return "climate"
            elif "singapore" in query_lower:
                return "singapore"
            elif "economic" in query_lower or "gdp" in query_lower:
                return "economics"
            elif "health" in query_lower or "medical" in query_lower:
                return "health"
            elif "education" in query_lower or "student" in query_lower:
                return "education"
            else:
                return "general"
        
        result = self.validator.validate_domain_routing_accuracy(
            good_domain_classifier, sample_size=10
        )
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == "domain_routing_validation"
        assert result.score > 0.8  # Should be high accuracy
        assert result.passed
        assert 'accuracy' in result.details
        
        # Test with poor classifier
        def poor_domain_classifier(query: str) -> str:
            return "general"  # Always returns general
        
        poor_result = self.validator.validate_domain_routing_accuracy(
            poor_domain_classifier, sample_size=10
        )
        
        assert poor_result.score < result.score
        assert not poor_result.passed
    
    def test_validate_singapore_first_strategy(self):
        """Test Singapore-first strategy validation"""
        
        # Mock Singapore detector
        def singapore_detector(query: str) -> bool:
            return "singapore" in query.lower()
        
        # Mock source ranker
        def source_ranker(query: str, sources: list) -> list:
            if "singapore" in query.lower():
                # Prioritize Singapore sources
                singapore_sources = [s for s in sources if s in ['data_gov_sg', 'singstat', 'lta_datamall']]
                other_sources = [s for s in sources if s not in ['data_gov_sg', 'singstat', 'lta_datamall']]
                return singapore_sources + other_sources
            else:
                return sources  # No special prioritization
        
        result = self.validator.validate_singapore_first_strategy(
            singapore_detector, source_ranker
        )
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == "singapore_first_validation"
        assert result.passed
        assert 'detection_accuracy' in result.details
        assert 'prioritization_accuracy' in result.details
        assert result.details['detection_accuracy'] > 0.8
    
    def test_run_regression_test(self):
        """Test regression testing"""
        
        # Mock metric function
        def current_metric_function() -> float:
            return 0.85
        
        # First run - should establish baseline
        result1 = self.validator.run_regression_test(
            "test_metric",
            current_metric_function,
            "baseline_test_metric"
        )
        
        assert isinstance(result1, RegressionTestResult)
        assert result1.passed  # First run should pass
        assert result1.current_score == 0.85
        
        # Second run with improved metric
        def improved_metric_function() -> float:
            return 0.90
        
        result2 = self.validator.run_regression_test(
            "test_metric_improved",
            improved_metric_function,
            "baseline_test_metric"
        )
        
        assert result2.passed
        assert result2.improvement > 0
        
        # Third run with degraded metric
        def degraded_metric_function() -> float:
            return 0.75
        
        result3 = self.validator.run_regression_test(
            "test_metric_degraded",
            degraded_metric_function,
            "baseline_test_metric",
            improvement_threshold=0.05
        )
        
        # Should fail if degradation is too large
        assert result3.improvement < 0
    
    def test_continuous_validation_suite(self):
        """Test complete validation suite"""
        
        # Mock functions
        def recommendation_function(query: str) -> list:
            return ["kaggle", "zenodo", "world_bank"]
        
        def domain_classifier(query: str) -> str:
            return "general"
        
        def singapore_detector(query: str) -> bool:
            return "singapore" in query.lower()
        
        def source_ranker(query: str, sources: list) -> list:
            return sources
        
        results = self.validator.continuous_validation_suite(
            recommendation_function,
            domain_classifier,
            singapore_detector,
            source_ranker
        )
        
        assert isinstance(results, dict)
        assert 'training_mappings' in results
        assert 'domain_routing' in results
        assert 'singapore_first' in results
        
        for test_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert result.test_name is not None
    
    def test_validation_history(self):
        """Test validation history retrieval"""
        
        # Create some test validation results
        def dummy_function(query: str) -> list:
            return ["kaggle", "zenodo"]
        
        # Run validation to create history
        self.validator.validate_against_training_mappings(dummy_function, sample_size=2)
        
        # Get history
        history = self.validator.get_validation_history("training_mappings_validation", days=1)
        
        assert len(history) > 0
        assert all(isinstance(result, ValidationResult) for result in history)
    
    def test_regression_history(self):
        """Test regression test history retrieval"""
        
        # Run regression test to create history
        def metric_function() -> float:
            return 0.8
        
        self.validator.run_regression_test(
            "test_regression",
            metric_function,
            "test_baseline"
        )
        
        # Get history
        history = self.validator.get_regression_history("test_regression", days=1)
        
        assert len(history) > 0
        assert all(isinstance(result, RegressionTestResult) for result in history)

def test_validation_result_dataclass():
    """Test ValidationResult dataclass"""
    result = ValidationResult(
        test_name="test",
        timestamp="2024-01-01T00:00:00",
        passed=True,
        score=0.85,
        expected_score=0.8,
        threshold=0.8,
        details={"key": "value"}
    )
    
    assert result.test_name == "test"
    assert result.passed == True
    assert result.score == 0.85

def test_regression_test_result_dataclass():
    """Test RegressionTestResult dataclass"""
    result = RegressionTestResult(
        test_name="regression_test",
        timestamp="2024-01-01T00:00:00",
        current_score=0.9,
        baseline_score=0.8,
        improvement=0.1,
        passed=True,
        threshold=0.05,
        details={"improvement_percentage": 12.5}
    )
    
    assert result.test_name == "regression_test"
    assert result.improvement == 0.1
    assert result.passed == True

if __name__ == "__main__":
    # Run basic functionality test
    print("Testing Automated Quality Validator...")
    
    # Test dataclasses
    print("✓ Testing ValidationResult dataclass...")
    test_validation_result_dataclass()
    
    print("✓ Testing RegressionTestResult dataclass...")
    test_regression_test_result_dataclass()
    
    # Test main functionality
    print("✓ Testing AutomatedQualityValidator...")
    test_validator = TestAutomatedQualityValidator()
    test_validator.setup_method()
    
    try:
        test_validator.test_validate_against_training_mappings()
        test_validator.test_validate_domain_routing_accuracy()
        test_validator.test_validate_singapore_first_strategy()
        test_validator.test_run_regression_test()
        test_validator.test_continuous_validation_suite()
        test_validator.test_validation_history()
        test_validator.test_regression_history()
    finally:
        test_validator.teardown_method()
    
    print("All tests passed! ✅")