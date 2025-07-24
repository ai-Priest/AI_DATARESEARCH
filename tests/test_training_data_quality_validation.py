#!/usr/bin/env python3
"""
Automated Testing for Training Data Quality Validation
Implements comprehensive tests for task 1.3: Training Data Quality Validation

This test suite validates:
- Training data coverage across domains
- Quality metrics for training example relevance scores  
- Singapore-first strategy representation
- Requirements 1.5, 1.6 compliance
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List
import pytest

from src.ml.training_data_quality_validator import (
    TrainingDataQualityValidator,
    ValidationResult,
    QualityMetrics,
    validate_training_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTrainingDataQualityValidation:
    """Comprehensive test suite for training data quality validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = TrainingDataQualityValidator()
        self.test_data_dir = Path("data/processed")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_test_training_data(self, examples: List[Dict]) -> str:
        """Create temporary training data file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"examples": examples}, f, indent=2)
            return f.name
    
    def test_domain_coverage_validation(self):
        """Test validation of training data coverage across domains"""
        logger.info("🧪 Testing domain coverage validation")
        
        # Test case 1: Complete domain coverage with proper source diversity
        complete_examples = []
        
        # Create examples with proper source diversity and Singapore examples
        domains_sources = {
            "psychology": ["kaggle", "zenodo"],
            "machine_learning": ["kaggle", "zenodo"], 
            "climate": ["world_bank", "zenodo"],
            "economics": ["world_bank", "kaggle"],
            "singapore": ["data_gov_sg", "singstat", "lta_datamall"],
            "health": ["world_bank", "zenodo"],
            "education": ["world_bank", "kaggle"]
        }
        
        for domain, sources in domains_sources.items():
            for i, source in enumerate(sources):
                for j in range(3):  # 3 examples per domain-source combination
                    is_singapore = domain == "singapore"
                    complete_examples.append({
                        "query": f"{domain} query {i}_{j}",
                        "domain": domain,
                        "source": source,
                        "relevance_score": 0.8 + (j * 0.05),  # Vary scores 0.8-0.9
                        "singapore_first": is_singapore,
                        "geographic_scope": "singapore" if is_singapore else "global"
                    })
        
        # Add extra Singapore examples to meet minimum requirement (8)
        for i in range(5):
            complete_examples.append({
                "query": f"singapore extra query {i}",
                "domain": "singapore",
                "source": "data_gov_sg",
                "relevance_score": 0.85,
                "singapore_first": True,
                "geographic_scope": "singapore"
            })
        
        test_file = self.create_test_training_data(complete_examples)
        result = self.validator.validate_training_data_coverage(complete_examples)
        
        assert result.passed, f"Complete domain coverage should pass: {result.issues}"
        assert result.score > 0.8, f"Coverage score should be high: {result.score}"
        assert result.details["domain_coverage"]["coverage_ratio"] == 1.0
        
        Path(test_file).unlink()  # Cleanup
        logger.info("✅ Complete domain coverage test passed")
        
        # Test case 2: Missing domains
        incomplete_examples = [
            {"query": "psychology research", "domain": "psychology", "source": "kaggle", "relevance_score": 0.9},
            {"query": "ml datasets", "domain": "machine_learning", "source": "kaggle", "relevance_score": 0.95},
            # Missing other domains
        ]
        
        result = self.validator.validate_training_data_coverage(incomplete_examples)
        
        assert not result.passed, "Incomplete domain coverage should fail"
        assert len(result.details["domain_coverage"]["missing_domains"]) > 0
        assert "Add more training examples for underrepresented domains" in result.recommendations
        
        logger.info("✅ Missing domain coverage test passed")
    
    def test_relevance_score_quality_validation(self):
        """Test validation of relevance score quality and distribution"""
        logger.info("🧪 Testing relevance score quality validation")
        
        # Test case 1: Valid relevance scores with good distribution
        good_examples = [
            {"query": "high quality query", "domain": "psychology", "source": "kaggle", "relevance_score": 0.95},
            {"query": "high quality query 2", "domain": "psychology", "source": "zenodo", "relevance_score": 0.88},
            {"query": "medium quality query", "domain": "climate", "source": "world_bank", "relevance_score": 0.75},
            {"query": "medium quality query 2", "domain": "climate", "source": "world_bank", "relevance_score": 0.72},
            {"query": "acceptable quality", "domain": "economics", "source": "world_bank", "relevance_score": 0.68},
        ]
        
        result = self.validator.validate_relevance_scores(good_examples)
        
        assert result.passed, f"Good relevance scores should pass: {result.issues}"
        assert result.score > 0.6, f"Relevance score should be reasonable: {result.score}"
        
        logger.info("✅ Good relevance scores test passed")
        
        # Test case 2: Invalid relevance scores
        bad_examples = [
            {"query": "invalid high", "domain": "psychology", "source": "kaggle", "relevance_score": 1.5},  # Too high
            {"query": "invalid low", "domain": "psychology", "source": "kaggle", "relevance_score": -0.1},  # Too low
            {"query": "valid", "domain": "psychology", "source": "kaggle", "relevance_score": 0.8},
        ]
        
        result = self.validator.validate_relevance_scores(bad_examples)
        
        assert not result.passed, "Invalid relevance scores should fail"
        assert "Found 2 invalid relevance scores" in result.issues
        assert "Ensure all relevance scores are between 0.0 and 1.0" in result.recommendations
        
        logger.info("✅ Invalid relevance scores test passed")
        
        # Test case 3: Poor score distribution (too many low quality)
        poor_distribution_examples = [
            {"query": f"low quality {i}", "domain": "psychology", "source": "kaggle", "relevance_score": 0.2}
            for i in range(10)
        ]
        
        result = self.validator.validate_relevance_scores(poor_distribution_examples)
        
        assert not result.passed, "Poor score distribution should fail"
        assert any("Too many low-quality examples" in issue for issue in result.issues)
        
        logger.info("✅ Poor score distribution test passed")
    
    def test_singapore_first_strategy_validation(self):
        """Test validation of Singapore-first strategy representation"""
        logger.info("🧪 Testing Singapore-first strategy validation")
        
        # Test case 1: Proper Singapore-first representation
        singapore_examples = [
            {
                "query": "singapore housing data",
                "domain": "singapore", 
                "source": "data_gov_sg",
                "relevance_score": 0.96,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore transport statistics", 
                "domain": "singapore",
                "source": "lta_datamall",
                "relevance_score": 0.94,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore demographics",
                "domain": "singapore",
                "source": "singstat", 
                "relevance_score": 0.92,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore economy",
                "domain": "singapore",
                "source": "singstat",
                "relevance_score": 0.88,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore health data",
                "domain": "singapore", 
                "source": "data_gov_sg",
                "relevance_score": 0.85,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            # Add some global examples for balance
            {
                "query": "global climate data",
                "domain": "climate",
                "source": "world_bank", 
                "relevance_score": 0.90,
                "singapore_first": False,
                "geographic_scope": "global"
            },
            {
                "query": "psychology research",
                "domain": "psychology",
                "source": "kaggle",
                "relevance_score": 0.92,
                "singapore_first": False,
                "geographic_scope": "global"
            }
        ]
        
        result = self.validator.validate_singapore_first_strategy_representation(singapore_examples)
        
        assert result.passed, f"Proper Singapore-first strategy should pass: {result.issues}"
        assert result.score > 0.8, f"Singapore-first score should be high: {result.score}"
        assert result.details["singapore_examples_count"] >= 5
        assert result.details["singapore_source_validation"]["correct_prioritization_ratio"] >= 0.8
        
        logger.info("✅ Proper Singapore-first strategy test passed")
        
        # Test case 2: Poor Singapore source prioritization
        poor_singapore_examples = [
            {
                "query": "singapore housing data",
                "domain": "singapore",
                "source": "kaggle",  # Wrong source for Singapore query
                "relevance_score": 0.96,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore transport",
                "domain": "singapore", 
                "source": "world_bank",  # Wrong source for Singapore query
                "relevance_score": 0.30,  # Low relevance is correct for wrong source
                "singapore_first": True,
                "geographic_scope": "singapore"
            }
        ]
        
        result = self.validator.validate_singapore_first_strategy_representation(poor_singapore_examples)
        
        # This should still pass because low relevance for wrong sources is correct
        assert result.details["singapore_source_validation"]["correct_prioritization_ratio"] >= 0.5
        
        logger.info("✅ Singapore source prioritization test passed")
        
        # Test case 3: Insufficient Singapore query diversity
        limited_diversity_examples = [
            {
                "query": "singapore data",
                "domain": "singapore",
                "source": "data_gov_sg",
                "relevance_score": 0.95,
                "singapore_first": True,
                "geographic_scope": "singapore"
            },
            {
                "query": "singapore statistics", 
                "domain": "singapore",
                "source": "singstat",
                "relevance_score": 0.93,
                "singapore_first": True,
                "geographic_scope": "singapore"
            }
        ]
        
        result = self.validator.validate_singapore_first_strategy_representation(limited_diversity_examples)
        
        assert result.details["singapore_query_diversity"]["unique_query_types"] < 4
        if not result.passed:
            assert any("Limited Singapore query diversity" in issue for issue in result.issues)
        
        logger.info("✅ Singapore query diversity test passed")
    
    def test_comprehensive_validation_integration(self):
        """Test comprehensive validation integration"""
        logger.info("🧪 Testing comprehensive validation integration")
        
        # Create a comprehensive test dataset
        comprehensive_examples = []
        
        # Add examples for all domains with proper distribution
        domains_sources = {
            "psychology": ["kaggle", "zenodo"],
            "machine_learning": ["kaggle", "zenodo"], 
            "climate": ["world_bank", "zenodo"],
            "economics": ["world_bank"],
            "singapore": ["data_gov_sg", "singstat", "lta_datamall"],
            "health": ["world_bank", "zenodo"],
            "education": ["world_bank", "zenodo"]
        }
        
        for domain, sources in domains_sources.items():
            for i, source in enumerate(sources):
                for j in range(6):  # 6 examples per domain-source combination
                    is_singapore = domain == "singapore"
                    
                    # Create diverse Singapore queries
                    if is_singapore:
                        singapore_query_types = [
                            "singapore housing data",
                            "singapore transport statistics", 
                            "singapore demographics info",
                            "singapore economy indicators",
                            "singapore health data"
                        ]
                        query = singapore_query_types[j % len(singapore_query_types)]
                    else:
                        query = f"{domain} query {i}_{j}"
                    
                    comprehensive_examples.append({
                        "query": query,
                        "domain": domain,
                        "source": source,
                        "relevance_score": 0.8 + (j * 0.02),  # Vary scores 0.8-0.9
                        "singapore_first": is_singapore,
                        "geographic_scope": "singapore" if is_singapore else "global"
                    })
        
        test_file = self.create_test_training_data(comprehensive_examples)
        
        # Run comprehensive validation
        validation_results = validate_training_data(test_file)
        
        # Verify overall results
        assert "overall" in validation_results
        assert "coverage" in validation_results
        assert "relevance" in validation_results
        assert "singapore_first" in validation_results
        
        overall_result = validation_results["overall"]
        assert isinstance(overall_result, ValidationResult)
        assert overall_result.score > 0.7, f"Overall score should be good: {overall_result.score}"
        
        # Verify coverage results
        coverage_result = validation_results["coverage"]
        assert coverage_result.passed, f"Coverage should pass: {coverage_result.issues}"
        
        # Verify relevance results  
        relevance_result = validation_results["relevance"]
        assert relevance_result.passed, f"Relevance should pass: {relevance_result.issues}"
        
        # Verify Singapore-first results
        singapore_result = validation_results["singapore_first"]
        assert singapore_result.passed, f"Singapore-first should pass: {singapore_result.issues}"
        
        Path(test_file).unlink()  # Cleanup
        logger.info("✅ Comprehensive validation integration test passed")
    
    def test_requirements_compliance(self):
        """Test compliance with specific requirements 1.5 and 1.6"""
        logger.info("🧪 Testing requirements compliance (1.5, 1.6)")
        
        # Requirement 1.5: Quality metrics for training example relevance scores
        test_examples = [
            {"query": "test query", "domain": "psychology", "source": "kaggle", "relevance_score": 0.95},
            {"query": "test query 2", "domain": "climate", "source": "world_bank", "relevance_score": 0.88},
        ]
        
        result = self.validator.validate_relevance_scores(test_examples)
        
        # Should have quality metrics
        assert "score_distribution" in result.details
        assert "domain_relevance" in result.details
        assert "high_quality_ratio" in result.details["score_distribution"]
        assert "average_score" in result.details["score_distribution"]
        
        logger.info("✅ Requirement 1.5 compliance test passed")
        
        # Requirement 1.6: Singapore-first strategy representation
        singapore_test_examples = [
            {
                "query": "singapore housing",
                "domain": "singapore",
                "source": "data_gov_sg", 
                "relevance_score": 0.95,
                "singapore_first": True,
                "geographic_scope": "singapore"
            }
        ]
        
        result = self.validator.validate_singapore_first_strategy_representation(singapore_test_examples)
        
        # Should validate Singapore-first strategy
        assert "singapore_source_validation" in result.details
        assert "singapore_query_diversity" in result.details
        assert "geographic_consistency" in result.details
        
        logger.info("✅ Requirement 1.6 compliance test passed")
    
    def test_real_training_data_validation(self):
        """Test validation against real training data files"""
        logger.info("🧪 Testing real training data validation")
        
        # Test against existing training data files
        test_files = [
            "data/processed/enhanced_training_data.json",
            "data/processed/test_enhanced_training_data.json"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                logger.info(f"Validating {file_path}")
                
                validation_results = validate_training_data(file_path)
                
                # Basic validation checks
                assert "overall" in validation_results
                assert "coverage" in validation_results
                assert "relevance" in validation_results
                assert "singapore_first" in validation_results
                
                overall_result = validation_results["overall"]
                logger.info(f"Overall validation score: {overall_result.score:.2f}")
                logger.info(f"Overall passed: {overall_result.passed}")
                
                if overall_result.issues:
                    logger.info(f"Issues found: {len(overall_result.issues)}")
                    for issue in overall_result.issues[:3]:  # Show first 3 issues
                        logger.info(f"  - {issue}")
                
                # The validation should complete without errors
                assert isinstance(overall_result.score, float)
                assert overall_result.score >= 0.0
                
                logger.info(f"✅ {file_path} validation completed")
        
        logger.info("✅ Real training data validation test passed")
    
    def test_automated_quality_monitoring(self):
        """Test automated quality monitoring capabilities"""
        logger.info("🧪 Testing automated quality monitoring")
        
        # Create test data with known quality issues
        problematic_examples = [
            # Invalid relevance scores
            {"query": "bad score 1", "domain": "psychology", "source": "kaggle", "relevance_score": 1.5},
            {"query": "bad score 2", "domain": "psychology", "source": "kaggle", "relevance_score": -0.1},
            
            # Missing domain coverage
            {"query": "only psychology 1", "domain": "psychology", "source": "kaggle", "relevance_score": 0.8},
            {"query": "only psychology 2", "domain": "psychology", "source": "kaggle", "relevance_score": 0.8},
            
            # Poor Singapore representation
            {"query": "singapore query", "domain": "singapore", "source": "kaggle", "relevance_score": 0.3, "singapore_first": False}
        ]
        
        test_file = self.create_test_training_data(problematic_examples)
        validation_results = validate_training_data(test_file)
        
        # Should detect multiple issues
        overall_result = validation_results["overall"]
        assert not overall_result.passed, "Problematic data should fail validation"
        assert len(overall_result.issues) > 0, "Should detect quality issues"
        assert len(overall_result.recommendations) > 0, "Should provide recommendations"
        
        # Should detect specific issue types
        all_issues = " ".join(overall_result.issues)
        assert "invalid relevance scores" in all_issues or "relevance" in all_issues
        
        Path(test_file).unlink()  # Cleanup
        logger.info("✅ Automated quality monitoring test passed")


def run_training_data_quality_tests():
    """Run all training data quality validation tests"""
    print("🧪 Running Training Data Quality Validation Tests")
    print("=" * 60)
    
    test_suite = TestTrainingDataQualityValidation()
    test_suite.setup_method()
    
    tests = [
        ("Domain Coverage Validation", test_suite.test_domain_coverage_validation),
        ("Relevance Score Quality Validation", test_suite.test_relevance_score_quality_validation),
        ("Singapore-First Strategy Validation", test_suite.test_singapore_first_strategy_validation),
        ("Comprehensive Validation Integration", test_suite.test_comprehensive_validation_integration),
        ("Requirements Compliance", test_suite.test_requirements_compliance),
        ("Real Training Data Validation", test_suite.test_real_training_data_validation),
        ("Automated Quality Monitoring", test_suite.test_automated_quality_monitoring),
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            failed_tests += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"🎉 Test Results: {passed_tests} passed, {failed_tests} failed")
    
    if failed_tests == 0:
        print("🎉 ALL TRAINING DATA QUALITY VALIDATION TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {failed_tests} tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_training_data_quality_tests()
    exit(0 if success else 1)