"""
Training Data Quality Validator - Comprehensive validation system
Validates training data coverage, quality metrics, and Singapore-first strategy representation
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for training data validation"""
    ndcg_at_3: float
    relevance_accuracy: float  # How well recommendations match training_mappings.md
    domain_routing_accuracy: float  # Correct domain classification rate
    singapore_first_accuracy: float  # Correct local vs global routing
    user_satisfaction_score: float
    recommendation_diversity: float
    
    def meets_quality_threshold(self, threshold: float = 0.7) -> bool:
        """Check if quality meets minimum threshold"""
        return (self.ndcg_at_3 >= threshold and 
                self.relevance_accuracy >= threshold and
                self.domain_routing_accuracy >= threshold)

@dataclass
class ValidationResult:
    """Result of training data validation"""
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    details: Dict[str, any]

class TrainingDataQualityValidator:
    """
    Comprehensive validation system for training data quality
    Ensures proper coverage across domains and Singapore-first strategy
    """
    
    def __init__(self):
        self.required_domains = {
            "psychology", "machine_learning", "climate", "economics", 
            "singapore", "health", "education"
        }
        self.required_sources = {
            "kaggle", "zenodo", "world_bank", "data_gov_sg", 
            "singstat", "lta_datamall"
        }
        self.singapore_sources = {"data_gov_sg", "singstat", "lta_datamall"}
        self.quality_thresholds = {
            "min_examples_per_domain": 5,
            "min_singapore_examples": 8,
            "min_relevance_score": 0.0,  # Allow low scores for negative examples
            "max_relevance_score": 1.0,
            "min_domain_coverage": 0.8,
            "min_source_diversity": 0.6
        }
    
    def validate_training_data_coverage(self, training_examples: List[Dict]) -> ValidationResult:
        """Validate training data coverage across domains and sources"""
        issues = []
        recommendations = []
        details = {}
        
        # Domain coverage validation
        domain_coverage = self._validate_domain_coverage(training_examples)
        details["domain_coverage"] = domain_coverage
        
        if domain_coverage["coverage_ratio"] < self.quality_thresholds["min_domain_coverage"]:
            issues.append(f"Domain coverage too low: {domain_coverage['coverage_ratio']:.2f} < {self.quality_thresholds['min_domain_coverage']}")
            recommendations.append("Add more training examples for underrepresented domains")
        
        # Source diversity validation
        source_diversity = self._validate_source_diversity(training_examples)
        details["source_diversity"] = source_diversity
        
        if source_diversity["diversity_score"] < self.quality_thresholds["min_source_diversity"]:
            issues.append(f"Source diversity too low: {source_diversity['diversity_score']:.2f}")
            recommendations.append("Include examples from more diverse data sources")
        
        # Singapore-first strategy validation
        singapore_validation = self._validate_singapore_first_strategy(training_examples)
        details["singapore_first"] = singapore_validation
        
        if singapore_validation["singapore_examples"] < self.quality_thresholds["min_singapore_examples"]:
            issues.append(f"Insufficient Singapore examples: {singapore_validation['singapore_examples']} < {self.quality_thresholds['min_singapore_examples']}")
            recommendations.append("Add more Singapore-specific training examples")
        
        # Calculate overall score
        score = self._calculate_coverage_score(domain_coverage, source_diversity, singapore_validation)
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def validate_relevance_scores(self, training_examples: List[Dict]) -> ValidationResult:
        """Validate relevance score quality and distribution"""
        issues = []
        recommendations = []
        details = {}
        
        relevance_scores = [ex.get("relevance_score", 0.0) for ex in training_examples]
        
        # Basic score validation
        invalid_scores = [score for score in relevance_scores 
                         if score < self.quality_thresholds["min_relevance_score"] or 
                            score > self.quality_thresholds["max_relevance_score"]]
        
        if invalid_scores:
            issues.append(f"Found {len(invalid_scores)} invalid relevance scores")
            recommendations.append("Ensure all relevance scores are between 0.0 and 1.0")
        
        # Score distribution analysis
        score_distribution = self._analyze_score_distribution(relevance_scores)
        details["score_distribution"] = score_distribution
        
        # Check for balanced distribution
        if score_distribution["high_quality_ratio"] < 0.4:
            issues.append(f"Too few high-quality examples: {score_distribution['high_quality_ratio']:.2f}")
            recommendations.append("Add more high-relevance training examples (score >= 0.8)")
        
        if score_distribution["low_quality_ratio"] > 0.3:
            issues.append(f"Too many low-quality examples: {score_distribution['low_quality_ratio']:.2f}")
            recommendations.append("Review and improve low-relevance examples")
        
        # Domain-specific relevance validation
        domain_relevance = self._validate_domain_specific_relevance(training_examples)
        details["domain_relevance"] = domain_relevance
        
        for domain, metrics in domain_relevance.items():
            if metrics["avg_relevance"] < 0.6:
                issues.append(f"Low average relevance for {domain}: {metrics['avg_relevance']:.2f}")
                recommendations.append(f"Improve relevance scores for {domain} domain examples")
        
        score = self._calculate_relevance_score(score_distribution, domain_relevance)
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def validate_singapore_first_strategy_representation(self, training_examples: List[Dict]) -> ValidationResult:
        """Validate Singapore-first strategy representation in training data"""
        issues = []
        recommendations = []
        details = {}
        
        # Identify Singapore-related examples
        singapore_examples = [ex for ex in training_examples 
                            if ex.get("singapore_first", False) or 
                               ex.get("geographic_scope") == "singapore"]
        
        details["singapore_examples_count"] = len(singapore_examples)
        details["total_examples"] = len(training_examples)
        details["singapore_ratio"] = len(singapore_examples) / len(training_examples) if training_examples else 0
        
        # Validate Singapore source prioritization
        singapore_source_validation = self._validate_singapore_source_prioritization(singapore_examples)
        details["singapore_source_validation"] = singapore_source_validation
        
        if singapore_source_validation["correct_prioritization_ratio"] < 0.8:
            issues.append(f"Singapore source prioritization too low: {singapore_source_validation['correct_prioritization_ratio']:.2f}")
            recommendations.append("Ensure Singapore queries prioritize local sources (data.gov.sg, singstat, lta)")
        
        # Validate query diversity for Singapore examples
        singapore_query_diversity = self._validate_singapore_query_diversity(singapore_examples)
        details["singapore_query_diversity"] = singapore_query_diversity
        
        if singapore_query_diversity["unique_query_types"] < 4:
            issues.append(f"Limited Singapore query diversity: {singapore_query_diversity['unique_query_types']} types")
            recommendations.append("Add more diverse Singapore-specific query types (housing, transport, demographics, etc.)")
        
        # Validate geographic scope consistency
        geographic_consistency = self._validate_geographic_scope_consistency(training_examples)
        details["geographic_consistency"] = geographic_consistency
        
        if geographic_consistency["consistency_score"] < 0.9:
            issues.append(f"Geographic scope inconsistency: {geographic_consistency['consistency_score']:.2f}")
            recommendations.append("Review geographic scope assignments for consistency")
        
        score = self._calculate_singapore_first_score(singapore_source_validation, singapore_query_diversity, geographic_consistency)
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def run_comprehensive_validation(self, training_data_file: str) -> Dict[str, ValidationResult]:
        """Run comprehensive validation on training data file"""
        logger.info(f"🔍 Starting comprehensive validation of {training_data_file}")
        
        # Load training data
        training_data = self._load_training_data(training_data_file)
        if not training_data:
            return {"error": ValidationResult(False, 0.0, ["Could not load training data"], [], {})}
        
        training_examples = training_data.get("examples", [])
        
        # Run all validation checks
        validation_results = {}
        
        # Coverage validation
        validation_results["coverage"] = self.validate_training_data_coverage(training_examples)
        
        # Relevance score validation
        validation_results["relevance"] = self.validate_relevance_scores(training_examples)
        
        # Singapore-first strategy validation
        validation_results["singapore_first"] = self.validate_singapore_first_strategy_representation(training_examples)
        
        # Overall validation summary
        validation_results["overall"] = self._create_overall_validation_summary(validation_results)
        
        logger.info(f"✅ Comprehensive validation complete")
        return validation_results
    
    def _validate_domain_coverage(self, training_examples: List[Dict]) -> Dict[str, any]:
        """Validate coverage across required domains"""
        domain_counts = Counter(ex.get("domain", "unknown") for ex in training_examples)
        
        covered_domains = set(domain_counts.keys()) & self.required_domains
        missing_domains = self.required_domains - covered_domains
        
        # Check minimum examples per domain
        insufficient_domains = []
        for domain in covered_domains:
            if domain_counts[domain] < self.quality_thresholds["min_examples_per_domain"]:
                insufficient_domains.append((domain, domain_counts[domain]))
        
        return {
            "total_domains": len(self.required_domains),
            "covered_domains": len(covered_domains),
            "missing_domains": list(missing_domains),
            "insufficient_domains": insufficient_domains,
            "coverage_ratio": len(covered_domains) / len(self.required_domains),
            "domain_counts": dict(domain_counts)
        }
    
    def _validate_source_diversity(self, training_examples: List[Dict]) -> Dict[str, any]:
        """Validate diversity of data sources in training examples"""
        source_counts = Counter(ex.get("source", "unknown") for ex in training_examples)
        
        covered_sources = set(source_counts.keys()) & self.required_sources
        missing_sources = self.required_sources - covered_sources
        
        # Calculate diversity score (Shannon entropy)
        total_examples = len(training_examples)
        diversity_score = 0.0
        if total_examples > 0:
            for count in source_counts.values():
                p = count / total_examples
                if p > 0:
                    diversity_score -= p * np.log2(p)
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(self.required_sources))
            diversity_score = diversity_score / max_entropy if max_entropy > 0 else 0.0
        
        return {
            "total_sources": len(self.required_sources),
            "covered_sources": len(covered_sources),
            "missing_sources": list(missing_sources),
            "diversity_score": diversity_score,
            "source_counts": dict(source_counts)
        }
    
    def _validate_singapore_first_strategy(self, training_examples: List[Dict]) -> Dict[str, any]:
        """Validate Singapore-first strategy representation"""
        singapore_examples = [ex for ex in training_examples 
                            if ex.get("singapore_first", False) or 
                               ex.get("geographic_scope") == "singapore"]
        
        singapore_source_examples = [ex for ex in singapore_examples 
                                   if ex.get("source") in self.singapore_sources]
        
        return {
            "singapore_examples": len(singapore_examples),
            "singapore_source_examples": len(singapore_source_examples),
            "singapore_source_ratio": len(singapore_source_examples) / len(singapore_examples) if singapore_examples else 0,
            "total_examples": len(training_examples),
            "singapore_representation": len(singapore_examples) / len(training_examples) if training_examples else 0
        }
    
    def _analyze_score_distribution(self, relevance_scores: List[float]) -> Dict[str, any]:
        """Analyze distribution of relevance scores"""
        if not relevance_scores:
            return {"high_quality_ratio": 0, "medium_quality_ratio": 0, "low_quality_ratio": 0}
        
        high_quality = sum(1 for score in relevance_scores if score >= 0.8)
        medium_quality = sum(1 for score in relevance_scores if 0.5 <= score < 0.8)
        low_quality = sum(1 for score in relevance_scores if score < 0.5)
        
        total = len(relevance_scores)
        
        return {
            "high_quality_ratio": high_quality / total,
            "medium_quality_ratio": medium_quality / total,
            "low_quality_ratio": low_quality / total,
            "average_score": np.mean(relevance_scores),
            "median_score": np.median(relevance_scores),
            "std_score": np.std(relevance_scores)
        }
    
    def _validate_domain_specific_relevance(self, training_examples: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Validate relevance scores within each domain"""
        domain_relevance = defaultdict(list)
        
        for ex in training_examples:
            domain = ex.get("domain", "unknown")
            relevance = ex.get("relevance_score", 0.0)
            domain_relevance[domain].append(relevance)
        
        domain_metrics = {}
        for domain, scores in domain_relevance.items():
            if scores:
                domain_metrics[domain] = {
                    "avg_relevance": np.mean(scores),
                    "min_relevance": np.min(scores),
                    "max_relevance": np.max(scores),
                    "example_count": len(scores)
                }
        
        return domain_metrics
    
    def _validate_singapore_source_prioritization(self, singapore_examples: List[Dict]) -> Dict[str, any]:
        """Validate that Singapore examples prioritize Singapore sources"""
        if not singapore_examples:
            return {"correct_prioritization_ratio": 0, "total_singapore_examples": 0}
        
        correct_prioritization = 0
        for ex in singapore_examples:
            source = ex.get("source", "")
            relevance = ex.get("relevance_score", 0.0)
            
            # Check if Singapore source has high relevance or non-Singapore source has low relevance
            if (source in self.singapore_sources and relevance >= 0.8) or \
               (source not in self.singapore_sources and relevance <= 0.6):
                correct_prioritization += 1
        
        return {
            "correct_prioritization": correct_prioritization,
            "total_singapore_examples": len(singapore_examples),
            "correct_prioritization_ratio": correct_prioritization / len(singapore_examples)
        }
    
    def _validate_singapore_query_diversity(self, singapore_examples: List[Dict]) -> Dict[str, any]:
        """Validate diversity of Singapore query types"""
        singapore_query_types = set()
        
        for ex in singapore_examples:
            query = ex.get("query", "").lower()
            
            # Categorize Singapore queries
            if any(word in query for word in ["housing", "hdb", "property"]):
                singapore_query_types.add("housing")
            elif any(word in query for word in ["transport", "lta", "traffic", "mrt"]):
                singapore_query_types.add("transport")
            elif any(word in query for word in ["demographics", "population", "census"]):
                singapore_query_types.add("demographics")
            elif any(word in query for word in ["economy", "economic", "gdp"]):
                singapore_query_types.add("economy")
            elif any(word in query for word in ["health", "healthcare", "medical"]):
                singapore_query_types.add("health")
            else:
                singapore_query_types.add("general")
        
        return {
            "unique_query_types": len(singapore_query_types),
            "query_types": list(singapore_query_types),
            "total_singapore_examples": len(singapore_examples)
        }
    
    def _validate_geographic_scope_consistency(self, training_examples: List[Dict]) -> Dict[str, any]:
        """Validate consistency of geographic scope assignments"""
        consistent_examples = 0
        total_examples = len(training_examples)
        
        for ex in training_examples:
            query = ex.get("query", "").lower()
            geographic_scope = ex.get("geographic_scope", "")
            singapore_first = ex.get("singapore_first", False)
            
            # Check consistency
            has_singapore_keywords = any(word in query for word in ["singapore", "sg", "singstat", "hdb", "lta"])
            
            if has_singapore_keywords:
                # Should have singapore scope and singapore_first=True
                if geographic_scope == "singapore" and singapore_first:
                    consistent_examples += 1
            else:
                # Should not have singapore_first=True
                if not singapore_first:
                    consistent_examples += 1
        
        return {
            "consistent_examples": consistent_examples,
            "total_examples": total_examples,
            "consistency_score": consistent_examples / total_examples if total_examples > 0 else 0
        }
    
    def _calculate_coverage_score(self, domain_coverage: Dict, source_diversity: Dict, singapore_validation: Dict) -> float:
        """Calculate overall coverage score"""
        domain_score = domain_coverage["coverage_ratio"]
        diversity_score = source_diversity["diversity_score"]
        singapore_score = min(1.0, singapore_validation["singapore_representation"] * 5)  # Scale up singapore representation
        
        return (domain_score * 0.4 + diversity_score * 0.4 + singapore_score * 0.2)
    
    def _calculate_relevance_score(self, score_distribution: Dict, domain_relevance: Dict) -> float:
        """Calculate overall relevance quality score"""
        distribution_score = (score_distribution["high_quality_ratio"] * 0.6 + 
                            score_distribution["medium_quality_ratio"] * 0.3 + 
                            (1 - score_distribution["low_quality_ratio"]) * 0.1)
        
        # Average domain relevance score
        domain_scores = [metrics["avg_relevance"] for metrics in domain_relevance.values()]
        domain_score = np.mean(domain_scores) if domain_scores else 0.0
        
        return (distribution_score * 0.6 + domain_score * 0.4)
    
    def _calculate_singapore_first_score(self, source_validation: Dict, query_diversity: Dict, geographic_consistency: Dict) -> float:
        """Calculate Singapore-first strategy score"""
        source_score = source_validation["correct_prioritization_ratio"]
        diversity_score = min(1.0, query_diversity["unique_query_types"] / 5)  # Normalize to max 5 types
        consistency_score = geographic_consistency["consistency_score"]
        
        return (source_score * 0.5 + diversity_score * 0.3 + consistency_score * 0.2)
    
    def _create_overall_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> ValidationResult:
        """Create overall validation summary"""
        all_issues = []
        all_recommendations = []
        overall_score = 0.0
        passed_count = 0
        
        for category, result in validation_results.items():
            if category != "overall":
                all_issues.extend([f"{category}: {issue}" for issue in result.issues])
                all_recommendations.extend([f"{category}: {rec}" for rec in result.recommendations])
                overall_score += result.score
                if result.passed:
                    passed_count += 1
        
        total_categories = len(validation_results) - 1  # Exclude 'overall'
        overall_score = overall_score / total_categories if total_categories > 0 else 0.0
        overall_passed = passed_count == total_categories
        
        return ValidationResult(
            passed=overall_passed,
            score=overall_score,
            issues=all_issues,
            recommendations=all_recommendations,
            details={
                "categories_passed": passed_count,
                "total_categories": total_categories,
                "pass_rate": passed_count / total_categories if total_categories > 0 else 0
            }
        )
    
    def _load_training_data(self, file_path: str) -> Optional[Dict]:
        """Load training data from file"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Training data file not found: {file_path}")
                return None
            
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult], output_file: str):
        """Generate comprehensive validation report"""
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_summary": {
                "passed": validation_results["overall"].passed,
                "score": validation_results["overall"].score,
                "total_issues": len(validation_results["overall"].issues),
                "total_recommendations": len(validation_results["overall"].recommendations)
            },
            "category_results": {},
            "detailed_issues": validation_results["overall"].issues,
            "recommendations": validation_results["overall"].recommendations
        }
        
        for category, result in validation_results.items():
            if category != "overall":
                report["category_results"][category] = {
                    "passed": result.passed,
                    "score": result.score,
                    "issues": result.issues,
                    "recommendations": result.recommendations,
                    "details": result.details
                }
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Validation report saved: {output_file}")
        return report


# Convenience functions
def validate_training_data(training_data_file: str, report_file: Optional[str] = None) -> Dict[str, ValidationResult]:
    """Quick function to validate training data"""
    validator = TrainingDataQualityValidator()
    results = validator.run_comprehensive_validation(training_data_file)
    
    if report_file:
        validator.generate_validation_report(results, report_file)
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Validate training data
    results = validate_training_data(
        "data/processed/enhanced_training_data.json",
        "data/processed/training_validation_report.json"
    )
    
    print(f"✅ Validation complete!")
    print(f"Overall passed: {results['overall'].passed}")
    print(f"Overall score: {results['overall'].score:.2f}")
    if results['overall'].issues:
        print(f"Issues found: {len(results['overall'].issues)}")
        for issue in results['overall'].issues[:3]:  # Show first 3 issues
            print(f"  - {issue}")