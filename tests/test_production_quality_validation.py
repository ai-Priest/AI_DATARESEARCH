"""
Production Quality Validation Test
Comprehensive testing against training_mappings.md expectations
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.ai.integrated_query_processor import create_integrated_query_processor
# from src.ai.automated_quality_validator import create_automated_quality_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionQualityValidator:
    """Production quality validation against training mappings"""
    
    def __init__(self, training_mappings_path: str = "training_mappings.md"):
        self.training_mappings_path = training_mappings_path
        self.processor = create_integrated_query_processor(training_mappings_path)
        # self.validator = create_automated_quality_validator(training_mappings_path)
        
        # Load training mappings for validation
        self.training_mappings = self._load_training_mappings()
        
        logger.info(f"🎯 ProductionQualityValidator initialized")
        logger.info(f"  Training mappings loaded: {len(self.training_mappings)}")
    
    def _load_training_mappings(self) -> Dict[str, List[Dict]]:
        """Load training mappings from file"""
        mappings = {}
        
        try:
            if not Path(self.training_mappings_path).exists():
                logger.warning(f"Training mappings file not found: {self.training_mappings_path}")
                return mappings
            
            with open(self.training_mappings_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            current_domain = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Detect domain headers
                if line.startswith('## ') and 'Queries' in line:
                    current_domain = line.replace('## ', '').replace(' Queries', '').lower()
                    mappings[current_domain] = []
                    continue
                
                # Parse mapping lines
                if line.startswith('- ') and '→' in line and current_domain:
                    try:
                        # Parse: query → source (score) - explanation
                        parts = line[2:].split('→')
                        if len(parts) == 2:
                            query = parts[0].strip()
                            rest = parts[1].strip()
                            
                            # Extract source and score
                            if '(' in rest and ')' in rest:
                                source_part = rest.split('(')[0].strip()
                                score_part = rest.split('(')[1].split(')')[0]
                                explanation = rest.split(')', 1)[1].strip(' -')
                                
                                try:
                                    score = float(score_part)
                                    mappings[current_domain].append({
                                        'query': query,
                                        'source': source_part,
                                        'score': score,
                                        'explanation': explanation,
                                        'domain': current_domain
                                    })
                                except ValueError:
                                    continue
                    except Exception as e:
                        logger.debug(f"Could not parse mapping line: {line} - {e}")
            
            logger.info(f"✅ Loaded {sum(len(v) for v in mappings.values())} training mappings")
            return mappings
            
        except Exception as e:
            logger.error(f"Error loading training mappings: {e}")
            return mappings
    
    async def validate_psychology_kaggle_routing(self) -> Dict[str, any]:
        """Validate that psychology queries correctly route to Kaggle/Zenodo"""
        logger.info("🧪 Testing Psychology → Kaggle/Zenodo routing...")
        
        psychology_queries = [
            "psychology research datasets",
            "mental health data",
            "behavioral psychology",
            "cognitive psychology research",
            "psychological studies data"
        ]
        
        results = {
            'total_queries': len(psychology_queries),
            'correct_routing': 0,
            'incorrect_routing': 0,
            'details': []
        }
        
        for query in psychology_queries:
            result = self.processor.process_query(query)
            
            # Check if classified as psychology domain
            is_psychology_domain = 'psychology' in result.classification.domain.lower()
            
            # Check if top sources include Kaggle or Zenodo
            top_sources = [s.get('name', '').lower() for s in result.recommended_sources[:3]]
            has_kaggle_zenodo = any('kaggle' in source or 'zenodo' in source for source in top_sources)
            
            is_correct = is_psychology_domain and has_kaggle_zenodo
            
            if is_correct:
                results['correct_routing'] += 1
                status = "✅ CORRECT"
            else:
                results['incorrect_routing'] += 1
                status = "❌ INCORRECT"
            
            details = {
                'query': query,
                'domain_detected': result.classification.domain,
                'top_sources': top_sources[:3],
                'is_psychology_domain': is_psychology_domain,
                'has_kaggle_zenodo': has_kaggle_zenodo,
                'status': status
            }
            
            results['details'].append(details)
            
            logger.info(f"   {status}: '{query}' → {result.classification.domain} → {top_sources[:2]}")
        
        accuracy = results['correct_routing'] / results['total_queries']
        results['accuracy'] = accuracy
        
        logger.info(f"✅ Psychology routing accuracy: {accuracy:.1%} ({results['correct_routing']}/{results['total_queries']})")
        
        return results
    
    async def validate_singapore_first_strategy(self) -> Dict[str, any]:
        """Validate Singapore-first strategy with local government data prioritization"""
        logger.info("🧪 Testing Singapore-first strategy...")
        
        singapore_queries = [
            "singapore housing data",
            "sg transport statistics", 
            "singapore government datasets",
            "singapore population data",
            "singapore economic indicators"
        ]
        
        results = {
            'total_queries': len(singapore_queries),
            'correct_singapore_first': 0,
            'incorrect_singapore_first': 0,
            'details': []
        }
        
        for query in singapore_queries:
            result = self.processor.process_query(query)
            
            # Check if Singapore-first strategy is applied
            singapore_first_applied = result.classification.singapore_first_applicable
            
            # Check if top sources are Singapore government sources
            top_sources = [s.get('name', '').lower() for s in result.recommended_sources[:3]]
            singapore_sources = ['data.gov.sg', 'singstat', 'lta']
            has_singapore_sources = any(any(sg_source in source for sg_source in singapore_sources) 
                                      for source in top_sources)
            
            is_correct = singapore_first_applied and has_singapore_sources
            
            if is_correct:
                results['correct_singapore_first'] += 1
                status = "✅ CORRECT"
            else:
                results['incorrect_singapore_first'] += 1
                status = "❌ INCORRECT"
            
            details = {
                'query': query,
                'singapore_first_applied': singapore_first_applied,
                'top_sources': top_sources[:3],
                'has_singapore_sources': has_singapore_sources,
                'status': status
            }
            
            results['details'].append(details)
            
            logger.info(f"   {status}: '{query}' → SG-first: {singapore_first_applied} → {top_sources[:2]}")
        
        accuracy = results['correct_singapore_first'] / results['total_queries']
        results['accuracy'] = accuracy
        
        logger.info(f"✅ Singapore-first accuracy: {accuracy:.1%} ({results['correct_singapore_first']}/{results['total_queries']})")
        
        return results
    
    async def validate_training_mappings_compliance(self) -> Dict[str, any]:
        """Validate comprehensive compliance with training_mappings.md expectations"""
        logger.info("🧪 Testing training mappings compliance...")
        
        results = {
            'total_mappings': 0,
            'compliant_mappings': 0,
            'non_compliant_mappings': 0,
            'domain_results': {},
            'details': []
        }
        
        for domain, mappings in self.training_mappings.items():
            domain_results = {
                'total': len(mappings),
                'compliant': 0,
                'non_compliant': 0,
                'details': []
            }
            
            logger.info(f"   Testing {domain} domain ({len(mappings)} mappings)...")
            
            for mapping in mappings:
                query = mapping['query']
                expected_source = mapping['source'].lower()
                expected_score = mapping['score']
                
                # Process query
                result = self.processor.process_query(query)
                
                # Check domain classification
                detected_domain = result.classification.domain.lower()
                domain_match = domain in detected_domain or detected_domain in domain
                
                # Check source recommendations
                top_sources = [s.get('name', '').lower() for s in result.recommended_sources[:5]]
                source_match = any(expected_source in source or source in expected_source 
                                 for source in top_sources)
                
                # Determine compliance based on expected score
                if expected_score >= 0.8:
                    # High-quality mappings should have both domain and source match
                    is_compliant = domain_match and source_match
                elif expected_score >= 0.6:
                    # Medium-quality mappings should have at least domain match
                    is_compliant = domain_match
                else:
                    # Low-quality mappings should not be top recommendations
                    is_compliant = not (source_match and top_sources.index(next((s for s in top_sources if expected_source in s), '')) == 0)
                
                if is_compliant:
                    domain_results['compliant'] += 1
                    results['compliant_mappings'] += 1
                    status = "✅ COMPLIANT"
                else:
                    domain_results['non_compliant'] += 1
                    results['non_compliant_mappings'] += 1
                    status = "❌ NON-COMPLIANT"
                
                detail = {
                    'query': query,
                    'expected_source': expected_source,
                    'expected_score': expected_score,
                    'detected_domain': detected_domain,
                    'top_sources': top_sources[:3],
                    'domain_match': domain_match,
                    'source_match': source_match,
                    'status': status
                }
                
                domain_results['details'].append(detail)
                results['details'].append(detail)
                
                results['total_mappings'] += 1
            
            domain_accuracy = domain_results['compliant'] / domain_results['total'] if domain_results['total'] > 0 else 0
            domain_results['accuracy'] = domain_accuracy
            results['domain_results'][domain] = domain_results
            
            logger.info(f"     {domain}: {domain_accuracy:.1%} ({domain_results['compliant']}/{domain_results['total']})")
        
        overall_accuracy = results['compliant_mappings'] / results['total_mappings'] if results['total_mappings'] > 0 else 0
        results['overall_accuracy'] = overall_accuracy
        
        logger.info(f"✅ Overall training mappings compliance: {overall_accuracy:.1%} ({results['compliant_mappings']}/{results['total_mappings']})")
        
        return results
    
    async def validate_quality_thresholds(self) -> Dict[str, any]:
        """Validate that system meets quality thresholds"""
        logger.info("🧪 Testing quality thresholds...")
        
        test_queries = [
            "psychology research",
            "singapore housing",
            "climate data",
            "machine learning",
            "health statistics",
            "economic indicators"
        ]
        
        results = {
            'total_queries': len(test_queries),
            'meets_threshold': 0,
            'below_threshold': 0,
            'quality_scores': [],
            'details': []
        }
        
        quality_threshold = 0.7
        
        for query in test_queries:
            result = self.processor.process_query(query)
            
            # Calculate quality score based on confidence and classification
            quality_score = result.processing_confidence
            
            meets_threshold = quality_score >= quality_threshold
            
            if meets_threshold:
                results['meets_threshold'] += 1
                status = "✅ MEETS THRESHOLD"
            else:
                results['below_threshold'] += 1
                status = "❌ BELOW THRESHOLD"
            
            results['quality_scores'].append(quality_score)
            
            detail = {
                'query': query,
                'quality_score': quality_score,
                'threshold': quality_threshold,
                'meets_threshold': meets_threshold,
                'status': status
            }
            
            results['details'].append(detail)
            
            logger.info(f"   {status}: '{query}' → Quality: {quality_score:.2f}")
        
        avg_quality = sum(results['quality_scores']) / len(results['quality_scores'])
        results['average_quality'] = avg_quality
        results['threshold_compliance_rate'] = results['meets_threshold'] / results['total_queries']
        
        logger.info(f"✅ Average quality score: {avg_quality:.2f}")
        logger.info(f"✅ Threshold compliance rate: {results['threshold_compliance_rate']:.1%}")
        
        return results
    
    async def validate_response_time_performance(self) -> Dict[str, any]:
        """Validate response time performance under quality-first approach"""
        logger.info("🧪 Testing response time performance...")
        
        test_queries = [
            "research datasets",
            "singapore data",
            "climate indicators",
            "psychology studies",
            "economic data",
            "health research",
            "education statistics",
            "transport data"
        ]
        
        results = {
            'total_queries': len(test_queries),
            'within_limit': 0,
            'over_limit': 0,
            'response_times': [],
            'details': []
        }
        
        time_limit = 4.0  # 4 seconds as per requirements
        
        for query in test_queries:
            start_time = time.time()
            result = self.processor.process_query(query)
            response_time = time.time() - start_time
            
            within_limit = response_time <= time_limit
            
            if within_limit:
                results['within_limit'] += 1
                status = "✅ WITHIN LIMIT"
            else:
                results['over_limit'] += 1
                status = "❌ OVER LIMIT"
            
            results['response_times'].append(response_time)
            
            detail = {
                'query': query,
                'response_time': response_time,
                'time_limit': time_limit,
                'within_limit': within_limit,
                'status': status
            }
            
            results['details'].append(detail)
            
            logger.info(f"   {status}: '{query}' → {response_time:.2f}s")
        
        avg_response_time = sum(results['response_times']) / len(results['response_times'])
        max_response_time = max(results['response_times'])
        min_response_time = min(results['response_times'])
        
        results['average_response_time'] = avg_response_time
        results['max_response_time'] = max_response_time
        results['min_response_time'] = min_response_time
        results['performance_compliance_rate'] = results['within_limit'] / results['total_queries']
        
        logger.info(f"✅ Average response time: {avg_response_time:.2f}s")
        logger.info(f"✅ Max response time: {max_response_time:.2f}s")
        logger.info(f"✅ Performance compliance rate: {results['performance_compliance_rate']:.1%}")
        
        return results
    
    async def run_comprehensive_validation(self) -> Dict[str, any]:
        """Run comprehensive production quality validation"""
        logger.info("🚀 Starting Comprehensive Production Quality Validation")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all validation tests
        psychology_results = await self.validate_psychology_kaggle_routing()
        singapore_results = await self.validate_singapore_first_strategy()
        mappings_results = await self.validate_training_mappings_compliance()
        quality_results = await self.validate_quality_thresholds()
        performance_results = await self.validate_response_time_performance()
        
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            'validation_timestamp': time.time(),
            'total_validation_time': total_time,
            'psychology_routing': psychology_results,
            'singapore_first_strategy': singapore_results,
            'training_mappings_compliance': mappings_results,
            'quality_thresholds': quality_results,
            'response_time_performance': performance_results
        }
        
        # Calculate overall scores
        overall_scores = {
            'psychology_routing_accuracy': psychology_results['accuracy'],
            'singapore_first_accuracy': singapore_results['accuracy'],
            'training_mappings_compliance': mappings_results['overall_accuracy'],
            'quality_threshold_compliance': quality_results['threshold_compliance_rate'],
            'performance_compliance': performance_results['performance_compliance_rate']
        }
        
        comprehensive_results['overall_scores'] = overall_scores
        
        # Calculate final validation score
        final_score = sum(overall_scores.values()) / len(overall_scores)
        comprehensive_results['final_validation_score'] = final_score
        
        # Determine validation status
        validation_passed = final_score >= 0.8  # 80% threshold for production readiness
        comprehensive_results['validation_passed'] = validation_passed
        
        # Generate summary
        logger.info("\n📊 Comprehensive Production Quality Validation Summary:")
        logger.info("=" * 55)
        logger.info(f"Psychology → Kaggle/Zenodo routing: {psychology_results['accuracy']:.1%}")
        logger.info(f"Singapore-first strategy: {singapore_results['accuracy']:.1%}")
        logger.info(f"Training mappings compliance: {mappings_results['overall_accuracy']:.1%}")
        logger.info(f"Quality threshold compliance: {quality_results['threshold_compliance_rate']:.1%}")
        logger.info(f"Performance compliance: {performance_results['performance_compliance_rate']:.1%}")
        logger.info(f"\nFinal Validation Score: {final_score:.1%}")
        logger.info(f"Validation Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        logger.info(f"Total Validation Time: {total_time:.2f}s")
        
        return comprehensive_results


async def test_production_quality_validation():
    """Run production quality validation test"""
    logger.info("🧪 Running Production Quality Validation Test")
    
    validator = ProductionQualityValidator()
    results = await validator.run_comprehensive_validation()
    
    # Validate results meet production standards
    assert results['validation_passed'], f"Production validation failed with score: {results['final_validation_score']:.1%}"
    
    # Validate specific requirements
    assert results['psychology_routing']['accuracy'] >= 0.7, "Psychology routing accuracy below 70%"
    assert results['singapore_first_strategy']['accuracy'] >= 0.8, "Singapore-first accuracy below 80%"
    assert results['training_mappings_compliance']['overall_accuracy'] >= 0.6, "Training mappings compliance below 60%"
    
    logger.info("✅ Production Quality Validation Test PASSED")
    
    return results


if __name__ == "__main__":
    async def main():
        """Run production quality validation"""
        results = await test_production_quality_validation()
        
        print(f"\n🎉 Production Quality Validation completed!")
        print(f"Final Score: {results['final_validation_score']:.1%}")
        print(f"Status: {'✅ PRODUCTION READY' if results['validation_passed'] else '❌ NEEDS IMPROVEMENT'}")
    
    asyncio.run(main())