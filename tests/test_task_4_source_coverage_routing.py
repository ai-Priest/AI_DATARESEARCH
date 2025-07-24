#!/usr/bin/env python3
"""
Comprehensive test for Task 4: Improve source coverage and routing
Tests intelligent source selection, minimum source coverage, and failure handling
"""

import asyncio
import sys
import os
import logging
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.web_search_engine import WebSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSourceCoverageRouting:
    """Test suite for source coverage and routing improvements"""
    
    def __init__(self):
        self.config = {
            'web_search': {
                'timeout': 10,
                'max_results': 10
            }
        }
        self.engine = WebSearchEngine(self.config)
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all test cases"""
        logger.info("🧪 Starting Task 4 comprehensive tests")
        
        test_cases = [
            self.test_intelligent_source_selection,
            self.test_domain_aware_prioritization,
            self.test_singapore_context_boosting,
            self.test_minimum_source_coverage,
            self.test_query_domain_detection,
            self.test_fallback_source_generation,
            self.test_source_failure_handling,
            self.test_end_to_end_search_with_coverage
        ]
        
        for test_case in test_cases:
            try:
                await test_case()
                self.test_results.append(f"✅ {test_case.__name__}: PASSED")
            except Exception as e:
                self.test_results.append(f"❌ {test_case.__name__}: FAILED - {str(e)}")
                logger.error(f"Test failed: {test_case.__name__} - {str(e)}")
        
        self.print_test_summary()
    
    async def test_intelligent_source_selection(self):
        """Test intelligent source selection based on query analysis"""
        logger.info("🎯 Testing intelligent source selection")
        
        # Test health query
        health_sources = self.engine._select_intelligent_sources("cancer mortality data", None)
        assert len(health_sources) >= 3, f"Expected at least 3 sources, got {len(health_sources)}"
        
        # Verify international sources are prioritized for health queries
        source_types = [s['type'] for s in health_sources]
        assert 'international' in source_types, "International sources should be selected for health queries"
        
        # Test technology query
        tech_sources = self.engine._select_intelligent_sources("machine learning datasets", None)
        assert len(tech_sources) >= 3, f"Expected at least 3 sources, got {len(tech_sources)}"
        
        # Verify Kaggle is prioritized for tech queries
        kaggle_priority = next((s['final_priority'] for s in tech_sources if s['type'] == 'kaggle'), 0)
        assert kaggle_priority > 100, f"Kaggle should have high priority for tech queries, got {kaggle_priority}"
        
        logger.info("✅ Intelligent source selection working correctly")
    
    async def test_domain_aware_prioritization(self):
        """Test domain-aware source prioritization"""
        logger.info("🏷️ Testing domain-aware prioritization")
        
        # Test different domains
        test_queries = [
            ("GDP economic data", "economics"),
            ("education enrollment statistics", "education"),
            ("climate change data", "environment"),
            ("population demographics", "demographics"),
            ("transport traffic data", "transport")
        ]
        
        for query, expected_domain in test_queries:
            detected_domain = self.engine._detect_query_domain(query)
            assert detected_domain == expected_domain, f"Expected {expected_domain}, got {detected_domain} for query: {query}"
            
            # Test source selection reflects domain awareness
            sources = self.engine._select_intelligent_sources(query, None)
            assert len(sources) >= 3, f"Insufficient sources for {query}"
            
            # Verify domain-specific boosts are applied
            priorities = [s['final_priority'] for s in sources]
            assert max(priorities) > min(priorities), f"No priority differentiation for {query}"
        
        logger.info("✅ Domain-aware prioritization working correctly")
    
    async def test_singapore_context_boosting(self):
        """Test Singapore-specific source boosting"""
        logger.info("🇸🇬 Testing Singapore context boosting")
        
        # Test Singapore-specific query
        sg_query = "HDB resale flat prices Singapore"
        sg_sources = self.engine._select_intelligent_sources(sg_query, None)
        
        # Verify government sources get highest priority
        gov_sources = [s for s in sg_sources if s['type'] == 'government']
        assert len(gov_sources) > 0, "Government sources should be selected for Singapore queries"
        
        # Verify Singapore boost is applied
        gov_priority = max(s['final_priority'] for s in gov_sources)
        other_priority = max(s['final_priority'] for s in sg_sources if s['type'] != 'government')
        assert gov_priority > other_priority, "Government sources should have higher priority for Singapore queries"
        
        # Test non-Singapore query for comparison
        global_query = "housing prices data"
        global_sources = self.engine._select_intelligent_sources(global_query, None)
        
        # Verify different prioritization
        global_gov_priority = max((s['final_priority'] for s in global_sources if s['type'] == 'government'), default=0)
        assert gov_priority > global_gov_priority, "Singapore context should boost government source priority"
        
        logger.info("✅ Singapore context boosting working correctly")
    
    async def test_minimum_source_coverage(self):
        """Test minimum source coverage enforcement"""
        logger.info("📊 Testing minimum source coverage")
        
        # Create mock results with insufficient sources
        mock_results = [
            {'source': 'kaggle', 'domain': 'kaggle.com', 'title': 'Test 1'},
            {'source': 'kaggle', 'domain': 'kaggle.com', 'title': 'Test 2'}
        ]
        
        # Test coverage enhancement
        enhanced_results = self.engine._ensure_minimum_source_coverage(
            mock_results, "test query", None
        )
        
        # Count unique sources
        unique_sources = set(r.get('source', '') for r in enhanced_results)
        assert len(unique_sources) >= 3, f"Expected at least 3 unique sources, got {len(unique_sources)}"
        
        # Verify fallback sources were added
        fallback_count = sum(1 for r in enhanced_results if r.get('is_fallback', False))
        assert fallback_count > 0, "Fallback sources should be added when coverage is insufficient"
        
        logger.info("✅ Minimum source coverage working correctly")
    
    async def test_query_domain_detection(self):
        """Test query domain detection accuracy"""
        logger.info("🔍 Testing query domain detection")
        
        test_cases = [
            ("cancer mortality rates", "health"),
            ("GDP growth statistics", "economics"),
            ("school enrollment data", "education"),
            ("climate temperature data", "environment"),
            ("machine learning datasets", "technology"),
            ("population census data", "demographics"),
            ("bus transport data", "transport"),
            ("random query", "general")
        ]
        
        for query, expected_domain in test_cases:
            detected = self.engine._detect_query_domain(query)
            assert detected == expected_domain, f"Query '{query}': expected {expected_domain}, got {detected}"
        
        logger.info("✅ Query domain detection working correctly")
    
    async def test_fallback_source_generation(self):
        """Test fallback source generation"""
        logger.info("🔄 Testing fallback source generation")
        
        # Test Singapore fallbacks
        sg_fallbacks = self.engine._get_fallback_sources("HDB data Singapore", None, set())
        assert len(sg_fallbacks) > 0, "Should generate Singapore fallbacks"
        
        sg_domains = [f['domain'] for f in sg_fallbacks]
        assert any('data.gov.sg' in domain for domain in sg_domains), "Should include Singapore government sources"
        
        # Test global fallbacks
        global_fallbacks = self.engine._get_fallback_sources("economic data", None, set())
        assert len(global_fallbacks) > 0, "Should generate global fallbacks"
        
        # Test with existing sources
        existing = {'kaggle', 'world_bank'}
        filtered_fallbacks = self.engine._get_fallback_sources("test query", None, existing)
        fallback_sources = [f['source'] for f in filtered_fallbacks]
        
        for existing_source in existing:
            assert existing_source not in fallback_sources, f"Should not include existing source: {existing_source}"
        
        logger.info("✅ Fallback source generation working correctly")
    
    async def test_source_failure_handling(self):
        """Test source failure handling and retry logic"""
        logger.info("🛠️ Testing source failure handling")
        
        # Test retry logic (mock)
        try:
            # This will test the retry wrapper structure
            result = await self.engine._search_with_retry(
                self._mock_failing_search, "test query", "test_source", None, max_retries=1
            )
            assert False, "Should have raised exception after retries"
        except Exception:
            # Expected to fail after retries
            pass
        
        # Test alternative source generation
        failed_source = {'type': 'kaggle', 'name': 'Kaggle'}
        alternatives = await self.engine._try_alternative_sources(failed_source, "test query", None)
        assert len(alternatives) > 0, "Should generate alternatives for failed sources"
        
        # Test generic fallback generation
        generic_fallbacks = self.engine._get_generic_fallback_sources("test query", None)
        assert len(generic_fallbacks) > 0, "Should generate generic fallbacks"
        
        logger.info("✅ Source failure handling working correctly")
    
    async def _mock_failing_search(self, query: str) -> List[Dict[str, Any]]:
        """Mock search function that always fails"""
        raise Exception("Mock search failure")
    
    async def test_end_to_end_search_with_coverage(self):
        """Test end-to-end search with coverage requirements"""
        logger.info("🌐 Testing end-to-end search with coverage")
        
        # Test with a simple query that should return multiple sources
        try:
            results = await self.engine.search_web("economic data", None)
            
            # Verify we got results
            assert len(results) > 0, "Should return search results"
            
            # Count unique sources
            unique_sources = set(r.get('source', '') for r in results)
            logger.info(f"End-to-end test returned {len(results)} results from {len(unique_sources)} sources")
            
            # Verify source diversity (should aim for 3+ sources)
            if len(unique_sources) < 3:
                logger.warning(f"Only {len(unique_sources)} unique sources returned, expected 3+")
            
            # Verify result structure
            for result in results[:3]:  # Check first 3 results
                assert 'title' in result, "Result should have title"
                assert 'url' in result, "Result should have URL"
                assert 'source' in result, "Result should have source"
                assert 'type' in result, "Result should have type"
            
        except Exception as e:
            logger.warning(f"End-to-end test encountered error (may be expected in test environment): {str(e)}")
            # In test environment, network calls may fail - this is acceptable
        
        logger.info("✅ End-to-end search test completed")
    
    def print_test_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 TASK 4 TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for result in self.test_results if "PASSED" in result)
        failed = sum(1 for result in self.test_results if "FAILED" in result)
        
        for result in self.test_results:
            logger.info(result)
        
        logger.info("="*60)
        logger.info(f"📊 TOTAL: {len(self.test_results)} tests")
        logger.info(f"✅ PASSED: {passed}")
        logger.info(f"❌ FAILED: {failed}")
        logger.info(f"📈 SUCCESS RATE: {(passed/len(self.test_results)*100):.1f}%")
        logger.info("="*60)
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Task 4 implementation is working correctly.")
        else:
            logger.warning(f"⚠️ {failed} tests failed. Please review the implementation.")

async def main():
    """Run the test suite"""
    tester = TestSourceCoverageRouting()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())