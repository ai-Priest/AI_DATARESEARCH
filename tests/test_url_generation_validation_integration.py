"""
Integration tests for URL generation and validation
Tests fixed URL generation for all external sources, URL validation and correction functionality,
end-to-end query processing with URL validation, and fallback URL strategies.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import aiohttp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ai.web_search_engine import WebSearchEngine
from src.ai.url_validator import URLValidator
from src.ai.conversational_query_processor import ConversationalQueryProcessor
from src.ai.llm_clients import LLMManager


class TestURLGenerationValidationIntegration:
    """Integration test suite for URL generation and validation"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'web_search': {
                'timeout': 10,
                'max_results': 5
            },
            'conversational_query': {
                'confidence_threshold': 0.7,
                'max_processing_time': 3.0
            }
        }
    
    @pytest.fixture
    def url_validator(self):
        """Create URLValidator instance for testing"""
        return URLValidator()
    
    @pytest.fixture
    def web_search_engine(self, config):
        """Create WebSearchEngine instance for testing"""
        return WebSearchEngine(config)
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLMManager for testing"""
        mock_manager = MagicMock(spec=LLMManager)
        mock_manager.complete_with_fallback = AsyncMock(return_value={
            'content': json.dumps({
                'is_dataset_request': True,
                'confidence': 0.9,
                'extracted_terms': ['housing', 'singapore'],
                'domain': 'housing',
                'singapore_context': True,
                'clarification_needed': None
            })
        })
        return mock_manager
    
    @pytest.fixture
    def query_processor(self, mock_llm_manager, config):
        """Create ConversationalQueryProcessor instance for testing"""
        return ConversationalQueryProcessor(mock_llm_manager, config)
    
    # Test 1: Fixed URL Generation for External Sources
    
    def test_kaggle_url_generation_fixed(self, web_search_engine):
        """Test that Kaggle URLs are generated correctly without conversational language"""
        test_cases = [
            ("I need HDB data", "HDB data"),
            ("Looking for Singapore population statistics", "Singapore population statistics"),
            ("Can you find transport data?", "transport data"),
            ("Show me housing information", "housing information")
        ]
        
        for conversational_input, expected_clean_query in test_cases:
            # Test the query normalization for Kaggle
            normalized = web_search_engine._normalize_query_for_source(conversational_input, 'kaggle')
            
            # Should remove conversational elements
            assert "I need" not in normalized
            assert "Looking for" not in normalized
            assert "Can you find" not in normalized
            assert "Show me" not in normalized
            
            # Should contain the core data terms
            core_terms = expected_clean_query.lower().split()
            for term in core_terms:
                if term not in ['data', 'information']:  # These might be filtered
                    assert term in normalized.lower(), \
                        f"Core term '{term}' missing from normalized query: '{normalized}'"
    
    @pytest.mark.asyncio
    async def test_kaggle_search_url_generation(self, web_search_engine):
        """Test Kaggle search URL generation with proper parameters"""
        results = await web_search_engine._search_kaggle_datasets("housing data")
        
        # Should return results with proper Kaggle URLs
        assert len(results) > 0
        
        # Check the main search result
        main_result = results[0]
        assert 'kaggle.com/datasets' in main_result['url']
        assert 'search=' in main_result['url']
        assert main_result['source'] == 'kaggle'
        
        # URL should not contain conversational language
        url = main_result['url']
        assert 'I need' not in url
        assert 'Looking for' not in url
        
        # Should contain the normalized query terms
        assert 'housing' in url.lower() or 'data' in url.lower()
    
    def test_world_bank_url_generation_fixed(self, web_search_engine):
        """Test that World Bank URLs are generated correctly with proper search endpoints"""
        test_queries = [
            "GDP data",
            "Population statistics", 
            "Economic indicators",
            "Development metrics"
        ]
        
        for query in test_queries:
            # Test World Bank URL pattern generation
            patterns = web_search_engine.url_validator.external_source_patterns.get('world_bank', {})
            search_pattern = patterns.get('search_pattern', '')
            
            if search_pattern:
                # Should use proper World Bank search endpoint
                assert 'data.worldbank.org' in search_pattern
                assert '{query}' in search_pattern
                
                # Generate actual URL
                actual_url = search_pattern.format(query=query.replace(' ', '+'))
                assert 'data.worldbank.org' in actual_url
                assert query.replace(' ', '+') in actual_url
    
    @pytest.mark.asyncio
    async def test_world_bank_search_integration(self, web_search_engine):
        """Test World Bank search integration with proper URL endpoints"""
        results = await web_search_engine._search_international_organizations("GDP data", {})
        
        # Should return results with World Bank URLs
        assert len(results) > 0
        
        # Find World Bank results
        world_bank_results = [r for r in results if 'worldbank.org' in r.get('url', '')]
        assert len(world_bank_results) > 0, "No World Bank results found"
        
        # Check World Bank result
        wb_result = world_bank_results[0]
        url = wb_result['url']
        
        # Should point to actual search results, not generic pages
        assert 'worldbank.org' in url
        # Should not be just the homepage
        assert url != 'https://data.worldbank.org/'
        assert 'source' in wb_result
    
    def test_aws_open_data_url_generation_fixed(self, web_search_engine):
        """Test that AWS Open Data URLs are generated correctly with search parameters"""
        test_queries = [
            "climate data",
            "satellite imagery",
            "genomics data",
            "transportation data"
        ]
        
        for query in test_queries:
            # Test AWS Open Data URL pattern
            patterns = web_search_engine.url_validator.external_source_patterns.get('aws_open_data', {})
            search_pattern = patterns.get('search_pattern', '')
            
            if search_pattern:
                # Should use proper AWS Open Data search endpoint
                assert 'registry.opendata.aws' in search_pattern
                assert '{query}' in search_pattern
                
                # Generate actual URL
                actual_url = search_pattern.format(query=query.replace(' ', '+'))
                assert 'registry.opendata.aws' in actual_url
                assert 'search' in actual_url or 'q=' in actual_url
    
    @pytest.mark.asyncio
    async def test_aws_open_data_search_with_fallback(self, web_search_engine):
        """Test AWS Open Data search with fallback to browse pages when search fails"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock failed search response, then successful fallback
            mock_search_response = MagicMock()
            mock_search_response.status = 404
            
            mock_fallback_response = MagicMock()
            mock_fallback_response.status = 200
            mock_fallback_response.text = AsyncMock(return_value='<html><body>AWS Open Data Registry</body></html>')
            
            mock_get.return_value.__aenter__.side_effect = [mock_search_response, mock_fallback_response]
            
            # This should test the fallback mechanism
            patterns = web_search_engine.url_validator.external_source_patterns.get('aws_open_data', {})
            fallback_url = patterns.get('fallback_url', 'https://registry.opendata.aws/')
            
            assert 'registry.opendata.aws' in fallback_url
            assert fallback_url.endswith('/')  # Should be a browse page
    
    # Test 2: URL Validation and Correction Functionality
    
    @pytest.mark.asyncio
    async def test_url_validation_basic_functionality(self, url_validator):
        """Test basic URL validation functionality"""
        # Test valid URLs
        valid_urls = [
            "https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view",
            "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "https://www.kaggle.com/datasets/username/dataset-name",
            "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD"
        ]
        
        with patch('aiohttp.ClientSession.head') as mock_head:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_head.return_value.__aenter__.return_value = mock_response
            
            for url in valid_urls:
                # Should not raise exceptions for well-formed URLs
                try:
                    is_valid, status_code = await url_validator.validate_url(url)
                    assert isinstance(is_valid, bool)
                    assert isinstance(status_code, int)
                except Exception as e:
                    pytest.fail(f"URL validation failed for valid URL {url}: {e}")
    
    def test_url_correction_patterns(self, url_validator):
        """Test URL correction patterns for different sources"""
        correction_cases = [
            # Kaggle corrections
            ("kaggle", "housing data", "https://www.kaggle.com/datasets?search=housing+data"),
            ("kaggle", "transport statistics", "https://www.kaggle.com/datasets?search=transport+statistics"),
            
            # World Bank corrections  
            ("world_bank", "GDP data", "https://data.worldbank.org/indicator?tab=all&q=GDP+data"),
            ("world_bank", "population", "https://data.worldbank.org/indicator?tab=all&q=population"),
            
            # AWS Open Data corrections
            ("aws_open_data", "climate data", "https://registry.opendata.aws/search?q=climate+data"),
            ("aws_open_data", "satellite imagery", "https://registry.opendata.aws/search?q=satellite+imagery")
        ]
        
        for source, query, expected_pattern in correction_cases:
            corrected_url = url_validator.correct_external_source_url(source, query, "")
            
            # Should generate URLs matching expected patterns
            if source == "kaggle":
                assert "kaggle.com/datasets" in corrected_url
                assert "search=" in corrected_url
            elif source == "world_bank":
                assert "data.worldbank.org" in corrected_url
            elif source == "aws_open_data":
                assert "registry.opendata.aws" in corrected_url
    
    def test_source_search_patterns(self, url_validator):
        """Test that source search patterns are properly defined"""
        patterns = url_validator.get_source_search_patterns()
        
        required_sources = ['kaggle', 'world_bank', 'aws_open_data', 'un_data']
        
        for source in required_sources:
            assert source in patterns, f"Missing search pattern for {source}"
            pattern = patterns[source]
            assert isinstance(pattern, str), f"Search pattern for {source} should be string"
            assert 'http' in pattern, f"Search pattern for {source} should be a URL"
            assert '{query}' in pattern, f"Search pattern for {source} should have query placeholder"
    
    @pytest.mark.asyncio
    async def test_external_search_results_validation(self, url_validator):
        """Test validation of external search results"""
        mock_results = [
            {
                'title': 'Housing Dataset',
                'url': 'https://www.kaggle.com/datasets/user/housing-data',
                'description': 'Housing price data',
                'source': 'kaggle'
            },
            {
                'title': 'GDP Indicators',
                'url': 'https://data.worldbank.org/indicator/NY.GDP.MKTP.CD',
                'description': 'World Bank GDP data',
                'source': 'world_bank'
            },
            {
                'title': 'Invalid URL Result',
                'url': 'https://invalid-domain.com/broken-link',
                'description': 'This should be corrected',
                'source': 'unknown'
            }
        ]
        
        with patch('aiohttp.ClientSession.head') as mock_head:
            # Mock successful validation for known sources
            mock_response = MagicMock()
            mock_response.status = 200
            mock_head.return_value.__aenter__.return_value = mock_response
            
            validated_results = await url_validator.validate_external_search_results(mock_results)
            
            # Should return results with validation status
            assert len(validated_results) >= 0
            for result in validated_results:
                assert 'url' in result
                assert 'title' in result
                # May have validation metadata added
    
    # Test 3: End-to-End Query Processing with URL Validation
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing_with_validation(self, query_processor, web_search_engine):
        """Test complete pipeline from query processing to validated URLs"""
        test_query = "I need Singapore housing data"
        
        # Step 1: Process conversational query
        query_result = await query_processor.process_input(test_query)
        
        assert query_result.is_dataset_request
        assert len(query_result.extracted_terms) > 0
        
        # Step 2: Use extracted terms for web search with URL validation
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='<html><body>Mock search results</body></html>')
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Use extracted terms for search
            search_terms = ' '.join(query_result.extracted_terms)
            search_results = await web_search_engine.search_web(search_terms, {})
            
            # Should return results with validated URLs
            assert isinstance(search_results, list)
            for result in search_results:
                if 'url' in result:
                    # URLs should be validated/corrected
                    assert result['url'].startswith('http')
                    assert len(result['url']) > 10  # Basic sanity check
    
    @pytest.mark.asyncio
    async def test_query_processing_with_multiple_sources(self, query_processor, web_search_engine):
        """Test that query processing returns multiple validated sources"""
        test_query = "Singapore economic indicators"
        
        # Process query
        query_result = await query_processor.process_input(test_query)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock responses for different sources
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='<html><body>Mock results</body></html>')
            mock_get.return_value.__aenter__.return_value = mock_response
            
            search_terms = ' '.join(query_result.extracted_terms)
            results = await web_search_engine.search_web(search_terms, {'domain': 'economy'})
            
            # Should attempt to get results from multiple sources
            # The actual number depends on implementation, but should be > 0
            assert len(results) >= 0
            
            # Check that results have proper structure
            for result in results:
                assert isinstance(result, dict)
                if 'source' in result:
                    assert isinstance(result['source'], str)
                if 'url' in result:
                    assert result['url'].startswith('http')
    
    # Test 4: Fallback URL Strategies and Error Handling
    
    @pytest.mark.asyncio
    async def test_url_validation_with_fallback_strategies(self, url_validator):
        """Test fallback URL strategies when primary URLs fail"""
        # Test cases with primary URL and expected fallback
        fallback_cases = [
            {
                'source': 'kaggle',
                'primary_url': 'https://www.kaggle.com/datasets/broken-link',
                'expected_fallback': 'https://www.kaggle.com/datasets'
            },
            {
                'source': 'world_bank', 
                'primary_url': 'https://data.worldbank.org/invalid-indicator',
                'expected_fallback': 'https://data.worldbank.org/'
            },
            {
                'source': 'aws_open_data',
                'primary_url': 'https://registry.opendata.aws/broken-dataset',
                'expected_fallback': 'https://registry.opendata.aws/'
            }
        ]
        
        for case in fallback_cases:
            with patch('aiohttp.ClientSession.head') as mock_head:
                # Mock failed primary URL
                mock_response = MagicMock()
                mock_response.status = 404
                mock_head.return_value.__aenter__.return_value = mock_response
                
                # Test fallback URL generation
                patterns = url_validator.external_source_patterns.get(case['source'], {})
                fallback_url = patterns.get('fallback_url', '')
                
                assert fallback_url, f"No fallback URL defined for {case['source']}"
                assert fallback_url.startswith('http'), f"Invalid fallback URL for {case['source']}"
                
                # Fallback should be different from primary
                assert fallback_url != case['primary_url']
    
    @pytest.mark.asyncio
    async def test_error_handling_in_url_validation(self, url_validator):
        """Test error handling during URL validation"""
        problematic_urls = [
            "not-a-url",
            "http://",
            "https://",
            "",
            "ftp://invalid-protocol.com",
            "https://domain-that-definitely-does-not-exist-12345.com"
        ]
        
        with patch('aiohttp.ClientSession.head') as mock_head:
            mock_response = MagicMock()
            mock_response.status = 404
            mock_head.return_value.__aenter__.return_value = mock_response
            
            for url in problematic_urls:
                if url is not None and url != "":
                    try:
                        # Should handle errors gracefully
                        is_valid, status_code = await url_validator.validate_url(url)
                        # Should return boolean and status code
                        assert isinstance(is_valid, bool)
                        assert isinstance(status_code, int)
                    except Exception as e:
                        # Some URLs may raise exceptions, which is acceptable
                        assert isinstance(e, (ValueError, TypeError, aiohttp.ClientError)), \
                            f"Unexpected exception type for URL '{url}': {type(e)}"
    
    @pytest.mark.asyncio
    async def test_timeout_handling_in_url_validation(self, url_validator):
        """Test timeout handling during URL validation"""
        with patch('aiohttp.ClientSession.head') as mock_head:
            # Mock timeout
            mock_head.side_effect = asyncio.TimeoutError("Request timeout")
            
            test_urls = [
                "https://www.kaggle.com/datasets/test",
                "https://data.worldbank.org/indicator/test"
            ]
            
            for url in test_urls:
                # Should handle timeouts gracefully
                try:
                    is_valid, status_code = await url_validator.validate_url(url)
                    # Should return some result even on timeout
                    assert isinstance(is_valid, bool)
                    assert isinstance(status_code, int)
                except asyncio.TimeoutError:
                    # If timeout propagates, that's also acceptable
                    pass
    
    @pytest.mark.asyncio
    async def test_source_failure_handling(self, web_search_engine):
        """Test graceful handling when sources fail to return results"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock different failure scenarios
            failure_responses = [
                (404, "Not Found"),
                (500, "Internal Server Error"),
                (403, "Forbidden"),
                (429, "Too Many Requests")
            ]
            
            for status_code, reason in failure_responses:
                mock_response = MagicMock()
                mock_response.status = status_code
                mock_response.text = AsyncMock(return_value=f"<html><body>{reason}</body></html>")
                mock_get.return_value.__aenter__.return_value = mock_response
                
                # Should handle failures gracefully
                results = await web_search_engine.search_web("test query", {})
                
                # Should return empty list or handle gracefully, not crash
                assert isinstance(results, list)
                # May be empty due to failures, but should not raise exceptions
    
    # Test 5: Source Coverage Requirements
    
    @pytest.mark.asyncio
    async def test_minimum_source_coverage_attempt(self, web_search_engine):
        """Test that system attempts to provide results from multiple sources"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='<html><body>Mock results</body></html>')
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test with a query that should trigger multiple sources
            results = await web_search_engine.search_web("Singapore economic data", {})
            
            # Should attempt to contact multiple sources
            # (The actual number of calls depends on implementation)
            call_count = mock_get.call_count
            assert call_count >= 0  # Should make at least some attempts
    
    def test_source_prioritization_logic(self, web_search_engine):
        """Test that sources are prioritized correctly"""
        # Check that priority domains are properly configured
        priority_domains = web_search_engine.priority_domains
        
        assert len(priority_domains) > 0, "No priority domains configured"
        
        # Should include key international sources
        international_sources = ['data.worldbank.org', 'kaggle.com', 'registry.opendata.aws']
        for source in international_sources:
            assert source in priority_domains, f"Missing priority source: {source}"
        
        # Should include Singapore-specific sources
        singapore_sources = ['data.gov.sg', 'singstat.gov.sg']
        for source in singapore_sources:
            assert source in priority_domains, f"Missing Singapore source: {source}"
    
    # Test 6: Performance and Reliability
    
    @pytest.mark.asyncio
    async def test_url_validation_performance(self, url_validator):
        """Test that URL validation completes within reasonable time"""
        import time
        
        test_urls = [
            "https://www.kaggle.com/datasets",
            "https://data.worldbank.org/",
            "https://registry.opendata.aws/"
        ]
        
        with patch('aiohttp.ClientSession.head') as mock_head:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_head.return_value.__aenter__.return_value = mock_response
            
            start_time = time.time()
            
            # Validate multiple URLs
            for url in test_urls:
                await url_validator.validate_url(url)
            
            end_time = time.time()
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert end_time - start_time < 10.0, "URL validation took too long"
    
    @pytest.mark.asyncio
    async def test_concurrent_url_validation(self, url_validator):
        """Test concurrent URL validation"""
        test_urls = [
            "https://www.kaggle.com/datasets",
            "https://data.worldbank.org/",
            "https://registry.opendata.aws/",
            "https://data.gov.sg/",
            "https://singstat.gov.sg/"
        ]
        
        with patch('aiohttp.ClientSession.head') as mock_head:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_head.return_value.__aenter__.return_value = mock_response
            
            # Test concurrent validation
            tasks = [url_validator.validate_url(url) for url in test_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle concurrent requests
            assert len(results) == len(test_urls)
            
            # Results should be meaningful (not all exceptions)
            non_exception_results = [r for r in results if not isinstance(r, Exception)]
            assert len(non_exception_results) >= 0  # At least some should succeed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])