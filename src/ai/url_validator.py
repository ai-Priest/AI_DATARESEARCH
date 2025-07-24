"""
URL Validator and Corrector for Singapore Dataset Links and External Sources
Ensures all dataset URLs are active and working, with support for external sources
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class URLValidator:
    """Validates and corrects dataset URLs to ensure they work"""
    
    def __init__(self):
        self.timeout = 10
        self.user_agent = "AI Dataset Research Assistant/2.0"
        
        # External source URL patterns for correction
        self.external_source_patterns = self._initialize_external_patterns()
        
        # Known working URLs for key Singapore datasets
        self.url_corrections = {
            # HDB Data and Housing
            "hdb_resale_prices": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "hdb_resale_flat_prices": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "resale_flat_prices": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "housing_prices": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "housing": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "property_prices": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            "residential_property": "https://tablebuilder.singstat.gov.sg/table/TS/M212161",
            
            # Transport Data - User-accessible government data portals
            "lta_bus_arrival": "https://data.gov.sg/search?query=transport",
            "lta_traffic_incidents": "https://data.gov.sg/search?query=traffic", 
            "lta_taxi_availability": "https://data.gov.sg/search?query=transport",
            "bus_routes": "https://data.gov.sg/search?query=bus",
            "train_service_alerts": "https://data.gov.sg/search?query=train",
            
            # Population & Demographics
            "population_trends": "https://tablebuilder.singstat.gov.sg/table/TS/M810001",
            "demographic_profile": "https://tablebuilder.singstat.gov.sg/table/TS/M810001",
            
            # Economic Data
            "gdp_data": "https://tablebuilder.singstat.gov.sg/table/TS/M015721",
            "cpi_data": "https://tablebuilder.singstat.gov.sg/table/TS/M212881",
            "employment_data": "https://tablebuilder.singstat.gov.sg/table/TS/M182001",
            
            # Education
            "school_directory": "https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view",
            "education_statistics": "https://tablebuilder.singstat.gov.sg/table/TS/M850001",
            
            # Healthcare
            "hospital_bed_occupancy": "https://data.gov.sg/datasets/d_c2294012eac8b4446d33d44f3e58e49a/view",
            
            # Weather & Environment
            "weather_data": "https://data.gov.sg/datasets/d_31253b1c6ba96e4dd2b8218db4e7c0d5/view",
            "air_quality": "https://data.gov.sg/datasets/d_31253b1c6ba96e4dd2b8218db4e7c0d5/view",
        }
        
        # URL patterns for different data sources
        self.url_patterns = {
            "data.gov.sg": "https://data.gov.sg/datasets/{dataset_id}/view",
            "tablebuilder.singstat.gov.sg": "https://tablebuilder.singstat.gov.sg/table/TS/{table_id}",
            "datamall2.mytransport.sg": "https://datamall2.mytransport.sg/ltaodataservice/{endpoint}",
        }
    
    def _initialize_external_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize URL patterns for external data sources"""
        return {
            'kaggle': {
                'search_pattern': 'https://www.kaggle.com/datasets?search={query}',
                'dataset_pattern': 'https://www.kaggle.com/datasets/{username}/{dataset_name}',
                'competition_pattern': 'https://www.kaggle.com/competitions/{competition_name}',
                'fallback_url': 'https://www.kaggle.com/datasets',
                'validation_domains': ['kaggle.com', 'www.kaggle.com']
            },
            'world_bank': {
                'search_pattern': 'https://data.worldbank.org/indicator?tab=all&q={query}',
                'indicator_pattern': 'https://data.worldbank.org/indicator/{indicator_code}',
                'country_pattern': 'https://data.worldbank.org/country/{country_code}',
                'fallback_url': 'https://data.worldbank.org/',
                'validation_domains': ['data.worldbank.org', 'worldbank.org']
            },
            'aws_open_data': {
                'search_pattern': 'https://registry.opendata.aws/search?q={query}',
                'dataset_pattern': 'https://registry.opendata.aws/{dataset_name}',
                'fallback_url': 'https://registry.opendata.aws/',
                'validation_domains': ['registry.opendata.aws']
            },
            'un_data': {
                'search_pattern': 'https://data.un.org/Search.aspx?q={query}',
                'dataset_pattern': 'https://data.un.org/Data.aspx?d={dataset_id}',
                'fallback_url': 'https://data.un.org/',
                'validation_domains': ['data.un.org', 'unstats.un.org']
            },
            'who': {
                'search_pattern': 'https://www.who.int/data/gho/data/themes',
                'indicator_pattern': 'https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}',
                'fallback_url': 'https://www.who.int/data/gho',
                'validation_domains': ['who.int', 'www.who.int']
            },
            'oecd': {
                'search_pattern': 'https://data.oecd.org/searchresults/?q={query}',
                'dataset_pattern': 'https://data.oecd.org/{category}/{indicator}.htm',
                'fallback_url': 'https://data.oecd.org/',
                'validation_domains': ['data.oecd.org', 'oecd.org']
            },
            'huggingface': {
                'search_pattern': 'https://huggingface.co/datasets?search={query}',
                'dataset_pattern': 'https://huggingface.co/datasets/{username}/{dataset_name}',
                'fallback_url': 'https://huggingface.co/datasets',
                'validation_domains': ['huggingface.co']
            }
        }
    
    def correct_external_source_url(self, source: str, query: str, current_url: str) -> str:
        """
        Correct and validate external source URLs
        
        Args:
            source: Source name (kaggle, world_bank, aws_open_data, etc.)
            query: Search query or dataset identifier
            current_url: Current URL (may be broken)
            
        Returns:
            Corrected, working URL
        """
        source_lower = source.lower()
        
        # Handle source name variations
        source_mapping = {
            'kaggle': 'kaggle',
            'world_bank': 'world_bank', 
            'worldbank': 'world_bank',
            'aws': 'aws_open_data',
            'aws_open_data': 'aws_open_data',
            'un': 'un_data',
            'un_data': 'un_data',
            'who': 'who',
            'oecd': 'oecd',
            'huggingface': 'huggingface',
            'hf': 'huggingface'
        }
        
        normalized_source = source_mapping.get(source_lower, source_lower)
        
        if normalized_source not in self.external_source_patterns:
            logger.warning(f"Unknown external source: {source}")
            return current_url
        
        pattern_config = self.external_source_patterns[normalized_source]
        
        # Clean and normalize the query
        clean_query = self._clean_query_for_url(query)
        
        # Try to fix the current URL first
        fixed_url = self._fix_external_url(current_url, normalized_source, clean_query)
        if fixed_url != current_url:
            logger.info(f"🔧 External URL fixed: {current_url} → {fixed_url}")
            return fixed_url
        
        # Generate new URL using search pattern
        try:
            search_url = pattern_config['search_pattern'].format(query=quote_plus(clean_query))
            logger.info(f"🆕 External URL generated for {source}: {search_url}")
            return search_url
        except Exception as e:
            logger.warning(f"Failed to generate URL for {source}: {str(e)}")
            return pattern_config['fallback_url']
    
    def _clean_query_for_url(self, query: str) -> str:
        """Clean query for URL generation"""
        if not query:
            return ""
        
        # Remove conversational language
        clean_query = query.lower().strip()
        
        # Remove common conversational patterns
        patterns_to_remove = [
            r'^(i need|i want|looking for|find me|get me|show me)\s+',
            r'^(can you|could you|please)\s+',
            r'\s+(please|thanks|thank you)$',
            r'\s+(data|dataset|datasets)$'
        ]
        
        for pattern in patterns_to_remove:
            clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE).strip()
        
        # Clean up multiple spaces
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        return clean_query if clean_query else query
    
    def _fix_external_url(self, url: str, source: str, query: str) -> str:
        """Fix common issues in external source URLs"""
        if not url or not url.startswith('http'):
            return url
        
        pattern_config = self.external_source_patterns.get(source, {})
        valid_domains = pattern_config.get('validation_domains', [])
        
        parsed_url = urlparse(url)
        
        # Check if domain is valid for this source
        if valid_domains and parsed_url.netloc not in valid_domains:
            # Domain mismatch - generate new URL
            return pattern_config.get('search_pattern', '').format(query=quote_plus(query))
        
        # Source-specific URL fixes
        if source == 'kaggle':
            return self._fix_kaggle_url(url, query)
        elif source == 'world_bank':
            return self._fix_world_bank_url(url, query)
        elif source == 'aws_open_data':
            return self._fix_aws_url(url, query)
        elif source == 'un_data':
            return self._fix_un_data_url(url, query)
        
        return url
    
    def _fix_kaggle_url(self, url: str, query: str) -> str:
        """Fix Kaggle-specific URL issues"""
        # Ensure proper Kaggle search format
        if 'kaggle.com' in url:
            if '/datasets' not in url:
                return f"https://www.kaggle.com/datasets?search={quote_plus(query)}"
            elif 'search=' not in url and query:
                # Add search parameter if missing
                separator = '&' if '?' in url else '?'
                return f"{url}{separator}search={quote_plus(query)}"
        return url
    
    def _fix_world_bank_url(self, url: str, query: str) -> str:
        """Fix World Bank-specific URL issues"""
        if 'worldbank.org' in url:
            # Ensure we're using the data portal
            if 'data.worldbank.org' not in url:
                return f"https://data.worldbank.org/indicator?tab=all&q={quote_plus(query)}"
            # Fix search parameter format
            if '/indicator' in url and 'q=' not in url and query:
                separator = '&' if '?' in url else '?'
                return f"{url}{separator}q={quote_plus(query)}"
        return url
    
    def _fix_aws_url(self, url: str, query: str) -> str:
        """Fix AWS Open Data-specific URL issues"""
        if 'opendata.aws' in url:
            # Ensure proper registry domain
            if 'registry.opendata.aws' not in url:
                return f"https://registry.opendata.aws/search?q={quote_plus(query)}"
            # Add search parameter if missing
            if '/search' not in url and query:
                return f"https://registry.opendata.aws/search?q={quote_plus(query)}"
        return url
    
    def _fix_un_data_url(self, url: str, query: str) -> str:
        """Fix UN Data-specific URL issues"""
        if 'un.org' in url:
            # Ensure proper data portal
            if 'data.un.org' not in url:
                return f"https://data.un.org/Search.aspx?q={quote_plus(query)}"
            # Fix search parameter format
            if 'Search.aspx' not in url and query:
                return f"https://data.un.org/Search.aspx?q={quote_plus(query)}"
        return url
    
    def get_source_search_patterns(self) -> Dict[str, str]:
        """Get search URL patterns for all supported external sources"""
        patterns = {}
        for source, config in self.external_source_patterns.items():
            patterns[source] = config.get('search_pattern', '')
        return patterns
    
    async def validate_external_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Validate and correct search result URLs from external sources with real-time validation
        
        Args:
            results: List of search results with URLs
            
        Returns:
            List of validated and corrected results
        """
        validated_results = []
        validation_tasks = []
        
        # Create validation tasks for concurrent processing
        for i, result in enumerate(results):
            if result.get('url'):
                validation_tasks.append(self._validate_single_result(result, i))
        
        # Execute validations concurrently with error handling
        try:
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for i, validation_result in enumerate(validation_results):
                if isinstance(validation_result, Exception):
                    logger.error(f"Validation task {i} failed: {str(validation_result)}")
                    # Create error result
                    error_result = results[i].copy() if i < len(results) else {}
                    error_result.update({
                        'url_status': 'validation_error',
                        'status_code': 0,
                        'validation_timestamp': datetime.now().isoformat(),
                        'validation_error': str(validation_result)
                    })
                    validated_results.append(error_result)
                else:
                    validated_results.append(validation_result)
                    
        except Exception as e:
            logger.error(f"Critical error in validation process: {str(e)}")
            # Return original results with error status
            for result in results:
                error_result = result.copy()
                error_result.update({
                    'url_status': 'critical_error',
                    'status_code': 0,
                    'validation_timestamp': datetime.now().isoformat(),
                    'validation_error': str(e)
                })
                validated_results.append(error_result)
        
        return validated_results
    
    async def _validate_single_result(self, result: Dict, index: int) -> Dict:
        """
        Validate a single search result with comprehensive error handling
        
        Args:
            result: Single search result dictionary
            index: Result index for logging
            
        Returns:
            Validated result dictionary
        """
        try:
            # Create a copy to avoid modifying original
            validated_result = result.copy()
            
            original_url = result.get('url', '')
            source = result.get('source', '').lower()
            title = result.get('title', '')
            
            if not original_url:
                validated_result.update({
                    'url_status': 'no_url',
                    'status_code': 0,
                    'validation_timestamp': datetime.now().isoformat()
                })
                return validated_result
            
            # Real-time URL validation
            is_valid, status_code = await self.validate_url_with_retry(original_url)
            
            if is_valid:
                validated_result.update({
                    'url_status': 'verified',
                    'status_code': status_code,
                    'validation_timestamp': datetime.now().isoformat()
                })
                logger.debug(f"✅ URL verified [{index}]: {original_url}")
            else:
                # Attempt URL correction
                corrected_result = await self._attempt_url_correction(
                    validated_result, source, title, original_url, status_code
                )
                validated_result = corrected_result
                logger.info(f"🔧 URL correction attempted [{index}]: {original_url} → {corrected_result.get('url')}")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Error validating single result [{index}]: {str(e)}")
            error_result = result.copy()
            error_result.update({
                'url_status': 'validation_exception',
                'status_code': 0,
                'validation_timestamp': datetime.now().isoformat(),
                'validation_error': str(e)
            })
            return error_result
    
    async def _attempt_url_correction(self, result: Dict, source: str, title: str, 
                                    original_url: str, original_status: int) -> Dict:
        """
        Attempt to correct a failed URL with multiple strategies
        
        Args:
            result: Result dictionary to update
            source: Source name
            title: Result title
            original_url: Original URL that failed
            original_status: HTTP status code from original validation
            
        Returns:
            Updated result dictionary
        """
        try:
            # Strategy 1: Source-specific URL correction
            if source in self.external_source_patterns:
                query = self._extract_query_from_title(title)
                corrected_url = self.correct_external_source_url(source, query, original_url)
                
                if corrected_url != original_url:
                    # Validate the corrected URL
                    corrected_valid, corrected_status = await self.validate_url_with_retry(corrected_url)
                    
                    if corrected_valid:
                        result.update({
                            'url': corrected_url,
                            'url_status': 'corrected_verified',
                            'status_code': corrected_status,
                            'validation_timestamp': datetime.now().isoformat(),
                            'original_url': original_url
                        })
                        return result
            
            # Strategy 2: Fallback to source homepage
            fallback_url = self._get_external_source_fallback(source, title)
            if fallback_url:
                fallback_valid, fallback_status = await self.validate_url_with_retry(fallback_url)
                
                if fallback_valid:
                    result.update({
                        'url': fallback_url,
                        'url_status': 'fallback_verified',
                        'status_code': fallback_status,
                        'validation_timestamp': datetime.now().isoformat(),
                        'original_url': original_url
                    })
                    return result
            
            # Strategy 3: Mark as failed but keep original
            result.update({
                'url_status': 'failed_validation',
                'status_code': original_status,
                'validation_timestamp': datetime.now().isoformat(),
                'correction_attempted': True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in URL correction: {str(e)}")
            result.update({
                'url_status': 'correction_error',
                'status_code': original_status,
                'validation_timestamp': datetime.now().isoformat(),
                'correction_error': str(e)
            })
            return result
    
    async def validate_url_with_retry(self, url: str, max_retries: int = 2) -> Tuple[bool, int]:
        """
        Validate URL with retry logic and comprehensive error handling
        
        Args:
            url: URL to validate
            max_retries: Maximum number of retry attempts
            
        Returns:
            (is_valid, status_code)
        """
        for attempt in range(max_retries + 1):
            try:
                is_valid, status_code = await self.validate_url(url)
                
                # If successful or client error (4xx), don't retry
                if is_valid or (400 <= status_code < 500):
                    return is_valid, status_code
                
                # For server errors (5xx) or network issues, retry with backoff
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                    logger.debug(f"Retrying URL validation in {wait_time}s: {url}")
                    await asyncio.sleep(wait_time)
                    continue
                
                return is_valid, status_code
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 0.5
                    logger.debug(f"Validation error, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Final validation attempt failed for {url}: {str(e)}")
                    return False, 0
        
        return False, 0
    
    def _get_external_source_fallback(self, source: str, title: str) -> Optional[str]:
        """
        Get fallback URL for external sources when direct links fail
        
        Args:
            source: Source name
            title: Result title for context
            
        Returns:
            Fallback URL or None
        """
        source_lower = source.lower()
        
        # Map source variations to standard names
        source_mapping = {
            'kaggle': 'kaggle',
            'world_bank': 'world_bank',
            'worldbank': 'world_bank',
            'aws': 'aws_open_data',
            'aws_open_data': 'aws_open_data',
            'un': 'un_data',
            'un_data': 'un_data',
            'who': 'who',
            'oecd': 'oecd',
            'huggingface': 'huggingface',
            'hf': 'huggingface'
        }
        
        normalized_source = source_mapping.get(source_lower, source_lower)
        
        if normalized_source in self.external_source_patterns:
            return self.external_source_patterns[normalized_source].get('fallback_url')
        
        return None
    
    def _extract_query_from_title(self, title: str) -> str:
        """Extract search query from result title"""
        if not title:
            return ""
        
        # Remove common prefixes and suffixes
        clean_title = title.lower()
        
        # Remove source prefixes
        prefixes_to_remove = [
            'kaggle datasets:', 'world bank:', 'aws open data:', 
            'un data:', 'who data:', 'oecd data:', 'hugging face:'
        ]
        
        for prefix in prefixes_to_remove:
            if clean_title.startswith(prefix):
                clean_title = clean_title[len(prefix):].strip()
                break
        
        # Remove common suffixes
        suffixes_to_remove = [
            '- kaggle', '- world bank', '- aws', '- un', '- who', '- oecd'
        ]
        
        for suffix in suffixes_to_remove:
            if clean_title.endswith(suffix):
                clean_title = clean_title[:-len(suffix)].strip()
                break
        
        return clean_title

    def correct_url(self, dataset_id: str, current_url: str, title: str = "") -> str:
        """
        Correct and validate dataset URLs
        
        Args:
            dataset_id: Dataset identifier
            current_url: Current URL (may be broken)
            title: Dataset title for context
            
        Returns:
            Corrected, working URL
        """
        # Clean up dataset_id and title for matching
        clean_id = dataset_id.lower().replace("-", "_").replace(" ", "_")
        clean_title = title.lower().replace("-", "_").replace(" ", "_")
        
        # Check direct ID mappings first
        if clean_id in self.url_corrections:
            corrected_url = self.url_corrections[clean_id]
            logger.info(f"🔗 URL corrected for {dataset_id}: {corrected_url}")
            return corrected_url
        
        # Check title-based mappings
        for key, url in self.url_corrections.items():
            if key in clean_title or any(word in clean_title for word in key.split("_")):
                logger.info(f"🔗 URL corrected via title match for {title}: {url}")
                return url
        
        # Fix incomplete URLs
        fixed_url = self._fix_incomplete_url(current_url)
        if fixed_url != current_url:
            logger.info(f"🔧 URL fixed: {current_url} → {fixed_url}")
            return fixed_url
        
        # Generate URL based on source patterns
        generated_url = self._generate_url_from_patterns(dataset_id, current_url, title)
        if generated_url:
            logger.info(f"🆕 URL generated: {generated_url}")
            return generated_url
        
        # Return original if no corrections found
        logger.warning(f"⚠️ No URL correction found for {dataset_id}")
        return current_url
    
    def _fix_incomplete_url(self, url: str) -> str:
        """Fix common URL issues"""
        if not url:
            return url
            
        # Add protocol if missing
        if url.startswith("data.gov.sg") or url.startswith("tablebuilder.singstat.gov.sg"):
            return f"https://{url}"
        elif url.startswith("www."):
            return f"https://{url}"
        elif not url.startswith(("http://", "https://")):
            if "gov.sg" in url:
                return f"https://{url}"
        
        return url
    
    def _generate_url_from_patterns(self, dataset_id: str, current_url: str, title: str) -> Optional[str]:
        """Generate URL based on known patterns"""
        
        # For data.gov.sg datasets
        if "data.gov.sg" in current_url or not current_url.startswith("http"):
            if dataset_id.startswith("d_"):
                return f"https://data.gov.sg/datasets/{dataset_id}/view"
        
        # For SingStat data - common tables
        singstat_mappings = {
            "hdb": "M212161",
            "resale": "M212161", 
            "population": "M810001",
            "gdp": "M015721",
            "cpi": "M212881",
            "employment": "M182001",
            "education": "M850001",
            "transport": "M454001",
        }
        
        # For transport data, use data.gov.sg search instead of API endpoints
        title_lower = title.lower() if title else ''
        if any(word in title_lower for word in ['bus', 'traffic', 'transport', 'lta', 'taxi']):
            if 'arrival' in title_lower or 'bus' in title_lower:
                return "https://data.gov.sg/search?query=bus"
            elif 'traffic' in title_lower:
                return "https://data.gov.sg/search?query=traffic"
            elif 'taxi' in title_lower:
                return "https://data.gov.sg/search?query=taxi"
            else:
                return "https://data.gov.sg/search?query=transport"
        
        title_lower = title.lower()
        for keyword, table_id in singstat_mappings.items():
            if keyword in title_lower:
                return f"https://tablebuilder.singstat.gov.sg/table/TS/{table_id}"
        
        return None
    
    async def validate_url(self, url: str) -> Tuple[bool, int]:
        """
        Validate if URL is accessible with comprehensive error handling
        
        Returns:
            (is_valid, status_code)
        """
        # Check for empty or invalid URLs first
        if not url or str(url).lower() in ['nan', 'none', '']:
            return False, 0
        
        # Additional pandas-safe check for NaN values
        try:
            if pd.isna(url):
                return False, 0
        except (NameError, AttributeError):
            # Fallback if pd is not available
            if str(url) == 'nan':
                return False, 0
        
        # Basic URL format validation
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format: {url}")
            return False, 0
            
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': self.user_agent},
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            ) as session:
                # Try HEAD request first (faster)
                try:
                    async with session.head(url) as response:
                        is_valid = response.status < 400
                        if is_valid or response.status in [405, 501]:  # Method not allowed - try GET
                            return is_valid, response.status
                except aiohttp.ClientError:
                    # HEAD failed, try GET request
                    pass
                
                # Fallback to GET request if HEAD fails
                try:
                    async with session.get(url) as response:
                        is_valid = response.status < 400
                        return is_valid, response.status
                except aiohttp.ClientError as e:
                    logger.warning(f"GET request failed for {url}: {str(e)}")
                    return False, 0
                    
        except asyncio.TimeoutError:
            logger.warning(f"URL validation timeout for {url}")
            return False, 408  # Request Timeout
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"Connection error for {url}: {str(e)}")
            return False, 0
        except aiohttp.InvalidURL as e:
            logger.warning(f"Invalid URL {url}: {str(e)}")
            return False, 0
        except Exception as e:
            logger.error(f"Unexpected error validating {url}: {str(e)}")
            return False, 0
    
    async def validate_and_correct_dataset(self, dataset: Dict) -> Dict:
        """
        Validate and correct a complete dataset entry
        
        Args:
            dataset: Dataset dictionary with url, dataset_id, title, etc.
            
        Returns:
            Updated dataset with corrected URL
        """
        original_url = dataset.get('url', '')
        dataset_id = dataset.get('dataset_id', '')
        title = dataset.get('title', '')
        
        # First, try to correct the URL
        corrected_url = self.correct_url(dataset_id, original_url, title)
        
        # Validate the corrected URL
        is_valid, status_code = await self.validate_url(corrected_url)
        
        if is_valid:
            dataset['url'] = corrected_url
            dataset['url_status'] = 'verified'
            dataset['status_code'] = status_code
        else:
            # Try fallback options
            fallback_url = self._get_fallback_url(dataset_id, title)
            if fallback_url:
                fallback_valid, fallback_status = await self.validate_url(fallback_url)
                if fallback_valid:
                    dataset['url'] = fallback_url
                    dataset['url_status'] = 'fallback_verified'
                    dataset['status_code'] = fallback_status
                else:
                    # Final fallback to source page
                    source_page = self.get_source_page_url(dataset)
                    dataset['url'] = source_page
                    dataset['url_status'] = 'source_page_fallback'
                    dataset['status_code'] = 200  # Assume source pages work
            else:
                # Direct source page fallback
                source_page = self.get_source_page_url(dataset)
                dataset['url'] = source_page
                dataset['url_status'] = 'source_page_fallback'
                dataset['status_code'] = 200
        
        return dataset
    
    def _get_fallback_url(self, dataset_id: str, title: str) -> Optional[str]:
        """Get fallback URL options - source pages when direct links fail"""
        
        title_lower = title.lower()
        
        # Smart source page fallbacks
        if any(word in title_lower for word in ['housing', 'hdb', 'resale', 'property']):
            return "https://tablebuilder.singstat.gov.sg/"
        elif any(word in title_lower for word in ['transport', 'bus', 'mrt', 'traffic', 'lta']):
            return "https://data.gov.sg/search?query=transport"
        elif any(word in title_lower for word in ['population', 'demographic', 'census']):
            return "https://tablebuilder.singstat.gov.sg/"
        elif any(word in title_lower for word in ['gdp', 'economy', 'economic', 'cpi']):
            return "https://tablebuilder.singstat.gov.sg/"
        elif any(word in title_lower for word in ['education', 'school', 'student']):
            return "https://data.gov.sg/search?query=education"
        elif any(word in title_lower for word in ['employment', 'job', 'work']):
            return "https://tablebuilder.singstat.gov.sg/"
        elif any(word in title_lower for word in ['weather', 'climate', 'environment']):
            return "https://data.gov.sg/search?query=weather"
        else:
            # Default to main government data portal
            return "https://data.gov.sg/"
    
    def get_source_page_url(self, dataset: Dict) -> str:
        """Get the main source page for a dataset when direct links fail"""
        source = dataset.get('source', '').lower()
        agency = dataset.get('agency', '').lower()
        title = dataset.get('title', '').lower()
        
        # Agency-specific source pages
        if 'singstat' in source or 'department of statistics' in agency:
            return "https://tablebuilder.singstat.gov.sg/"
        elif 'lta' in agency or 'land transport' in agency:
            return "https://data.gov.sg/search?query=transport"
        elif 'data.gov.sg' in source:
            return "https://data.gov.sg/"
        elif 'hdb' in agency or 'housing' in agency:
            return "https://tablebuilder.singstat.gov.sg/"
        elif 'ura' in agency or 'urban redevelopment' in agency:
            return "https://data.gov.sg/search?query=urban"
        else:
            # Category-based fallback
            return self._get_fallback_url('', title)
    
    def get_validation_statistics(self) -> Dict[str, int]:
        """
        Get validation statistics for monitoring and reporting
        
        Returns:
            Dictionary with validation statistics
        """
        # This would be enhanced with actual statistics tracking in a production system
        # For now, return basic structure that can be populated
        return {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'corrected_urls': 0,
            'fallback_urls_used': 0,
            'validation_errors': 0,
            'average_response_time_ms': 0
        }
    
    def log_validation_failure(self, url: str, source: str, error: str, status_code: int):
        """
        Log validation failures for monitoring and debugging
        
        Args:
            url: Failed URL
            source: Source name
            error: Error description
            status_code: HTTP status code
        """
        logger.warning(
            f"URL validation failure - Source: {source}, URL: {url}, "
            f"Status: {status_code}, Error: {error}"
        )
        
        # In a production system, this could also:
        # - Send metrics to monitoring system
        # - Store in database for analysis
        # - Trigger alerts for high failure rates
    
    def get_source_health_status(self) -> Dict[str, Dict[str, any]]:
        """
        Get health status for all external sources
        
        Returns:
            Dictionary with health status for each source
        """
        health_status = {}
        
        for source, config in self.external_source_patterns.items():
            health_status[source] = {
                'name': source,
                'base_url': config.get('fallback_url', ''),
                'search_pattern': config.get('search_pattern', ''),
                'status': 'unknown',  # Would be populated by actual health checks
                'last_check': None,
                'response_time_ms': None,
                'error_rate': 0.0
            }
        
        return health_status
    
    async def perform_health_check(self, source: str) -> Dict[str, any]:
        """
        Perform health check on a specific external source
        
        Args:
            source: Source name to check
            
        Returns:
            Health check results
        """
        if source not in self.external_source_patterns:
            return {
                'source': source,
                'status': 'unknown_source',
                'error': f'Unknown source: {source}'
            }
        
        config = self.external_source_patterns[source]
        fallback_url = config.get('fallback_url', '')
        
        if not fallback_url:
            return {
                'source': source,
                'status': 'no_fallback_url',
                'error': 'No fallback URL configured'
            }
        
        try:
            start_time = datetime.now()
            is_valid, status_code = await self.validate_url_with_retry(fallback_url, max_retries=1)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            
            return {
                'source': source,
                'url': fallback_url,
                'status': 'healthy' if is_valid else 'unhealthy',
                'status_code': status_code,
                'response_time_ms': round(response_time, 2),
                'timestamp': end_time.isoformat()
            }
            
        except Exception as e:
            return {
                'source': source,
                'url': fallback_url,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_all_source_patterns(self) -> Dict[str, Dict[str, any]]:
        """
        Validate all external source patterns for monitoring
        
        Returns:
            Validation results for all sources
        """
        validation_results = {}
        
        # Create health check tasks for all sources
        health_check_tasks = [
            self.perform_health_check(source) 
            for source in self.external_source_patterns.keys()
        ]
        
        try:
            # Execute health checks concurrently
            results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                source = list(self.external_source_patterns.keys())[i]
                
                if isinstance(result, Exception):
                    validation_results[source] = {
                        'source': source,
                        'status': 'check_failed',
                        'error': str(result),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    validation_results[source] = result
                    
        except Exception as e:
            logger.error(f"Critical error in source pattern validation: {str(e)}")
            # Return error status for all sources
            for source in self.external_source_patterns.keys():
                validation_results[source] = {
                    'source': source,
                    'status': 'critical_error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return validation_results


# Global instance for easy use
url_validator = URLValidator()