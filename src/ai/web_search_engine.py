"""
Web Search Engine for Dataset Discovery
Integrates web search to find additional data sources beyond local datasets
"""
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp

from .url_validator import URLValidator

try:
    from bs4 import BeautifulSoup
except ImportError:
    try:
        from beautifulsoup4 import BeautifulSoup
    except ImportError:
        print("Warning: BeautifulSoup not available, web scraping will be limited")
        BeautifulSoup = None

logger = logging.getLogger(__name__)


class WebSearchEngine:
    """
    Web search engine that finds relevant data sources and research materials
    Complements the local dataset search with external sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_config = config.get('web_search', {})
        self.timeout = self.search_config.get('timeout', 10)
        self.max_results = self.search_config.get('max_results', 5)
        self.user_agent = "AI Dataset Research Assistant/2.0"
        
        # Initialize URL validator for result validation
        self.url_validator = URLValidator()
        
        # Prioritized data source domains - Global sources first, then regional
        self.priority_domains = [
            # International Organizations
            'data.worldbank.org',
            'data.un.org', 
            'unstats.un.org',
            'who.int',
            'imf.org',
            'oecd.org',
            'unesco.org',
            'unicef.org',
            'fao.org',
            'wto.org',
            
            # Global Data Platforms & Repositories
            'kaggle.com',
            'huggingface.co/datasets',
            'registry.opendata.aws',
            'github.com',
            'zenodo.org',
            'figshare.com',
            'datacite.org',
            'dryad.org',
            'mendeley.com/datasets',
            'osf.io',
            'dataverse.harvard.edu',
            'data.gov',
            'eurostat.ec.europa.eu',
            'ourworldindata.org',
            'gapminder.org',
            'google.com/publicdata',
            'opendatasoft.com',
            'socrata.com',
            
            # Regional/National (including Singapore)
            'data.gov.sg',
            'singstat.gov.sg', 
            'lta.gov.sg',
            'onemap.sg'
        ]
        
    async def search_web(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the web for relevant data sources and research materials with intelligent source selection
        
        Args:
            query: Search query
            context: Additional context (domain, location, etc.)
            
        Returns:
            List of web search results with metadata
        """
        try:
            start_time = time.time()
            
            # Enhance query for data-focused search
            enhanced_query = self._enhance_query_for_data_search(query, context)
            
            # Intelligent source selection based on query analysis
            selected_sources = self._select_intelligent_sources(query, context)
            logger.info(f"🎯 Selected {len(selected_sources)} intelligent sources for query: {query}")
            
            # Perform parallel searches with failure handling and retry logic
            search_results = await self._execute_searches_with_failure_handling(
                selected_sources, query, enhanced_query, context
            )
            
            # Combine results from successful searches
            combined_results = []
            failed_sources = []
            
            for i, (source_config, result_set) in enumerate(zip(selected_sources, search_results)):
                if isinstance(result_set, Exception):
                    # Log the failure and track for fallback
                    logger.warning(f"❌ Source '{source_config['type']}' failed: {str(result_set)}")
                    failed_sources.append(source_config)
                elif isinstance(result_set, list):
                    combined_results.extend(result_set)
                    logger.info(f"✅ Source '{source_config['type']}' returned {len(result_set)} results")
                else:
                    logger.warning(f"⚠️ Source '{source_config['type']}' returned unexpected result type")
                    failed_sources.append(source_config)
            
            # Handle failed sources with fallback strategies
            if failed_sources:
                logger.info(f"🔄 Applying fallback strategies for {len(failed_sources)} failed sources")
                fallback_results = await self._handle_failed_sources(failed_sources, query, context)
                combined_results.extend(fallback_results)
            
            # Apply intelligent source coverage requirements
            coverage_results = self._ensure_minimum_source_coverage(combined_results, query, context)
            
            # Rank and filter results with domain-aware prioritization
            ranked_results = self._rank_search_results_with_domain_awareness(coverage_results, query, context)
            
            # Validate and correct URLs before returning results
            validated_results = await self._validate_and_correct_result_urls(ranked_results[:self.max_results])
            
            search_time = time.time() - start_time
            logger.info(f"🌐 Web search completed: {len(validated_results)} validated results from {len(selected_sources)} sources in {search_time:.2f}s")
            
            return validated_results
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {str(e)}")
            return []
    
    def _select_intelligent_sources(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Intelligently select sources based on query analysis and domain awareness
        Implements query-type based source prioritization and Singapore-specific boosting
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            List of source configurations with priorities
        """
        query_lower = query.lower()
        
        # Detect Singapore context
        singapore_keywords = [
            'singapore', 'sg', 'hdb', 'mrt', 'coe', 'cpf', 'lta', 'singstat',
            'data.gov.sg', 'resale flat', 'bto', 'public housing', 'hawker',
            'void deck', 'wet market', 'ccb', 'east coast', 'orchard road'
        ]
        has_singapore_context = any(keyword in query_lower for keyword in singapore_keywords)
        
        # Detect query domain/type
        query_domain = self._detect_query_domain(query)
        
        # Base source configurations
        all_sources = [
            {'type': 'kaggle', 'priority': 100, 'domain_boost': 0, 'singapore_boost': 0},
            {'type': 'public_data', 'priority': 90, 'domain_boost': 0, 'singapore_boost': 0},
            {'type': 'academic', 'priority': 85, 'domain_boost': 0, 'singapore_boost': 0},
            {'type': 'international', 'priority': 80, 'domain_boost': 0, 'singapore_boost': 0},
            {'type': 'government', 'priority': 75, 'domain_boost': 0, 'singapore_boost': 0},
            {'type': 'duckduckgo', 'priority': 60, 'domain_boost': 0, 'singapore_boost': 0}
        ]
        
        # Apply domain-specific boosts
        domain_boosts = {
            'health': {
                'international': 40,  # WHO, health organizations
                'academic': 30,       # Medical research
                'government': 25      # National health data
            },
            'economics': {
                'international': 45,  # World Bank, IMF, OECD
                'academic': 20,       # Economic research
                'government': 30      # National economic data
            },
            'education': {
                'international': 35,  # UNESCO, education orgs
                'academic': 40,       # Educational research
                'government': 35      # National education data
            },
            'environment': {
                'international': 40,  # Climate organizations
                'academic': 35,       # Environmental research
                'public_data': 30     # Environmental datasets
            },
            'technology': {
                'kaggle': 50,         # ML/tech datasets
                'academic': 30,       # Tech research
                'public_data': 25     # Tech platforms
            },
            'demographics': {
                'international': 45,  # UN Population, demographics
                'government': 40,     # Census data
                'academic': 25        # Demographic research
            },
            'transport': {
                'government': 50,     # Transport authorities
                'public_data': 30,    # Transport datasets
                'international': 20   # Global transport data
            }
        }
        
        # Apply Singapore-specific boosts
        singapore_boosts = {
            'government': 60,     # Singapore government sources
            'international': 20,  # International orgs with Singapore data
            'academic': 15,       # Singapore research
            'public_data': 10,    # Public platforms with Singapore data
            'kaggle': 5,          # Kaggle Singapore datasets
            'duckduckgo': 5       # General web search
        }
        
        # Calculate final priorities
        for source in all_sources:
            source_type = source['type']
            
            # Apply domain boost
            if query_domain in domain_boosts:
                source['domain_boost'] = domain_boosts[query_domain].get(source_type, 0)
            
            # Apply Singapore boost
            if has_singapore_context:
                source['singapore_boost'] = singapore_boosts.get(source_type, 0)
            
            # Calculate final priority
            source['final_priority'] = (source['priority'] + 
                                      source['domain_boost'] + 
                                      source['singapore_boost'])
        
        # Sort by final priority and select top sources
        all_sources.sort(key=lambda x: x['final_priority'], reverse=True)
        
        # Ensure minimum of 3 sources, maximum of 6 for performance
        min_sources = 3
        max_sources = 6
        
        # Always include top sources, ensure diversity
        selected_sources = []
        source_types_included = set()
        
        # First pass: include highest priority sources
        for source in all_sources:
            if len(selected_sources) < max_sources:
                selected_sources.append(source)
                source_types_included.add(source['type'])
        
        # Second pass: ensure minimum coverage if needed
        if len(selected_sources) < min_sources:
            for source in all_sources:
                if (len(selected_sources) < min_sources and 
                    source['type'] not in source_types_included):
                    selected_sources.append(source)
                    source_types_included.add(source['type'])
        
        # Log source selection reasoning
        logger.info(f"🎯 Source selection for query '{query}':")
        logger.info(f"   Domain: {query_domain}")
        logger.info(f"   Singapore context: {has_singapore_context}")
        for source in selected_sources:
            logger.info(f"   {source['type']}: priority={source['final_priority']} "
                       f"(base={source['priority']}, domain={source['domain_boost']}, "
                       f"singapore={source['singapore_boost']})")
        
        return selected_sources
    
    def _detect_query_domain(self, query: str) -> str:
        """
        Detect the domain/type of the query for intelligent source selection
        
        Args:
            query: Search query
            
        Returns:
            Detected domain type
        """
        query_lower = query.lower()
        
        # Health domain
        health_keywords = [
            'health', 'medical', 'disease', 'mortality', 'life expectancy', 'covid',
            'pandemic', 'hospital', 'medicine', 'healthcare', 'cancer', 'death',
            'birth', 'vaccination', 'epidemic', 'clinical', 'patient'
        ]
        if any(keyword in query_lower for keyword in health_keywords):
            return 'health'
        
        # Economics domain
        economics_keywords = [
            'gdp', 'economic', 'economy', 'growth', 'inflation', 'trade', 'finance',
            'financial', 'market', 'investment', 'business', 'commerce', 'price',
            'cost', 'income', 'salary', 'wage', 'employment', 'unemployment'
        ]
        if any(keyword in query_lower for keyword in economics_keywords):
            return 'economics'
        
        # Education domain - exclude tech-related learning
        education_keywords = [
            'education', 'school', 'university', 'student', 'academic',
            'literacy', 'enrollment', 'graduation', 'teacher', 'curriculum'
        ]
        # Only match education if it's not tech-related learning
        if (any(keyword in query_lower for keyword in education_keywords) and
            not any(tech_keyword in query_lower for tech_keyword in ['machine learning', 'data science', 'artificial intelligence', 'deep learning'])):
            return 'education'
        
        # Environment domain
        environment_keywords = [
            'environment', 'climate', 'weather', 'temperature', 'pollution',
            'carbon', 'emission', 'renewable', 'energy', 'sustainability',
            'green', 'ecological', 'conservation'
        ]
        if any(keyword in query_lower for keyword in environment_keywords):
            return 'environment'
        
        # Technology domain - check for ML/AI/tech keywords first
        technology_keywords = [
            'machine learning', 'ml', 'ai', 'artificial intelligence', 'data science',
            'technology', 'tech', 'computer', 'software', 'digital', 'internet',
            'blockchain', 'cryptocurrency', 'bitcoin', 'neural network', 'deep learning',
            'algorithm', 'programming', 'coding', 'dataset', 'model training'
        ]
        # Check for compound terms first (machine learning, data science)
        if any(keyword in query_lower for keyword in ['machine learning', 'data science', 'artificial intelligence', 'deep learning']):
            return 'technology'
        if any(keyword in query_lower for keyword in technology_keywords):
            return 'technology'
        
        # Demographics domain
        demographics_keywords = [
            'population', 'demographic', 'census', 'age', 'gender', 'race',
            'ethnicity', 'migration', 'birth rate', 'death rate', 'urbanization'
        ]
        if any(keyword in query_lower for keyword in demographics_keywords):
            return 'demographics'
        
        # Transport domain
        transport_keywords = [
            'transport', 'transportation', 'traffic', 'bus', 'train', 'mrt',
            'subway', 'car', 'vehicle', 'road', 'highway', 'aviation', 'airport'
        ]
        if any(keyword in query_lower for keyword in transport_keywords):
            return 'transport'
        
        # Default to general
        return 'general'
    
    def _enhance_query_for_data_search(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance query to focus on data sources and datasets"""
        
        # Add data-focused terms
        data_terms = ["dataset", "data", "statistics", "research data", "open data"]
        
        # Add global organization terms for better discovery
        global_terms = ["World Bank", "UN", "WHO", "OECD", "IMF", "UNESCO"]
        
        # Check if query already mentions global organizations
        query_lower = query.lower()
        has_global_org = any(org.lower() in query_lower for org in global_terms)
        
        if has_global_org:
            # Already global-focused, just add data terms
            enhanced_query = f"{query} ({' OR '.join(data_terms)})"
        elif context and context.get('singapore_focus', False):
            # Explicitly Singapore-focused
            singapore_terms = ["Singapore", "SG", "data.gov.sg"]
            enhanced_query = f"{query} ({' OR '.join(singapore_terms)}) ({' OR '.join(data_terms)})"
        else:
            # Default to global search with international organizations
            enhanced_query = f"{query} ({' OR '.join(global_terms[:3])}) ({' OR '.join(data_terms)})"
            
        return enhanced_query
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (privacy-focused)"""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': self.user_agent}
            ) as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_duckduckgo_results(html)
                        
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")
            
        return []
    
    def _parse_duckduckgo_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo search results"""
        results = []
        
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup not available, cannot parse web results")
            return results
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result links
            for result in soup.find_all('a', class_='result__a'):
                title = result.get_text(strip=True)
                url = result.get('href')
                
                if url and title:
                    # Find description
                    description = ""
                    result_container = result.find_parent('div', class_='result')
                    if result_container:
                        desc_elem = result_container.find('a', class_='result__snippet')
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'source': 'duckduckgo',
                        'type': 'web_search'
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to parse DuckDuckGo results: {str(e)}")
            
        return results
    
    def _normalize_query_for_source(self, query: str, source: str) -> str:
        """
        Normalize conversational query for specific external sources
        Removes conversational language and extracts core search terms
        """
        normalized = query.lower().strip()
        
        # Remove common conversational patterns in order
        conversational_patterns = [
            # Complex patterns first
            r'^(can you|could you|please)\s+(find me|get me|show me)\s+',
            r'^(i need|i want|i\'m looking for|looking for)\s+',
            r'^(find me|get me|show me|search for)\s+',
            r'^(can you|could you|please)\s+',
            r'^(find|search)\s+',
            # Remove remaining filler words
            r'\b(some|any)\s+',
            r'\s+(related to|regarding|concerning)\s+',
            # Remove trailing words
            r'\s+(please|thanks|thank you)$',
            r'\s+(data|dataset|datasets)$'
        ]
        
        for pattern in conversational_patterns:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE).strip()
        
        # Handle "about" separately to preserve spacing
        normalized = re.sub(r'\s+about\s+', ' ', normalized, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Source-specific processing
        if source == 'kaggle':
            # For Kaggle, ensure we have meaningful search terms
            # If the result is too short or generic, keep more of the original
            if len(normalized.split()) < 1 or normalized in ['me', 'it', 'this', 'that']:
                # Extract key nouns from original query
                # Simple noun extraction - look for meaningful words
                words = query.lower().split()
                meaningful_words = []
                skip_words = {'i', 'need', 'want', 'looking', 'for', 'find', 'me', 'get', 'show', 
                             'search', 'can', 'you', 'could', 'please', 'some', 'any', 'about',
                             'data', 'dataset', 'datasets', 'thanks', 'thank', 'you'}
                
                for word in words:
                    clean_word = re.sub(r'[^\w]', '', word)
                    if clean_word and clean_word not in skip_words and len(clean_word) > 2:
                        meaningful_words.append(clean_word)
                
                if meaningful_words:
                    normalized = ' '.join(meaningful_words)
                    
        elif source == 'world_bank':
            # World Bank prefers broader economic/development terms
            # Keep country names and economic indicators
            # World Bank search works better with broader terms
            if len(normalized.split()) > 3:
                # For longer queries, focus on key economic/development terms
                key_terms = []
                words = normalized.split()
                economic_keywords = ['gdp', 'economy', 'economic', 'growth', 'development', 'poverty', 
                                   'education', 'health', 'population', 'employment', 'trade', 'finance']
                
                for word in words:
                    if word in economic_keywords or len(word) > 4:  # Keep longer words and economic terms
                        key_terms.append(word)
                
                if key_terms:
                    normalized = ' '.join(key_terms[:3])  # Limit to 3 key terms
                    
        elif source == 'aws':
            # AWS Open Data works better with specific domain terms
            # Extract meaningful keywords and remove conversational noise
            if normalized:
                words = normalized.split()
                meaningful_words = []
                # More conservative skip words - only remove clearly conversational terms
                skip_words = {'i', 'need', 'want', 'looking', 'for', 'find', 'me', 'get', 'show', 
                             'search', 'can', 'you', 'could', 'please', 'some', 'any', 'about',
                             'data', 'dataset', 'datasets', 'thanks', 'thank', 'you', 'what', 
                             'are', 'available', 'my', 'project'}
                
                for word in words:
                    clean_word = re.sub(r'[^\w-]', '', word)  # Keep hyphens for terms like covid-19
                    if clean_word and clean_word not in skip_words and len(clean_word) > 1:  # Allow 2-letter words
                        meaningful_words.append(clean_word)
                
                if meaningful_words:
                    # Allow more terms for AWS search - up to 8 key terms
                    normalized = ' '.join(meaningful_words[:8])
        
        return normalized if normalized else query  # Fallback to original if empty
    
    def _generate_world_bank_url(self, query: str, query_type: str = 'general') -> str:
        """
        Generate World Bank URL with proper search parameters using working endpoints
        Uses domain-specific URL patterns for different query types to ensure actual search results
        
        Args:
            query: The search query
            query_type: Type of query ('economic', 'education', 'health', 'population', 'general')
            
        Returns:
            Valid World Bank URL pointing to actual search results
        """
        # Clean the query for URL encoding
        normalized_query = self._normalize_query_for_source(query, 'world_bank')
        query_lower = normalized_query.lower() if normalized_query else query.lower()
        
        # Domain-specific URL patterns for different query types
        # These are verified working World Bank topic pages that provide actual search results
        
        # Economic/Finance queries - use economy and growth topic
        if (query_type == 'economic' or 
            any(term in query_lower for term in ['gdp', 'economic', 'economy', 'growth', 'inflation', 
                                                'trade', 'financial', 'finance', 'market', 'investment'])):
            return "https://data.worldbank.org/topic/economy-and-growth"
        
        # Education queries - use education topic
        elif (query_type == 'education' or 
              any(term in query_lower for term in ['education', 'school', 'enrollment', 'literacy', 
                                                  'learning', 'student', 'academic', 'university'])):
            return "https://data.worldbank.org/topic/education"
        
        # Health queries - use health topic
        elif (query_type == 'health' or 
              any(term in query_lower for term in ['health', 'mortality', 'life expectancy', 'disease', 
                                                  'medical', 'healthcare', 'hospital', 'medicine'])):
            return "https://data.worldbank.org/topic/health"
        
        # Population/Demographics queries - use health topic (contains demographic data)
        elif (query_type == 'population' or 
              any(term in query_lower for term in ['population', 'demographic', 'birth', 'death', 
                                                  'migration', 'age', 'census'])):
            return "https://data.worldbank.org/topic/health"
        
        # Poverty queries - use poverty topic
        elif any(term in query_lower for term in ['poverty', 'poor', 'income inequality', 'welfare']):
            return "https://data.worldbank.org/topic/poverty"
        
        # Employment/Labor queries - use labor and social protection topic
        elif any(term in query_lower for term in ['employment', 'unemployment', 'labor', 'job', 
                                                 'work', 'social protection']):
            return "https://data.worldbank.org/topic/labor-and-social-protection"
        
        # Gender queries - use gender topic
        elif any(term in query_lower for term in ['gender', 'women', 'female', 'male', 'equality']):
            return "https://data.worldbank.org/topic/gender"
        
        # Environment queries - use environment topic
        elif any(term in query_lower for term in ['environment', 'climate', 'pollution', 'energy', 
                                                 'carbon', 'emission', 'renewable']):
            return "https://data.worldbank.org/topic/environment"
        
        # Agriculture queries - use agriculture and rural development topic
        elif any(term in query_lower for term in ['agriculture', 'farming', 'rural', 'crop', 
                                                 'livestock', 'food security']):
            return "https://data.worldbank.org/topic/agriculture-and-rural-development"
        
        # Infrastructure queries - use infrastructure topic
        elif any(term in query_lower for term in ['infrastructure', 'transport', 'road', 'railway', 
                                                 'telecommunication', 'internet', 'broadband']):
            return "https://data.worldbank.org/topic/infrastructure"
        
        # Urban development queries - use urban development topic
        elif any(term in query_lower for term in ['urban', 'city', 'housing', 'construction', 
                                                 'real estate']):
            return "https://data.worldbank.org/topic/urban-development"
        
        # General queries or no specific match - use main indicators page
        else:
            return "https://data.worldbank.org/indicator"
    
    def _validate_world_bank_url(self, url: str, query: str) -> str:
        """
        Validate World Bank URLs and provide fallback strategies
        Uses verified working endpoints only and fixes broken datacatalog URLs
        
        Args:
            url: The URL to validate
            query: Original query for context
            
        Returns:
            Validated URL or fallback URL
        """
        # List of verified working World Bank endpoints (tested and confirmed)
        valid_endpoints = [
            "https://data.worldbank.org/topic/",         # Working topic pages
            "https://data.worldbank.org/indicator",      # Working indicators page
            "https://data.worldbank.org/country/",       # Working country pages
            "https://databank.worldbank.org/",           # Working databank
            "https://data.worldbank.org/products/wdi"   # Working WDI page
        ]
        
        # Check if URL uses a verified working endpoint
        if any(url.startswith(endpoint) for endpoint in valid_endpoints):
            return url
        
        # Fix broken datacatalog URLs - these are known to be broken
        if "datacatalog.worldbank.org" in url:
            logger.warning(f"Replacing broken World Bank datacatalog URL: {url}")
            return self._generate_world_bank_url(query, 'general')
        
        # Fix any other broken or invalid URLs
        if not url.startswith("https://data.worldbank.org/"):
            logger.warning(f"Invalid World Bank URL detected: {url}, using fallback")
            return self._generate_world_bank_url(query, 'general')
        
        # Check for invalid paths within the valid domain
        valid_paths = ['/topic/', '/indicator', '/country/', '/products/']
        if not any(path in url for path in valid_paths):
            logger.warning(f"Invalid World Bank URL path detected: {url}, using fallback")
            return self._generate_world_bank_url(query, 'general')
        
        # URL appears valid
        return url

    def _generate_aws_open_data_url(self, query: str) -> str:
        """
        Generate AWS Open Data URL with proper search parameters or fallback to browse pages
        AWS Registry supports direct search parameters via ?search= parameter
        Enhanced with better normalization and fallback handling
        """
        normalized_query = self._normalize_query_for_source(query, 'aws')
        
        # Use direct search parameters for better results
        if normalized_query and len(normalized_query.strip()) > 0:
            # Primary approach: Use search parameters for direct query matching
            # Ensure the search query is meaningful (not just single characters or stop words)
            query_words = normalized_query.strip().split()
            if len(query_words) >= 1 and not all(len(word) <= 1 for word in query_words):
                search_url = f"https://registry.opendata.aws/?search={quote_plus(normalized_query)}"
                return search_url
        
        # Enhanced fallback: Topic-based routing for AWS Open Data with verified working tags
        query_lower = query.lower()
        
        # Geospatial and Earth observation data
        if any(term in query_lower for term in ['satellite', 'imagery', 'landsat', 'sentinel', 'earth', 'geospatial', 'mapping', 'gis', 'remote sensing', 'modis']):
            return "https://registry.opendata.aws/tag/geospatial/"
        
        # Climate and weather data
        elif any(term in query_lower for term in ['weather', 'climate', 'meteorological', 'noaa', 'atmospheric', 'precipitation', 'temperature', 'forecast']):
            return "https://registry.opendata.aws/tag/climate/"
        
        # Life sciences and genomics
        elif any(term in query_lower for term in ['genomics', 'dna', 'genetic', 'bioinformatics', 'biology', 'life sciences', 'protein', 'sequencing', 'biomedical']):
            return "https://registry.opendata.aws/tag/life-sciences/"
        
        # Transportation and mobility
        elif any(term in query_lower for term in ['transportation', 'traffic', 'mobility', 'transit', 'logistics', 'vehicle', 'road', 'highway', 'transport']):
            return "https://registry.opendata.aws/tag/transportation/"
        
        # Economics and finance
        elif any(term in query_lower for term in ['economics', 'economic', 'finance', 'financial', 'business', 'market', 'trade', 'commerce', 'gdp']):
            return "https://registry.opendata.aws/tag/economics/"
        
        # Health and medical data
        elif any(term in query_lower for term in ['health', 'medical', 'healthcare', 'disease', 'medicine', 'hospital', 'patient', 'clinical', 'covid', 'pandemic']):
            return "https://registry.opendata.aws/tag/health/"
        
        # Machine learning and AI
        elif any(term in query_lower for term in ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural', 'model', 'training']):
            return "https://registry.opendata.aws/tag/machine-learning/"
        
        # Astronomy and space
        elif any(term in query_lower for term in ['astronomy', 'space', 'cosmic', 'stellar', 'planetary', 'universe', 'galaxy', 'telescope']):
            return "https://registry.opendata.aws/tag/astronomy/"
        
        # Energy and utilities
        elif any(term in query_lower for term in ['energy', 'power', 'electricity', 'renewable', 'solar', 'wind', 'utility']):
            return "https://registry.opendata.aws/tag/energy/"
        
        # Government and public sector
        elif any(term in query_lower for term in ['government', 'public', 'census', 'demographic', 'policy', 'regulation']):
            return "https://registry.opendata.aws/tag/government/"
        
        else:
            # Final fallback to main browse page
            return "https://registry.opendata.aws/"
    
    async def _validate_aws_url(self, url: str, query: str) -> str:
        """
        Validate AWS Open Data URL and provide fallback if URL is not accessible
        Enhanced with better error handling and fallback strategies
        
        Args:
            url: The URL to validate
            query: Original query for context in fallback
            
        Returns:
            Validated URL or fallback URL
        """
        # Basic URL format validation first
        if not url or not url.startswith("https://registry.opendata.aws"):
            logger.warning(f"Invalid AWS URL format: {url}")
            return self._get_aws_fallback_url(query)
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8),  # Increased timeout for better reliability
                headers={'User-Agent': self.user_agent}
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Additional check: ensure the response contains actual content
                        content = await response.text()
                        if len(content) > 1000:  # Basic check for meaningful content
                            return url
                        else:
                            logger.warning(f"AWS URL returned minimal content: {url}")
                            return self._get_aws_intelligent_fallback_url(query)
                    elif response.status == 404:
                        logger.warning(f"AWS URL not found (404): {url}")
                        return self._get_aws_intelligent_fallback_url(query)
                    else:
                        logger.warning(f"AWS URL validation failed with status {response.status}: {url}")
                        return self._get_aws_intelligent_fallback_url(query)
                        
        except asyncio.TimeoutError:
            logger.warning(f"AWS URL validation timeout: {url}")
            return self._get_aws_intelligent_fallback_url(query)
        except Exception as e:
            logger.warning(f"AWS URL validation error: {str(e)} for URL: {url}")
            return self._get_aws_intelligent_fallback_url(query)
    
    def _get_aws_intelligent_fallback_url(self, query: str) -> str:
        """
        Get intelligent fallback URL for AWS Open Data when primary URL fails
        Uses enhanced logic to provide the best alternative
        
        Args:
            query: Original query for context
            
        Returns:
            Best fallback URL based on query content
        """
        # First try tag-based fallback
        tag_url = self._get_aws_tag_fallback_url(query)
        if tag_url != "https://registry.opendata.aws/":
            return tag_url
        
        # If no specific tag matches, return main page
        return self._get_aws_fallback_url(query)
    
    def _get_aws_tag_fallback_url(self, query: str) -> str:
        """
        Get tag-based fallback URL for AWS Open Data when search parameters fail
        Enhanced with additional categories and better matching
        
        Args:
            query: Original query for context
            
        Returns:
            Tag-based URL or main page if no suitable tag found
        """
        query_lower = query.lower()
        
        # Enhanced topic-based routing for fallback with more categories
        # Geospatial and Earth observation data
        if any(term in query_lower for term in ['satellite', 'imagery', 'landsat', 'sentinel', 'earth', 'geospatial', 'mapping', 'gis', 'remote sensing', 'modis']):
            return "https://registry.opendata.aws/tag/geospatial/"
        
        # Climate and weather data
        elif any(term in query_lower for term in ['weather', 'climate', 'meteorological', 'noaa', 'atmospheric', 'precipitation', 'temperature', 'forecast']):
            return "https://registry.opendata.aws/tag/climate/"
        
        # Life sciences and genomics
        elif any(term in query_lower for term in ['genomics', 'dna', 'genetic', 'bioinformatics', 'biology', 'life sciences', 'protein', 'sequencing', 'biomedical']):
            return "https://registry.opendata.aws/tag/life-sciences/"
        
        # Transportation and mobility
        elif any(term in query_lower for term in ['transportation', 'traffic', 'mobility', 'transit', 'logistics', 'vehicle', 'road', 'highway', 'transport']):
            return "https://registry.opendata.aws/tag/transportation/"
        
        # Economics and finance
        elif any(term in query_lower for term in ['economics', 'economic', 'finance', 'financial', 'business', 'market', 'trade', 'commerce', 'gdp']):
            return "https://registry.opendata.aws/tag/economics/"
        
        # Health and medical data
        elif any(term in query_lower for term in ['health', 'medical', 'healthcare', 'disease', 'medicine', 'hospital', 'patient', 'clinical', 'covid', 'pandemic']):
            return "https://registry.opendata.aws/tag/health/"
        
        # Machine learning and AI
        elif any(term in query_lower for term in ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural', 'model', 'training']):
            return "https://registry.opendata.aws/tag/machine-learning/"
        
        # Astronomy and space
        elif any(term in query_lower for term in ['astronomy', 'space', 'cosmic', 'stellar', 'planetary', 'universe', 'galaxy', 'telescope']):
            return "https://registry.opendata.aws/tag/astronomy/"
        
        # Energy and utilities (if this tag exists)
        elif any(term in query_lower for term in ['energy', 'power', 'electricity', 'renewable', 'solar', 'wind', 'utility']):
            return "https://registry.opendata.aws/tag/energy/"
        
        # Government and public sector (if this tag exists)
        elif any(term in query_lower for term in ['government', 'public', 'census', 'demographic', 'policy', 'regulation']):
            return "https://registry.opendata.aws/tag/government/"
        
        else:
            return "https://registry.opendata.aws/"

    def _get_aws_fallback_url(self, query: str) -> str:
        """
        Get fallback URL for AWS Open Data when primary URL fails
        
        Args:
            query: Original query for context
            
        Returns:
            Fallback URL (main AWS Open Data page)
        """
        # Always fallback to the main page which is guaranteed to work
        return "https://registry.opendata.aws/"

    async def _search_kaggle_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search Kaggle datasets specifically"""
        results = []
        
        try:
            # Normalize query to remove conversational language
            normalized_query = self._normalize_query_for_source(query, 'kaggle')
            
            # Direct Kaggle dataset search with normalized query
            kaggle_search_url = f"https://www.kaggle.com/datasets?search={quote_plus(normalized_query)}"
            
            results.append({
                'title': f'Kaggle Datasets: {normalized_query}',
                'url': kaggle_search_url,
                'description': f'Machine learning datasets and data science competitions related to {normalized_query}',
                'source': 'kaggle',
                'type': 'ml_dataset_platform',
                'domain': 'kaggle.com',
                'relevance_score': 1000  # High priority for accessible datasets
            })
            
            # Add specific popular Kaggle datasets if query matches
            query_lower = query.lower()
            if any(word in query_lower for word in ['housing', 'real estate', 'property', 'hdb']):
                results.append({
                    'title': 'House Prices Dataset (Kaggle)',
                    'url': 'https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques',
                    'description': 'Advanced regression techniques for house price prediction - popular ML competition dataset',
                    'source': 'kaggle',
                    'type': 'ml_dataset_platform',
                    'domain': 'kaggle.com',
                    'relevance_score': 95
                })
            
            if any(word in query_lower for word in ['titanic', 'survival', 'passenger']):
                results.append({
                    'title': 'Titanic Dataset (Kaggle)',
                    'url': 'https://www.kaggle.com/competitions/titanic',
                    'description': 'Machine learning from disaster - predict survival on the Titanic',
                    'source': 'kaggle',
                    'type': 'ml_dataset_platform',
                    'domain': 'kaggle.com',
                    'relevance_score': 95
                })
                
        except Exception as e:
            logger.warning(f"Kaggle search failed: {str(e)}")
            
        return results
    
    async def _search_public_data_platforms(self, query: str) -> List[Dict[str, Any]]:
        """Search public data platforms and repositories"""
        results = []
        
        # Major public data platforms
        platforms = [
            {
                'name': 'Hugging Face Datasets',
                'search_url': f'https://huggingface.co/datasets?search={quote_plus(query)}',
                'description': f'Machine learning datasets and NLP corpora for {query}',
                'domain': 'huggingface.co',
                'type': 'ml_dataset_platform',
                'relevance_score': 88
            },
            {
                'name': 'AWS Open Data',
                'search_url': self._generate_aws_open_data_url(query),
                'description': f'AWS public datasets and cloud-hosted data for {query}',
                'domain': 'registry.opendata.aws',
                'type': 'cloud_data_platform',
                'relevance_score': 85
            },
            {
                'name': 'Google Dataset Search',
                'search_url': f'https://datasetsearch.research.google.com/search?q={quote_plus(query)}',
                'description': f'Google\'s dataset search engine results for {query}',
                'domain': 'datasetsearch.research.google.com',
                'type': 'dataset_search_engine',
                'relevance_score': 92
            },
            {
                'name': 'Harvard Dataverse',
                'search_url': f'https://dataverse.harvard.edu/dataverse/harvard?q={quote_plus(query)}',
                'description': f'Research data repository from Harvard University for {query}',
                'domain': 'dataverse.harvard.edu',
                'type': 'academic_repository',
                'relevance_score': 85
            },
            {
                'name': 'Mendeley Data',
                'search_url': f'https://data.mendeley.com/research-data/?search={quote_plus(query)}',
                'description': f'Research datasets and scientific data for {query}',
                'domain': 'data.mendeley.com',
                'type': 'academic_repository',
                'relevance_score': 82
            },
            {
                'name': 'OpenDataSoft',
                'search_url': f'https://public.opendatasoft.com/explore/?q={quote_plus(query)}',
                'description': f'Public datasets and open data portal for {query}',
                'domain': 'public.opendatasoft.com',
                'type': 'open_data_platform',
                'relevance_score': 80
            }
        ]
        
        for platform in platforms:
            # Special handling for AWS Open Data with URL validation
            if platform['name'] == 'AWS Open Data':
                validated_url = await self._validate_aws_url(platform['search_url'], query)
                results.append({
                    'title': f"{platform['name']}: {query}",
                    'url': validated_url,
                    'description': platform['description'],
                    'source': platform['name'].lower().replace(' ', '_'),
                    'type': platform['type'],
                    'domain': platform['domain'],
                    'relevance_score': platform['relevance_score']
                })
            else:
                results.append({
                    'title': f"{platform['name']}: {query}",
                    'url': platform['search_url'],
                    'description': platform['description'],
                    'source': platform['name'].lower().replace(' ', '_'),
                    'type': platform['type'],
                    'domain': platform['domain'],
                    'relevance_score': platform['relevance_score']
                })
            
        return results
    
    async def _search_academic_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search academic and research data sources"""
        results = []
        
        # Academic search patterns
        academic_sources = [
            {
                'name': 'Zenodo',
                'search_url': f"https://zenodo.org/search?q={quote_plus(query + ' dataset')}",
                'domain': 'zenodo.org'
            },
            {
                'name': 'Figshare', 
                'search_url': f"https://figshare.com/search?q={quote_plus(query)}",
                'domain': 'figshare.com'
            },
            {
                'name': 'Dryad Digital Repository',
                'search_url': f"https://datadryad.org/search?q={quote_plus(query)}",
                'domain': 'datadryad.org'
            },
            {
                'name': 'Open Science Framework',
                'search_url': f"https://osf.io/search/?q={quote_plus(query)}",
                'domain': 'osf.io'
            }
        ]
        
        for source in academic_sources:
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={'User-Agent': self.user_agent}
                ) as session:
                    async with session.get(source['search_url']) as response:
                        if response.status == 200:
                            results.append({
                                'title': f"Search {source['name']} for: {query}",
                                'url': source['search_url'],
                                'description': f"Academic datasets and research data from {source['name']}",
                                'source': source['name'].lower(),
                                'type': 'academic_search',
                                'domain': source['domain']
                            })
                            
            except Exception as e:
                logger.warning(f"Academic search failed for {source['name']}: {str(e)}")
                
        return results
    
    async def _search_international_organizations(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search international organizations and global data sources"""
        results = []
        
        # Get direct links to international organization datasets
        intl_links = self._get_international_dataset_links(query)
        results.extend(intl_links)
        
        # International data portals with smart topic routing
        query_lower = query.lower()
        
        # Smart URL routing based on query topic - prioritizing Singapore then accessible platforms
        if any(term in query_lower for term in ['cancer', 'mortality', 'death', 'disease', 'health']):
            # Check if Singapore context first
            if any(sg_term in query_lower for sg_term in ['singapore', 'sg']):
                intl_portals = [
                    {
                        'name': 'Singapore Health Data Portal',
                        'search_url': f"https://data.gov.sg/search?query=health",
                        'domain': 'data.gov.sg',
                        'priority': 1500  # Highest for Singapore health
                    },
                    {
                        'name': 'Our World in Data - Cancer',
                        'search_url': f"https://ourworldindata.org/cancer",
                        'domain': 'ourworldindata.org',
                        'priority': 900
                    },
                    {
                        'name': 'WHO Global Health Observatory - Cancer Data',
                        'search_url': "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/estimated-number-of-deaths-from-cancer",
                        'domain': 'who.int',
                        'priority': 800
                    }
                ]
            else:
                intl_portals = [
                    {
                        'name': 'Our World in Data - Cancer',
                        'search_url': f"https://ourworldindata.org/cancer",
                        'domain': 'ourworldindata.org',
                        'priority': 500  # Lower than Kaggle for global queries
                    },
                    {
                        'name': 'WHO Global Health Observatory - Cancer Data',
                        'search_url': "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/estimated-number-of-deaths-from-cancer",
                        'domain': 'who.int',
                        'priority': 400
                    },
                    {
                        'name': 'Global Burden of Disease Study',
                        'search_url': "http://ghdx.healthdata.org/gbd-results-tool",
                        'domain': 'healthdata.org',
                        'priority': 300
                    },
                    {
                        'name': 'UN Data Portal',
                        'search_url': f"https://data.un.org/Search.aspx?q={quote_plus(query)}",
                        'domain': 'data.un.org',
                        'priority': 200
                    }
                ]
        else:
            # Smart routing for non-health queries based on topic relevance
            tech_terms = ['cryptocurrency', 'crypto', 'bitcoin', 'blockchain', 'technology', 'laptop', 'computer', 'software', 'ai', 'machine learning']
            finance_terms = ['price', 'prices', 'stock', 'market', 'trading', 'investment', 'finance', 'economy']
            education_terms = ['education', 'school', 'university', 'learning', 'student', 'academic']
            
            if any(term in query_lower for term in tech_terms):
                # Technology/Crypto queries - prioritize accessible data platforms
                intl_portals = [
                    {
                        'name': 'OECD Data',
                        'search_url': f"https://data.oecd.org/searchresults/?q={quote_plus(query)}",
                        'domain': 'oecd.org',
                        'priority': 300  # Lower priority for tech queries
                    }
                ]
            elif any(term in query_lower for term in education_terms):
                # Education queries - prioritize education-focused sources
                normalized_query = self._normalize_query_for_source(query, 'world_bank')
                intl_portals = [
                    {
                        'name': 'World Bank Open Data - Education',
                        'search_url': self._generate_world_bank_url(normalized_query, 'education'),
                        'fallback_url': "https://data.worldbank.org/indicator",
                        'domain': 'data.worldbank.org',
                        'priority': 600
                    },
                    {
                        'name': 'UNESCO Institute for Statistics',
                        'search_url': f"http://data.uis.unesco.org/",
                        'domain': 'unesco.org',
                        'priority': 650
                    },
                    {
                        'name': 'UN Data Portal',
                        'search_url': f"https://data.un.org/Search.aspx?q={quote_plus(query)}",
                        'domain': 'data.un.org',
                        'priority': 500
                    }
                ]
            elif any(term in query_lower for term in finance_terms):
                # Economic/Finance queries - prioritize financial institutions
                normalized_query = self._normalize_query_for_source(query, 'world_bank')
                intl_portals = [
                    {
                        'name': 'World Bank Open Data',
                        'search_url': self._generate_world_bank_url(normalized_query, 'economic'),
                        'fallback_url': "https://data.worldbank.org/indicator",
                        'domain': 'data.worldbank.org',
                        'priority': 600
                    },
                    {
                        'name': 'IMF Data',
                        'search_url': f"https://data.imf.org/",
                        'domain': 'imf.org',
                        'priority': 650
                    },
                    {
                        'name': 'OECD Data',
                        'search_url': f"https://data.oecd.org/searchresults/?q={quote_plus(query)}",
                        'domain': 'oecd.org',
                        'priority': 550
                    }
                ]
            else:
                # General queries - balanced approach
                normalized_query = self._normalize_query_for_source(query, 'world_bank')
                intl_portals = [
                    {
                        'name': 'World Bank Open Data',
                        'search_url': self._generate_world_bank_url(normalized_query, 'general'),
                        'fallback_url': "https://data.worldbank.org/indicator",
                        'domain': 'data.worldbank.org',
                        'priority': 600
                    },
                    {
                        'name': 'UN Data Portal',
                        'search_url': f"https://data.un.org/Search.aspx?q={quote_plus(query)}",
                        'domain': 'data.un.org',
                        'priority': 550
                    },
                    {
                        'name': 'OECD Data',
                        'search_url': f"https://data.oecd.org/searchresults/?q={quote_plus(query)}",
                        'domain': 'oecd.org',
                        'priority': 500
                    }
                ]
        
        for portal in intl_portals:
            # Validate World Bank URLs before adding to results
            search_url = portal['search_url']
            if 'worldbank' in portal['domain']:
                search_url = self._validate_world_bank_url(search_url, query)
            
            results.append({
                'title': f"Search {portal['name']} for {query} data",
                'url': search_url,
                'description': f"Global datasets and indicators from {portal['name']}",
                'source': 'international_organization',
                'type': 'global_data',
                'domain': portal['domain'],
                'relevance_score': portal['priority']
            })
            
        return results
    
    def _get_international_dataset_links(self, query: str) -> List[Dict[str, Any]]:
        """Get direct links to specific international datasets based on query"""
        results = []
        query_lower = query.lower()
        
        # Economic / GDP data
        if any(word in query_lower for word in ['gdp', 'economic', 'economy', 'growth', 'trade']):
            results.extend([
                {
                    'title': 'World Bank - GDP and Economic Indicators',
                    'url': self._generate_world_bank_url(query, 'economic'),
                    'description': 'Global GDP, economic growth, and development indicators from World Bank',
                    'source': 'world_bank',
                    'type': 'economic_data',
                    'domain': 'data.worldbank.org',
                    'relevance_score': 95
                },
                {
                    'title': 'IMF World Economic Outlook Database',
                    'url': 'https://www.imf.org/en/Publications/WEO/weo-database',
                    'description': 'International Monetary Fund economic data and forecasts',
                    'source': 'imf',
                    'type': 'economic_data',
                    'domain': 'imf.org',
                    'relevance_score': 90
                }
            ])
        
        # Health data - more specific URLs based on topic
        if any(word in query_lower for word in ['health', 'disease', 'mortality', 'life expectancy', 'covid']):
            # Determine specific WHO URL based on keywords
            who_url = 'https://www.who.int/data/gho'
            who_title = 'WHO Global Health Observatory'
            
            if any(word in query_lower for word in ['air pollution', 'pollution', 'air quality']):
                who_url = 'https://www.who.int/data/gho/data/themes/air-pollution'
                who_title = 'WHO Air Pollution Data'
            elif any(word in query_lower for word in ['covid', 'coronavirus', 'pandemic']):
                who_url = 'https://covid19.who.int/data'
                who_title = 'WHO COVID-19 Dashboard'
            elif any(word in query_lower for word in ['mortality', 'death', 'deaths']):
                who_url = 'https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates'
                who_title = 'WHO Mortality and Health Estimates'
            elif any(word in query_lower for word in ['mental health', 'depression', 'suicide']):
                who_url = 'https://www.who.int/data/gho/data/themes/mental-disorders'
                who_title = 'WHO Mental Health Data'
            elif any(word in query_lower for word in ['nutrition', 'malnutrition', 'obesity']):
                who_url = 'https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/malnutrition'
                who_title = 'WHO Nutrition Data'
                
            results.extend([
                {
                    'title': who_title,
                    'url': who_url,
                    'description': 'Global health statistics and indicators from World Health Organization',
                    'source': 'who',
                    'type': 'health_data',
                    'domain': 'who.int',
                    'relevance_score': 95
                },
                {
                    'title': 'Our World in Data - Health',
                    'url': 'https://ourworldindata.org/health-meta',
                    'description': 'Research and data on global health trends and outcomes',
                    'source': 'ourworldindata',
                    'type': 'health_data',
                    'domain': 'ourworldindata.org',
                    'relevance_score': 85
                }
            ])
        
        # Population / Demographics
        if any(word in query_lower for word in ['population', 'demographic', 'census', 'migration', 'urbanization']):
            results.extend([
                {
                    'title': 'UN Population Division Data',
                    'url': 'https://population.un.org/wpp/',
                    'description': 'World population prospects and demographic data from United Nations',
                    'source': 'un_population',
                    'type': 'demographic_data',
                    'domain': 'population.un.org',
                    'relevance_score': 95
                },
                {
                    'title': 'World Bank - Population Data',
                    'url': self._generate_world_bank_url(query, 'population'),
                    'description': 'Global population and demographic indicators',
                    'source': 'world_bank',
                    'type': 'demographic_data', 
                    'domain': 'data.worldbank.org',
                    'relevance_score': 90
                }
            ])
        
        # Education data
        if any(word in query_lower for word in ['education', 'literacy', 'school', 'university', 'learning']):
            results.extend([
                {
                    'title': 'UNESCO Institute for Statistics',
                    'url': 'http://uis.unesco.org/en/home',
                    'description': 'Global education statistics and indicators from UNESCO',
                    'source': 'unesco',
                    'type': 'education_data',
                    'domain': 'unesco.org',
                    'relevance_score': 95
                },
                {
                    'title': 'World Bank - Education Data',
                    'url': self._generate_world_bank_url(query, 'education'),
                    'description': 'Global education indicators and development data',
                    'source': 'world_bank',
                    'type': 'education_data',
                    'domain': 'data.worldbank.org',
                    'relevance_score': 90
                }
            ])
        
        # Environment / Climate data
        if any(word in query_lower for word in ['climate', 'environment', 'temperature', 'emissions', 'carbon']):
            results.extend([
                {
                    'title': 'World Bank - Climate Data',
                    'url': 'https://climatedata.worldbank.org/',
                    'description': 'Global climate and environmental data from World Bank',
                    'source': 'world_bank',
                    'type': 'climate_data',
                    'domain': 'climatedata.worldbank.org',
                    'relevance_score': 95
                },
                {
                    'title': 'Our World in Data - Environment',
                    'url': 'https://ourworldindata.org/environmental-change',
                    'description': 'Environmental change and climate data visualizations',
                    'source': 'ourworldindata',
                    'type': 'climate_data',
                    'domain': 'ourworldindata.org',
                    'relevance_score': 90
                }
            ])
        
        return results

    async def _search_government_portals(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search government data portals with direct dataset links"""
        results = []
        
        # Direct dataset links based on query keywords
        direct_links = self._get_direct_dataset_links(query)
        results.extend(direct_links)
        
        # Fallback to search portals
        gov_portals = [
            {
                'name': 'Singapore Open Data',
                'search_url': f"https://data.gov.sg/search?q={quote_plus(query)}",
                'domain': 'data.gov.sg'
            },
            {
                'name': 'SingStat TableBuilder',
                'search_url': f"https://tablebuilder.singstat.gov.sg/",
                'domain': 'singstat.gov.sg'
            }
        ]
        
        for portal in gov_portals:
            results.append({
                'title': f"Browse {portal['name']} for {query} data",
                'url': portal['search_url'],
                'description': f"Official government datasets from {portal['name']}",
                'source': 'government_portal',
                'type': 'government_data',
                'domain': portal['domain'],
                'relevance_score': 60
            })
            
        return results
    
    def _get_direct_dataset_links(self, query: str) -> List[Dict[str, Any]]:
        """Get direct links to specific datasets based on query"""
        results = []
        query_lower = query.lower()
        
        # HDB / Housing datasets
        if any(word in query_lower for word in ['hdb', 'housing', 'resale', 'property', 'flat']):
            results.append({
                'title': 'HDB Resale Flat Prices (Official)',
                'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M212161',
                'description': 'Official HDB resale flat transaction data from Department of Statistics Singapore',
                'source': 'singstat',
                'type': 'government_data',
                'domain': 'singstat.gov.sg',
                'relevance_score': 95
            })
        
        # Transport datasets
        if any(word in query_lower for word in ['transport', 'bus', 'mrt', 'train', 'traffic', 'lta']):
            results.extend([
                {
                    'title': 'Singapore Transport Data Collection',
                    'url': 'https://data.gov.sg/search?query=transport',
                    'description': 'Official transport datasets including bus routes, traffic data, and public transport information',
                    'source': 'data.gov.sg',
                    'type': 'government_data',
                    'domain': 'data.gov.sg',
                    'relevance_score': 90
                },
                {
                    'title': 'Transport Statistics (SingStat)',
                    'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M454001',
                    'description': 'Singapore transport statistics and trends from Department of Statistics',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'relevance_score': 85
                },
                {
                    'title': 'Land Transport Authority (LTA)',
                    'url': 'https://www.lta.gov.sg/',
                    'description': 'Official LTA website with transport policies, projects, and information',
                    'source': 'lta',
                    'type': 'government_data',
                    'domain': 'lta.gov.sg',
                    'relevance_score': 80
                }
            ])
        
        # Population / Demographics
        if any(word in query_lower for word in ['population', 'demographic', 'census', 'people']):
            results.append({
                'title': 'Singapore Population Trends',
                'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M810001',
                'description': 'Official population statistics and demographic data',
                'source': 'singstat',
                'type': 'government_data',
                'domain': 'singstat.gov.sg',
                'relevance_score': 90
            })
        
        # Economic data
        if any(word in query_lower for word in ['gdp', 'economy', 'economic', 'growth', 'cpi', 'inflation']):
            results.extend([
                {
                    'title': 'Singapore GDP Statistics',
                    'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M015721',
                    'description': 'Gross Domestic Product data and economic indicators',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'relevance_score': 88
                },
                {
                    'title': 'Consumer Price Index (CPI)',
                    'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M212881',
                    'description': 'Inflation and consumer price data',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'relevance_score': 85
                }
            ])
        
        # Employment
        if any(word in query_lower for word in ['employment', 'job', 'work', 'labor', 'unemployment']):
            results.append({
                'title': 'Employment Statistics',
                'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M182001',
                'description': 'Employment rates, job market data, and workforce statistics',
                'source': 'singstat',
                'type': 'government_data',
                'domain': 'singstat.gov.sg',
                'relevance_score': 88
            })
        
        # Education
        if any(word in query_lower for word in ['education', 'school', 'student', 'university']):
            results.extend([
                {
                    'title': 'School Directory and Information',
                    'url': 'https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view',
                    'description': 'Official directory of schools in Singapore',
                    'source': 'data.gov.sg',
                    'type': 'government_data',
                    'domain': 'data.gov.sg',
                    'relevance_score': 90
                },
                {
                    'title': 'Education Statistics',
                    'url': 'https://tablebuilder.singstat.gov.sg/table/TS/M850001',
                    'description': 'Student enrollment and education sector data',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'relevance_score': 85
                }
            ])
        
        # Climate/Weather
        if any(word in query_lower for word in ['weather', 'climate', 'temperature', 'rainfall', 'environment']):
            results.append({
                'title': 'Weather and Climate Data',
                'url': 'https://data.gov.sg/datasets/d_31253b1c6ba96e4dd2b8218db4e7c0d5/view',
                'description': 'Singapore weather data, rainfall, and climate statistics',
                'source': 'data.gov.sg',
                'type': 'government_data',
                'domain': 'data.gov.sg',
                'relevance_score': 88
            })
        
        return results
    
    def _ensure_minimum_source_coverage(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ensure minimum source coverage (3+ sources when possible)
        Add fallback sources if insufficient coverage
        
        Args:
            results: Current search results
            query: Original query
            context: Search context
            
        Returns:
            Results with ensured minimum source coverage
        """
        if not results:
            return results
        
        # Count unique sources
        unique_sources = set()
        source_counts = {}
        
        for result in results:
            source = result.get('source', 'unknown')
            domain = result.get('domain', '')
            
            # Create unique source identifier
            source_id = f"{source}_{domain}" if domain else source
            unique_sources.add(source_id)
            
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        current_source_count = len(unique_sources)
        min_required_sources = 3
        
        logger.info(f"📊 Source coverage analysis: {current_source_count} unique sources found")
        for source, count in source_counts.items():
            logger.info(f"   {source}: {count} results")
        
        # If we have sufficient coverage, return as is
        if current_source_count >= min_required_sources:
            logger.info(f"✅ Sufficient source coverage: {current_source_count}/{min_required_sources}")
            return results
        
        # Need to add more sources - identify missing source types
        logger.warning(f"⚠️ Insufficient source coverage: {current_source_count}/{min_required_sources}")
        
        existing_source_types = set(result.get('source', '') for result in results)
        
        # Define fallback sources in priority order
        fallback_sources = self._get_fallback_sources(query, context, existing_source_types)
        
        # Add fallback results
        enhanced_results = list(results)  # Copy original results
        
        for fallback_source in fallback_sources:
            if len(set(result.get('source', '') for result in enhanced_results)) >= min_required_sources:
                break
                
            # Add fallback result
            fallback_result = self._create_fallback_result(fallback_source, query)
            if fallback_result:
                enhanced_results.append(fallback_result)
                logger.info(f"➕ Added fallback source: {fallback_source['name']}")
        
        final_source_count = len(set(result.get('source', '') for result in enhanced_results))
        logger.info(f"📈 Final source coverage: {final_source_count} sources")
        
        return enhanced_results
    
    def _get_fallback_sources(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]], 
        existing_sources: set
    ) -> List[Dict[str, Any]]:
        """
        Get fallback sources to ensure minimum coverage
        
        Args:
            query: Search query
            context: Search context
            existing_sources: Already included source types
            
        Returns:
            List of fallback source configurations
        """
        query_lower = query.lower()
        
        # Detect Singapore context
        singapore_keywords = ['singapore', 'sg', 'hdb', 'mrt', 'singstat', 'data.gov.sg']
        has_singapore_context = any(keyword in query_lower for keyword in singapore_keywords)
        
        # Define fallback sources with priorities
        fallback_sources = []
        
        # Singapore-specific fallbacks
        if has_singapore_context:
            if 'data.gov.sg' not in existing_sources:
                fallback_sources.append({
                    'name': 'Singapore Open Data',
                    'url': f"https://data.gov.sg/search?q={quote_plus(query)}",
                    'source': 'data.gov.sg',
                    'type': 'government_data',
                    'domain': 'data.gov.sg',
                    'priority': 95
                })
            
            if 'singstat' not in existing_sources:
                fallback_sources.append({
                    'name': 'SingStat TableBuilder',
                    'url': 'https://tablebuilder.singstat.gov.sg/',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'priority': 90
                })
        
        # Global fallbacks based on query domain
        query_domain = self._detect_query_domain(query)
        
        if query_domain == 'health' and 'who' not in existing_sources:
            fallback_sources.append({
                'name': 'WHO Global Health Observatory',
                'url': 'https://www.who.int/data/gho',
                'source': 'who',
                'type': 'health_data',
                'domain': 'who.int',
                'priority': 85
            })
        
        if query_domain == 'economics' and 'world_bank' not in existing_sources:
            fallback_sources.append({
                'name': 'World Bank Open Data',
                'url': 'https://data.worldbank.org/',
                'source': 'world_bank',
                'type': 'economic_data',
                'domain': 'data.worldbank.org',
                'priority': 85
            })
        
        if 'kaggle' not in existing_sources:
            fallback_sources.append({
                'name': 'Kaggle Datasets',
                'url': f"https://www.kaggle.com/search?q={quote_plus(query)}",
                'source': 'kaggle',
                'type': 'ml_dataset_platform',
                'domain': 'kaggle.com',
                'priority': 80
            })
        
        if 'ourworldindata' not in existing_sources:
            fallback_sources.append({
                'name': 'Our World in Data',
                'url': f"https://ourworldindata.org/search?q={quote_plus(query)}",
                'source': 'ourworldindata',
                'type': 'global_data',
                'domain': 'ourworldindata.org',
                'priority': 75
            })
        
        if 'aws' not in existing_sources:
            fallback_sources.append({
                'name': 'AWS Open Data',
                'url': f"https://registry.opendata.aws/?search={quote_plus(query)}",
                'source': 'aws',
                'type': 'cloud_data_platform',
                'domain': 'registry.opendata.aws',
                'priority': 70
            })
        
        # Sort by priority
        fallback_sources.sort(key=lambda x: x['priority'], reverse=True)
        
        return fallback_sources
    
    def _create_fallback_result(self, source_config: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """
        Create a fallback result from source configuration
        
        Args:
            source_config: Source configuration
            query: Search query
            
        Returns:
            Fallback result or None if creation fails
        """
        try:
            return {
                'title': f"Search {source_config['name']} for: {query}",
                'url': source_config['url'],
                'description': f"Browse {source_config['name']} for datasets related to '{query}'",
                'source': source_config['source'],
                'type': source_config['type'],
                'domain': source_config['domain'],
                'relevance_score': source_config['priority'],
                'is_fallback': True
            }
        except Exception as e:
            logger.warning(f"Failed to create fallback result for {source_config.get('name', 'unknown')}: {str(e)}")
            return None
    
    async def _execute_searches_with_failure_handling(
        self,
        selected_sources: List[Dict[str, Any]],
        query: str,
        enhanced_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Execute searches with comprehensive failure handling and retry logic
        
        Args:
            selected_sources: List of source configurations
            query: Original query
            enhanced_query: Enhanced query for search
            context: Search context
            
        Returns:
            List of results or exceptions for each source
        """
        search_tasks = []
        
        # Create search tasks with retry wrapper
        for source_config in selected_sources:
            if source_config['type'] == 'duckduckgo':
                task = self._search_with_retry(
                    self._search_duckduckgo, enhanced_query, source_config['type']
                )
            elif source_config['type'] == 'kaggle':
                task = self._search_with_retry(
                    self._search_kaggle_datasets, query, source_config['type']
                )
            elif source_config['type'] == 'public_data':
                task = self._search_with_retry(
                    self._search_public_data_platforms, query, source_config['type']
                )
            elif source_config['type'] == 'academic':
                task = self._search_with_retry(
                    self._search_academic_sources, query, source_config['type']
                )
            elif source_config['type'] == 'international':
                task = self._search_with_retry(
                    self._search_international_organizations, query, source_config['type'], context
                )
            elif source_config['type'] == 'government':
                task = self._search_with_retry(
                    self._search_government_portals, query, source_config['type'], context
                )
            else:
                # Unknown source type, create empty result
                task = asyncio.create_task(self._create_empty_result())
            
            search_tasks.append(task)
        
        # Execute all searches with timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=30.0  # 30 second total timeout
            )
            return results
        except asyncio.TimeoutError:
            logger.error("⏰ Search execution timed out after 30 seconds")
            # Return partial results if available
            completed_tasks = [task for task in search_tasks if task.done()]
            partial_results = []
            for task in completed_tasks:
                try:
                    partial_results.append(task.result())
                except Exception as e:
                    partial_results.append(e)
            
            # Fill remaining with exceptions
            while len(partial_results) < len(search_tasks):
                partial_results.append(asyncio.TimeoutError("Search timed out"))
            
            return partial_results
    
    async def _search_with_retry(
        self,
        search_func,
        query: str,
        source_type: str,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Execute search function with exponential backoff retry logic
        
        Args:
            search_func: Search function to execute
            query: Search query
            source_type: Type of source for logging
            context: Optional context parameter
            max_retries: Maximum number of retry attempts
            
        Returns:
            Search results or raises exception after all retries
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Calculate delay for exponential backoff
                if attempt > 0:
                    delay = min(2 ** (attempt - 1), 8)  # Max 8 second delay
                    logger.info(f"🔄 Retrying {source_type} search (attempt {attempt + 1}/{max_retries + 1}) after {delay}s delay")
                    await asyncio.sleep(delay)
                
                # Execute search with timeout
                if context is not None:
                    result = await asyncio.wait_for(
                        search_func(query, context),
                        timeout=15.0  # 15 second per-source timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        search_func(query),
                        timeout=15.0
                    )
                
                # Validate result
                if isinstance(result, list):
                    logger.info(f"✅ {source_type} search succeeded on attempt {attempt + 1}")
                    return result
                else:
                    raise ValueError(f"Invalid result type: {type(result)}")
                    
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"⏰ {source_type} search timed out on attempt {attempt + 1}")
                
            except Exception as e:
                last_exception = e
                logger.warning(f"❌ {source_type} search failed on attempt {attempt + 1}: {str(e)}")
        
        # All retries exhausted
        logger.error(f"💥 {source_type} search failed after {max_retries + 1} attempts")
        raise last_exception or Exception(f"{source_type} search failed")
    
    async def _create_empty_result(self) -> List[Dict[str, Any]]:
        """Create empty result for unknown source types"""
        return []
    
    async def _handle_failed_sources(
        self,
        failed_sources: List[Dict[str, Any]],
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Handle failed sources with intelligent fallback strategies
        
        Args:
            failed_sources: List of source configurations that failed
            query: Original query
            context: Search context
            
        Returns:
            List of fallback results
        """
        fallback_results = []
        
        # Analyze failed sources to determine best fallback strategy
        failed_types = set(source['type'] for source in failed_sources)
        
        logger.info(f"🔄 Handling {len(failed_sources)} failed sources: {failed_types}")
        
        # Strategy 1: Try alternative sources of the same type
        for failed_source in failed_sources:
            alternative_results = await self._try_alternative_sources(
                failed_source, query, context
            )
            fallback_results.extend(alternative_results)
        
        # Strategy 2: Add generic fallback sources if still insufficient
        if len(fallback_results) < len(failed_sources):
            generic_fallbacks = self._get_generic_fallback_sources(query, context)
            for fallback_source in generic_fallbacks:
                if len(fallback_results) >= len(failed_sources):
                    break
                
                fallback_result = self._create_fallback_result(fallback_source, query)
                if fallback_result:
                    fallback_results.append(fallback_result)
        
        logger.info(f"🔄 Generated {len(fallback_results)} fallback results for failed sources")
        return fallback_results
    
    async def _try_alternative_sources(
        self,
        failed_source: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Try alternative sources when a primary source fails
        
        Args:
            failed_source: Configuration of the failed source
            query: Search query
            context: Search context
            
        Returns:
            List of alternative results
        """
        source_type = failed_source['type']
        alternatives = []
        
        # Define alternative sources for each type
        source_alternatives = {
            'kaggle': [
                {
                    'name': 'Hugging Face Datasets',
                    'url': f"https://huggingface.co/datasets?search={quote_plus(query)}",
                    'source': 'huggingface',
                    'type': 'ml_dataset_platform',
                    'domain': 'huggingface.co',
                    'priority': 85
                }
            ],
            'international': [
                {
                    'name': 'Our World in Data',
                    'url': f"https://ourworldindata.org/search?q={quote_plus(query)}",
                    'source': 'ourworldindata',
                    'type': 'global_data',
                    'domain': 'ourworldindata.org',
                    'priority': 80
                }
            ],
            'academic': [
                {
                    'name': 'Google Dataset Search',
                    'url': f"https://datasetsearch.research.google.com/search?query={quote_plus(query)}",
                    'source': 'google_dataset_search',
                    'type': 'dataset_search_engine',
                    'domain': 'datasetsearch.research.google.com',
                    'priority': 85
                }
            ],
            'public_data': [
                {
                    'name': 'Data.gov',
                    'url': f"https://catalog.data.gov/dataset?q={quote_plus(query)}",
                    'source': 'data.gov',
                    'type': 'government_data',
                    'domain': 'data.gov',
                    'priority': 75
                }
            ]
        }
        
        # Get alternatives for the failed source type
        if source_type in source_alternatives:
            for alt_config in source_alternatives[source_type]:
                alt_result = self._create_fallback_result(alt_config, query)
                if alt_result:
                    alternatives.append(alt_result)
                    logger.info(f"➕ Added alternative for failed {source_type}: {alt_config['name']}")
        
        return alternatives
    
    def _get_generic_fallback_sources(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get generic fallback sources when specific alternatives aren't available
        
        Args:
            query: Search query
            context: Search context
            
        Returns:
            List of generic fallback source configurations
        """
        query_lower = query.lower()
        
        # Detect Singapore context for appropriate fallbacks
        singapore_keywords = ['singapore', 'sg', 'hdb', 'mrt', 'singstat', 'data.gov.sg']
        has_singapore_context = any(keyword in query_lower for keyword in singapore_keywords)
        
        generic_fallbacks = []
        
        if has_singapore_context:
            # Singapore-focused generic fallbacks
            generic_fallbacks.extend([
                {
                    'name': 'Singapore Open Data Portal',
                    'url': 'https://data.gov.sg/',
                    'source': 'data.gov.sg',
                    'type': 'government_data',
                    'domain': 'data.gov.sg',
                    'priority': 90
                },
                {
                    'name': 'Department of Statistics Singapore',
                    'url': 'https://www.singstat.gov.sg/',
                    'source': 'singstat',
                    'type': 'government_data',
                    'domain': 'singstat.gov.sg',
                    'priority': 85
                }
            ])
        
        # Global generic fallbacks
        generic_fallbacks.extend([
            {
                'name': 'Kaggle Datasets',
                'url': 'https://www.kaggle.com/datasets',
                'source': 'kaggle',
                'type': 'ml_dataset_platform',
                'domain': 'kaggle.com',
                'priority': 80
            },
            {
                'name': 'Google Dataset Search',
                'url': 'https://datasetsearch.research.google.com/',
                'source': 'google_dataset_search',
                'type': 'dataset_search_engine',
                'domain': 'datasetsearch.research.google.com',
                'priority': 75
            },
            {
                'name': 'Our World in Data',
                'url': 'https://ourworldindata.org/',
                'source': 'ourworldindata',
                'type': 'global_data',
                'domain': 'ourworldindata.org',
                'priority': 70
            }
        ])
        
        # Sort by priority
        generic_fallbacks.sort(key=lambda x: x['priority'], reverse=True)
        
        return generic_fallbacks
    
    def _rank_search_results_with_domain_awareness(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank search results with domain awareness and enhanced Singapore context detection
        Implements intelligent prioritization based on query domain and context
        """
        return self._rank_search_results(results, query, context)
    
    def _rank_search_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rank search results by relevance and data source quality with enhanced domain awareness"""
        
        # Check if query has Singapore context - enhanced detection
        query_lower = query.lower()
        singapore_keywords = [
            'singapore', 'sg', 'hdb', 'mrt', 'coe', 'cpf', 'lta', 'singstat',
            'data.gov.sg', 'resale flat', 'bto', 'public housing', 'hawker',
            'void deck', 'wet market', 'ccb', 'east coast', 'orchard road'
        ]
        
        # Enhanced context: only if explicit Singapore terms mentioned
        has_singapore_context = (
            any(keyword in query_lower for keyword in singapore_keywords) or
            ('housing singapore' in query_lower or 'hdb' in query_lower or 'singapore property' in query_lower)
        )
        
        # Detect query domain for enhanced scoring
        query_domain = self._detect_query_domain(query)
        
        def calculate_score(result: Dict[str, Any]) -> float:
            score = 0.0
            
            # Domain priority scoring - reorder for Singapore context
            domain = self._extract_domain(result.get('url', ''))
            result_source = result.get('source', '')
            result_type = result.get('type', '')
            
            if has_singapore_context:
                # Singapore government domains get highest priority
                singapore_domains = ['data.gov.sg', 'singstat.gov.sg', 'lta.gov.sg', 'onemap.sg']
                if domain in singapore_domains:
                    score += 50 * (100 + singapore_domains.index(domain))  # High priority for Singapore
                elif domain in self.priority_domains:
                    # Global domains get lower priority when Singapore context
                    priority_index = self.priority_domains.index(domain)
                    score += 20 * (len(self.priority_domains) - priority_index)
            else:
                # Default global priority
                if domain in self.priority_domains:
                    score += 50 * (len(self.priority_domains) - self.priority_domains.index(domain))
            
            # Domain-specific source boosting based on query type
            domain_source_boosts = {
                'health': {
                    'who': 40, 'ourworldindata': 35, 'academic_search': 30,
                    'government_portal': 25, 'international_organization': 35
                },
                'economics': {
                    'world_bank': 45, 'imf': 40, 'oecd': 35, 'international_organization': 40,
                    'government_portal': 30, 'academic_search': 25
                },
                'education': {
                    'unesco': 40, 'academic_search': 35, 'international_organization': 35,
                    'government_portal': 30, 'ourworldindata': 25
                },
                'environment': {
                    'international_organization': 40, 'academic_search': 35, 'ourworldindata': 35,
                    'aws': 30, 'government_portal': 25
                },
                'technology': {
                    'kaggle': 50, 'aws': 40, 'academic_search': 35,
                    'ml_dataset_platform': 45, 'cloud_data_platform': 35
                },
                'demographics': {
                    'international_organization': 45, 'government_portal': 40, 'academic_search': 30,
                    'ourworldindata': 35, 'world_bank': 35
                },
                'transport': {
                    'government_portal': 50, 'aws': 30, 'international_organization': 25,
                    'academic_search': 20
                }
            }
            
            # Apply domain-specific boost
            if query_domain in domain_source_boosts:
                source_boosts = domain_source_boosts[query_domain]
                
                # Check source name
                for boost_source, boost_value in source_boosts.items():
                    if (boost_source in result_source.lower() or 
                        boost_source in result_type.lower() or
                        boost_source in domain.lower()):
                        score += boost_value
                        logger.debug(f"Applied domain boost: {boost_source} +{boost_value} for {query_domain}")
                        break
            
            # Type-based scoring - prioritize Singapore sources when Singapore context detected
            if has_singapore_context:
                type_scores = {
                    'government_data': 120,       # Singapore government data (highest for local context)
                    'ml_dataset_platform': 115,  # Kaggle, Hugging Face (2nd highest - accessible datasets)
                    'academic_search': 110,      # Zenodo, Figshare (accessible research)
                    'global_data': 100,          # International organizations 
                    'dataset_search_engine': 95, # Google Dataset Search
                    'economic_data': 90,         # World Bank, IMF economic data
                    'health_data': 90,           # WHO, health organizations
                    'demographic_data': 90,      # UN Population, census data
                    'education_data': 90,        # UNESCO, education statistics
                    'climate_data': 90,          # Climate and environmental data
                    'cloud_data_platform': 85,  # AWS Open Data
                    'academic_repository': 80,   # Harvard Dataverse, Mendeley
                    'open_data_platform': 75,   # OpenDataSoft
                    'web_search': 60             # General web search
                }
            else:
                type_scores = {
                    'ml_dataset_platform': 120,  # Kaggle, Hugging Face (HIGHEST priority - most accessible)
                    'academic_search': 115,      # Zenodo, Figshare (accessible research data)
                    'global_data': 105,          # International organizations 
                    'economic_data': 100,        # World Bank, IMF economic data
                    'health_data': 100,          # WHO, health organizations
                    'demographic_data': 100,     # UN Population, census data
                    'education_data': 100,       # UNESCO, education statistics
                    'climate_data': 100,         # Climate and environmental data
                    'dataset_search_engine': 92, # Google Dataset Search
                    'cloud_data_platform': 88,   # AWS Open Data
                    'academic_repository': 85,    # Harvard Dataverse, Mendeley
                    'open_data_platform': 78,    # OpenDataSoft
                    'web_search': 60              # General web search
                }
            score += type_scores.get(result.get('type', 'web_search'), 60)
            
            # Title relevance
            title = result.get('title', '').lower()
            query_words = query.lower().split()
            title_matches = sum(1 for word in query_words if word in title)
            score += title_matches * 20
            
            # Description relevance
            description = result.get('description', '').lower()
            desc_matches = sum(1 for word in query_words if word in description)
            score += desc_matches * 10
            
            # Data-related keywords bonus - context-aware
            if has_singapore_context:
                data_keywords = [
                    'dataset', 'data', 'statistics', 'research', 'open data',
                    'singapore', 'data.gov.sg', 'singstat', 'lta', 'hdb', 'government'
                ]
            else:
                data_keywords = [
                    'dataset', 'data', 'statistics', 'research', 'open data',
                    'indicators', 'world bank', 'united nations', 'who', 'unesco',
                    'oecd', 'imf', 'global', 'international', 'kaggle', 'machine learning',
                    'repository', 'public dataset', 'csv', 'json', 'api'
                ]
            for keyword in data_keywords:
                if keyword in title or keyword in description:
                    score += 15
            
            return score
        
        # Sort by calculated score
        scored_results = [(result, calculate_score(result)) for result in results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Add score to results and return
        ranked_results = []
        for result, score in scored_results:
            result['relevance_score'] = score
            ranked_results.append(result)
            
        return ranked_results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return ""
    
    async def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a web page"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': self.user_agent}
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        if BeautifulSoup is None:
                            return {'url': url, 'status': 'parser_unavailable'}
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract metadata
                        title = soup.find('title')
                        title_text = title.get_text(strip=True) if title else ""
                        
                        # Meta description
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        description = meta_desc.get('content', '') if meta_desc else ""
                        
                        # Keywords
                        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                        keywords = meta_keywords.get('content', '') if meta_keywords else ""
                        
                        return {
                            'title': title_text,
                            'description': description,
                            'keywords': keywords,
                            'url': url,
                            'status': 'accessible'
                        }
                        
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {url}: {str(e)}")
            
        return {
            'url': url,
            'status': 'inaccessible',
            'error': str(e) if 'e' in locals() else 'Unknown error'
        }

    
    async def _validate_and_correct_result_urls(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and correct URLs in search results with comprehensive monitoring and fallback strategies
        
        Args:
            results: List of search results with URLs
            
        Returns:
            List of validated and corrected results
        """
        if not results:
            return results
        
        try:
            validation_start_time = time.time()
            logger.info(f"🔍 Starting URL validation for {len(results)} results")
            
            # Use the enhanced URL validator for external sources with real-time validation
            validated_results = await self.url_validator.validate_external_search_results(results)
            
            # Apply intelligent fallback strategies for failed validations
            final_results = []
            validation_stats = {
                'verified': 0,
                'corrected_verified': 0,
                'fallback_verified': 0,
                'final_fallback': 0,
                'failed': 0,
                'errors': 0
            }
            
            for result in validated_results:
                url_status = result.get('url_status', 'unknown')
                original_url = result.get('original_url', result.get('url', ''))
                current_url = result.get('url', '')
                source = result.get('source', 'unknown')
                
                if url_status in ['verified']:
                    # URL is working perfectly
                    validation_stats['verified'] += 1
                    final_results.append(result)
                    
                elif url_status in ['corrected_verified']:
                    # URL was corrected and now works
                    validation_stats['corrected_verified'] += 1
                    result['description'] = f"{result.get('description', '')} (URL corrected for better access)"
                    final_results.append(result)
                    logger.info(f"✅ URL corrected for {source}: {original_url} → {current_url}")
                    
                elif url_status in ['fallback_verified']:
                    # Using fallback URL that works
                    validation_stats['fallback_verified'] += 1
                    result['description'] = f"{result.get('description', '')} (Using verified source page)"
                    final_results.append(result)
                    logger.info(f"🔄 Fallback URL used for {source}: {current_url}")
                    
                elif url_status in ['failed_validation', 'failed']:
                    # Try final fallback strategy
                    fallback_result = self._apply_final_fallback_strategy(result)
                    if fallback_result:
                        validation_stats['final_fallback'] += 1
                        final_results.append(fallback_result)
                        logger.warning(f"⚠️ Final fallback applied for {source}: {fallback_result.get('url')}")
                    else:
                        validation_stats['failed'] += 1
                        # Log the failure for monitoring
                        self.url_validator.log_validation_failure(
                            current_url, source, f"URL validation failed with status: {url_status}", 
                            result.get('status_code', 0)
                        )
                        logger.warning(f"❌ URL validation failed for {source}, result excluded: {current_url}")
                        
                elif url_status in ['validation_error', 'correction_error', 'critical_error']:
                    # Handle validation errors
                    validation_stats['errors'] += 1
                    error_msg = result.get('validation_error', 'Unknown validation error')
                    self.url_validator.log_validation_failure(current_url, source, error_msg, 0)
                    
                    # Try final fallback for error cases
                    fallback_result = self._apply_final_fallback_strategy(result)
                    if fallback_result:
                        validation_stats['final_fallback'] += 1
                        final_results.append(fallback_result)
                        logger.warning(f"🔄 Error recovery fallback for {source}: {fallback_result.get('url')}")
                    else:
                        logger.error(f"💥 Critical validation error for {source}, result excluded: {error_msg}")
                        
                else:
                    # Unknown status, apply conservative approach
                    result['url_status'] = 'unverified'
                    final_results.append(result)
                    logger.debug(f"❓ Unknown validation status '{url_status}' for {source}, keeping original")
            
            # Calculate validation performance metrics
            validation_time = time.time() - validation_start_time
            total_results = len(results)
            final_count = len(final_results)
            success_rate = (validation_stats['verified'] + validation_stats['corrected_verified'] + 
                          validation_stats['fallback_verified']) / max(total_results, 1) * 100
            
            # Log comprehensive validation summary
            logger.info(f"🔍 URL Validation Complete:")
            logger.info(f"   ✅ Verified: {validation_stats['verified']}")
            logger.info(f"   🔧 Corrected: {validation_stats['corrected_verified']}")
            logger.info(f"   🔄 Fallback: {validation_stats['fallback_verified']}")
            logger.info(f"   ⚠️ Final Fallback: {validation_stats['final_fallback']}")
            logger.info(f"   ❌ Failed: {validation_stats['failed']}")
            logger.info(f"   💥 Errors: {validation_stats['errors']}")
            logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
            logger.info(f"   ⏱️ Time: {validation_time:.2f}s")
            logger.info(f"   📈 Results: {final_count}/{total_results}")
            
            # Log performance metrics for monitoring
            self._log_url_validation_performance(final_results, validation_time, validation_stats)
            
            return final_results
            
        except Exception as e:
            logger.error(f"💥 Critical error during URL validation: {str(e)}")
            # Log the critical error for monitoring
            for result in results:
                source = result.get('source', 'unknown')
                url = result.get('url', '')
                self.url_validator.log_validation_failure(url, source, f"Critical validation error: {str(e)}", 0)
            
            # Return original results with error status
            for result in results:
                result['url_status'] = 'validation_system_error'
                result['validation_error'] = str(e)
            
            return results
    
    def _apply_final_fallback_strategy(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply final fallback strategy for failed URL validations
        
        Args:
            result: Search result with failed URL
            
        Returns:
            Modified result with fallback URL or None if no fallback possible
        """
        source = result.get('source', '').lower()
        domain = result.get('domain', '')
        
        # Define fallback URLs for major sources
        fallback_urls = {
            'kaggle': 'https://www.kaggle.com/datasets',
            'world_bank': 'https://data.worldbank.org/',
            'aws': 'https://registry.opendata.aws/',
            'un': 'https://data.un.org/',
            'who': 'https://www.who.int/data/gho',
            'oecd': 'https://data.oecd.org/',
            'huggingface': 'https://huggingface.co/datasets'
        }
        
        # Try to find appropriate fallback
        fallback_url = None
        for key, url in fallback_urls.items():
            if key in source or key in domain:
                fallback_url = url
                break
        
        if fallback_url:
            result['url'] = fallback_url
            result['url_status'] = 'final_fallback'
            result['description'] = f"{result.get('description', '')} (Browse {source} datasets - specific link unavailable)"
            logger.info(f"🔄 Applied final fallback for {source}: {fallback_url}")
            return result
        
        return None
    
    def _log_url_validation_performance(self, results: List[Dict[str, Any]], validation_time: float, 
                                       validation_stats: Dict[str, int]):
        """
        Log comprehensive URL validation performance metrics for monitoring and analytics
        
        Args:
            results: Validated results
            validation_time: Time taken for validation
            validation_stats: Detailed validation statistics
        """
        try:
            # Collect detailed status information
            status_counts = {}
            source_performance = {}
            
            for result in results:
                status = result.get('url_status', 'unknown')
                source = result.get('source', 'unknown')
                response_time = result.get('validation_timestamp', '')
                
                # Count by status
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Track per-source performance
                if source not in source_performance:
                    source_performance[source] = {
                        'total': 0,
                        'verified': 0,
                        'corrected': 0,
                        'fallback': 0,
                        'failed': 0
                    }
                
                source_performance[source]['total'] += 1
                if status in ['verified']:
                    source_performance[source]['verified'] += 1
                elif status in ['corrected_verified']:
                    source_performance[source]['corrected'] += 1
                elif status in ['fallback_verified', 'final_fallback']:
                    source_performance[source]['fallback'] += 1
                else:
                    source_performance[source]['failed'] += 1
            
            # Calculate performance metrics
            total_results = len(results)
            success_count = validation_stats['verified'] + validation_stats['corrected_verified'] + validation_stats['fallback_verified']
            success_rate = (success_count / max(total_results, 1)) * 100 if total_results > 0 else 0
            avg_time_per_result = validation_time / max(total_results, 1)
            
            # Log detailed performance metrics
            logger.info(f"📊 Detailed URL Validation Performance:")
            logger.info(f"   ⏱️ Total Time: {validation_time:.2f}s")
            logger.info(f"   📈 Avg Time/Result: {avg_time_per_result:.3f}s")
            logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
            logger.info(f"   🎯 Results Processed: {total_results}")
            
            # Log per-source performance
            logger.info(f"📋 Source Performance Breakdown:")
            for source, stats in source_performance.items():
                source_success_rate = ((stats['verified'] + stats['corrected'] + stats['fallback']) / 
                                     max(stats['total'], 1)) * 100
                logger.info(f"   {source}: {source_success_rate:.1f}% success ({stats['total']} total)")
            
            # Create comprehensive performance data for monitoring systems
            performance_data = {
                'timestamp': time.time(),
                'validation_time_seconds': validation_time,
                'avg_time_per_result_seconds': avg_time_per_result,
                'total_results': total_results,
                'success_rate_percent': success_rate,
                'detailed_stats': validation_stats,
                'status_breakdown': status_counts,
                'source_performance': source_performance,
                'system_health': {
                    'validation_errors': validation_stats.get('errors', 0),
                    'critical_failures': validation_stats.get('failed', 0),
                    'fallback_usage_rate': (validation_stats.get('fallback_verified', 0) + 
                                          validation_stats.get('final_fallback', 0)) / max(total_results, 1) * 100
                }
            }
            
            # Log structured data for monitoring systems
            logger.debug(f"Structured URL validation metrics: {json.dumps(performance_data, indent=2)}")
            
            # Alert on poor performance
            if success_rate < 50:
                logger.warning(f"⚠️ Low URL validation success rate: {success_rate:.1f}%")
            
            if validation_stats.get('errors', 0) > total_results * 0.2:
                logger.warning(f"⚠️ High validation error rate: {validation_stats['errors']}/{total_results}")
            
            if validation_time > 30:  # More than 30 seconds for validation
                logger.warning(f"⚠️ Slow URL validation performance: {validation_time:.2f}s")
            
            # This could be extended to:
            # - Send metrics to monitoring systems (Prometheus, DataDog, etc.)
            # - Store in database for trend analysis
            # - Trigger alerts for performance degradation
            # - Update health check endpoints
            
        except Exception as e:
            logger.warning(f"Failed to log URL validation performance: {str(e)}")
            # Ensure we don't fail the main validation process due to logging issues