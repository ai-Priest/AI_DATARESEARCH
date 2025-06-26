"""
Web Search Engine for Dataset Discovery
Integrates web search to find additional data sources beyond local datasets
"""
import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp

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
            
            # Global Data Platforms
            'kaggle.com',
            'data.gov',
            'eurostat.ec.europa.eu',
            'github.com',
            'zenodo.org',
            'figshare.com',
            'datacite.org',
            'ourworldindata.org',
            'gapminder.org',
            
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
        Search the web for relevant data sources and research materials
        
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
            
            # Perform parallel searches across multiple strategies
            search_tasks = [
                self._search_duckduckgo(enhanced_query),
                self._search_academic_sources(query),
                self._search_international_organizations(query, context),
                self._search_government_portals(query, context)
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and rank results
            combined_results = []
            for result_set in results:
                if isinstance(result_set, list):
                    combined_results.extend(result_set)
            
            # Rank and filter results
            ranked_results = self._rank_search_results(combined_results, query)
            
            search_time = time.time() - start_time
            logger.info(f"🌐 Web search completed: {len(ranked_results)} results in {search_time:.2f}s")
            
            return ranked_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {str(e)}")
            return []
    
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
        
        # International data portals
        intl_portals = [
            {
                'name': 'World Bank Open Data',
                'search_url': f"https://data.worldbank.org/search?q={quote_plus(query)}",
                'domain': 'data.worldbank.org',
                'priority': 95
            },
            {
                'name': 'UN Data Portal',
                'search_url': f"https://data.un.org/Search.aspx?q={quote_plus(query)}",
                'domain': 'data.un.org',
                'priority': 95
            },
            {
                'name': 'UN Statistics',
                'search_url': f"https://unstats.un.org/UNSDWebsite/",
                'domain': 'unstats.un.org',
                'priority': 90
            },
            {
                'name': 'WHO Global Health Observatory',
                'search_url': f"https://www.who.int/data/gho",
                'domain': 'who.int',
                'priority': 90
            },
            {
                'name': 'OECD Data',
                'search_url': f"https://data.oecd.org/searchresults/?q={quote_plus(query)}",
                'domain': 'oecd.org',
                'priority': 85
            },
            {
                'name': 'IMF Data',
                'search_url': f"https://data.imf.org/",
                'domain': 'imf.org',
                'priority': 85
            },
            {
                'name': 'Eurostat',
                'search_url': f"https://ec.europa.eu/eurostat/web/main/search/-/search/estatsearch?text={quote_plus(query)}",
                'domain': 'eurostat.ec.europa.eu',
                'priority': 80
            },
            {
                'name': 'Our World in Data',
                'search_url': f"https://ourworldindata.org/search?q={quote_plus(query)}",
                'domain': 'ourworldindata.org',
                'priority': 75
            }
        ]
        
        for portal in intl_portals:
            results.append({
                'title': f"Search {portal['name']} for {query} data",
                'url': portal['search_url'],
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
                    'url': 'https://data.worldbank.org/topic/economy-and-growth',
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
        
        # Health data
        if any(word in query_lower for word in ['health', 'disease', 'mortality', 'life expectancy', 'covid']):
            results.extend([
                {
                    'title': 'WHO Global Health Observatory',
                    'url': 'https://www.who.int/data/gho',
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
                    'url': 'https://data.worldbank.org/topic/health',
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
                    'url': 'https://data.worldbank.org/topic/education',
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
    
    def _rank_search_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Rank search results by relevance and data source quality"""
        
        def calculate_score(result: Dict[str, Any]) -> float:
            score = 0.0
            
            # Domain priority scoring
            domain = self._extract_domain(result.get('url', ''))
            if domain in self.priority_domains:
                score += 50 * (len(self.priority_domains) - self.priority_domains.index(domain))
            
            # Type-based scoring - prioritize global sources
            type_scores = {
                'global_data': 110,           # New: International organizations
                'economic_data': 105,         # World Bank, IMF economic data
                'health_data': 105,           # WHO, health organizations
                'demographic_data': 105,      # UN Population, census data
                'education_data': 105,        # UNESCO, education statistics
                'climate_data': 105,          # Climate and environmental data
                'government_data': 100,       # National government data
                'academic_search': 80,        # Academic repositories
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
            
            # Data-related keywords bonus - enhanced for global sources
            data_keywords = [
                'dataset', 'data', 'statistics', 'research', 'open data',
                'indicators', 'world bank', 'united nations', 'who', 'unesco',
                'oecd', 'imf', 'global', 'international'
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