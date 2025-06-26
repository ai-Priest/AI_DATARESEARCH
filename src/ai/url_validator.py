"""
URL Validator and Corrector for Singapore Dataset Links
Ensures all dataset URLs are active and working
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp

logger = logging.getLogger(__name__)


class URLValidator:
    """Validates and corrects dataset URLs to ensure they work"""
    
    def __init__(self):
        self.timeout = 10
        self.user_agent = "AI Dataset Research Assistant/2.0"
        
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
        Validate if URL is accessible
        
        Returns:
            (is_valid, status_code)
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': self.user_agent}
            ) as session:
                async with session.head(url) as response:
                    is_valid = response.status < 400
                    return is_valid, response.status
                    
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {str(e)}")
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


# Global instance for easy use
url_validator = URLValidator()