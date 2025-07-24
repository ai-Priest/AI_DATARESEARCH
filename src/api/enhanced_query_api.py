"""
Enhanced Query API - Quality-First Approach
Integrates the enhanced query router with the API layer
"""
import logging
import time
from typing import Dict, List
import asyncio

# Import the enhanced query router
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ..ai.enhanced_query_router import EnhancedQueryRouter

logger = logging.getLogger(__name__)

class EnhancedQueryAPI:
    """API integration for enhanced query routing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.router = EnhancedQueryRouter(self.config)
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        logger.info("🚀 EnhancedQueryAPI initialized")
    
    async def process_query(self, query: str, context: Dict = None) -> Dict:
        """Process a query with quality-first approach"""
        start_time = time.time()
        context = context or {}
        
        # Check cache first
        cache_key = f"{query}".lower()
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.info(f"🔍 Cache hit for query: {query}")
                return cache_entry['result']
        
        # Classify query
        classification = self.router.classify_query(query)
        
        # Prepare response
        response = {
            'query': query,
            'classification': {
                'domain': classification.domain,
                'geographic_scope': classification.geographic_scope,
                'intent': classification.intent,
                'singapore_first': classification.singapore_first_applicable,
                'confidence': classification.confidence
            },
            'sources': [
                {
                    'name': source,
                    'relevance': 0.8,  # Simplified for now
                    'reason': f'Recommended for {classification.domain} queries'
                }
                for source in classification.recommended_sources
            ],
            'processing_time': time.time() - start_time
        }
        
        # Cache result
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'result': response
        }
        
        logger.info(f"✅ Processed query: {query}")
        logger.info(f"  Domain: {classification.domain}")
        logger.info(f"  Singapore-first: {classification.singapore_first_applicable}")
        logger.info(f"  Processing time: {response['processing_time']:.3f}s")
        
        return response

if __name__ == "__main__":
    async def test_api():
        logging.basicConfig(level=logging.INFO)
        api = EnhancedQueryAPI()
        
        # Test queries
        test_queries = [
            "psychology research data",
            "singapore housing",
            "climate change indicators"
        ]
        
        print("🧪 Testing Enhanced Query API\n")
        
        for query in test_queries:
            print(f"📝 Query: {query}")
            result = await api.process_query(query)
            print(f"  Domain: {result['classification']['domain']}")
            print(f"  Singapore-first: {result['classification']['singapore_first']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            print(f"  Top sources: {', '.join([s['name'] for s in result['sources'][:3]])}")
            print()
        
        print("✅ Enhanced Query API testing complete!")
    
    asyncio.run(test_api())