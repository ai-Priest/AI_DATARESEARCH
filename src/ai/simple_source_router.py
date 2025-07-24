"""
Simple Source Priority Router for testing - AI Implementation
Part of the AI implementation layer for basic query routing
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class SimpleSourceRouter:
    """Simple router for testing source priority routing"""
    
    def __init__(self):
        self.sources = {
            'kaggle': {'quality': 0.85, 'domains': ['ml', 'psychology']},
            'zenodo': {'quality': 0.80, 'domains': ['psychology', 'climate']},
            'world_bank': {'quality': 0.90, 'domains': ['economics', 'climate']},
            'data_gov_sg': {'quality': 0.95, 'domains': ['singapore'], 'singapore': True},
            'singstat': {'quality': 0.95, 'domains': ['singapore'], 'singapore': True},
            'lta_datamall': {'quality': 0.90, 'domains': ['singapore', 'transport'], 'singapore': True}
        }
        
    def route_query(self, query: str, domain: str, singapore_first: bool = False) -> List[Dict]:
        """Route query to appropriate sources"""
        results = []
        
        for source_name, source_info in self.sources.items():
            relevance = 0.5
            
            # Domain match
            if domain in source_info.get('domains', []):
                relevance += 0.3
                
            # Singapore priority
            if singapore_first and source_info.get('singapore', False):
                relevance += 0.2
                
            # Quality influence
            relevance = 0.7 * relevance + 0.3 * source_info['quality']
            
            results.append({
                'source': source_name,
                'relevance_score': min(relevance, 1.0),
                'routing_reason': 'simple_routing'
            })
        
        # Sort by relevance
        results.sort(key=lambda x: -x['relevance_score'])
        return results[:5]

if __name__ == "__main__":
    print("🧪 Testing Simple Source Router - AI Implementation")
    
    router = SimpleSourceRouter()
    
    # Test Singapore query
    sources = router.route_query("singapore housing", "singapore", True)
    print(f"\nSingapore housing query:")
    for source in sources[:3]:
        print(f"  - {source['source']}: {source['relevance_score']:.2f}")
    
    # Test psychology query
    sources = router.route_query("psychology data", "psychology", False)
    print(f"\nPsychology query:")
    for source in sources[:3]:
        print(f"  - {source['source']}: {source['relevance_score']:.2f}")
    
    print("\n✅ Simple router test complete!")