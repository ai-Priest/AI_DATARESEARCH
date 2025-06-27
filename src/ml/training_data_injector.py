"""
Training Data Injector - Add direct training data without waiting for user feedback
Allows manual injection of query-source mappings to improve neural network performance
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class TrainingDataInjector:
    """Inject training data directly into the feedback system for immediate learning"""
    
    def __init__(self, feedback_file: str = "data/feedback/user_feedback.json"):
        self.feedback_file = Path(feedback_file)
        self.feedback_data = []
        self._load_existing_feedback()
    
    def _load_existing_feedback(self):
        """Load existing feedback data"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.feedback_data)} existing feedback entries")
            else:
                logger.info("📝 Starting with fresh feedback system")
        except Exception as e:
            logger.warning(f"Could not load existing feedback: {e}")
            self.feedback_data = []
    
    def _save_feedback(self):
        """Save feedback data to file"""
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
        logger.info(f"💾 Saved {len(self.feedback_data)} feedback entries")
    
    def inject_query_source_mappings(self, mappings: List[Tuple[str, str, float, str]]):
        """
        Inject direct query-source mappings as high-quality training data
        
        Args:
            mappings: List of (query, source, relevance_score, reason) tuples
            
        Example:
            mappings = [
                ("psychology", "kaggle", 0.95, "Best for research datasets"),
                ("psychology", "zenodo", 0.90, "Academic repositories"),
                ("psychology", "world_bank", 0.30, "Limited psychology data"),
                ("machine learning", "kaggle", 0.98, "ML competitions and datasets"),
                ("climate data", "world_bank", 0.85, "Global climate indicators")
            ]
        """
        timestamp = datetime.now()
        
        for query, source, relevance, reason in mappings:
            
            # Create synthetic feedback entry that mimics real user behavior
            feedback_entry = {
                "id": f"training_injection_{query}_{source}_{timestamp.timestamp()}",
                "user_id": "training_system",
                "session_id": f"training_session_{timestamp.strftime('%Y%m%d')}",
                "timestamp": timestamp.isoformat(),
                "query": query,
                "interaction_type": "synthetic_training",
                "dataset_source": source,
                "relevance_score": relevance,
                "training_reason": reason,
                "synthetic_feedback": True,
                
                # Simulate user rating based on relevance score
                "rating": int(relevance * 5),  # Convert 0-1 to 1-5 scale
                "satisfaction_score": relevance,
                "meets_threshold": relevance >= 0.85,
                
                # Add context for neural training
                "source_quality_signals": {
                    "has_search_functionality": source in ["kaggle", "zenodo"],
                    "academic_focused": source in ["zenodo", "arxiv"],
                    "commercial_platform": source in ["kaggle"],
                    "government_source": source in ["world_bank", "data_gov_sg", "singstat"],
                    "research_oriented": source in ["zenodo", "kaggle", "arxiv"]
                }
            }
            
            self.feedback_data.append(feedback_entry)
            logger.info(f"💉 Injected training data: {query} → {source} (relevance: {relevance})")
        
        self._save_feedback()
        return len(mappings)
    
    def inject_from_markdown_file(self, markdown_file: str):
        """
        Parse markdown file and inject training data
        
        Expected format:
        # Training Data Mappings
        
        ## Psychology Queries
        - psychology → kaggle (0.95) - Best platform for psychology datasets
        - psychology → zenodo (0.90) - Academic repository
        - psychology → world_bank (0.30) - Limited psychology data
        
        ## Machine Learning Queries  
        - machine learning → kaggle (0.98) - ML competitions and datasets
        """
        
        md_path = Path(markdown_file)
        if not md_path.exists():
            logger.error(f"Markdown file not found: {markdown_file}")
            return 0
        
        with open(md_path, 'r') as f:
            content = f.read()
        
        mappings = []
        
        # Parse markdown content
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for mapping lines: "- query → source (score) - reason"
            if line.startswith('- ') and '→' in line and '(' in line and ')' in line:
                try:
                    # Parse: "- psychology → kaggle (0.95) - Best platform"
                    parts = line[2:].split('→')  # Remove "- " prefix
                    query = parts[0].strip()
                    
                    rest = parts[1].strip()
                    source_and_score = rest.split('(')[0].strip()
                    score_part = rest.split('(')[1].split(')')[0]
                    reason_part = rest.split(')')[1].strip()
                    
                    source = source_and_score
                    relevance = float(score_part)
                    reason = reason_part.lstrip('- ').strip()
                    
                    mappings.append((query, source, relevance, reason))
                    
                except Exception as e:
                    logger.warning(f"Could not parse line: {line} - {e}")
                    continue
        
        if mappings:
            count = self.inject_query_source_mappings(mappings)
            logger.info(f"📝 Injected {count} mappings from {markdown_file}")
            return count
        else:
            logger.warning(f"No valid mappings found in {markdown_file}")
            return 0
    
    def inject_domain_expertise(self, domain_mappings: Dict[str, Dict[str, float]]):
        """
        Inject domain-specific source preferences
        
        Args:
            domain_mappings: Dictionary mapping domains to source preferences
        """
        injected_count = 0
        
        for domain, source_scores in domain_mappings.items():
            # Create multiple synthetic queries for each domain
            domain_queries = self._generate_domain_queries(domain)
            
            for query in domain_queries:
                mappings = [(query, source, score, f"Domain expertise: {domain}") 
                           for source, score in source_scores.items()]
                injected_count += self.inject_query_source_mappings(mappings)
        
        return injected_count
    
    def _generate_domain_queries(self, domain: str) -> List[str]:
        """Generate realistic queries for a domain"""
        query_templates = {
            "psychology": [
                "psychology research data",
                "mental health statistics", 
                "behavioral psychology datasets",
                "cognitive psychology data",
                "psychological research studies"
            ],
            "economics": [
                "economic indicators",
                "gdp data", 
                "financial statistics",
                "economic development data",
                "trade statistics"
            ],
            "singapore_data": [
                "singapore statistics",
                "singapore government data",
                "singapore demographics", 
                "singapore housing data",
                "singapore transport data"
            ],
            "machine_learning": [
                "machine learning datasets",
                "ml training data",
                "artificial intelligence data",
                "deep learning datasets", 
                "neural network training data"
            ],
            "climate": [
                "climate change data",
                "weather statistics",
                "environmental data",
                "temperature records",
                "climate indicators"
            ]
        }
        
        return query_templates.get(domain, [domain])


# Quick injection functions
def inject_from_file(markdown_file: str = "training_mappings.md"):
    """Quick function to inject from markdown file"""
    injector = TrainingDataInjector()
    return injector.inject_from_markdown_file(markdown_file)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Inject from markdown file
    count = inject_from_file()
    print(f"✅ Injected {count} training mappings!")