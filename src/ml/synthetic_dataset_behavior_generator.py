"""
Global Synthetic Dataset Discovery User Behavior Generator
Creates realistic user behavior data for universal ML evaluation of dataset recommendation systems.
Generates domain-agnostic evaluation scenarios applicable to any regional datasets.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SyntheticDatasetBehaviorGenerator:
    """Generates realistic user behavior for global dataset discovery evaluation."""
    
    def __init__(self, datasets_df: pd.DataFrame):
        self.datasets_df = datasets_df
        # Global dataset discovery terms (not region-specific)
        self.global_terms = [
            'data', 'dataset', 'statistics', 'research', 'analysis', 'metrics',
            'government', 'public', 'open data', 'census', 'survey', 'indicators'
        ]
        
        # Global dataset discovery search patterns
        self.search_patterns = {
            'housing': ['housing prices', 'real estate market', 'property data', 'rental costs', 'home sales'],
            'transport': ['transportation data', 'traffic patterns', 'public transit', 'mobility statistics', 'vehicle data'],
            'economics': ['economic indicators', 'gdp data', 'employment statistics', 'trade data', 'inflation rates'],
            'demographics': ['population data', 'demographic statistics', 'census data', 'age distribution', 'migration patterns'],
            'education': ['education statistics', 'school data', 'student performance', 'enrollment data', 'academic metrics'],
            'health': ['health statistics', 'medical data', 'disease surveillance', 'healthcare metrics', 'public health'],
            'environment': ['environmental data', 'climate statistics', 'air quality', 'pollution data', 'sustainability metrics'],
            'urban': ['urban planning', 'city development', 'land use data', 'zoning information', 'municipal data'],
            'finance': ['financial data', 'budget statistics', 'revenue data', 'expenditure analysis', 'fiscal indicators'],
            'technology': ['technology adoption', 'digital infrastructure', 'innovation metrics', 'tech statistics', 'connectivity data']
        }
        
        # Global user personas for dataset discovery
        self.user_personas = [
            {'id': 'researcher_001', 'type': 'academic_researcher', 'expertise': 'high', 'domain': 'urban_planning'},
            {'id': 'analyst_002', 'type': 'government_analyst', 'expertise': 'high', 'domain': 'economics'},
            {'id': 'student_003', 'type': 'graduate_student', 'expertise': 'medium', 'domain': 'demographics'},
            {'id': 'journalist_004', 'type': 'data_journalist', 'expertise': 'medium', 'domain': 'housing'},
            {'id': 'consultant_005', 'type': 'business_consultant', 'expertise': 'medium', 'domain': 'transport'},
            {'id': 'citizen_006', 'type': 'interested_citizen', 'expertise': 'low', 'domain': 'general'},
            {'id': 'developer_007', 'type': 'data_scientist', 'expertise': 'high', 'domain': 'technology'},
            {'id': 'researcher_008', 'type': 'market_researcher', 'expertise': 'high', 'domain': 'economics'},
            {'id': 'environmentalist_009', 'type': 'environmental_researcher', 'expertise': 'high', 'domain': 'environment'},
            {'id': 'health_analyst_010', 'type': 'public_health_analyst', 'expertise': 'high', 'domain': 'health'},
            {'id': 'educator_011', 'type': 'education_researcher', 'expertise': 'medium', 'domain': 'education'},
            {'id': 'policy_maker_012', 'type': 'policy_analyst', 'expertise': 'high', 'domain': 'finance'}
        ]
    
    def generate_realistic_search_query(self, domain: str, expertise: str) -> str:
        """Generate a realistic search query based on domain and expertise."""
        
        base_queries = self.search_patterns.get(domain, ['dataset discovery'])
        query = random.choice(base_queries)
        
        # Add complexity based on expertise
        if expertise == 'high':
            # Experts use more specific terms
            modifiers = ['detailed', 'historical', 'quarterly', 'annual', 'trend analysis', 'longitudinal', 'comprehensive']
            if random.random() < 0.4:
                query = f"{random.choice(modifiers)} {query}"
        elif expertise == 'low':
            # Beginners use simpler terms
            query = query.replace('statistics', 'data').replace('indicators', 'info').replace('metrics', 'numbers')
        
        # Sometimes add generic research context
        research_modifiers = ['research', 'analysis', 'study', 'report', 'information']
        if random.random() < 0.3:
            query = f"{query} {random.choice(research_modifiers)}"
            
        return query
    
    def find_relevant_datasets(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find datasets relevant to the query using simple keyword matching."""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_datasets = []
        
        for _, dataset in self.datasets_df.iterrows():
            # Calculate relevance score
            title = str(dataset.get('title', '')).lower()
            description = str(dataset.get('description', '')).lower()
            category = str(dataset.get('category', '')).lower()
            
            # Count matching words
            title_matches = len(query_words.intersection(set(title.split())))
            desc_matches = len(query_words.intersection(set(description.split())))
            cat_matches = len(query_words.intersection(set(category.split())))
            
            # Weight matches
            relevance_score = (title_matches * 3 + desc_matches * 1 + cat_matches * 2) / len(query_words)
            
            # Add some randomness for realism
            relevance_score += random.uniform(-0.1, 0.1)
            
            if relevance_score > 0:
                scored_datasets.append({
                    'dataset_id': dataset.get('id', f"dataset_{len(scored_datasets)}"),
                    'title': dataset.get('title', 'Unknown'),
                    'description': dataset.get('description', ''),
                    'category': dataset.get('category', 'general'),
                    'relevance_score': max(0, relevance_score),
                    'quality_score': dataset.get('quality_score', 0.7)
                })
        
        # Sort by relevance and return top_k
        scored_datasets.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_datasets[:top_k]
    
    def simulate_user_interactions(self, user: Dict, query: str, recommendations: List[Dict]) -> Dict:
        """Simulate realistic user interactions with recommendations."""
        
        interactions = {
            'clicked_datasets': [],
            'viewed_datasets': [],
            'downloaded_datasets': [],
            'session_duration': 0,
            'refinement_count': 0,
            'success_indicators': {}
        }
        
        expertise = user['expertise']
        
        # Simulate viewing behavior
        view_probability = 0.8 if expertise == 'high' else 0.6 if expertise == 'medium' else 0.4
        
        for i, rec in enumerate(recommendations):
            if random.random() < view_probability * (0.9 ** i):  # Decreasing probability down the list
                interactions['viewed_datasets'].append({
                    'dataset_id': rec['dataset_id'],
                    'title': rec['title'],
                    'position': i + 1,
                    'relevance_score': rec['relevance_score']
                })
                
                # Simulate clicking based on relevance
                click_prob = min(0.9, rec['relevance_score'] * 0.7)
                if expertise == 'high':
                    click_prob *= 1.2  # Experts are better at identifying relevance
                elif expertise == 'low':
                    click_prob *= 0.8  # Beginners are less discerning
                    
                if random.random() < click_prob:
                    interactions['clicked_datasets'].append({
                        'dataset_id': rec['dataset_id'],
                        'title': rec['title'],
                        'position': i + 1,
                        'relevance_score': rec['relevance_score'],
                        'click_timestamp': datetime.now().isoformat()
                    })
                    
                    # High relevance items might be downloaded
                    if rec['relevance_score'] > 0.6 and random.random() < 0.3:
                        interactions['downloaded_datasets'].append({
                            'dataset_id': rec['dataset_id'],
                            'title': rec['title'],
                            'relevance_score': rec['relevance_score']
                        })
        
        # Calculate session metrics
        interactions['session_duration'] = random.uniform(1.5, 8.0)  # minutes
        interactions['refinement_count'] = random.randint(1, 5)
        
        # Success indicators
        interactions['success_indicators'] = {
            'found_relevant': len(interactions['clicked_datasets']) > 0,
            'high_engagement': len(interactions['viewed_datasets']) >= 3,
            'task_completed': len(interactions['downloaded_datasets']) > 0,
            'satisfaction_score': min(1.0, len(interactions['clicked_datasets']) * 0.3 + 
                                    len(interactions['downloaded_datasets']) * 0.5)
        }
        
        return interactions
    
    def generate_user_session(self, user: Dict) -> Dict:
        """Generate a complete user session with realistic search behavior."""
        
        # Generate search query
        query = self.generate_realistic_search_query(user['domain'], user['expertise'])
        
        # Find relevant datasets
        recommendations = self.find_relevant_datasets(query, top_k=8)
        
        # Simulate interactions
        interactions = self.simulate_user_interactions(user, query, recommendations)
        
        # Create session object
        session = {
            'session_id': f"session_{user['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'user_id': user['id'],
            'user_type': user['type'],
            'user_expertise': user['expertise'],
            'search_intent': query,
            'original_query': query,
            'recommendations': recommendations,
            'user_interactions': interactions,
            'timestamp': datetime.now().isoformat(),
            'domain': user['domain']
        }
        
        return session
    
    def generate_evaluation_dataset(self, num_sessions: int = 50) -> Dict:
        """Generate a complete evaluation dataset with multiple user sessions."""
        
        logger.info(f"🎯 Generating {num_sessions} synthetic dataset discovery sessions...")
        
        sessions = []
        user_behavior_metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'average_satisfaction': 0,
            'domain_distribution': {},
            'expertise_distribution': {}
        }
        
        for i in range(num_sessions):
            # Select random user persona
            user = random.choice(self.user_personas)
            
            # Generate session
            session = self.generate_user_session(user)
            sessions.append(session)
            
            # Update metrics
            user_behavior_metrics['total_sessions'] += 1
            if session['user_interactions']['success_indicators']['found_relevant']:
                user_behavior_metrics['successful_sessions'] += 1
            
            # Track distributions
            domain = user['domain']
            expertise = user['expertise']
            user_behavior_metrics['domain_distribution'][domain] = \
                user_behavior_metrics['domain_distribution'].get(domain, 0) + 1
            user_behavior_metrics['expertise_distribution'][expertise] = \
                user_behavior_metrics['expertise_distribution'].get(expertise, 0) + 1
        
        # Calculate final metrics
        user_behavior_metrics['success_rate'] = \
            user_behavior_metrics['successful_sessions'] / user_behavior_metrics['total_sessions']
        user_behavior_metrics['average_satisfaction'] = \
            sum(s['user_interactions']['success_indicators']['satisfaction_score'] for s in sessions) / len(sessions)
        
        evaluation_dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_sessions': num_sessions,
                'generator_version': '2.0',
                'domain': 'global_dataset_discovery',
                'description': 'Global dataset discovery evaluation with diverse user personas and domain scenarios'
            },
            'sessions': sessions,
            'aggregated_metrics': user_behavior_metrics
        }
        
        logger.info(f"✅ Generated dataset with {num_sessions} sessions")
        logger.info(f"📊 Success rate: {user_behavior_metrics['success_rate']:.1%}")
        logger.info(f"📈 Average satisfaction: {user_behavior_metrics['average_satisfaction']:.2f}")
        
        return evaluation_dataset

def create_synthetic_evaluation_data():
    """Create synthetic evaluation data for dataset discovery."""
    
    # Load current datasets
    try:
        datasets_df = pd.read_csv("data/processed/singapore_datasets.csv")
        logger.info(f"📊 Loaded {len(datasets_df)} datasets for synthesis")
    except FileNotFoundError:
        # Create minimal dataset if file not found
        datasets_df = pd.DataFrame({
            'id': [f'dataset_{i}' for i in range(20)],
            'title': [f'Singapore Dataset {i}' for i in range(20)],
            'description': [f'Government dataset about various topics {i}' for i in range(20)],
            'category': ['general'] * 20,
            'quality_score': [0.7] * 20
        })
        logger.warning("Using minimal synthetic datasets")
    
    # Generate evaluation data
    generator = SyntheticDatasetBehaviorGenerator(datasets_df)
    evaluation_data = generator.generate_evaluation_dataset(num_sessions=100)
    
    # Save to file
    output_path = "data/processed/synthetic_dataset_discovery_behavior.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    logger.info(f"💾 Saved synthetic evaluation data to: {output_path}")
    
    return evaluation_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_synthetic_evaluation_data()