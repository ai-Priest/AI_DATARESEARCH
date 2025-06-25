"""
Simple CSV-based search fallback when multimodal search is not available
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from .url_validator import url_validator

logger = logging.getLogger(__name__)

class SimpleSearchEngine:
    """Simple search engine that reads directly from CSV files"""
    
    def __init__(self):
        self.datasets = None
        self.load_datasets()
    
    def load_datasets(self):
        """Load datasets from CSV files and Singapore metadata JSON"""
        try:
            all_datasets = []
            
            # Load Singapore datasets CSV
            singapore_csv_path = Path("data/processed/singapore_datasets.csv")
            if singapore_csv_path.exists():
                singapore_df = pd.read_csv(singapore_csv_path)
                all_datasets.append(singapore_df)
                logger.info(f"✅ Loaded {len(singapore_df)} Singapore datasets from CSV")
            else:
                logger.warning("⚠️ Singapore datasets CSV not found")
            
            # Load global datasets CSV  
            global_csv_path = Path("data/processed/global_datasets.csv")
            if global_csv_path.exists():
                global_df = pd.read_csv(global_csv_path)
                all_datasets.append(global_df)
                logger.info(f"✅ Loaded {len(global_df)} global datasets from CSV")
            else:
                logger.warning("⚠️ Global datasets CSV not found")
            
            # Combine all CSV datasets
            if all_datasets:
                self.datasets = pd.concat(all_datasets, ignore_index=True)
                logger.info(f"📊 Combined CSV datasets: {len(self.datasets)} total")
            else:
                self.datasets = pd.DataFrame()
                logger.error("❌ No CSV datasets found")
            
            # Also load Singapore-specific metadata JSON
            metadata_path = Path("models/dl/singapore_dataset_metadata.json")
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        singapore_metadata = json.load(f)
                    
                    # Convert to DataFrame
                    singapore_df = pd.DataFrame(singapore_metadata['datasets'])
                    
                    if not singapore_df.empty:
                        # Add missing columns to match main dataset structure
                        for col in self.datasets.columns:
                            if col not in singapore_df.columns:
                                if col == 'quality_score':
                                    singapore_df[col] = 0.8  # Default good quality
                                elif col == 'url':
                                    singapore_df[col] = singapore_df.get('source', 'https://data.gov.sg') + '/datasets/' + singapore_df['dataset_id']
                                else:
                                    singapore_df[col] = 'Unknown'
                        
                        # Add Singapore datasets to the main dataframe
                        existing_ids = set(self.datasets['dataset_id'].values) if not self.datasets.empty else set()
                        new_datasets = singapore_df[~singapore_df['dataset_id'].isin(existing_ids)]
                        
                        if len(new_datasets) > 0:
                            self.datasets = pd.concat([self.datasets, new_datasets], ignore_index=True)
                            logger.info(f"✅ Added {len(new_datasets)} Singapore-specific datasets from metadata")
                        
                except Exception as e:
                    logger.warning(f"Failed to load Singapore metadata: {e}")
            
            if self.datasets.empty:
                logger.error("❌ No datasets loaded")
            else:
                logger.info(f"📊 Total datasets available: {len(self.datasets)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            self.datasets = pd.DataFrame()
    
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        if self.datasets is None or self.datasets.empty:
            return []
        
        try:
            # Convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            
            # Singapore-specific query expansion
            query_expanded = self.expand_singapore_query(query_lower)
            
            # Score datasets based on text matches
            scores = []
            for idx, row in self.datasets.iterrows():
                score = self.calculate_relevance_score(row, query_expanded, query_lower)
                if score > 0:
                    scores.append((idx, score))
            
            # Sort by score and get top results
            scores.sort(key=lambda x: x[1], reverse=True)
            top_scores = scores[:top_k]
            
            # Format results
            results = []
            for idx, score in top_scores:
                row = self.datasets.iloc[idx]
                result = self.format_search_result(row, score, query)
                results.append(result)
            
            logger.info(f"🔍 Simple search found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def expand_singapore_query(self, query: str) -> str:
        """Expand Singapore-specific abbreviations"""
        expansions = {
            'hdb': 'hdb housing development board flat public housing resale bto',
            'mrt': 'mrt mass rapid transit train subway transport station',
            'cpf': 'cpf central provident fund retirement savings',
            'coe': 'coe certificate of entitlement vehicle car',
            'lta': 'lta land transport authority traffic transport',
            'ura': 'ura urban redevelopment authority planning',
            'bto': 'bto build to order hdb flat housing',
            'resale': 'resale hdb flat property housing',
            'transport': 'transport transportation traffic mrt bus taxi lta'
        }
        
        expanded = query
        for abbr, expansion in expansions.items():
            if abbr in query:
                expanded += f" {expansion}"
        
        return expanded
    
    def calculate_relevance_score(self, row: pd.Series, query_expanded: str, query_original: str) -> float:
        """Calculate relevance score for a dataset"""
        score = 0.0
        
        # Get text fields
        title = str(row.get('title', '')).lower()
        description = str(row.get('description', '')).lower()
        category = str(row.get('category', '')).lower()
        agency = str(row.get('agency', '')).lower()
        tags = str(row.get('tags', '')).lower()
        
        # Combine all searchable text
        searchable_text = f"{title} {description} {category} {agency} {tags}"
        
        # Query terms (both original and expanded)
        query_terms = set(query_original.split() + query_expanded.split())
        
        # Score based on matches
        for term in query_terms:
            if len(term) < 2:  # Skip very short terms
                continue
                
            # Title matches (highest weight)
            if term in title:
                score += 3.0
            
            # Description matches
            if term in description:
                score += 1.5
            
            # Category matches
            if term in category:
                score += 2.0
            
            # Agency matches
            if term in agency:
                score += 1.0
            
            # Tags matches
            if term in tags:
                score += 1.0
        
        # Quality boost
        try:
            quality_score = float(row.get('quality_score', 0.0))
            if quality_score > 0:
                score += quality_score * 0.5
        except (ValueError, TypeError):
            # Handle non-numeric quality scores
            pass
        
        # Transport-specific boosts
        if 'transport' in query_original or 'mrt' in query_original:
            if 'transport' in category or 'lta' in agency.lower():
                score += 2.0
        
        # Housing-specific boosts
        if 'housing' in query_original or 'hdb' in query_original:
            if 'housing' in category or 'hdb' in title:
                score += 2.0
        
        return score
    
    def safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            return float(value) if not pd.isna(value) else default
        except (ValueError, TypeError):
            return default
    
    def format_search_result(self, row: pd.Series, score: float, query: str) -> Dict[str, Any]:
        """Format a search result"""
        # Handle potential NaN values
        def safe_get(field, default=''):
            value = row.get(field, default)
            if pd.isna(value):
                return default
            return str(value)
        
        # Ensure we have valid titles and descriptions
        title = safe_get('title', 'Dataset')
        if not title or title == 'Dataset':
            # Try to generate a better title from description or other fields
            description = safe_get('description', '')
            if description:
                # Extract first meaningful part of description
                desc_words = description.split()[:5]
                title = ' '.join(desc_words) + "..."
            else:
                title = f"{safe_get('agency', 'Unknown')} Dataset"
        
        description = safe_get('description', 'No description available')
        
        # Correct the URL using our validator
        original_url = safe_get('url')
        dataset_id = safe_get('dataset_id')
        corrected_url = url_validator.correct_url(dataset_id, original_url, title)
        
        return {
            'dataset_id': dataset_id,
            'title': title,
            'description': description,
            'source': safe_get('source'),
            'agency': safe_get('agency'),
            'category': safe_get('category'),
            'quality_score': self.safe_float(row.get('quality_score', 0.0)),
            'last_updated': safe_get('last_updated'),
            'format': safe_get('format'),
            'url': corrected_url,
            'multimodal_score': min(1.0, score / 10.0),  # Normalize score to 0-1
            'score_breakdown': {
                'simple_search': score
            },
            'search_metadata': {
                'query': query,
                'search_mode': 'simple_csv',
                'engine': 'SimpleSearchEngine'
            }
        }

def create_simple_search_engine():
    """Factory function to create simple search engine"""
    return SimpleSearchEngine()