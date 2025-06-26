# ML Preprocessing Module - Advanced Data Preparation for ML Training
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logger = logging.getLogger(__name__)


class MLDataPreprocessor:
    """
    Advanced data preprocessing for ML training with feature engineering,
    quality filtering, and optimization techniques.
    """
    
    def __init__(self, config: Dict):
        """Initialize preprocessor with configuration"""
        self.config = config
        self.data_config = config.get('data_processing', {})
        self.quality_filters = self.data_config.get('quality_filters', {})
        self.feature_config = self.data_config.get('feature_engineering', {})
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Feature weights for text combination
        self.field_weights = self.feature_config.get('field_weights', {
            'title': 0.3,
            'description': 0.4, 
            'tags': 0.2,
            'category': 0.1
        })
        
        logger.info("🔄 MLDataPreprocessor initialized")
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load and validate input datasets"""
        try:
            input_paths = self.data_config.get('input_paths', {})
            
            # Load Singapore datasets
            sg_path = input_paths.get('singapore_datasets', 'data/processed/singapore_datasets.csv')
            sg_df = pd.read_csv(sg_path)
            logger.info(f"📊 Loaded {len(sg_df)} Singapore datasets")
            
            # Load global datasets  
            global_path = input_paths.get('global_datasets', 'data/processed/global_datasets.csv')
            global_df = pd.read_csv(global_path) if Path(global_path).exists() else pd.DataFrame()
            
            if not global_df.empty:
                logger.info(f"🌍 Loaded {len(global_df)} global datasets")
            
            # Load ground truth
            gt_path = input_paths.get('ground_truth', 'data/processed/intelligent_ground_truth.json')
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
            logger.info(f"🎯 Loaded {len(ground_truth)} ground truth scenarios")
            
            return sg_df, global_df, ground_truth
            
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            raise
    
    def combine_datasets(self, sg_df: pd.DataFrame, global_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and standardize datasets from multiple sources"""
        try:
            datasets = [sg_df]
            
            if not global_df.empty:
                # Standardize global dataset columns to match Singapore format
                global_df = self._standardize_columns(global_df)
                datasets.append(global_df)
            
            # Combine datasets
            combined_df = pd.concat(datasets, ignore_index=True)
            
            # Add source indicators
            combined_df['is_singapore'] = combined_df['source'].str.contains('data.gov.sg|LTA|OneMap|SingStat', na=False)
            combined_df['is_global'] = ~combined_df['is_singapore']
            
            logger.info(f"✅ Combined {len(combined_df)} total datasets")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Failed to combine datasets: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and formats across different sources"""
        try:
            # Ensure required columns exist
            required_columns = ['dataset_id', 'title', 'description', 'source', 'category']
            
            for col in required_columns:
                if col not in df.columns:
                    if col == 'category':
                        df[col] = 'general'  # Default category
                    elif col == 'dataset_id':
                        df[col] = df.index.astype(str)  # Generate IDs
                    else:
                        df[col] = ''  # Empty string for text fields
            
            # Standardize data types
            df['quality_score'] = pd.to_numeric(df.get('quality_score', 0.5), errors='coerce').fillna(0.5)
            df['title'] = df['title'].astype(str).fillna('')
            df['description'] = df['description'].astype(str).fillna('')
            df['tags'] = df['tags'].astype(str).fillna('')
            df['category'] = df['category'].astype(str).fillna('general')
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Column standardization failed: {e}")
            return df
    
    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to remove low-quality datasets"""
        try:
            original_count = len(df)
            
            # Quality score filter
            min_quality = self.quality_filters.get('min_quality_score', 0.3)
            df = df[df['quality_score'] >= min_quality]
            
            # Required fields filter
            if self.quality_filters.get('require_title', True):
                df = df[df['title'].str.len() > 0]
            
            if self.quality_filters.get('require_description', True):
                min_desc_len = self.quality_filters.get('min_description_length', 10)
                df = df[df['description'].str.len() >= min_desc_len]
            
            # Title length filter
            max_title_len = self.quality_filters.get('max_title_length', 200)
            df = df[df['title'].str.len() <= max_title_len]
            
            filtered_count = len(df)
            removed_count = original_count - filtered_count
            
            logger.info(f"🔍 Quality filters: {removed_count} datasets removed, {filtered_count} retained")
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"❌ Quality filtering failed: {e}")
            return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for ML training"""
        try:
            # 1. Combined text features with weighted importance
            df = self._create_combined_text(df)
            
            # 2. Text quality features
            df = self._extract_text_features(df)
            
            # 3. Metadata features
            df = self._extract_metadata_features(df)
            
            # 4. Source credibility features
            df = self._extract_credibility_features(df)
            
            # 5. Temporal features
            df = self._extract_temporal_features(df)
            
            # 6. Categorical encoding
            df = self._encode_categorical_features(df)
            
            logger.info("✅ Feature engineering completed")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return df
    
    def _create_combined_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create semantically weighted combined text for ML training"""
        try:
            combine_fields = self.feature_config.get('combine_fields', ['title', 'description', 'tags', 'category'])
            
            # Ensure all fields exist
            for field in combine_fields:
                if field not in df.columns:
                    df[field] = ''
            
            # Create semantically weighted combination without repetition
            combined_texts = []
            
            for idx, row in df.iterrows():
                text_parts = []
                
                # Title gets highest priority - clean and enhance
                title = str(row.get('title', '')).strip()
                if title and title != 'nan':
                    # Clean title and extract key terms
                    clean_title = self._clean_and_enhance_text(title, is_title=True)
                    text_parts.append(clean_title)
                
                # Description gets second priority
                description = str(row.get('description', '')).strip()
                if description and description != 'nan' and len(description) > 10:
                    clean_desc = self._clean_and_enhance_text(description, is_description=True)
                    text_parts.append(clean_desc)
                
                # Tags and category as supplementary
                tags = str(row.get('tags', '')).strip()
                if tags and tags != 'nan':
                    clean_tags = self._clean_and_enhance_text(tags, is_tags=True)
                    text_parts.append(clean_tags)
                
                category = str(row.get('category', '')).strip()
                if category and category != 'nan':
                    clean_category = self._clean_and_enhance_text(category, is_category=True)
                    text_parts.append(clean_category)
                
                # Combine with semantic structure
                combined = ' '.join(text_parts).strip()
                combined = self._apply_semantic_normalization(combined)
                combined_texts.append(combined)
            
            df['combined_text'] = combined_texts
            
            # Alternative: Simple concatenation (fallback)
            if df['combined_text'].str.len().mean() < 10:
                df['combined_text'] = (
                    df['title'].fillna('') + ' ' + 
                    df['description'].fillna('') + ' ' + 
                    df['tags'].fillna('') + ' ' + 
                    df['category'].fillna('')
                ).str.strip()
            
            logger.info(f"📝 Combined text created, average length: {df['combined_text'].str.len().mean():.1f} chars")
            return df
            
        except Exception as e:
            logger.error(f"❌ Combined text creation failed: {e}")
            # Fallback to simple concatenation
            df['combined_text'] = (
                df['title'].fillna('') + ' ' + 
                df['description'].fillna('')
            ).str.strip()
            return df
    
    def _clean_and_enhance_text(self, text: str, is_title=False, is_description=False, is_tags=False, is_category=False) -> str:
        """Clean and enhance text for better semantic matching"""
        import re
        
        if not text or text == 'nan':
            return ''
        
        # Basic cleaning
        cleaned = text.strip().lower()
        
        # Remove special characters but keep meaningful punctuation
        cleaned = re.sub(r'[^\w\s\-\(\)\/]', ' ', cleaned)
        
        # Normalize common abbreviations and terms for better semantic matching
        replacements = {
            # Currency and economic terms
            'sgd': 'singapore dollars',
            'usd': 'us dollars',
            'ppp': 'purchasing power parity',
            'gdp': 'gross domestic product',
            'gni': 'gross national income',
            'bop': 'balance of payments',
            
            # Singapore agencies and organizations
            'ura': 'urban redevelopment authority',
            'lta': 'land transport authority',
            'hdb': 'housing development board',
            'moh': 'ministry of health',
            'mom': 'ministry of manpower',
            'moe': 'ministry of education',
            'mnd': 'ministry of national development',
            'mti': 'ministry of trade and industry',
            'singstat': 'singapore statistics',
            'nea': 'national environment agency',
            'pub': 'public utilities board',
            
            # Transportation
            'mrt': 'mass rapid transit',
            'lrt': 'light rail transit',
            'bus': 'public bus transport',
            'taxi': 'taxi services',
            
            # Environmental and health
            'co2': 'carbon dioxide emissions',
            'pm25': 'particulate matter air pollution',
            'pm10': 'particulate matter air pollution',
            'no2': 'nitrogen dioxide pollution',
            'o3': 'ozone pollution',
            
            # International organizations
            'wb': 'world bank',
            'who': 'world health organization',
            'oecd': 'organisation economic cooperation development',
            'imf': 'international monetary fund',
            'unesco': 'united nations educational scientific cultural organization',
            'undp': 'united nations development programme',
            
            # Technology and data terms
            'api': 'application programming interface',
            'iot': 'internet of things',
            'gis': 'geographic information system',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            
            # Urban planning and housing
            'bto': 'build to order housing',
            'ec': 'executive condominium',
            'private': 'private housing',
            'public': 'public housing',
            'resale': 'resale housing market',
            
            # Demographic terms
            'elderly': 'senior citizens aging population',
            'youth': 'young people demographics',
            'workforce': 'employment labor market',
            'immigration': 'foreign workers migration'
        }
        
        for abbrev, full_form in replacements.items():
            cleaned = re.sub(r'\b' + abbrev + r'\b', full_form, cleaned)
        
        # Extract semantic keywords based on context
        if is_title:
            # For titles, preserve key domain terms
            cleaned = self._enhance_title_semantics(cleaned)
        elif is_description:
            # For descriptions, extract key concepts
            cleaned = self._extract_key_concepts(cleaned)
        elif is_tags:
            # For tags, normalize and expand
            cleaned = self._normalize_tags(cleaned)
        elif is_category:
            # For categories, expand semantic meaning
            cleaned = self._expand_category_semantics(cleaned)
        
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _enhance_title_semantics(self, title: str) -> str:
        """Enhance title with semantic context"""
        # Add context for common patterns
        if 'population' in title:
            title += ' demographic statistics'
        if 'gdp' in title or 'gross domestic product' in title:
            title += ' economic indicators'
        if 'poverty' in title:
            title += ' social economic indicators'
        if 'mortality' in title:
            title += ' health statistics'
        if 'transport' in title or 'traffic' in title:
            title += ' infrastructure data'
        if 'housing' in title or 'property' in title:
            title += ' real estate market'
        if 'employment' in title or 'labor' in title:
            title += ' workforce statistics'
        
        return title
    
    def _extract_key_concepts(self, description: str) -> str:
        """Extract key concepts from description"""
        import re

        # Extract sentences that contain key indicators
        sentences = re.split(r'[.!?]+', description)
        key_concepts = []
        
        concept_patterns = [
            r'\b(measures?|calculates?|shows?|indicates?|represents?)\s+\w+',
            r'\b(data|statistics|information)\s+(?:on|about|for)\s+\w+',
            r'\b(percentage|proportion|rate|ratio|index)\s+of\s+\w+',
            r'\b(annual|monthly|quarterly|daily)\s+\w+',
            r'\b(total|average|median|maximum|minimum)\s+\w+'
        ]
        
        for sentence in sentences:
            for pattern in concept_patterns:
                matches = re.findall(pattern, sentence.lower())
                if matches:
                    key_concepts.extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        # If no concepts found, take first meaningful sentence
        if not key_concepts and sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                key_concepts.append(first_sentence[:100])  # First 100 chars
        
        return ' '.join(key_concepts[:3])  # Top 3 concepts
    
    def _normalize_tags(self, tags: str) -> str:
        """Normalize and expand tag meanings"""
        # Split tags and clean each
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Expand common tag abbreviations
        expanded_tags = []
        for tag in tag_list:
            if tag in ['wb', 'world bank']:
                expanded_tags.append('development indicators global data')
            elif tag in ['singapore', 'sg']:
                expanded_tags.append('singapore government data')
            elif tag in ['econ', 'economic']:
                expanded_tags.append('economic development indicators')
            elif tag in ['stats', 'statistics']:
                expanded_tags.append('statistical data analysis')
            else:
                expanded_tags.append(tag)
        
        return ' '.join(expanded_tags)
    
    def _expand_category_semantics(self, category: str) -> str:
        """Expand category with semantic meaning"""
        category_mappings = {
            'economic_development': 'economic growth development indicators financial statistics',
            'economic_finance': 'financial economic monetary banking statistics',
            'transport': 'transportation infrastructure mobility traffic data',
            'housing': 'residential property real estate housing market',
            'health': 'healthcare medical health outcomes statistics',
            'education': 'educational academic learning statistics',
            'environment': 'environmental climate sustainability data',
            'demographics': 'population demographic social statistics',
            'general': 'general public government data'
        }
        
        return category_mappings.get(category, category)
    
    def _apply_semantic_normalization(self, text: str) -> str:
        """Apply final semantic normalization"""
        import re

        # Remove redundant words
        text = re.sub(r'\b(data|dataset|statistics|information)\s+\1\b', r'\1', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure minimum meaningful length
        if len(text) < 10:
            text += ' statistical dataset'
        
        return text
    
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract quantitative features from text fields"""
        try:
            # Title features
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
            df['title_has_numbers'] = df['title'].str.contains(r'\d', na=False).astype(int)
            
            # Description features  
            df['description_length'] = df['description'].str.len()
            df['description_word_count'] = df['description'].str.split().str.len()
            df['description_sentence_count'] = df['description'].str.count(r'[.!?]')
            
            # Tags features
            df['tag_count'] = df['tags'].str.count(',') + 1
            df['tag_count'] = df['tag_count'].where(df['tags'].str.len() > 0, 0)
            
            # Combined text features
            df['combined_text_length'] = df['combined_text'].str.len()
            df['combined_text_word_count'] = df['combined_text'].str.split().str.len()
            
            # Text quality indicators
            df['text_quality_score'] = (
                (df['title_length'] / 100).clip(0, 1) * 0.3 +
                (df['description_length'] / 500).clip(0, 1) * 0.4 +
                (df['tag_count'] / 5).clip(0, 1) * 0.3
            )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Text feature extraction failed: {e}")
            return df
    
    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from metadata fields"""
        try:
            # Completeness score
            required_fields = ['title', 'description', 'source', 'category']
            df['metadata_completeness'] = sum(
                df[field].notna() & (df[field] != '') for field in required_fields if field in df.columns
            ) / len(required_fields)
            
            # Quality score normalization
            if 'quality_score' in df.columns:
                df['quality_score_normalized'] = df['quality_score'].fillna(0.5)
            else:
                df['quality_score_normalized'] = 0.5
            
            # Geographic coverage features
            if 'geographic_coverage' in df.columns:
                df['is_singapore_focused'] = df['geographic_coverage'].str.contains('Singapore', na=False).astype(int)
                df['is_global_coverage'] = df['geographic_coverage'].str.contains('Global|World', na=False).astype(int)
            else:
                df['is_singapore_focused'] = df['is_singapore'].astype(int) if 'is_singapore' in df.columns else 0
                df['is_global_coverage'] = df['is_global'].astype(int) if 'is_global' in df.columns else 0
            
            # Format features
            if 'format' in df.columns:
                df['is_csv_format'] = df['format'].str.contains('CSV', na=False).astype(int)
                df['is_json_format'] = df['format'].str.contains('JSON', na=False).astype(int)
                df['is_api_format'] = df['format'].str.contains('API', na=False).astype(int)
            else:
                df['is_csv_format'] = 0
                df['is_json_format'] = 0
                df['is_api_format'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Metadata feature extraction failed: {e}")
            return df
    
    def _extract_credibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract source credibility and reliability features"""
        try:
            # Government source indicator
            gov_patterns = r'data\.gov\.sg|gov\.sg|government|ministry|authority|board|agency'
            df['is_government_source'] = df['source'].str.contains(gov_patterns, case=False, na=False).astype(int)
            
            # International organization indicator
            intl_patterns = r'world bank|united nations|un data|who|imf|oecd'
            df['is_international_org'] = df['source'].str.contains(intl_patterns, case=False, na=False).astype(int)
            
            # Academic source indicator
            academic_patterns = r'university|institute|research|academic'
            df['is_academic_source'] = df['source'].str.contains(academic_patterns, case=False, na=False).astype(int)
            
            # Source credibility score
            df['source_credibility_score'] = (
                df['is_government_source'] * 0.9 +
                df['is_international_org'] * 0.8 +
                df['is_academic_source'] * 0.7
            ).clip(0, 1)
            
            # Default credibility for other sources
            df['source_credibility_score'] = df['source_credibility_score'].where(
                df['source_credibility_score'] > 0, 0.5
            )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Credibility feature extraction failed: {e}")
            return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and update frequency features"""
        try:
            # Update frequency encoding
            if 'frequency' in df.columns:
                frequency_scores = {
                    'real-time': 1.0, 'daily': 0.9, 'weekly': 0.7, 
                    'monthly': 0.5, 'quarterly': 0.3, 'annual': 0.2, 'unknown': 0.1
                }
                
                df['update_frequency_score'] = df['frequency'].str.lower().map(frequency_scores).fillna(0.1)
            else:
                df['update_frequency_score'] = 0.5
            
            # Recency score (if last_updated available)
            if 'last_updated' in df.columns:
                try:
                    df['last_updated_date'] = pd.to_datetime(df['last_updated'], errors='coerce')
                    current_date = pd.Timestamp.now()
                    days_since_update = (current_date - df['last_updated_date']).dt.days
                    
                    # Recency score: 1.0 for recent data, decreasing over time
                    df['recency_score'] = np.exp(-days_since_update / 365).clip(0, 1)  # Exponential decay over 1 year
                except:
                    df['recency_score'] = 0.5
            else:
                df['recency_score'] = 0.5
            
            # Temporal coverage features
            if 'coverage_start' in df.columns and 'coverage_end' in df.columns:
                try:
                    df['coverage_start_date'] = pd.to_datetime(df['coverage_start'], errors='coerce')
                    df['coverage_end_date'] = pd.to_datetime(df['coverage_end'], errors='coerce')
                    df['temporal_span_years'] = (
                        (df['coverage_end_date'] - df['coverage_start_date']).dt.days / 365
                    ).fillna(0)
                except:
                    df['temporal_span_years'] = 0
            else:
                df['temporal_span_years'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Temporal feature extraction failed: {e}")
            return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML algorithms"""
        try:
            categorical_features = ['category', 'source', 'agency']
            
            for feature in categorical_features:
                if feature in df.columns:
                    # Clean and standardize categories
                    df[feature] = df[feature].fillna('unknown').astype(str).str.lower().str.strip()
                    
                    # Label encoding for categories with many unique values
                    if df[feature].nunique() > 10:
                        le = LabelEncoder()
                        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
                        self.label_encoders[feature] = le
                    
                    # One-hot encoding for categories with few unique values
                    else:
                        dummies = pd.get_dummies(df[feature], prefix=f'{feature}')
                        df = pd.concat([df, dummies], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Categorical encoding failed: {e}")
            return df
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final ML-ready feature matrix"""
        try:
            # Select numerical features for ML algorithms
            ml_features = [
                # Text features
                'title_length', 'title_word_count', 'title_has_numbers',
                'description_length', 'description_word_count', 'description_sentence_count',
                'tag_count', 'combined_text_length', 'combined_text_word_count',
                'text_quality_score',
                
                # Metadata features  
                'metadata_completeness', 'quality_score_normalized',
                'is_singapore_focused', 'is_global_coverage',
                'is_csv_format', 'is_json_format', 'is_api_format',
                
                # Credibility features
                'is_government_source', 'is_international_org', 'is_academic_source',
                'source_credibility_score',
                
                # Temporal features
                'update_frequency_score', 'recency_score', 'temporal_span_years'
            ]
            
            # Filter features that actually exist in the dataframe
            available_features = [f for f in ml_features if f in df.columns]
            
            if not available_features:
                logger.warning("⚠️ No ML features found, using basic features")
                available_features = ['quality_score_normalized', 'text_quality_score']
                
                # Create basic features if they don't exist
                if 'quality_score_normalized' not in df.columns:
                    df['quality_score_normalized'] = df.get('quality_score', 0.5)
                if 'text_quality_score' not in df.columns:
                    df['text_quality_score'] = 0.5
            
            # Create feature matrix
            feature_matrix = df[available_features].fillna(0)
            
            # Normalize features
            if self.data_config.get('normalize_features', True):
                feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
                feature_df = pd.DataFrame(
                    feature_matrix_scaled, 
                    columns=available_features,
                    index=df.index
                )
            else:
                feature_df = feature_matrix
            
            # Add feature matrix to original dataframe
            df = pd.concat([df, feature_df.add_suffix('_normalized')], axis=1)
            
            logger.info(f"✅ Created {len(available_features)} ML features")
            return df
            
        except Exception as e:
            logger.error(f"❌ ML feature creation failed: {e}")
            return df
    
    def validate_preprocessing(self, df: pd.DataFrame) -> Dict:
        """Validate preprocessing results and generate summary"""
        try:
            validation_results = {
                'dataset_count': len(df),
                'required_fields_present': all(
                    col in df.columns for col in ['dataset_id', 'title', 'description', 'combined_text']
                ),
                'average_quality_score': df.get('quality_score_normalized', pd.Series([0.5])).mean(),
                'text_coverage': {
                    'has_title': (df['title'].str.len() > 0).sum(),
                    'has_description': (df['description'].str.len() > 0).sum(),
                    'has_combined_text': (df['combined_text'].str.len() > 0).sum()
                },
                'feature_summary': {
                    'text_features': len([c for c in df.columns if 'length' in c or 'count' in c]),
                    'categorical_features': len([c for c in df.columns if 'encoded' in c]),
                    'quality_features': len([c for c in df.columns if 'score' in c]),
                    'total_features': len(df.columns)
                },
                'data_quality': {
                    'missing_values': df.isnull().sum().sum(),
                    'duplicate_titles': df['title'].duplicated().sum() if 'title' in df.columns else 0,
                    'empty_descriptions': (df['description'].str.len() == 0).sum() if 'description' in df.columns else 0
                }
            }
            
            # Quality assessment
            validation_results['quality_assessment'] = 'PASS' if (
                validation_results['dataset_count'] >= 10 and
                validation_results['required_fields_present'] and
                validation_results['average_quality_score'] >= 0.3
            ) else 'FAIL'
            
            logger.info(f"✅ Preprocessing validation: {validation_results['quality_assessment']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Preprocessing validation failed: {e}")
            return {'quality_assessment': 'FAIL', 'error': str(e)}
    
    def process_complete_pipeline(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Execute complete preprocessing pipeline"""
        try:
            logger.info("🚀 Starting complete preprocessing pipeline")
            
            # Step 1: Load datasets
            sg_df, global_df, ground_truth = self.load_datasets()
            
            # Step 2: Combine datasets
            combined_df = self.combine_datasets(sg_df, global_df)
            
            # Step 3: Apply quality filters
            filtered_df = self.apply_quality_filters(combined_df)
            
            # Step 4: Engineer features
            featured_df = self.engineer_features(filtered_df)
            
            # Step 5: Create ML features
            final_df = self.create_ml_features(featured_df)
            
            # Step 6: Validate results
            validation_results = self.validate_preprocessing(final_df)
            
            logger.info("🎉 Complete preprocessing pipeline finished successfully")
            
            return final_df, ground_truth, validation_results
            
        except Exception as e:
            logger.error(f"❌ Complete preprocessing pipeline failed: {e}")
            raise


def create_preprocessor(config: Dict) -> MLDataPreprocessor:
    """Factory function to create preprocessor with configuration"""
    return MLDataPreprocessor(config)