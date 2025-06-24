"""
Dataset Preview Card Generator
Creates rich preview cards with metadata, statistics, and integration guidance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import re
import json
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class DatasetPreviewGenerator:
    """Generates comprehensive preview cards for datasets."""
    
    def __init__(self):
        self.preview_templates = {
            'government': {
                'style': 'official',
                'color_scheme': 'blue',
                'trust_level': 'high',
                'integration_complexity': 'medium'
            },
            'commercial': {
                'style': 'business',
                'color_scheme': 'green', 
                'trust_level': 'medium',
                'integration_complexity': 'low'
            },
            'academic': {
                'style': 'research',
                'color_scheme': 'purple',
                'trust_level': 'high',
                'integration_complexity': 'high'
            }
        }
        
        self.singapore_agencies = {
            'hdb': 'Housing Development Board',
            'lta': 'Land Transport Authority',
            'ura': 'Urban Redevelopment Authority',
            'nea': 'National Environment Agency',
            'mom': 'Ministry of Manpower',
            'moh': 'Ministry of Health',
            'moe': 'Ministry of Education'
        }
    
    def generate_preview_card(self, dataset: Dict, similarity_score: float = None,
                            explanation: Dict = None, user_context: Dict = None) -> Dict:
        """
        Generate comprehensive preview card for a dataset.
        
        Args:
            dataset: Dataset information
            similarity_score: Relevance score for current query
            explanation: Explanation object from RecommendationExplainer
            user_context: User preferences and context
            
        Returns:
            Complete preview card data structure
        """
        logger.info(f"🎨 Generating preview card for: {dataset.get('title', 'Unknown')}")
        
        preview_card = {
            'dataset_id': dataset.get('id', f"dataset_{hash(dataset.get('title', ''))}"),
            'title': dataset.get('title', 'Untitled Dataset'),
            'header': self._generate_header(dataset, similarity_score),
            'content': self._generate_content(dataset),
            'metadata': self._generate_metadata(dataset),
            'statistics': self._generate_statistics(dataset),
            'integration_guide': self._generate_integration_guide(dataset),
            'visual_elements': self._generate_visual_elements(dataset),
            'actions': self._generate_action_buttons(dataset, user_context),
            'explanation': self._format_explanation(explanation),
            'preview_type': self._determine_preview_type(dataset)
        }
        
        logger.info("✅ Preview card generated successfully")
        return preview_card
    
    def _generate_header(self, dataset: Dict, similarity_score: float = None) -> Dict:
        """Generate the header section of the preview card."""
        
        title = dataset.get('title', 'Untitled Dataset')
        source = dataset.get('source', 'Unknown Source')
        category = dataset.get('category', 'General').title()
        
        # Determine source type and styling
        source_info = self._analyze_source(source)
        
        # Generate quality indicators
        quality_score = dataset.get('quality_score', 0)
        quality_indicators = self._generate_quality_indicators(quality_score)
        
        header = {
            'title': title,
            'title_truncated': title[:60] + "..." if len(title) > 60 else title,
            'source': source,
            'source_type': source_info['type'],
            'source_display': source_info['display_name'],
            'category': category,
            'category_icon': self._get_category_icon(category.lower()),
            'similarity_score': similarity_score,
            'similarity_display': self._format_similarity_score(similarity_score),
            'quality_indicators': quality_indicators,
            'trust_badge': self._generate_trust_badge(source_info),
            'last_updated': self._extract_temporal_info(dataset),
            'header_style': source_info.get('style', 'default')
        }
        
        return header
    
    def _generate_content(self, dataset: Dict) -> Dict:
        """Generate the main content section."""
        
        description = str(dataset.get('description', '')).strip()
        tags = str(dataset.get('tags', '')).strip()
        
        # Process description
        if description and description != 'nan':
            description_processed = self._process_description(description)
        else:
            description_processed = {
                'full': 'No description available for this dataset.',
                'summary': 'No description available.',
                'key_points': []
            }
        
        # Process tags
        tag_list = self._process_tags(tags)
        
        content = {
            'description': description_processed,
            'tags': tag_list,
            'key_highlights': self._extract_key_highlights(dataset),
            'content_type': self._determine_content_type(dataset),
            'data_scope': self._analyze_data_scope(dataset)
        }
        
        return content
    
    def _generate_metadata(self, dataset: Dict) -> Dict:
        """Generate metadata section with technical details."""
        
        metadata = {
            'format': self._detect_data_format(dataset),
            'size_estimate': self._estimate_data_size(dataset),
            'update_frequency': self._detect_update_frequency(dataset),
            'geographic_coverage': self._analyze_geographic_coverage(dataset),
            'temporal_coverage': self._analyze_temporal_coverage(dataset),
            'access_level': self._determine_access_level(dataset),
            'license': self._extract_license_info(dataset),
            'api_availability': self._check_api_availability(dataset)
        }
        
        return metadata
    
    def _generate_statistics(self, dataset: Dict) -> Dict:
        """Generate statistical overview of the dataset."""
        
        # Extract any numerical information from description
        description = str(dataset.get('description', ''))
        numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', description)
        
        statistics = {
            'estimated_records': self._estimate_record_count(dataset),
            'data_points_mentioned': len(numbers),
            'complexity_score': self._calculate_complexity_score(dataset),
            'completeness_score': self._calculate_completeness_score(dataset),
            'freshness_score': self._calculate_freshness_score(dataset),
            'usage_indicators': self._extract_usage_indicators(dataset)
        }
        
        return statistics
    
    def _generate_integration_guide(self, dataset: Dict) -> Dict:
        """Generate integration guidance for developers."""
        
        source_type = self._analyze_source(dataset.get('source', ''))['type']
        
        # Generate integration complexity assessment
        complexity_factors = []
        if source_type == 'government':
            complexity_factors.append('May require API key registration')
        
        description = str(dataset.get('description', '')).lower()
        if 'api' in description:
            complexity_factors.append('API access available')
        if 'csv' in description or 'excel' in description:
            complexity_factors.append('Downloadable format available')
        if 'real-time' in description or 'live' in description:
            complexity_factors.append('Real-time data updates')
        
        integration_guide = {
            'difficulty_level': self._assess_integration_difficulty(dataset),
            'recommended_tools': self._suggest_integration_tools(dataset),
            'data_pipeline_steps': self._generate_pipeline_steps(dataset),
            'complexity_factors': complexity_factors,
            'estimated_setup_time': self._estimate_setup_time(dataset),
            'code_examples': self._generate_code_snippets(dataset),
            'common_use_cases': self._suggest_use_cases(dataset)
        }
        
        return integration_guide
    
    def _generate_visual_elements(self, dataset: Dict) -> Dict:
        """Generate visual styling and elements for the card."""
        
        source_info = self._analyze_source(dataset.get('source', ''))
        category = dataset.get('category', 'general').lower()
        
        # Color scheme based on source and category
        color_scheme = self._determine_color_scheme(source_info, category)
        
        visual_elements = {
            'primary_color': color_scheme['primary'],
            'secondary_color': color_scheme['secondary'],
            'accent_color': color_scheme['accent'],
            'card_style': source_info.get('style', 'default'),
            'icon_set': self._select_icon_set(category),
            'background_pattern': self._select_background_pattern(source_info),
            'border_style': self._determine_border_style(dataset),
            'typography': self._select_typography(source_info)
        }
        
        return visual_elements
    
    def _generate_action_buttons(self, dataset: Dict, user_context: Dict = None) -> List[Dict]:
        """Generate action buttons for the preview card."""
        
        actions = []
        
        # Primary actions
        actions.append({
            'type': 'primary',
            'label': 'View Details',
            'action': 'view_dataset',
            'icon': 'eye',
            'tooltip': 'View complete dataset information'
        })
        
        # Check if downloadable
        if self._is_downloadable(dataset):
            actions.append({
                'type': 'secondary',
                'label': 'Download',
                'action': 'download_dataset',
                'icon': 'download',
                'tooltip': 'Download dataset files'
            })
        
        # API access
        if self._has_api_access(dataset):
            actions.append({
                'type': 'secondary',
                'label': 'API Docs',
                'action': 'view_api',
                'icon': 'code',
                'tooltip': 'View API documentation'
            })
        
        # User-specific actions
        if user_context:
            # Bookmark action
            bookmarked = user_context.get('bookmarked_datasets', [])
            dataset_id = dataset.get('id', '')
            
            if dataset_id in bookmarked:
                actions.append({
                    'type': 'tertiary',
                    'label': 'Bookmarked',
                    'action': 'remove_bookmark',
                    'icon': 'bookmark-filled',
                    'tooltip': 'Remove from bookmarks'
                })
            else:
                actions.append({
                    'type': 'tertiary',
                    'label': 'Bookmark',
                    'action': 'add_bookmark',
                    'icon': 'bookmark',
                    'tooltip': 'Add to bookmarks'
                })
        
        # Share action
        actions.append({
            'type': 'tertiary',
            'label': 'Share',
            'action': 'share_dataset',
            'icon': 'share',
            'tooltip': 'Share dataset link'
        })
        
        return actions
    
    def _format_explanation(self, explanation: Dict) -> Dict:
        """Format explanation data for the preview card."""
        
        if not explanation:
            return {
                'available': False,
                'summary': 'No explanation available',
                'confidence': 'unknown'
            }
        
        return {
            'available': True,
            'summary': explanation.get('explanation_text', ''),
            'confidence': explanation.get('confidence_level', 'medium'),
            'primary_reasons': explanation.get('primary_reasons', []),
            'secondary_reasons': explanation.get('secondary_reasons', []),
            'expandable': len(explanation.get('primary_reasons', [])) > 1
        }
    
    def _determine_preview_type(self, dataset: Dict) -> str:
        """Determine the type of preview card to display."""
        
        source = str(dataset.get('source', '')).lower()
        category = str(dataset.get('category', '')).lower()
        
        if any(agency in source for agency in self.singapore_agencies.keys()):
            return 'singapore_government'
        elif 'government' in source or 'ministry' in source:
            return 'government'
        elif 'university' in source or 'research' in source:
            return 'academic'
        elif category in ['transport', 'housing', 'population']:
            return 'singapore_focus'
        else:
            return 'general'
    
    # Helper methods for various analyses
    
    def _analyze_source(self, source: str) -> Dict:
        """Analyze the data source and determine its characteristics."""
        
        source_lower = source.lower()
        
        # Check for Singapore government agencies
        for abbrev, full_name in self.singapore_agencies.items():
            if abbrev in source_lower or full_name.lower() in source_lower:
                return {
                    'type': 'singapore_government',
                    'display_name': full_name,
                    'style': 'official',
                    'trust_level': 'very_high'
                }
        
        # Check for other government sources
        if any(keyword in source_lower for keyword in ['government', 'ministry', 'authority', 'board']):
            return {
                'type': 'government',
                'display_name': source,
                'style': 'official',
                'trust_level': 'high'
            }
        
        # Check for academic sources
        if any(keyword in source_lower for keyword in ['university', 'research', 'institute', 'academic']):
            return {
                'type': 'academic',
                'display_name': source,
                'style': 'research',
                'trust_level': 'high'
            }
        
        # Default to commercial/other
        return {
            'type': 'commercial',
            'display_name': source,
            'style': 'business',
            'trust_level': 'medium'
        }
    
    def _process_description(self, description: str) -> Dict:
        """Process and enhance dataset description."""
        
        # Clean up description
        cleaned = re.sub(r'\s+', ' ', description).strip()
        
        # Generate summary (first sentence or first 150 chars)
        sentences = re.split(r'[.!?]+', cleaned)
        summary = sentences[0] if sentences else cleaned
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        # Extract key points
        key_points = []
        if len(sentences) > 1:
            for sentence in sentences[1:4]:  # Take up to 3 additional sentences
                sentence = sentence.strip()
                if len(sentence) > 10:
                    key_points.append(sentence)
        
        return {
            'full': cleaned,
            'summary': summary,
            'key_points': key_points,
            'word_count': len(cleaned.split())
        }
    
    def _process_tags(self, tags: str) -> List[Dict]:
        """Process and categorize tags."""
        
        if not tags or tags == 'nan':
            return []
        
        # Split tags and clean them
        tag_list = re.split(r'[,;\s]+', tags)
        tag_list = [tag.strip().lower() for tag in tag_list if tag.strip()]
        
        # Categorize tags
        categorized_tags = []
        for tag in tag_list[:8]:  # Limit to 8 tags
            category = self._categorize_tag(tag)
            categorized_tags.append({
                'text': tag,
                'category': category,
                'display_style': self._get_tag_style(category)
            })
        
        return categorized_tags
    
    def _categorize_tag(self, tag: str) -> str:
        """Categorize a tag into type groups."""
        
        tag_categories = {
            'domain': ['housing', 'transport', 'health', 'education', 'environment'],
            'geography': ['singapore', 'sg', 'asia', 'global'],
            'data_type': ['statistics', 'survey', 'census', 'monitoring', 'tracking'],
            'temporal': ['annual', 'monthly', 'daily', 'historical', 'current'],
            'agency': list(self.singapore_agencies.keys())
        }
        
        for category, keywords in tag_categories.items():
            if any(keyword in tag.lower() for keyword in keywords):
                return category
        
        return 'general'
    
    def _get_tag_style(self, category: str) -> str:
        """Get visual style for tag category."""
        
        styles = {
            'domain': 'primary',
            'geography': 'success',
            'data_type': 'info',
            'temporal': 'warning',
            'agency': 'official',
            'general': 'secondary'
        }
        
        return styles.get(category, 'secondary')
    
    def _determine_color_scheme(self, source_info: Dict, category: str) -> Dict:
        """Determine color scheme for the card."""
        
        # Singapore government - official blue
        if source_info['type'] == 'singapore_government':
            return {
                'primary': '#1e40af',    # Official blue
                'secondary': '#3b82f6',  # Lighter blue
                'accent': '#ef4444'      # Singapore red
            }
        
        # Category-based colors
        category_colors = {
            'housing': {'primary': '#059669', 'secondary': '#10b981', 'accent': '#f59e0b'},
            'transport': {'primary': '#7c3aed', 'secondary': '#8b5cf6', 'accent': '#06b6d4'},
            'environment': {'primary': '#16a34a', 'secondary': '#22c55e', 'accent': '#84cc16'},
            'population': {'primary': '#dc2626', 'secondary': '#ef4444', 'accent': '#f97316'}
        }
        
        if category in category_colors:
            return category_colors[category]
        
        # Default scheme
        return {
            'primary': '#6b7280',
            'secondary': '#9ca3af', 
            'accent': '#3b82f6'
        }
    
    def _estimate_record_count(self, dataset: Dict) -> str:
        """Estimate the number of records in the dataset."""
        
        description = str(dataset.get('description', '')).lower()
        
        # Look for explicit numbers
        numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*)\b', description)
        if numbers:
            largest_num = max([int(num.replace(',', '')) for num in numbers])
            if largest_num > 1000:
                return f"~{largest_num:,} records"
        
        # Estimate based on keywords
        if any(keyword in description for keyword in ['comprehensive', 'complete', 'all']):
            return "Large dataset (10K+ records)"
        elif any(keyword in description for keyword in ['sample', 'subset', 'limited']):
            return "Small dataset (<1K records)"
        else:
            return "Medium dataset (1K-10K records)"
    
    def _generate_quality_indicators(self, quality_score: float) -> Dict:
        """Generate quality indicators based on score."""
        indicators = {
            'score': quality_score,
            'level': 'medium',
            'badges': [],
            'warnings': []
        }
        
        if quality_score >= 0.9:
            indicators['level'] = 'excellent'
            indicators['badges'] = ['High Quality', 'Verified']
        elif quality_score >= 0.8:
            indicators['level'] = 'high'
            indicators['badges'] = ['Good Quality', 'Reliable']
        elif quality_score >= 0.6:
            indicators['level'] = 'medium'
            indicators['badges'] = ['Acceptable Quality']
        elif quality_score >= 0.4:
            indicators['level'] = 'low'
            indicators['warnings'] = ['Limited Quality']
        else:
            indicators['level'] = 'poor'
            indicators['warnings'] = ['Quality Issues', 'Use with Caution']
        
        return indicators
    
    def _get_category_icon(self, category: str) -> str:
        """Get icon for dataset category."""
        icons = {
            'housing': '🏠',
            'transport': '🚌',
            'environment': '🌱',
            'population': '👥',
            'economy': '💰',
            'health': '🏥',
            'education': '🎓',
            'government': '🏛️',
            'general': '📊'
        }
        return icons.get(category, '📄')
    
    def _format_similarity_score(self, similarity_score: float) -> str:
        """Format similarity score for display."""
        if similarity_score is None:
            return 'N/A'
        
        percentage = int(similarity_score * 100)
        if percentage >= 90:
            return f'{percentage}% (Excellent Match)'
        elif percentage >= 80:
            return f'{percentage}% (Very Good Match)'
        elif percentage >= 70:
            return f'{percentage}% (Good Match)'
        elif percentage >= 50:
            return f'{percentage}% (Fair Match)'
        else:
            return f'{percentage}% (Limited Match)'
    
    def _generate_trust_badge(self, source_info: Dict) -> Dict:
        """Generate trust badge information."""
        trust_level = source_info.get('trust_level', 'medium')
        
        badge_info = {
            'show_badge': trust_level in ['high', 'very_high'],
            'badge_text': '',
            'badge_color': '#6b7280',
            'tooltip': ''
        }
        
        if trust_level == 'very_high':
            badge_info.update({
                'badge_text': 'Official',
                'badge_color': '#059669',
                'tooltip': 'Singapore Government Official Data'
            })
        elif trust_level == 'high':
            badge_info.update({
                'badge_text': 'Verified',
                'badge_color': '#3b82f6',
                'tooltip': 'Verified Government Source'
            })
        
        return badge_info
    
    def _extract_temporal_info(self, dataset: Dict) -> Dict:
        """Extract temporal information from dataset."""
        description = str(dataset.get('description', '')).lower()
        
        # Look for temporal indicators
        temporal_patterns = {
            'daily': r'\b(daily|day|per day)\b',
            'weekly': r'\b(weekly|week|per week)\b',
            'monthly': r'\b(monthly|month|per month)\b',
            'quarterly': r'\b(quarterly|quarter)\b',
            'annual': r'\b(annual|yearly|year|\d{4})\b',
            'real-time': r'\b(real.?time|live|current)\b',
            'historical': r'\b(historical|archive|past)\b'
        }
        
        update_frequency = 'unknown'
        last_updated = 'Unknown'
        
        for freq, pattern in temporal_patterns.items():
            if re.search(pattern, description):
                update_frequency = freq
                break
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', description)
        if years:
            last_updated = f'Data includes {max(years)}'
        
        return {
            'update_frequency': update_frequency,
            'last_updated': last_updated,
            'temporal_coverage': years
        }
    
    def _extract_key_highlights(self, dataset: Dict) -> List[str]:
        """Extract key highlights from dataset."""
        highlights = []
        description = str(dataset.get('description', ''))
        
        # Look for quantitative highlights
        numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', description)
        if numbers:
            largest = max([float(n.replace(',', '')) for n in numbers])
            if largest >= 1000:
                highlights.append(f'Large dataset with {largest:,.0f}+ records')
        
        # Look for coverage highlights
        if any(term in description.lower() for term in ['comprehensive', 'complete', 'all']):
            highlights.append('Comprehensive coverage')
        
        if any(term in description.lower() for term in ['real-time', 'live', 'current']):
            highlights.append('Real-time updates')
        
        if any(term in description.lower() for term in ['historical', 'archive', 'time series']):
            highlights.append('Historical data available')
        
        return highlights[:3]
    
    def _determine_content_type(self, dataset: Dict) -> str:
        """Determine the type of content in dataset."""
        description = str(dataset.get('description', '')).lower()
        
        if any(term in description for term in ['api', 'endpoint', 'json']):
            return 'api_data'
        elif any(term in description for term in ['csv', 'excel', 'spreadsheet']):
            return 'tabular_data'
        elif any(term in description for term in ['map', 'spatial', 'geographic', 'coordinates']):
            return 'geospatial_data'
        elif any(term in description for term in ['time series', 'temporal', 'historical']):
            return 'time_series_data'
        elif any(term in description for term in ['survey', 'questionnaire', 'responses']):
            return 'survey_data'
        else:
            return 'general_data'
    
    def _analyze_data_scope(self, dataset: Dict) -> Dict:
        """Analyze the scope and coverage of the dataset."""
        description = str(dataset.get('description', '')).lower()
        title = str(dataset.get('title', '')).lower()
        
        scope = {
            'geographic_scope': 'unknown',
            'temporal_scope': 'unknown',
            'subject_scope': 'specific'
        }
        
        # Geographic scope
        if any(term in f'{title} {description}' for term in ['singapore', 'sg', 'national']):
            scope['geographic_scope'] = 'national'
        elif any(term in f'{title} {description}' for term in ['global', 'international', 'worldwide']):
            scope['geographic_scope'] = 'global'
        elif any(term in f'{title} {description}' for term in ['region', 'district', 'area']):
            scope['geographic_scope'] = 'regional'
        else:
            scope['geographic_scope'] = 'local'
        
        # Temporal scope
        if any(term in description for term in ['historical', 'archive', 'decades']):
            scope['temporal_scope'] = 'historical'
        elif any(term in description for term in ['current', 'latest', '2024', '2023']):
            scope['temporal_scope'] = 'current'
        elif any(term in description for term in ['time series', 'longitudinal']):
            scope['temporal_scope'] = 'time_series'
        
        return scope
    
    def _detect_data_format(self, dataset: Dict) -> str:
        """Detect the format of the dataset."""
        description = str(dataset.get('description', '')).lower()
        
        if 'json' in description:
            return 'JSON'
        elif any(term in description for term in ['csv', 'comma separated']):
            return 'CSV'
        elif any(term in description for term in ['excel', 'xlsx', 'xls']):
            return 'Excel'
        elif 'xml' in description:
            return 'XML'
        elif any(term in description for term in ['api', 'rest', 'endpoint']):
            return 'API'
        elif any(term in description for term in ['pdf', 'document']):
            return 'Document'
        else:
            return 'Mixed/Unknown'
    
    def _estimate_data_size(self, dataset: Dict) -> str:
        """Estimate the size of the dataset."""
        description = str(dataset.get('description', ''))
        
        # Look for explicit size mentions
        size_patterns = [
            (r'(\d+)\s*gb', lambda x: f'{x} GB'),
            (r'(\d+)\s*mb', lambda x: f'{x} MB'),
            (r'(\d+)\s*kb', lambda x: f'{x} KB'),
            (r'(\d{4,})\s*records?', lambda x: f'~{int(x):,} records')
        ]
        
        for pattern, formatter in size_patterns:
            match = re.search(pattern, description.lower())
            if match:
                return formatter(match.group(1))
        
        # Estimate based on description
        if any(term in description.lower() for term in ['large', 'comprehensive', 'extensive']):
            return 'Large (>100MB)'
        elif any(term in description.lower() for term in ['small', 'sample', 'subset']):
            return 'Small (<10MB)'
        else:
            return 'Medium (10-100MB)'
    
    def _detect_update_frequency(self, dataset: Dict) -> str:
        """Detect how frequently the dataset is updated."""
        description = str(dataset.get('description', '')).lower()
        
        if any(term in description for term in ['real-time', 'live', 'continuous']):
            return 'Real-time'
        elif any(term in description for term in ['daily', 'day']):
            return 'Daily'
        elif any(term in description for term in ['weekly', 'week']):
            return 'Weekly'
        elif any(term in description for term in ['monthly', 'month']):
            return 'Monthly'
        elif any(term in description for term in ['quarterly', 'quarter']):
            return 'Quarterly'
        elif any(term in description for term in ['annual', 'yearly', 'year']):
            return 'Annually'
        elif any(term in description for term in ['static', 'historical', 'archive']):
            return 'Static'
        else:
            return 'Unknown'
    
    def _analyze_geographic_coverage(self, dataset: Dict) -> str:
        """Analyze geographic coverage of the dataset."""
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        
        if any(term in text for term in ['singapore', 'sg', 'republic of singapore']):
            return 'Singapore'
        elif any(term in text for term in ['asia', 'asean', 'southeast asia']):
            return 'Asia/ASEAN'
        elif any(term in text for term in ['global', 'worldwide', 'international']):
            return 'Global'
        elif any(term in text for term in ['region', 'district', 'area']):
            return 'Regional'
        else:
            return 'Not specified'
    
    def _analyze_temporal_coverage(self, dataset: Dict) -> str:
        """Analyze temporal coverage of the dataset."""
        description = str(dataset.get('description', ''))
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', description)
        if len(years) >= 2:
            return f'{min(years)} - {max(years)}'
        elif len(years) == 1:
            return f'{years[0]} data'
        
        # Check for temporal keywords
        if any(term in description.lower() for term in ['current', 'latest', '2024']):
            return 'Current'
        elif any(term in description.lower() for term in ['historical', 'archive']):
            return 'Historical'
        else:
            return 'Not specified'
    
    def _determine_access_level(self, dataset: Dict) -> str:
        """Determine the access level of the dataset."""
        description = str(dataset.get('description', '')).lower()
        source = str(dataset.get('source', '')).lower()
        
        if any(term in f'{description} {source}' for term in ['public', 'open', 'free']):
            return 'Public'
        elif any(term in f'{description} {source}' for term in ['restricted', 'limited', 'approval']):
            return 'Restricted'
        elif any(term in f'{description} {source}' for term in ['private', 'confidential']):
            return 'Private'
        elif any(term in f'{description} {source}' for term in ['government', 'official']):
            return 'Government'
        else:
            return 'Unknown'
    
    def _extract_license_info(self, dataset: Dict) -> str:
        """Extract license information."""
        description = str(dataset.get('description', '')).lower()
        
        if 'creative commons' in description or 'cc by' in description:
            return 'Creative Commons'
        elif 'open data' in description:
            return 'Open Data License'
        elif 'government' in description:
            return 'Government License'
        elif any(term in description for term in ['license', 'licensed']):
            return 'Licensed'
        else:
            return 'Not specified'
    
    def _check_api_availability(self, dataset: Dict) -> bool:
        """Check if API access is available."""
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        return any(term in text for term in ['api', 'endpoint', 'rest', 'json api', 'web service'])
    
    def _calculate_complexity_score(self, dataset: Dict) -> float:
        """Calculate complexity score of the dataset."""
        score = 0.5  # Base score
        description = str(dataset.get('description', '')).lower()
        
        # Increase complexity for certain indicators
        if any(term in description for term in ['multiple tables', 'relational', 'complex']):
            score += 0.2
        if any(term in description for term in ['api', 'real-time', 'streaming']):
            score += 0.2
        if any(term in description for term in ['geospatial', 'coordinates', 'spatial']):
            score += 0.15
        if any(term in description for term in ['time series', 'temporal', 'longitudinal']):
            score += 0.15
        
        # Decrease for simple indicators
        if any(term in description for term in ['simple', 'basic', 'single table']):
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_completeness_score(self, dataset: Dict) -> float:
        """Calculate completeness score based on available metadata."""
        score = 0.0
        
        # Check presence of key fields
        if dataset.get('title') and dataset.get('title') != 'nan':
            score += 0.25
        if dataset.get('description') and len(str(dataset.get('description'))) > 20:
            score += 0.35
        if dataset.get('category') and dataset.get('category') != 'nan':
            score += 0.15
        if dataset.get('source') and dataset.get('source') != 'nan':
            score += 0.15
        if dataset.get('tags') and dataset.get('tags') != 'nan':
            score += 0.1
        
        return score
    
    def _calculate_freshness_score(self, dataset: Dict) -> float:
        """Calculate freshness score based on temporal indicators."""
        description = str(dataset.get('description', '')).lower()
        
        if any(term in description for term in ['2024', 'current', 'latest', 'real-time']):
            return 1.0
        elif '2023' in description:
            return 0.8
        elif '2022' in description:
            return 0.6
        elif any(term in description for term in ['2021', '2020']):
            return 0.4
        elif any(term in description for term in ['historical', 'archive']):
            return 0.2
        else:
            return 0.5  # Unknown
    
    def _extract_usage_indicators(self, dataset: Dict) -> List[str]:
        """Extract usage indicators from dataset description."""
        indicators = []
        description = str(dataset.get('description', '')).lower()
        
        usage_patterns = {
            'Research': ['research', 'study', 'analysis', 'academic'],
            'Business Intelligence': ['business', 'commercial', 'market', 'industry'],
            'Policy Making': ['policy', 'planning', 'government', 'decision'],
            'Public Information': ['public', 'citizen', 'transparency', 'open'],
            'Development': ['development', 'application', 'software', 'app']
        }
        
        for usage_type, keywords in usage_patterns.items():
            if any(keyword in description for keyword in keywords):
                indicators.append(usage_type)
        
        return indicators[:3]  # Limit to top 3
    
    def _assess_integration_difficulty(self, dataset: Dict) -> str:
        """Assess integration difficulty level."""
        description = str(dataset.get('description', '')).lower()
        
        difficulty_score = 0
        
        # Factors that increase difficulty
        if any(term in description for term in ['complex', 'multiple tables', 'relational']):
            difficulty_score += 2
        if any(term in description for term in ['authentication', 'api key', 'registration']):
            difficulty_score += 1
        if any(term in description for term in ['real-time', 'streaming', 'live']):
            difficulty_score += 2
        if any(term in description for term in ['geospatial', 'coordinates', 'spatial']):
            difficulty_score += 1
        
        # Factors that decrease difficulty
        if any(term in description for term in ['csv', 'simple', 'download']):
            difficulty_score -= 1
        if any(term in description for term in ['api', 'rest', 'json']):
            difficulty_score -= 0  # Neutral for modern APIs
        
        if difficulty_score >= 3:
            return 'Advanced'
        elif difficulty_score >= 1:
            return 'Intermediate'
        else:
            return 'Beginner'
    
    def _suggest_integration_tools(self, dataset: Dict) -> List[str]:
        """Suggest tools for dataset integration."""
        tools = []
        description = str(dataset.get('description', '')).lower()
        
        # Based on data format
        if any(term in description for term in ['csv', 'excel']):
            tools.extend(['pandas', 'Excel', 'Google Sheets'])
        if any(term in description for term in ['json', 'api']):
            tools.extend(['requests', 'curl', 'Postman'])
        if any(term in description for term in ['geospatial', 'coordinates']):
            tools.extend(['QGIS', 'ArcGIS', 'geopandas'])
        if any(term in description for term in ['database', 'sql']):
            tools.extend(['SQL', 'PostgreSQL', 'MySQL'])
        
        # Default tools
        if not tools:
            tools = ['Python', 'R', 'Excel']
        
        return tools[:4]  # Limit to 4 tools
    
    def _generate_pipeline_steps(self, dataset: Dict) -> List[str]:
        """Generate data pipeline steps."""
        steps = ['1. Access dataset']
        description = str(dataset.get('description', '')).lower()
        
        # Authentication step
        if any(term in description for term in ['api key', 'registration', 'authentication']):
            steps.append('2. Obtain API credentials')
            steps.append('3. Configure authentication')
        else:
            steps.append('2. Download/connect to data')
        
        # Processing steps
        if any(term in description for term in ['clean', 'preprocess', 'format']):
            steps.append(f'{len(steps)+1}. Clean and preprocess data')
        
        steps.append(f'{len(steps)+1}. Validate data quality')
        steps.append(f'{len(steps)+1}. Integrate into your system')
        
        return steps
    
    def _estimate_setup_time(self, dataset: Dict) -> str:
        """Estimate setup time for integration."""
        difficulty = self._assess_integration_difficulty(dataset)
        description = str(dataset.get('description', '')).lower()
        
        base_time = {
            'Beginner': 30,
            'Intermediate': 120,
            'Advanced': 480
        }.get(difficulty, 60)
        
        # Adjust based on specific factors
        if any(term in description for term in ['api key', 'registration']):
            base_time += 60
        if any(term in description for term in ['complex', 'multiple']):
            base_time += 120
        
        if base_time < 60:
            return f'{base_time} minutes'
        else:
            hours = base_time // 60
            return f'{hours} hour{"s" if hours > 1 else ""}'
    
    def _generate_code_snippets(self, dataset: Dict) -> Dict[str, str]:
        """Generate code snippets for common integrations."""
        snippets = {}
        description = str(dataset.get('description', '')).lower()
        
        # Python snippet for CSV
        if any(term in description for term in ['csv', 'download']):
            snippets['python_csv'] = '''import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')
print(df.head())'''
        
        # Python snippet for API
        if any(term in description for term in ['api', 'json']):
            snippets['python_api'] = '''import requests

# API call
response = requests.get('API_ENDPOINT')
data = response.json()
print(data)'''
        
        # R snippet
        snippets['r_basic'] = '''# Load dataset in R
library(readr)
data <- read_csv("dataset.csv")
head(data)'''
        
        return snippets
    
    def _suggest_use_cases(self, dataset: Dict) -> List[str]:
        """Suggest common use cases for the dataset."""
        use_cases = []
        category = str(dataset.get('category', '')).lower()
        description = str(dataset.get('description', '')).lower()
        
        category_use_cases = {
            'housing': [
                'Property market analysis',
                'Housing affordability studies',
                'Urban planning research',
                'Real estate investment analysis'
            ],
            'transport': [
                'Traffic flow optimization',
                'Public transport planning',
                'Route efficiency analysis',
                'Urban mobility studies'
            ],
            'population': [
                'Demographic research',
                'Population trend analysis',
                'Social policy development',
                'Market segmentation'
            ],
            'environment': [
                'Environmental monitoring',
                'Climate change research',
                'Sustainability assessment',
                'Policy impact evaluation'
            ]
        }
        
        # Get category-specific use cases
        for cat, cases in category_use_cases.items():
            if cat in category or cat in description:
                use_cases.extend(cases[:2])  # Top 2 per category
                break
        
        # Generic use cases if no specific category
        if not use_cases:
            use_cases = [
                'Data analysis and visualization',
                'Research and academic studies',
                'Business intelligence',
                'Policy development'
            ]
        
        return use_cases[:3]
    
    def _select_icon_set(self, category: str) -> str:
        """Select appropriate icon set for category."""
        icon_sets = {
            'housing': 'building',
            'transport': 'vehicle',
            'environment': 'nature',
            'population': 'people',
            'government': 'official',
            'economy': 'finance'
        }
        return icon_sets.get(category, 'general')
    
    def _select_background_pattern(self, source_info: Dict) -> str:
        """Select background pattern based on source."""
        source_type = source_info.get('type', 'general')
        
        patterns = {
            'singapore_government': 'official_grid',
            'government': 'institutional',
            'academic': 'research_dots',
            'commercial': 'business_lines'
        }
        return patterns.get(source_type, 'default')
    
    def _determine_border_style(self, dataset: Dict) -> str:
        """Determine border style for the card."""
        quality_score = dataset.get('quality_score', 0)
        
        if quality_score >= 0.9:
            return 'premium'
        elif quality_score >= 0.8:
            return 'enhanced'
        elif quality_score >= 0.6:
            return 'standard'
        else:
            return 'basic'
    
    def _select_typography(self, source_info: Dict) -> Dict:
        """Select typography settings."""
        source_type = source_info.get('type', 'general')
        
        typography = {
            'singapore_government': {
                'title_font': 'Inter',
                'body_font': 'Inter',
                'title_weight': 'semibold',
                'style': 'clean'
            },
            'academic': {
                'title_font': 'Merriweather',
                'body_font': 'Source Sans Pro',
                'title_weight': 'bold',
                'style': 'scholarly'
            },
            'commercial': {
                'title_font': 'Roboto',
                'body_font': 'Roboto',
                'title_weight': 'medium',
                'style': 'modern'
            }
        }
        
        return typography.get(source_type, typography['commercial'])
    
    def _is_downloadable(self, dataset: Dict) -> bool:
        """Check if dataset is downloadable."""
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        return any(term in text for term in ['download', 'csv', 'excel', 'file', 'export'])
    
    def _has_api_access(self, dataset: Dict) -> bool:
        """Check if dataset has API access."""
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        return any(term in text for term in ['api', 'endpoint', 'rest', 'web service', 'json api'])


def demo_preview_generator():
    """Demonstrate the dataset preview generator."""
    print("🎨 Initializing Dataset Preview Generator Demo")
    
    generator = DatasetPreviewGenerator()
    
    # Sample dataset
    sample_dataset = {
        'id': 'hdb_resale_001',
        'title': 'HDB Resale Flat Prices by Town and Flat Type',
        'description': 'Comprehensive data on HDB resale flat transactions including prices, locations, flat characteristics, and transaction details. Updated monthly with over 500,000 historical records from 2017 to present. Includes information on flat type, block number, street name, storey range, floor area, flat model, lease commence date, and resale price.',
        'category': 'housing',
        'source': 'Housing Development Board (HDB)',
        'quality_score': 0.94,
        'tags': 'housing, property, resale, HDB, singapore, prices, real estate'
    }
    
    # Sample explanation
    sample_explanation = {
        'explanation_text': '"HDB Resale Flat Prices by Town and Flat Type" is an excellent match for your query. Contains keywords: housing, data, singapore; Similar category: Housing; Recommended based on Singapore government data expertise.',
        'confidence_level': 'very_high',
        'primary_reasons': [
            'Contains keywords: housing, data, singapore',
            'Similar category: Housing',
            'Recommended based on Singapore government data expertise'
        ],
        'secondary_reasons': [
            'High quality dataset (score: 0.9)',
            'Same data source: Housing Development Board'
        ]
    }
    
    # Sample user context
    user_context = {
        'bookmarked_datasets': ['hdb_rental_001'],
        'preferred_query_terms': [('housing', 5), ('singapore', 4)]
    }
    
    # Generate preview card
    preview_card = generator.generate_preview_card(
        dataset=sample_dataset,
        similarity_score=0.89,
        explanation=sample_explanation,
        user_context=user_context
    )
    
    # Display preview card information
    print(f"\\n🎨 Generated Preview Card:")
    print(f"Title: {preview_card['header']['title']}")
    print(f"Source: {preview_card['header']['source_display']}")
    print(f"Type: {preview_card['preview_type']}")
    print(f"Similarity: {preview_card['header']['similarity_display']}")
    
    print(f"\\n📝 Content Summary:")
    print(f"{preview_card['content']['description']['summary']}")
    
    print(f"\\n🏷️ Tags:")
    for tag in preview_card['content']['tags']:
        print(f"  • {tag['text']} ({tag['category']})")
    
    print(f"\\n📊 Metadata:")
    for key, value in preview_card['metadata'].items():
        print(f"  {key}: {value}")
    
    print(f"\\n🔧 Integration Guide:")
    print(f"  Difficulty: {preview_card['integration_guide']['difficulty_level']}")
    print(f"  Setup Time: {preview_card['integration_guide']['estimated_setup_time']}")
    print(f"  Tools: {', '.join(preview_card['integration_guide']['recommended_tools'])}")
    
    print(f"\\n🎯 Actions:")
    for action in preview_card['actions']:
        print(f"  • {action['label']} ({action['type']})")
    
    print(f"\\n🎨 Visual Style:")
    print(f"  Primary Color: {preview_card['visual_elements']['primary_color']}")
    print(f"  Card Style: {preview_card['visual_elements']['card_style']}")
    
    print("\\n✅ Preview card demo complete!")


if __name__ == "__main__":
    demo_preview_generator()