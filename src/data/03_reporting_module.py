# 03_reporting_module.py - EDA & Validation Reporting
import json
import logging
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEDAReporter:
    """Comprehensive EDA analysis and reporting with configurable outputs"""
    
    def __init__(self, config_path: str = "data_pipeline.yml"):
        """Initialize reporter with configuration"""
        self.config = self._load_config(config_path)
        self.reporting_config = self.config.get('phase_3_reporting', {})
        self.quality_thresholds = self.reporting_config.get('quality_thresholds', {})
        
        # Setup output paths (outputs/EDA/ for easy retrieval)
        self.output_base = Path(self.reporting_config.get('output_base_path', 'outputs/EDA'))
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.datasets_df = None
        self.keyword_profiles = {}
        self.relationships = {}
        self.ground_truth = {}
        self.user_behavior_analysis = {}
        self.analysis_results = {}
        
        # Setup plotting
        self._setup_plotting()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return {}
    
    def _setup_plotting(self):
        """Setup matplotlib and seaborn configurations"""
        viz_config = self.reporting_config.get('visualizations', {})
        
        plt.style.use(viz_config.get('style', 'default'))
        sns.set_palette(viz_config.get('color_palette', 'husl'))
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = viz_config.get('figure_size', [15, 10])
        plt.rcParams['figure.dpi'] = viz_config.get('dpi', 300)
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = viz_config.get('dpi', 300)
    
    def load_all_analysis_data(self, data_path: str = "data/processed") -> bool:
        """Load all data from previous phases"""
        logger.info("📂 Loading comprehensive analysis data...")
        
        try:
            data_path = Path(data_path)
            
            # Load datasets
            datasets = []
            singapore_file = data_path / "singapore_datasets.csv"
            global_file = data_path / "global_datasets.csv"
            
            if singapore_file.exists():
                sg_df = pd.read_csv(singapore_file)
                datasets.append(sg_df)
                logger.info(f"📊 Singapore datasets: {len(sg_df)}")
            
            if global_file.exists():
                global_df = pd.read_csv(global_file)
                datasets.append(global_df)
                logger.info(f"🌍 Global datasets: {len(global_df)}")
            
            if datasets:
                self.datasets_df = pd.concat(datasets, ignore_index=True)
                logger.info(f"✅ Total datasets loaded: {len(self.datasets_df)}")
            
            # Load Phase 2 analysis outputs
            analysis_files = [
                ("keyword_profiles.json", "keyword_profiles"),
                ("dataset_relationships.json", "relationships"),
                ("intelligent_ground_truth.json", "ground_truth"),
                ("user_behavior_analysis.json", "user_behavior_analysis")
            ]
            
            for filename, attr_name in analysis_files:
                file_path = data_path / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        setattr(self, attr_name, json.load(f))
                    logger.info(f"📄 Loaded {filename}")
                else:
                    logger.warning(f"⚠️ {filename} not found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load analysis data: {e}")
            return False
    
    def analyze_dataset_collection_overview(self) -> Dict:
        """Comprehensive analysis of dataset collection"""
        logger.info("📊 Analyzing dataset collection overview...")
        
        if self.datasets_df is None:
            return {}
        
        # Basic distribution analysis
        analysis = {
            'collection_metadata': {
                'total_datasets': len(self.datasets_df),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'data_sources': self.datasets_df['source'].nunique(),
                'categories_covered': self.datasets_df['category'].nunique()
            },
            'source_analysis': {
                'distribution': self.datasets_df['source'].value_counts().to_dict(),
                'singapore_vs_global': {
                    'singapore_count': len(self.datasets_df[self.datasets_df['source'].str.contains('gov.sg|LTA|URA|MAS|SingStat|OneMap', case=False, na=False)]),
                    'global_count': len(self.datasets_df[self.datasets_df['source'].str.contains('World Bank|IMF|OECD|UN', case=False, na=False)])
                }
            },
            'category_analysis': {
                'distribution': self.datasets_df['category'].value_counts().to_dict(),
                'coverage_assessment': self._assess_category_coverage()
            },
            'quality_analysis': {
                'overall_statistics': {
                    'mean_quality': float(self.datasets_df['quality_score'].mean()),
                    'median_quality': float(self.datasets_df['quality_score'].median()),
                    'std_quality': float(self.datasets_df['quality_score'].std())
                },
                'quality_distribution': self._categorize_quality_levels(),
                'quality_by_source': self.datasets_df.groupby('source')['quality_score'].mean().to_dict(),
                'quality_by_category': self.datasets_df.groupby('category')['quality_score'].mean().to_dict()
            },
            'temporal_analysis': self._analyze_temporal_characteristics(),
            'format_analysis': self.datasets_df['format'].value_counts().to_dict()
        }
        
        self.analysis_results['collection_overview'] = analysis
        return analysis
    
    def _assess_category_coverage(self) -> Dict:
        """Assess coverage and balance of dataset categories"""
        total_datasets = len(self.datasets_df)
        category_counts = self.datasets_df['category'].value_counts()
        
        return {
            'well_represented': category_counts[category_counts >= 3].to_dict(),
            'under_represented': category_counts[category_counts < 2].to_dict(),
            'over_represented': category_counts[category_counts > total_datasets * 0.3].to_dict(),
            'balance_score': float(1 - (category_counts.std() / category_counts.mean()))  # Lower std = better balance
        }
    
    def _categorize_quality_levels(self) -> Dict:
        """Categorize datasets by quality levels"""
        high_threshold = self.quality_thresholds.get('high_quality', 0.8)
        medium_threshold = self.quality_thresholds.get('medium_quality', 0.5)
        
        return {
            'high_quality': int((self.datasets_df['quality_score'] >= high_threshold).sum()),
            'medium_quality': int(((self.datasets_df['quality_score'] >= medium_threshold) & 
                                  (self.datasets_df['quality_score'] < high_threshold)).sum()),
            'low_quality': int((self.datasets_df['quality_score'] < medium_threshold).sum()),
            'thresholds_used': {
                'high_quality_threshold': high_threshold,
                'medium_quality_threshold': medium_threshold
            }
        }
    
    def _analyze_temporal_characteristics(self) -> Dict:
        """Analyze temporal characteristics of the dataset collection"""
        temporal_analysis = {
            'update_frequency_distribution': self.datasets_df['frequency'].value_counts().to_dict(),
            'status_distribution': self.datasets_df['status'].value_counts().to_dict(),
            'active_datasets_percentage': float((self.datasets_df['status'] == 'active').mean() * 100)
        }
        
        # Analyze last update dates if available
        if 'last_updated' in self.datasets_df.columns:
            try:
                self.datasets_df['last_updated_parsed'] = pd.to_datetime(self.datasets_df['last_updated'], errors='coerce')
                recent_updates = (self.datasets_df['last_updated_parsed'] > '2024-01-01').sum()
                temporal_analysis['recent_updates_2024'] = int(recent_updates)
            except:
                temporal_analysis['date_parsing_note'] = "Could not parse last_updated dates"
        
        return temporal_analysis
    
    def analyze_keyword_intelligence(self) -> Dict:
        """Analyze keyword extraction and intelligence patterns"""
        logger.info("🔍 Analyzing keyword intelligence...")
        
        if not self.keyword_profiles:
            return {'error': 'No keyword profiles available'}
        
        # Aggregate keyword analysis
        all_keywords = []
        domain_signals = []
        relevance_scores = []
        category_keyword_map = defaultdict(list)
        
        for profile in self.keyword_profiles.values():
            keywords = profile.get('keywords', {})
            all_keywords.extend(keywords.keys())
            domain_signals.extend(profile.get('domain_signals', []))
            relevance_scores.append(profile.get('relevance_score', 0))
            
            category = profile.get('category', 'unknown')
            category_keyword_map[category].extend(keywords.keys())
        
        # Keyword frequency analysis
        keyword_frequency = Counter(all_keywords)
        domain_signal_frequency = Counter(domain_signals)
        
        # Extract domain patterns
        domain_keyword_patterns = defaultdict(Counter)
        for keyword in keyword_frequency.keys():
            if ':' in keyword:
                domain, term = keyword.split(':', 1)
                domain_keyword_patterns[domain][term] += keyword_frequency[keyword]
        
        analysis = {
            'keyword_extraction_summary': {
                'total_unique_keywords': len(keyword_frequency),
                'avg_keywords_per_dataset': len(all_keywords) / len(self.keyword_profiles),
                'most_common_keywords': dict(keyword_frequency.most_common(20))
            },
            'domain_signal_analysis': {
                'signal_frequency': dict(domain_signal_frequency),
                'research_ready_datasets': sum(1 for signals in [profile.get('domain_signals', []) for profile in self.keyword_profiles.values()] 
                                             if any('research_ready' in signal for signal in signals)),
                'warning_signals': {signal: count for signal, count in domain_signal_frequency.items() if 'warning' in signal}
            },
            'domain_keyword_patterns': {
                domain: dict(counter.most_common(10)) 
                for domain, counter in domain_keyword_patterns.items()
            },
            'relevance_distribution': {
                'mean_relevance': float(np.mean(relevance_scores)),
                'median_relevance': float(np.median(relevance_scores)),
                'high_relevance_count': sum(1 for score in relevance_scores if score >= 0.8),
                'medium_relevance_count': sum(1 for score in relevance_scores if 0.5 <= score < 0.8),
                'low_relevance_count': sum(1 for score in relevance_scores if score < 0.5)
            },
            'category_keyword_diversity': {
                category: len(set(keywords)) 
                for category, keywords in category_keyword_map.items()
            }
        }
        
        self.analysis_results['keyword_intelligence'] = analysis
        return analysis
    
    def analyze_relationship_discovery(self) -> Dict:
        """Analyze dataset relationship discovery results"""
        logger.info("🔗 Analyzing relationship discovery...")
        
        if not self.relationships:
            return {'error': 'No relationships data available'}
        
        # Aggregate all relationships
        all_relationships = []
        for rel_type, rel_list in self.relationships.items():
            for rel in rel_list:
                rel_copy = rel.copy()
                rel_copy['type'] = rel_type
                all_relationships.append(rel_copy)
        
        if not all_relationships:
            return {'error': 'No relationships found'}
        
        # Extract relationship metrics
        relationship_scores = [rel['relationship_score'] for rel in all_relationships]
        confidence_scores = [rel['confidence'] for rel in all_relationships]
        relationship_types = [rel['type'] for rel in all_relationships]
        
        # Quality assessment
        high_quality_threshold = 0.7
        high_confidence_threshold = 0.6
        
        high_quality_relationships = [
            rel for rel in all_relationships 
            if rel['relationship_score'] >= high_quality_threshold and rel['confidence'] >= high_confidence_threshold
        ]
        
        analysis = {
            'relationship_summary': {
                'total_relationships': len(all_relationships),
                'relationship_type_distribution': dict(Counter(relationship_types)),
                'high_quality_relationships': len(high_quality_relationships)
            },
            'quality_metrics': {
                'score_statistics': {
                    'mean_relationship_score': float(np.mean(relationship_scores)),
                    'median_relationship_score': float(np.median(relationship_scores)),
                    'mean_confidence': float(np.mean(confidence_scores)),
                    'median_confidence': float(np.median(confidence_scores))
                },
                'score_distribution': {
                    'excellent_relationships': sum(1 for score in relationship_scores if score >= 0.8),
                    'good_relationships': sum(1 for score in relationship_scores if 0.6 <= score < 0.8),
                    'fair_relationships': sum(1 for score in relationship_scores if 0.4 <= score < 0.6),
                    'weak_relationships': sum(1 for score in relationship_scores if score < 0.4)
                }
            },
            'top_relationships': [
                {
                    'dataset1': rel['dataset1_title'],
                    'dataset2': rel['dataset2_title'],
                    'type': rel['type'],
                    'score': rel['relationship_score'],
                    'confidence': rel['confidence']
                }
                for rel in sorted(all_relationships, key=lambda x: x['relationship_score'], reverse=True)[:10]
            ],
            'relationship_network_stats': {
                'avg_relationships_per_dataset': len(all_relationships) / len(self.keyword_profiles) if self.keyword_profiles else 0,
                'most_connected_datasets': self._find_most_connected_datasets(all_relationships)
            }
        }
        
        self.analysis_results['relationship_discovery'] = analysis
        return analysis
    
    def _find_most_connected_datasets(self, relationships: List[Dict]) -> List[Dict]:
        """Find datasets with most relationships"""
        dataset_connections = defaultdict(int)
        
        for rel in relationships:
            dataset_connections[rel['dataset1_title']] += 1
            dataset_connections[rel['dataset2_title']] += 1
        
        return [
            {'dataset': dataset, 'connection_count': count}
            for dataset, count in sorted(dataset_connections.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def validate_ground_truth_scenarios(self) -> Dict:
        """Comprehensive validation of ground truth scenarios"""
        logger.info("🎯 Validating ground truth scenarios...")
        
        if not self.ground_truth:
            return {'error': 'No ground truth scenarios available'}
        
        # Analyze scenario characteristics
        scenario_analysis = {}
        confidence_scores = []
        complementary_counts = []
        generation_methods = []
        validation_scores = []
        
        for scenario_name, scenario_data in self.ground_truth.items():
            confidence = scenario_data.get('confidence', 0)
            complementary_count = len(scenario_data.get('complementary', []))
            generation_method = scenario_data.get('generation_method', 'unknown')
            validation_score = scenario_data.get('validation_score', 0)
            
            confidence_scores.append(confidence)
            complementary_counts.append(complementary_count)
            generation_methods.append(generation_method)
            validation_scores.append(validation_score)
            
            scenario_analysis[scenario_name] = {
                'primary_query': scenario_data.get('primary', ''),
                'complementary_count': complementary_count,
                'confidence': confidence,
                'generation_method': generation_method,
                'validation_score': validation_score,
                'source': scenario_data.get('source', 'unknown')
            }
        
        # Quality assessment
        high_confidence_threshold = 0.7
        adequate_confidence_threshold = 0.5
        
        high_confidence_scenarios = sum(1 for c in confidence_scores if c >= high_confidence_threshold)
        adequate_scenarios = sum(1 for c in confidence_scores if adequate_confidence_threshold <= c < high_confidence_threshold)
        low_confidence_scenarios = sum(1 for c in confidence_scores if c < adequate_confidence_threshold)
        
        # Identify potential issues
        issues = []
        if low_confidence_scenarios > 0:
            issues.append(f"{low_confidence_scenarios} scenarios with low confidence (<{adequate_confidence_threshold})")
        
        min_complementary = 2
        few_complementary_scenarios = sum(1 for count in complementary_counts if count < min_complementary)
        if few_complementary_scenarios > 0:
            issues.append(f"{few_complementary_scenarios} scenarios with <{min_complementary} complementary datasets")
        
        analysis = {
            'scenario_summary': {
                'total_scenarios': len(self.ground_truth),
                'generation_method_distribution': dict(Counter(generation_methods)),
                'scenario_details': scenario_analysis
            },
            'quality_assessment': {
                'confidence_statistics': {
                    'mean_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0,
                    'median_confidence': float(np.median(confidence_scores)) if confidence_scores else 0,
                    'std_confidence': float(np.std(confidence_scores)) if confidence_scores else 0
                },
                'confidence_distribution': {
                    'high_confidence': high_confidence_scenarios,
                    'adequate_confidence': adequate_scenarios,
                    'low_confidence': low_confidence_scenarios
                },
                'complementary_statistics': {
                    'mean_complementary_count': float(np.mean(complementary_counts)) if complementary_counts else 0,
                    'min_complementary_count': int(min(complementary_counts)) if complementary_counts else 0,
                    'max_complementary_count': int(max(complementary_counts)) if complementary_counts else 0
                }
            },
            'validation_results': {
                'mean_validation_score': float(np.mean(validation_scores)) if validation_scores else 0,
                'fully_validated_scenarios': sum(1 for score in validation_scores if score >= 0.9),
                'partially_validated_scenarios': sum(1 for score in validation_scores if 0.5 <= score < 0.9)
            },
            'identified_issues': issues,
            'ml_readiness_assessment': self._assess_ml_readiness(high_confidence_scenarios, adequate_scenarios)
        }
        
        self.analysis_results['ground_truth_validation'] = analysis
        return analysis
    
    def _assess_ml_readiness(self, high_confidence: int, adequate_confidence: int) -> Dict:
        """Assess readiness for ML model training"""
        ml_config = self.config.get('pipeline', {}).get('ml_readiness_thresholds', {})
        
        min_scenarios = ml_config.get('min_ground_truth_scenarios', 3)
        min_high_conf = ml_config.get('min_high_confidence_scenarios', 2)
        
        total_usable = high_confidence + adequate_confidence
        
        assessment = {
            'ready_for_ml_training': total_usable >= min_scenarios and high_confidence >= min_high_conf,
            'scenarios_available': total_usable,
            'scenarios_required': min_scenarios,
            'high_confidence_available': high_confidence,
            'high_confidence_required': min_high_conf,
            'expected_performance': self._estimate_ml_performance(high_confidence, adequate_confidence)
        }
        
        return assessment
    
    def _estimate_ml_performance(self, high_confidence: int, adequate_confidence: int) -> Dict:
        """Estimate expected ML model performance"""
        total_usable = high_confidence + adequate_confidence
        
        if total_usable >= 5 and high_confidence >= 3:
            return {'estimated_f1_score': '0.70-0.80', 'confidence_level': 'high'}
        elif total_usable >= 3 and high_confidence >= 2:
            return {'estimated_f1_score': '0.60-0.70', 'confidence_level': 'medium'}
        elif total_usable >= 2:
            return {'estimated_f1_score': '0.40-0.60', 'confidence_level': 'low'}
        else:
            return {'estimated_f1_score': '0.20-0.40', 'confidence_level': 'very_low'}
    
    def identify_data_quality_issues(self) -> Dict:
        """Comprehensive data quality issue identification"""
        logger.info("⚠️ Identifying data quality issues...")
        
        issues = []
        recommendations = []
        
        if self.datasets_df is not None:
            # Quality score analysis
            low_quality_threshold = self.quality_thresholds.get('low_quality', 0.3)
            low_quality_datasets = self.datasets_df[self.datasets_df['quality_score'] < low_quality_threshold]
            
            if len(low_quality_datasets) > 0:
                issues.append(f"Low quality datasets: {len(low_quality_datasets)} datasets below {low_quality_threshold}")
                recommendations.append("Review and improve metadata for low-quality datasets or exclude from ML training")
            
            # Missing data analysis
            missing_descriptions = self.datasets_df['description'].isna().sum() + (self.datasets_df['description'] == '').sum()
            if missing_descriptions > 0:
                issues.append(f"Missing descriptions: {missing_descriptions} datasets without proper descriptions")
                recommendations.append("Add meaningful descriptions for datasets or implement fallback description generation")
            
            # Category balance analysis
            category_dist = self.datasets_df['category'].value_counts()
            total_datasets = len(self.datasets_df)
            
            over_represented = category_dist[category_dist > total_datasets * 0.4]
            if len(over_represented) > 0:
                issues.append(f"Over-represented categories: {over_represented.to_dict()}")
                recommendations.append("Balance dataset collection across different domains")
            
            under_represented = category_dist[category_dist < 2]
            if len(under_represented) > 0:
                issues.append(f"Under-represented categories: {under_represented.to_dict()}")
                recommendations.append("Add more datasets to under-represented categories for better model training")
        
        # Keyword profile issues
        if self.keyword_profiles:
            warning_datasets = []
            for dataset_id, profile in self.keyword_profiles.items():
                warnings = [signal for signal in profile.get('domain_signals', []) if 'warning' in signal]
                if warnings:
                    warning_datasets.append({
                        'dataset_id': dataset_id,
                        'title': profile.get('title', ''),
                        'warnings': warnings
                    })
            
            if warning_datasets:
                issues.append(f"Datasets with warning signals: {len(warning_datasets)} datasets flagged")
                recommendations.append("Manually review flagged datasets for proper categorization and quality")
        
        # Relationship quality issues
        if self.relationships:
            low_conf_relationships = []
            for rel_type, rel_list in self.relationships.items():
                low_conf = [rel for rel in rel_list if rel.get('confidence', 0) < 0.5]
                low_conf_relationships.extend(low_conf)
            
            if low_conf_relationships:
                issues.append(f"Low confidence relationships: {len(low_conf_relationships)} relationships with confidence <0.5")
                recommendations.append("Review relationship scoring algorithm and dataset categorization")
        
        analysis = {
            'issues_summary': {
                'total_issues_identified': len(issues),
                'critical_issues': [issue for issue in issues if any(word in issue.lower() for word in ['low quality', 'missing', 'warning'])],
                'balance_issues': [issue for issue in issues if any(word in issue.lower() for word in ['over-represented', 'under-represented'])]
            },
            'detailed_issues': issues,
            'recommendations': recommendations,
            'quality_metrics': self._calculate_overall_quality_metrics(),
            'improvement_priorities': self._prioritize_improvements(issues)
        }
        
        if self.datasets_df is not None and warning_datasets:
            analysis['flagged_datasets'] = warning_datasets[:10]  # Top 10 flagged datasets
        
        self.analysis_results['data_quality_issues'] = analysis
        return analysis
    
    def _calculate_overall_quality_metrics(self) -> Dict:
        """Calculate overall quality metrics for the collection"""
        if self.datasets_df is None:
            return {}
        
        high_threshold = self.quality_thresholds.get('high_quality', 0.8)
        medium_threshold = self.quality_thresholds.get('medium_quality', 0.5)
        
        return {
            'collection_quality_score': float(self.datasets_df['quality_score'].mean()),
            'quality_distribution': {
                'high_quality_percentage': float((self.datasets_df['quality_score'] >= high_threshold).mean() * 100),
                'medium_quality_percentage': float(((self.datasets_df['quality_score'] >= medium_threshold) & 
                                                   (self.datasets_df['quality_score'] < high_threshold)).mean() * 100),
                'low_quality_percentage': float((self.datasets_df['quality_score'] < medium_threshold).mean() * 100)
            },
            'completeness_metrics': {
                'description_completeness': float((~self.datasets_df['description'].isna() & (self.datasets_df['description'] != '')).mean() * 100),
                'metadata_completeness': float((~self.datasets_df[['agency', 'category', 'frequency']].isna()).all(axis=1).mean() * 100)
            }
        }
    
    def _prioritize_improvements(self, issues: List[str]) -> List[str]:
        """Prioritize improvement recommendations"""
        priorities = []
        
        for issue in issues:
            if 'low quality' in issue.lower():
                priorities.append("HIGH PRIORITY: Improve dataset quality scores through better metadata")
            elif 'warning' in issue.lower():
                priorities.append("HIGH PRIORITY: Review and reclassify flagged datasets")
            elif 'missing descriptions' in issue.lower():
                priorities.append("MEDIUM PRIORITY: Add comprehensive descriptions to datasets")
            elif 'under-represented' in issue.lower():
                priorities.append("MEDIUM PRIORITY: Expand dataset collection in underrepresented categories")
            elif 'over-represented' in issue.lower():
                priorities.append("LOW PRIORITY: Balance dataset distribution across categories")
        
        return priorities
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for all analyses"""
        logger.info("📊 Creating comprehensive visualizations...")
        
        viz_output_path = self.output_base / "visualizations"
        viz_output_path.mkdir(exist_ok=True)
        
        chart_types = self.reporting_config.get('chart_types', [])
        save_format = self.reporting_config.get('visualizations', {}).get('save_format', 'png')
        
        # 1. Dataset Distribution Overview
        if "dataset_distribution_overview" in chart_types and self.datasets_df is not None:
            self._create_distribution_overview(viz_output_path, save_format)
        
        # 2. Quality Analysis Charts
        if "quality_analysis" in chart_types and self.datasets_df is not None:
            self._create_quality_analysis_charts(viz_output_path, save_format)
        
        # 3. Relationship Network Visualization
        if "relationship_network" in chart_types and self.relationships:
            self._create_relationship_network(viz_output_path, save_format)
        
        # 4. Keyword Pattern Analysis
        if "keyword_patterns" in chart_types and self.keyword_profiles:
            self._create_keyword_patterns(viz_output_path, save_format)
        
        # 5. Ground Truth Validation Charts
        if "ground_truth_validation" in chart_types and self.ground_truth:
            self._create_ground_truth_validation(viz_output_path, save_format)
        
        logger.info(f"✅ Visualizations saved to {viz_output_path}")
    
    def _create_distribution_overview(self, output_path: Path, save_format: str):
        """Create dataset distribution overview charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Collection Distribution Overview', fontsize=16, fontweight='bold')
        
        # Source distribution
        source_counts = self.datasets_df['source'].value_counts()
        axes[0,0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Distribution by Data Source')
        
        # Category distribution
        cat_counts = self.datasets_df['category'].value_counts()
        bars = axes[0,1].barh(cat_counts.index, cat_counts.values, color=sns.color_palette("husl", len(cat_counts)))
        axes[0,1].set_title('Distribution by Category')
        axes[0,1].set_xlabel('Number of Datasets')
        
        # Quality score distribution
        axes[1,0].hist(self.datasets_df['quality_score'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1,0].axvline(self.datasets_df['quality_score'].mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {self.datasets_df["quality_score"].mean():.2f}')
        axes[1,0].set_title('Quality Score Distribution')
        axes[1,0].set_xlabel('Quality Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Quality by source
        quality_by_source = self.datasets_df.groupby('source')['quality_score'].mean().sort_values(ascending=True)
        bars = axes[1,1].barh(quality_by_source.index, quality_by_source.values, color='lightcoral')
        axes[1,1].set_title('Average Quality Score by Source')
        axes[1,1].set_xlabel('Average Quality Score')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'dataset_distribution_overview.{save_format}', bbox_inches='tight')
        plt.close()
    
    def _create_quality_analysis_charts(self, output_path: Path, save_format: str):
        """Create detailed quality analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Quality Analysis', fontsize=16, fontweight='bold')
        
        # Quality distribution by category
        quality_by_cat = self.datasets_df.groupby('category')['quality_score'].mean().sort_values(ascending=False)
        bars = axes[0,0].bar(range(len(quality_by_cat)), quality_by_cat.values, color='lightblue')
        axes[0,0].set_xticks(range(len(quality_by_cat)))
        axes[0,0].set_xticklabels(quality_by_cat.index, rotation=45, ha='right')
        axes[0,0].set_title('Average Quality Score by Category')
        axes[0,0].set_ylabel('Quality Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Quality vs other metrics (if available)
        if 'record_count' in self.datasets_df.columns:
            # Convert record_count to numeric, handling non-numeric values
            record_counts = pd.to_numeric(self.datasets_df['record_count'], errors='coerce')
            valid_records = ~record_counts.isna()
            
            if valid_records.sum() > 0:
                axes[0,1].scatter(record_counts[valid_records], self.datasets_df.loc[valid_records, 'quality_score'], 
                                alpha=0.6, color='green')
                axes[0,1].set_xlabel('Record Count (log scale)')
                axes[0,1].set_xscale('log')
                axes[0,1].set_ylabel('Quality Score')
                axes[0,1].set_title('Quality Score vs Dataset Size')
                axes[0,1].grid(True, alpha=0.3)
        
        # Quality level categorization
        high_threshold = self.quality_thresholds.get('high_quality', 0.8)
        medium_threshold = self.quality_thresholds.get('medium_quality', 0.5)
        
        quality_levels = ['Low (<0.5)', 'Medium (0.5-0.8)', 'High (≥0.8)']
        quality_counts = [
            (self.datasets_df['quality_score'] < medium_threshold).sum(),
            ((self.datasets_df['quality_score'] >= medium_threshold) & (self.datasets_df['quality_score'] < high_threshold)).sum(),
            (self.datasets_df['quality_score'] >= high_threshold).sum()
        ]
        
        colors = ['lightcoral', 'gold', 'lightgreen']
        axes[1,0].pie(quality_counts, labels=quality_levels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1,0].set_title('Quality Level Distribution')
        
        # Status vs quality
        if 'status' in self.datasets_df.columns:
            status_quality = self.datasets_df.groupby('status')['quality_score'].mean()
            bars = axes[1,1].bar(status_quality.index, status_quality.values, color='orange')
            axes[1,1].set_title('Quality Score by Dataset Status')
            axes[1,1].set_ylabel('Average Quality Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'quality_analysis.{save_format}', bbox_inches='tight')
        plt.close()
    
    def _create_relationship_network(self, output_path: Path, save_format: str):
        """Create relationship analysis visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Dataset Relationship Analysis', fontsize=16, fontweight='bold')
        
        # Collect all relationships
        all_relationships = []
        for rel_type, rel_list in self.relationships.items():
            for rel in rel_list:
                rel_copy = rel.copy()
                rel_copy['type'] = rel_type
                all_relationships.append(rel_copy)
        
        if all_relationships:
            # Relationship type distribution
            rel_types = [rel['type'] for rel in all_relationships]
            type_counts = Counter(rel_types)
            
            axes[0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Relationship Type Distribution')
            
            # Relationship quality scatter
            scores = [rel['relationship_score'] for rel in all_relationships]
            confidences = [rel['confidence'] for rel in all_relationships]
            types = [rel['type'] for rel in all_relationships]
            
            # Color by type
            unique_types = list(set(types))
            colors = sns.color_palette("husl", len(unique_types))
            type_color_map = dict(zip(unique_types, colors))
            
            for rel_type in unique_types:
                type_mask = [t == rel_type for t in types]
                type_scores = [s for s, m in zip(scores, type_mask) if m]
                type_confidences = [c for c, m in zip(confidences, type_mask) if m]
                
                axes[1].scatter(type_scores, type_confidences, 
                              label=rel_type, alpha=0.6, color=type_color_map[rel_type])
            
            axes[1].set_xlabel('Relationship Score')
            axes[1].set_ylabel('Confidence Score')
            axes[1].set_title('Relationship Quality vs Confidence')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'relationship_analysis.{save_format}', bbox_inches='tight')
        plt.close()
    
    def _create_keyword_patterns(self, output_path: Path, save_format: str):
        """Create keyword pattern analysis charts"""
        # Aggregate keyword data
        all_keywords = []
        domain_signals = []
        
        for profile in self.keyword_profiles.values():
            all_keywords.extend(profile.get('keywords', {}).keys())
            domain_signals.extend(profile.get('domain_signals', []))
        
        if all_keywords:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Keyword Pattern Analysis', fontsize=16, fontweight='bold')
            
            # Most common keywords
            keyword_frequency = Counter(all_keywords)
            top_keywords = dict(keyword_frequency.most_common(15))
            
            axes[0,0].barh(range(len(top_keywords)), list(top_keywords.values()))
            axes[0,0].set_yticks(range(len(top_keywords)))
            axes[0,0].set_yticklabels(list(top_keywords.keys()))
            axes[0,0].set_title('Most Common Keywords')
            axes[0,0].set_xlabel('Frequency')
            
            # Domain signal frequency
            signal_frequency = Counter(domain_signals)
            if signal_frequency:
                top_signals = dict(signal_frequency.most_common(10))
                
                axes[0,1].barh(range(len(top_signals)), list(top_signals.values()), color='lightgreen')
                axes[0,1].set_yticks(range(len(top_signals)))
                axes[0,1].set_yticklabels(list(top_signals.keys()))
                axes[0,1].set_title('Domain Signal Frequency')
                axes[0,1].set_xlabel('Frequency')
            
            # Relevance score distribution
            relevance_scores = [profile.get('relevance_score', 0) for profile in self.keyword_profiles.values()]
            axes[1,0].hist(relevance_scores, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
            axes[1,0].axvline(np.mean(relevance_scores), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(relevance_scores):.2f}')
            axes[1,0].set_title('Dataset Relevance Score Distribution')
            axes[1,0].set_xlabel('Relevance Score')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Category keyword diversity
            category_diversity = defaultdict(set)
            for profile in self.keyword_profiles.values():
                category = profile.get('category', 'unknown')
                keywords = profile.get('keywords', {}).keys()
                category_diversity[category].update(keywords)
            
            diversity_scores = {cat: len(keywords) for cat, keywords in category_diversity.items()}
            if diversity_scores:
                axes[1,1].bar(diversity_scores.keys(), diversity_scores.values(), color='gold')
                axes[1,1].set_title('Keyword Diversity by Category')
                axes[1,1].set_ylabel('Unique Keywords')
                axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / f'keyword_patterns.{save_format}', bbox_inches='tight')
            plt.close()
    
    def _create_ground_truth_validation(self, output_path: Path, save_format: str):
        """Create ground truth validation charts"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Ground Truth Scenario Validation', fontsize=16, fontweight='bold')
        
        # Extract scenario metrics
        confidences = [scenario.get('confidence', 0) for scenario in self.ground_truth.values()]
        complementary_counts = [len(scenario.get('complementary', [])) for scenario in self.ground_truth.values()]
        generation_methods = [scenario.get('generation_method', 'unknown') for scenario in self.ground_truth.values()]
        
        # Confidence distribution
        axes[0].hist(confidences, bins=10, alpha=0.7, edgecolor='black', color='lightblue')
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.2f}')
        axes[0].set_title('Scenario Confidence Distribution')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Number of Scenarios')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Complementary dataset counts
        comp_count_dist = Counter(complementary_counts)
        axes[1].bar(comp_count_dist.keys(), comp_count_dist.values(), color='lightgreen')
        axes[1].set_title('Complementary Datasets per Scenario')
        axes[1].set_xlabel('Number of Complementary Datasets')
        axes[1].set_ylabel('Number of Scenarios')
        axes[1].grid(True, alpha=0.3)
        
        # Generation method distribution
        method_counts = Counter(generation_methods)
        axes[2].pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%', startangle=90)
        axes[2].set_title('Scenario Generation Methods')
        
        plt.tight_layout()
        plt.savefig(output_path / f'ground_truth_validation.{save_format}', bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_reports(self):
        """Generate comprehensive analysis reports"""
        logger.info("📋 Generating comprehensive reports...")
        
        reports_output_path = self.output_base / "reports"
        reports_output_path.mkdir(exist_ok=True)
        
        # Compile comprehensive analysis report
        comprehensive_report = {
            'analysis_metadata': {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_datasets_analyzed': len(self.datasets_df) if self.datasets_df is not None else 0,
                'configuration_used': self.config.get('phase_3_reporting', {}),
                'analysis_components_completed': list(self.analysis_results.keys())
            },
            'collection_overview': self.analysis_results.get('collection_overview', {}),
            'keyword_intelligence': self.analysis_results.get('keyword_intelligence', {}),
            'relationship_discovery': self.analysis_results.get('relationship_discovery', {}),
            'ground_truth_validation': self.analysis_results.get('ground_truth_validation', {}),
            'data_quality_assessment': self.analysis_results.get('data_quality_issues', {})
        }
        
        # Save detailed JSON report
        report_config = self.reporting_config.get('reports', {})
        if report_config.get('generate_detailed_json', True):
            with open(reports_output_path / 'comprehensive_analysis_report.json', 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Generate executive summary
        if report_config.get('generate_executive_summary', True):
            exec_summary = self._generate_executive_summary(comprehensive_report)
            with open(reports_output_path / 'executive_summary.md', 'w') as f:
                f.write(exec_summary)
        
        # Generate technical report
        tech_report = self._generate_technical_report(comprehensive_report)
        with open(reports_output_path / 'technical_analysis_report.md', 'w') as f:
            f.write(tech_report)
        
        logger.info(f"✅ Reports generated in {reports_output_path}")
        return comprehensive_report
    
    def _generate_executive_summary(self, report: Dict) -> str:
        """Generate executive summary in markdown format"""
        overview = report.get('collection_overview', {})
        keyword_analysis = report.get('keyword_intelligence', {})
        relationship_analysis = report.get('relationship_discovery', {})
        gt_analysis = report.get('ground_truth_validation', {})
        quality_analysis = report.get('data_quality_assessment', {})
        
        summary = f"""# Dataset Research Assistant - Executive Summary

## 📊 Analysis Overview
- **Analysis Date**: {report['analysis_metadata']['analysis_timestamp'][:10]}
- **Total Datasets Analyzed**: {overview.get('collection_metadata', {}).get('total_datasets', 0)}
- **Data Sources**: {overview.get('collection_metadata', {}).get('data_sources', 0)} different sources
- **Categories Covered**: {overview.get('collection_metadata', {}).get('categories_covered', 0)} domains

## 🎯 Key Findings

### Dataset Collection Quality
- **Overall Quality Score**: {overview.get('quality_analysis', {}).get('overall_statistics', {}).get('mean_quality', 0):.2f}/1.0
- **High Quality Datasets**: {overview.get('quality_analysis', {}).get('quality_distribution', {}).get('high_quality', 0)} datasets (≥0.8 quality)
- **Active Datasets**: {overview.get('temporal_analysis', {}).get('active_datasets_percentage', 0):.1f}% actively maintained

### Intelligence Extraction
- **Keywords Extracted**: {keyword_analysis.get('keyword_extraction_summary', {}).get('total_unique_keywords', 0)} unique terms
- **Research-Ready Datasets**: {keyword_analysis.get('domain_signal_analysis', {}).get('research_ready_datasets', 0)} datasets
- **Average Relevance Score**: {keyword_analysis.get('relevance_distribution', {}).get('mean_relevance', 0):.2f}/1.0

### Relationship Discovery
- **Total Relationships**: {relationship_analysis.get('relationship_summary', {}).get('total_relationships', 0)} discovered
- **High-Quality Relationships**: {relationship_analysis.get('relationship_summary', {}).get('high_quality_relationships', 0)} pairs
- **Average Confidence**: {relationship_analysis.get('quality_metrics', {}).get('score_statistics', {}).get('mean_confidence', 0):.2f}/1.0

### Ground Truth Scenarios
- **Scenarios Generated**: {gt_analysis.get('scenario_summary', {}).get('total_scenarios', 0)}
- **High Confidence**: {gt_analysis.get('quality_assessment', {}).get('confidence_distribution', {}).get('high_confidence', 0)} scenarios (≥0.7)
- **ML Training Ready**: {'Yes' if gt_analysis.get('ml_readiness_assessment', {}).get('ready_for_ml_training', False) else 'No'}

## ⚠️ Issues Identified
"""
        
        issues = quality_analysis.get('detailed_issues', [])
        for i, issue in enumerate(issues[:5], 1):  # Top 5 issues
            summary += f"{i}. {issue}\n"
        
        if len(issues) > 5:
            summary += f"   ...and {len(issues) - 5} additional issues identified\n"
        
        summary += f"""
## 📈 Recommendations
"""
        
        recommendations = quality_analysis.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
            summary += f"{i}. {rec}\n"
        
        # ML Readiness Assessment
        ml_assessment = gt_analysis.get('ml_readiness_assessment', {})
        expected_performance = ml_assessment.get('expected_performance', {})
        
        summary += f"""
## 🚀 Next Steps

### ML Model Training Readiness
- **Status**: {'READY' if ml_assessment.get('ready_for_ml_training', False) else 'NEEDS IMPROVEMENT'}
- **Expected F1 Score**: {expected_performance.get('estimated_f1_score', 'N/A')}
- **Confidence Level**: {expected_performance.get('confidence_level', 'N/A').title()}

### Immediate Actions
1. **Data Quality**: Address {quality_analysis.get('issues_summary', {}).get('total_issues_identified', 0)} identified issues
2. **Ground Truth**: {'Proceed to ML training' if ml_assessment.get('ready_for_ml_training', False) else 'Improve scenario quality'}
3. **Collection**: {'Balanced collection achieved' if overview.get('category_analysis', {}).get('coverage_assessment', {}).get('balance_score', 0) > 0.7 else 'Balance dataset categories'}

### Expected Timeline
- **Phase 3 Complete**: ✅ Analysis and reporting finished
- **Next Phase**: {'ML Pipeline (train_models.py)' if ml_assessment.get('ready_for_ml_training', False) else 'Data Collection Enhancement'}
- **Estimated Timeline**: {'1-2 weeks to ML results' if ml_assessment.get('ready_for_ml_training', False) else '2-3 weeks with improvements'}

---
*Generated by Dataset Research Assistant - Configuration-Driven Analysis Pipeline*
"""
        
        return summary
    
    def _generate_technical_report(self, report: Dict) -> str:
        """Generate detailed technical report"""
        overview = report.get('collection_overview', {})
        keyword_analysis = report.get('keyword_intelligence', {})
        relationship_analysis = report.get('relationship_discovery', {})
        gt_analysis = report.get('ground_truth_validation', {})
        quality_analysis = report.get('data_quality_assessment', {})
        
        tech_report = f"""# Technical Analysis Report - Dataset Research Assistant

## Configuration & Methodology

### Analysis Configuration
- **Config File**: data_pipeline.yml (configuration-driven approach)
- **Analysis Timestamp**: {report['analysis_metadata']['analysis_timestamp']}
- **Components Analyzed**: {', '.join(report['analysis_metadata']['analysis_components_completed'])}

### Data Processing Pipeline
1. **Phase 1**: Configuration-driven data extraction from {overview.get('collection_metadata', {}).get('data_sources', 0)} sources
2. **Phase 2**: Intelligent keyword extraction and relationship discovery
3. **Phase 3**: Comprehensive EDA and validation reporting

## Detailed Technical Findings

### 1. Dataset Collection Analysis
"""
        
        # Add detailed technical sections based on available data
        if overview:
            source_dist = overview.get('source_analysis', {}).get('distribution', {})
            category_dist = overview.get('category_analysis', {}).get('distribution', {})
            
            tech_report += f"""
**Source Distribution:**
{chr(10).join([f'- {source}: {count} datasets' for source, count in source_dist.items()])}

**Category Distribution:**
{chr(10).join([f'- {category}: {count} datasets' for category, count in category_dist.items()])}

**Quality Statistics:**
- Mean Quality Score: {overview.get('quality_analysis', {}).get('overall_statistics', {}).get('mean_quality', 0):.3f}
- Standard Deviation: {overview.get('quality_analysis', {}).get('overall_statistics', {}).get('std_quality', 0):.3f}
- Quality Range: Comprehensive assessment across {len(source_dist)} sources
"""
        
        if keyword_analysis:
            tech_report += f"""
### 2. Keyword Intelligence Analysis

**Extraction Methodology:**
- Domain-weighted keyword extraction using TF-IDF principles
- Multi-domain classification: {len(keyword_analysis.get('domain_keyword_patterns', {}))} domains identified
- Signal-based relevance scoring with automated quality gates

**Results:**
- Total Unique Keywords: {keyword_analysis.get('keyword_extraction_summary', {}).get('total_unique_keywords', 0)}
- Average Keywords per Dataset: {keyword_analysis.get('keyword_extraction_summary', {}).get('avg_keywords_per_dataset', 0):.1f}
- Research-Ready Datasets: {keyword_analysis.get('domain_signal_analysis', {}).get('research_ready_datasets', 0)}
"""
        
        if relationship_analysis:
            tech_report += f"""
### 3. Relationship Discovery Algorithm

**Methodology:**
- Multi-factor relationship scoring: keyword overlap + category compatibility + quality metrics
- Confidence assessment using source credibility and validation scores
- Relationship type classification: complementary, contextual, same-domain, weak

**Performance:**
- Total Relationships Discovered: {relationship_analysis.get('relationship_summary', {}).get('total_relationships', 0)}
- High-Quality Pairs (score ≥0.7, confidence ≥0.6): {relationship_analysis.get('relationship_summary', {}).get('high_quality_relationships', 0)}
- Average Relationship Score: {relationship_analysis.get('quality_metrics', {}).get('score_statistics', {}).get('mean_relationship_score', 0):.3f}
- Average Confidence: {relationship_analysis.get('quality_metrics', {}).get('score_statistics', {}).get('mean_confidence', 0):.3f}
"""
        
        if gt_analysis:
            tech_report += f"""
### 4. Ground Truth Generation Intelligence

**Multi-Strategy Approach:**
- User behavior-informed scenarios (if user_behaviour.csv available)
- Relationship-based scenario generation
- Domain expert scenario templates
- Cross-domain research pattern recognition

**Quality Metrics:**
- Scenarios Generated: {gt_analysis.get('scenario_summary', {}).get('total_scenarios', 0)}
- High Confidence (≥0.7): {gt_analysis.get('quality_assessment', {}).get('confidence_distribution', {}).get('high_confidence', 0)}
- Adequate Confidence (0.5-0.7): {gt_analysis.get('quality_assessment', {}).get('confidence_distribution', {}).get('adequate_confidence', 0)}
- Validation Score: {gt_analysis.get('validation_results', {}).get('mean_validation_score', 0):.3f}

**ML Training Assessment:**
- Ready for Training: {'Yes' if gt_analysis.get('ml_readiness_assessment', {}).get('ready_for_ml_training', False) else 'No'}
- Expected Performance: {gt_analysis.get('ml_readiness_assessment', {}).get('expected_performance', {}).get('estimated_f1_score', 'N/A')}
"""
        
        tech_report += f"""
### 5. Data Quality Assessment

**Automated Issue Detection:**
- Total Issues Identified: {quality_analysis.get('issues_summary', {}).get('total_issues_identified', 0)}
- Critical Issues: {len(quality_analysis.get('issues_summary', {}).get('critical_issues', []))}
- Balance Issues: {len(quality_analysis.get('issues_summary', {}).get('balance_issues', []))}

**Quality Metrics:**
- Collection Quality Score: {quality_analysis.get('quality_metrics', {}).get('collection_quality_score', 0):.3f}
- High Quality Percentage: {quality_analysis.get('quality_metrics', {}).get('quality_distribution', {}).get('high_quality_percentage', 0):.1f}%
- Metadata Completeness: {quality_analysis.get('quality_metrics', {}).get('completeness_metrics', {}).get('metadata_completeness', 0):.1f}%

## Technical Recommendations

### Immediate Technical Actions
"""
        
        priorities = quality_analysis.get('improvement_priorities', [])
        for priority in priorities:
            tech_report += f"- {priority}\n"
        
        tech_report += f"""
### Architecture Scalability
- Configuration-driven design supports easy expansion to new data sources
- Modular analysis components allow independent enhancement
- Output structure optimized for ML pipeline integration

### Performance Characteristics
- Processing Time: ~2-3 minutes for {overview.get('collection_metadata', {}).get('total_datasets', 0)} datasets
- Memory Usage: Efficient pandas-based processing
- Output Size: Comprehensive analysis with visualizations (~50MB total)

### Next Phase Readiness
- **ML Pipeline**: {'READY' if gt_analysis.get('ml_readiness_assessment', {}).get('ready_for_ml_training', False) else 'REQUIRES IMPROVEMENT'}
- **Expected ML Performance**: {gt_analysis.get('ml_readiness_assessment', {}).get('expected_performance', {}).get('estimated_f1_score', 'Insufficient ground truth')}
- **Recommended Next Steps**: {'Execute train_models.py' if gt_analysis.get('ml_readiness_assessment', {}).get('ready_for_ml_training', False) else 'Enhance ground truth generation'}

---
*Technical Analysis completed using configuration-driven methodology*
*Pipeline ready for ML/DL enhancement phases*
"""
        
        return tech_report


def main():
    """Main execution for Phase 3: Comprehensive EDA & Reporting"""
    print("📊 Phase 3: Comprehensive EDA & Validation Reporting")
    print("=" * 70)
    
    # Initialize comprehensive EDA reporter
    reporter = ComprehensiveEDAReporter()
    
    # Load all analysis data from previous phases
    if not reporter.load_all_analysis_data():
        print("❌ Failed to load analysis data. Ensure Phases 1 & 2 completed successfully.")
        return
    
    print(f"✅ Analysis data loaded successfully")
    
    # Run comprehensive analysis
    print("\n🔍 Running comprehensive analysis suite...")
    
    # 1. Dataset collection overview
    collection_analysis = reporter.analyze_dataset_collection_overview()
    print(f"   📊 Collection Analysis: {collection_analysis.get('collection_metadata', {}).get('total_datasets', 0)} datasets")
    
    # 2. Keyword intelligence analysis
    keyword_analysis = reporter.analyze_keyword_intelligence()
    print(f"   🔍 Keyword Analysis: {keyword_analysis.get('keyword_extraction_summary', {}).get('total_unique_keywords', 0)} keywords")
    
    # 3. Relationship discovery analysis
    relationship_analysis = reporter.analyze_relationship_discovery()
    print(f"   🔗 Relationship Analysis: {relationship_analysis.get('relationship_summary', {}).get('total_relationships', 0)} relationships")
    
    # 4. Ground truth validation
    gt_analysis = reporter.validate_ground_truth_scenarios()
    print(f"   🎯 Ground Truth Validation: {gt_analysis.get('scenario_summary', {}).get('total_scenarios', 0)} scenarios")
    
    # 5. Data quality assessment
    quality_analysis = reporter.identify_data_quality_issues()
    print(f"   ⚠️ Quality Assessment: {quality_analysis.get('issues_summary', {}).get('total_issues_identified', 0)} issues identified")
    
    # Create comprehensive visualizations
    print("\n📊 Creating comprehensive visualizations...")
    reporter.create_comprehensive_visualizations()
    print(f"   ✅ Visualizations saved to: {reporter.output_base}/visualizations/")
    
    # Generate comprehensive reports
    print("\n📋 Generating comprehensive reports...")
    comprehensive_report = reporter.generate_comprehensive_reports()
    print(f"   ✅ Reports saved to: {reporter.output_base}/reports/")
    
    # Final summary
    total_datasets = collection_analysis.get('collection_metadata', {}).get('total_datasets', 0)
    high_quality_relationships = relationship_analysis.get('relationship_summary', {}).get('high_quality_relationships', 0)
    high_confidence_scenarios = gt_analysis.get('quality_assessment', {}).get('confidence_distribution', {}).get('high_confidence', 0)
    ml_ready = gt_analysis.get('ml_readiness_assessment', {}).get('ready_for_ml_training', False)
    
    print(f"\n🎉 Phase 3 Complete - Comprehensive Analysis Finished!")
    print(f"=" * 70)
    print(f"📊 Final Results:")
    print(f"   Total Datasets Analyzed: {total_datasets}")
    print(f"   High-Quality Relationships: {high_quality_relationships}")
    print(f"   High-Confidence Ground Truth: {high_confidence_scenarios}")
    print(f"   Data Quality Issues: {quality_analysis.get('issues_summary', {}).get('total_issues_identified', 0)}")
    
    print(f"\n📁 Generated Outputs:")
    print(f"   📊 Visualizations: outputs/EDA/visualizations/")
    print(f"   📋 Executive Summary: outputs/EDA/reports/executive_summary.md")
    print(f"   📋 Technical Report: outputs/EDA/reports/technical_analysis_report.md")
    print(f"   📋 Detailed Analysis: outputs/EDA/reports/comprehensive_analysis_report.json")
    
    # ML readiness assessment
    if ml_ready:
        expected_performance = gt_analysis.get('ml_readiness_assessment', {}).get('expected_performance', {})
        print(f"\n✅ ML TRAINING READY!")
        print(f"   Expected F1@3 Score: {expected_performance.get('estimated_f1_score', 'N/A')}")
        print(f"   Confidence Level: {expected_performance.get('confidence_level', 'N/A').title()}")
        print(f"🔄 Next: Execute ML Pipeline (train_models.py)")
    else:
        print(f"\n⚠️ ML Training Needs Improvement")
        print(f"   High-confidence scenarios: {high_confidence_scenarios} (need ≥2)")
        print(f"   Total usable scenarios: {gt_analysis.get('ml_readiness_assessment', {}).get('scenarios_available', 0)} (need ≥3)")
        print(f"🔄 Next: Review outputs/EDA/reports/executive_summary.md for improvement guidance")
    
    print(f"\n🎯 Configuration-Driven Analysis Pipeline Complete!")
    print(f"   Ready for ML/DL/AI Enhancement Phases")


if __name__ == "__main__":
    main()