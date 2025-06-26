# 02_analysis_module.py - Intelligent Analysis with User Behavior Integration
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserBehaviorAnalyzer:
    """Analyze user interaction patterns from platform analytics"""

    def __init__(self, config: Dict):
        self.config = config.get("phase_2_analysis", {})
        self.behavior_config = self.config.get("behavior_analysis", {})
        self.user_segments = {}
        self.interaction_patterns = {}

    def load_user_behavior_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess user behavior data"""
        logger.info(f"👥 Loading user behavior data from {file_path}")

        try:
            behavior_df = pd.read_csv(file_path)
            logger.info(f"📊 Loaded {len(behavior_df)} user interaction events")

            # Basic preprocessing
            behavior_df["EVENT_TIME"] = pd.to_datetime(behavior_df["EVENT_TIME"])
            behavior_df["hour"] = behavior_df["EVENT_TIME"].dt.hour
            behavior_df["day_of_week"] = behavior_df["EVENT_TIME"].dt.dayofweek

            return behavior_df

        except Exception as e:
            logger.error(f"❌ Failed to load user behavior data: {e}")
            return pd.DataFrame()

    def analyze_user_segments(self, behavior_df: pd.DataFrame) -> Dict:
        """Segment users based on interaction patterns"""
        logger.info("🔍 Analyzing user segments...")

        if behavior_df.empty:
            return {}

        # Calculate user activity metrics
        user_metrics = (
            behavior_df.groupby("SESSION_ID")
            .agg(
                {
                    "EVENT_ID": "count",
                    "EVENT_TYPE": lambda x: x.nunique(),
                    "EVENT_TIME": ["min", "max"],
                }
            )
            .round(2)
        )

        user_metrics.columns = [
            "total_events",
            "unique_event_types",
            "first_event",
            "last_event",
        ]
        user_metrics["session_duration"] = (
            user_metrics["last_event"] - user_metrics["first_event"]
        ).dt.total_seconds() / 60

        # Segment users based on activity level
        min_events_power_user = self.behavior_config.get("min_events_power_user", 15)

        def classify_user(row):
            if row["total_events"] >= min_events_power_user:
                return "power_user"
            elif row["total_events"] >= 5:
                return "casual_user"
            else:
                return "quick_browser"

        user_metrics["user_segment"] = user_metrics.apply(classify_user, axis=1)

        # Calculate segment statistics
        segment_stats = {
            "total_sessions": len(user_metrics),
            "segment_distribution": user_metrics["user_segment"]
            .value_counts()
            .to_dict(),
            "avg_events_per_session": float(user_metrics["total_events"].mean()),
            "avg_session_duration": float(user_metrics["session_duration"].mean()),
            "peak_activity_hour": int(behavior_df["hour"].mode().iloc[0]),
        }

        self.user_segments = segment_stats
        logger.info(
            f"✅ User segmentation complete: {segment_stats['total_sessions']} sessions analyzed"
        )
        return segment_stats

    def extract_search_patterns(self, behavior_df: pd.DataFrame) -> Dict:
        """Extract search and interaction patterns"""
        logger.info("🔍 Extracting search patterns...")

        if behavior_df.empty:
            return {}

        # Analyze event types
        event_type_dist = behavior_df["EVENT_TYPE"].value_counts().to_dict()

        # Analyze temporal patterns
        hourly_activity = behavior_df.groupby("hour")["EVENT_ID"].count().to_dict()
        daily_activity = (
            behavior_df.groupby("day_of_week")["EVENT_ID"].count().to_dict()
        )

        # Extract search-related patterns (if EVENT_PROPERTIES contains search info)
        search_patterns = {}
        if "EVENT_PROPERTIES" in behavior_df.columns:
            # Try to extract search terms from event properties
            search_events = behavior_df[
                behavior_df["EVENT_TYPE"].str.contains("search", case=False, na=False)
            ]
            if len(search_events) > 0:
                search_patterns["search_events_count"] = len(search_events)
                search_patterns["avg_searches_per_session"] = (
                    len(search_events) / behavior_df["SESSION_ID"].nunique()
                )

        patterns = {
            "event_type_distribution": event_type_dist,
            "hourly_activity": hourly_activity,
            "daily_activity": daily_activity,
            "search_patterns": search_patterns,
            "most_active_hour": int(behavior_df["hour"].mode().iloc[0]),
            "total_unique_sessions": int(behavior_df["SESSION_ID"].nunique()),
        }

        self.interaction_patterns = patterns
        return patterns


class DatasetIntelligenceEngine:
    """Advanced dataset analysis with keyword extraction and relationship discovery"""

    def __init__(self, config: Dict):
        self.config = config.get("phase_2_analysis", {})
        self.keyword_config = self.config.get("keyword_extraction", {})
        self.domain_keywords = self.config.get("domain_keywords", {})
        self.datasets_df = None
        self.keyword_profiles = {}
        self.relationship_matrix = {}

    def load_extracted_datasets(self, data_path: str) -> pd.DataFrame:
        """Load datasets from Phase 1 extraction"""
        logger.info(f"📂 Loading extracted datasets from {data_path}")

        try:
            data_path = Path(data_path)
            datasets = []

            # Load Singapore datasets
            sg_file = data_path / "singapore_datasets.csv"
            if sg_file.exists():
                sg_df = pd.read_csv(sg_file)
                datasets.append(sg_df)
                logger.info(f"📊 Loaded {len(sg_df)} Singapore datasets")

            # Load global datasets
            global_file = data_path / "global_datasets.csv"
            if global_file.exists():
                global_df = pd.read_csv(global_file)
                datasets.append(global_df)
                logger.info(f"🌍 Loaded {len(global_df)} global datasets")

            if datasets:
                self.datasets_df = pd.concat(datasets, ignore_index=True)
                logger.info(f"✅ Total datasets loaded: {len(self.datasets_df)}")
                return self.datasets_df
            else:
                logger.warning("⚠️ No datasets found")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            return pd.DataFrame()

    def extract_intelligent_keywords(self) -> Dict:
        """Extract intelligent keywords using domain knowledge and TF-IDF"""
        logger.info("🧠 Extracting intelligent keywords...")

        if self.datasets_df is None or self.datasets_df.empty:
            return {}

        keyword_profiles = {}
        domain_weights = self.keyword_config.get("domain_weights", {})

        for idx, dataset in self.datasets_df.iterrows():
            dataset_id = dataset["dataset_id"]

            # Combine all text fields
            text_content = self._combine_text_fields(dataset)

            # Extract domain-specific keywords
            domain_keywords = self._extract_domain_keywords(
                text_content, domain_weights
            )

            # Extract contextual keywords
            contextual_keywords = self._extract_contextual_keywords(text_content)

            # Calculate keyword importance scores
            keyword_scores = self._score_keywords(
                domain_keywords + contextual_keywords, text_content
            )

            # Identify research relevance signals
            research_signals = self._identify_research_signals(text_content, dataset)

            keyword_profiles[dataset_id] = {
                "title": dataset.get("title", ""),
                "category": dataset.get("category", "general"),
                "source": dataset.get("source", ""),
                "keywords": keyword_scores,
                "domain_signals": research_signals,
                "relevance_score": self._calculate_research_relevance(
                    dataset, keyword_scores
                ),
                "text_content": text_content[:200],  # First 200 chars for debugging
            }

        self.keyword_profiles = keyword_profiles
        logger.info(
            f"✅ Generated keyword profiles for {len(keyword_profiles)} datasets"
        )
        return keyword_profiles

    def _combine_text_fields(self, dataset: pd.Series) -> str:
        """Combine all relevant text fields for analysis"""
        fields = ["title", "description", "tags", "agency", "category"]
        text_parts = []

        for field in fields:
            value = str(dataset.get(field, "")).strip()
            if value and value != "nan":
                text_parts.append(value)

        return " ".join(text_parts).lower()

    def _extract_domain_keywords(self, text: str, domain_weights: Dict) -> List[str]:
        """Extract domain-specific keywords with weights"""
        keywords = []
        words = re.findall(r"\b\w{3,}\b", text.lower())

        for domain, domain_terms in self.domain_keywords.items():
            weight = domain_weights.get(domain, 1.0)

            for word in words:
                if word in domain_terms:
                    keywords.append(f"{domain}:{word}:{weight}")

        return keywords

    def _extract_contextual_keywords(self, text: str) -> List[str]:
        """Extract contextual keywords (geographic, temporal, type indicators)"""
        keywords = []

        # Geographic keywords
        geo_terms = [
            "singapore",
            "global",
            "international",
            "national",
            "regional",
            "local",
        ]
        # Temporal keywords
        temporal_terms = [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "annual",
            "real-time",
            "historical",
        ]
        # Data type keywords
        type_terms = [
            "survey",
            "census",
            "index",
            "indicator",
            "statistics",
            "report",
            "analysis",
        ]

        words = re.findall(r"\b\w{3,}\b", text.lower())

        for word in words:
            if word in geo_terms:
                keywords.append(f"geographic:{word}:0.8")
            elif word in temporal_terms:
                keywords.append(f"temporal:{word}:0.6")
            elif word in type_terms:
                keywords.append(f"data_type:{word}:0.5")

        return keywords

    def _score_keywords(self, keywords: List[str], text: str) -> Dict[str, float]:
        """Score keywords by frequency and importance"""
        keyword_scores = {}

        for keyword in keywords:
            parts = keyword.split(":")
            if len(parts) >= 3:
                domain, term, weight = parts[0], parts[1], float(parts[2])
            else:
                domain, term, weight = "general", keyword, 1.0

            # Count frequency in text
            frequency = text.lower().count(term)

            # Calculate final score
            score = frequency * weight
            keyword_scores[f"{domain}:{term}"] = score

        return keyword_scores

    def _identify_research_signals(self, text: str, dataset: pd.Series) -> List[str]:
        """Identify signals relevant for research applications"""
        signals = []

        # Strong domain signals
        if any(term in text for term in ["housing", "hdb", "property", "resale"]):
            signals.append("housing_research_ready")
        if any(term in text for term in ["transport", "traffic", "mrt", "mobility"]):
            signals.append("transport_research_ready")
        if any(term in text for term in ["health", "medical", "hospital", "disease"]):
            signals.append("health_research_ready")
        if any(term in text for term in ["economic", "gdp", "employment", "finance"]):
            signals.append("economics_research_ready")

        # Data quality signals
        if dataset.get("quality_score", 0) >= 0.8:
            signals.append("high_quality_data")
        if "active" in str(dataset.get("status", "")).lower():
            signals.append("actively_maintained")

        # Source credibility signals
        source = str(dataset.get("source", "")).lower()
        if "gov" in source:
            signals.append("government_source")
        if any(org in source for org in ["world bank", "un", "oecd", "imf"]):
            signals.append("international_organization")

        # Warning signals
        if any(term in text for term in ["private", "restricted", "limited"]):
            signals.append("access_limitation_warning")
        if any(
            term in text
            for term in ["newspaper", "news article", "press release", "editorial"]
        ):
            signals.append("media_content_warning")
        elif "media" in text and not any(
            term in text
            for term in [
                "telecommunications",
                "communication",
                "hotspot",
                "wireless",
                "development authority",
                "infocomm",
            ]
        ):
            signals.append("media_content_warning")

        return signals

    def _calculate_research_relevance(
        self, dataset: pd.Series, keyword_scores: Dict
    ) -> float:
        """Calculate how relevant this dataset is for research"""
        score = 0.5  # Base score

        # Quality boost
        quality = dataset.get("quality_score", 0)
        score += quality * 0.2

        # Source credibility boost
        source = str(dataset.get("source", "")).lower()
        if "gov" in source:
            score += 0.15
        elif any(org in source for org in ["world bank", "un", "oecd"]):
            score += 0.1

        # Keyword richness boost
        if keyword_scores:
            avg_keyword_score = np.mean(list(keyword_scores.values()))
            score += min(avg_keyword_score / 10, 0.15)  # Cap at 0.15

        # Recency boost
        if dataset.get("status") == "active":
            score += 0.1

        return min(1.0, score)

    def discover_dataset_relationships(self) -> Dict:
        """Discover intelligent relationships between datasets"""
        logger.info("🔗 Discovering dataset relationships...")

        if not self.keyword_profiles:
            return {}

        relationships = defaultdict(list)
        dataset_ids = list(self.keyword_profiles.keys())

        for i, dataset1_id in enumerate(dataset_ids):
            for dataset2_id in dataset_ids[i + 1 :]:
                dataset1 = self.keyword_profiles[dataset1_id]
                dataset2 = self.keyword_profiles[dataset2_id]

                # Calculate relationship strength and type
                rel_score, rel_type = self._calculate_relationship_strength(
                    dataset1, dataset2
                )

                if rel_score >= 0.3:  # Minimum threshold
                    relationship = {
                        "dataset1_id": dataset1_id,
                        "dataset1_title": dataset1["title"],
                        "dataset2_id": dataset2_id,
                        "dataset2_title": dataset2["title"],
                        "relationship_score": rel_score,
                        "relationship_type": rel_type,
                        "confidence": self._calculate_confidence(
                            dataset1, dataset2, rel_score
                        ),
                    }

                    relationships[rel_type].append(relationship)

        # Sort relationships by score within each type
        for rel_type in relationships:
            relationships[rel_type].sort(
                key=lambda x: x["relationship_score"], reverse=True
            )

        logger.info(
            f"✅ Discovered {sum(len(v) for v in relationships.values())} relationships"
        )
        return dict(relationships)

    def _are_complementary_categories_enhanced(self, cat1: str, cat2: str) -> bool:
        complementary_pairs = {
            ("economic_development", "transport"),
            ("economic_development", "geospatial"),
            ("economic_development", "sustainable_development"),
            ("transport", "geospatial"),
            ("economic_finance", "economic_development"),
            ("economic_finance", "economic_statistics"),
            ("economic_development", "statistics"),
            ("transport", "statistics"),
            ("geospatial", "statistics"),
        }
        return (cat1, cat2) in complementary_pairs or (
            cat2,
            cat1,
        ) in complementary_pairs

    def _are_contextual_categories(self, cat1: str, cat2: str) -> bool:
        """Check if categories provide useful context for each other"""
        contextual_pairs = {
            ("economics", "development"),
            ("health", "environment"),
            ("education", "development"),
            ("transport", "planning"),
            ("demographics", "development"),
        }

        return (cat1, cat2) in contextual_pairs or (cat2, cat1) in contextual_pairs

    def _calculate_quality_boost(self, dataset1: Dict, dataset2: Dict) -> float:
        """Calculate quality boost for relationship score"""
        avg_relevance = (dataset1["relevance_score"] + dataset2["relevance_score"]) / 2
        return min(avg_relevance, 0.5)  # Cap boost at 0.5

    def _calculate_confidence(
        self, dataset1: Dict, dataset2: Dict, rel_score: float
    ) -> float:
        """Calculate confidence in the relationship"""
        base_confidence = rel_score

        # Boost for high-quality datasets
        quality_boost = self._calculate_quality_boost(dataset1, dataset2) * 0.2

        # Boost for government/official sources
        source_boost = 0
        for dataset in [dataset1, dataset2]:
            if any(
                signal in dataset["domain_signals"]
                for signal in ["government_source", "international_organization"]
            ):
                source_boost += 0.05

        # Penalty for warning signals
        warning_penalty = 0
        for dataset in [dataset1, dataset2]:
            if any("warning" in signal for signal in dataset["domain_signals"]):
                warning_penalty += 0.1

        final_confidence = (
            base_confidence + quality_boost + source_boost - warning_penalty
        )
        return max(0.0, min(1.0, final_confidence))

    def _calculate_source_alignment_boost(
        self, dataset1: Dict, dataset2: Dict
    ) -> float:
        source1 = dataset1.get("source", "").lower()
        source2 = dataset2.get("source", "").lower()

        if "data.gov.sg" in source1 and "data.gov.sg" in source2:
            return 0.8

        intl_orgs = ["world bank", "imf", "oecd", "un"]
        if any(org in source1 for org in intl_orgs) and any(
            org in source2 for org in intl_orgs
        ):
            return 0.7

        if ("data.gov.sg" in source1 and any(org in source2 for org in intl_orgs)) or (
            "data.gov.sg" in source2 and any(org in source1 for org in intl_orgs)
        ):
            return 0.6

        return 0.0

    def _calculate_relationship_strength(
        self, dataset1: Dict, dataset2: Dict
    ) -> Tuple[float, str]:
        """Calculate relationship strength and type between two datasets"""

        # Initialize scores
        category_score = 0.0
        keyword_score = 0.0
        source_score = 0.0
        geographic_score = 0.0

        # Get dataset attributes
        cat1 = dataset1.get("category", "").lower()
        cat2 = dataset2.get("category", "").lower()
        keywords1 = set(dataset1.get("keywords", {}).keys())
        keywords2 = set(dataset2.get("keywords", {}).keys())

        # 1. Category relationship analysis
        if cat1 == cat2:
            category_score = 0.8
            relationship_type = "same_domain"
        elif self._are_complementary_categories_enhanced(cat1, cat2):
            category_score = 0.6
            relationship_type = "complementary"
        elif self._are_contextual_categories(cat1, cat2):
            category_score = 0.4
            relationship_type = "contextual"
        else:
            category_score = 0.1
            relationship_type = "unrelated"

        # 2. Keyword overlap analysis
        if keywords1 and keywords2:
            common_keywords = keywords1.intersection(keywords2)
            total_keywords = keywords1.union(keywords2)

            if total_keywords:
                keyword_overlap = len(common_keywords) / len(total_keywords)
                keyword_score = keyword_overlap * 0.7

                # Boost for domain-specific keyword matches
                domain_keywords = [
                    k
                    for k in common_keywords
                    if any(
                        domain in k
                        for domain in ["economic", "transport", "health", "housing"]
                    )
                ]
                if domain_keywords:
                    keyword_score += len(domain_keywords) * 0.1

        # 3. Source alignment boost
        source_score = self._calculate_source_alignment_boost(dataset1, dataset2)

        # 4. Geographic compatibility
        geo1 = dataset1.get("source", "").lower()
        geo2 = dataset2.get("source", "").lower()

        # Both Singapore sources
        if "singapore" in geo1 and "singapore" in geo2:
            geographic_score = 0.3
        # Both global sources
        elif any(org in geo1 for org in ["world bank", "imf", "oecd", "un"]) and any(
            org in geo2 for org in ["world bank", "imf", "oecd", "un"]
        ):
            geographic_score = 0.25
        # Singapore + Global (complementary)
        elif (
            "singapore" in geo1
            and any(org in geo2 for org in ["world bank", "imf", "oecd", "un"])
        ) or (
            "singapore" in geo2
            and any(org in geo1 for org in ["world bank", "imf", "oecd", "un"])
        ):
            geographic_score = 0.2

        # 5. Quality boost for high-quality datasets
        quality_boost = self._calculate_quality_boost(dataset1, dataset2) * 0.15

        # Calculate final relationship score
        base_score = (
            category_score * 0.4
            + keyword_score * 0.3
            + source_score * 0.2
            + geographic_score * 0.1
        )
        final_score = min(1.0, base_score + quality_boost)

        # Adjust relationship type based on final score
        if final_score >= 0.7:
            if relationship_type == "unrelated":
                relationship_type = "contextual"
        elif final_score < 0.3:
            relationship_type = "weak_connection"

        return final_score, relationship_type


class IntelligentGroundTruthGenerator:
    """Generate intelligent ground truth scenarios using user behavior and dataset analysis"""

    def __init__(
        self,
        config: Dict,
        user_analyzer: UserBehaviorAnalyzer,
        dataset_engine: DatasetIntelligenceEngine,
    ):
        self.config = config.get("phase_2_analysis", {})
        self.gt_config = self.config.get("ground_truth_generation", {})
        self.user_analyzer = user_analyzer
        self.dataset_engine = dataset_engine
        self.relationships = {}

    def generate_intelligent_ground_truth(self) -> Dict:
        """Generate intelligent ground truth using all available intelligence"""
        logger.info("🎯 Generating intelligent ground truth scenarios...")

        # Load the actual datasets to work with real titles
        if (
            self.dataset_engine.datasets_df is None
            or self.dataset_engine.datasets_df.empty
        ):
            logger.warning("❌ No datasets available for ground truth generation")
            return {}

        ground_truth_scenarios = {}

        # Strategy 1: Same-source clustering (highest confidence)
        same_source_scenarios = self._generate_realistic_same_source_scenarios()
        ground_truth_scenarios.update(same_source_scenarios)
        logger.info(f"✅ Same-source scenarios: {len(same_source_scenarios)}")

        # Strategy 2: Same-category clustering
        same_category_scenarios = self._generate_realistic_same_category_scenarios()
        ground_truth_scenarios.update(same_category_scenarios)
        logger.info(f"✅ Same-category scenarios: {len(same_category_scenarios)}")

        # Strategy 3: Cross-domain logical connections
        cross_domain_scenarios = self._generate_realistic_cross_domain_scenarios()
        ground_truth_scenarios.update(cross_domain_scenarios)
        logger.info(f"✅ Cross-domain scenarios: {len(cross_domain_scenarios)}")

        # Validate all scenarios
        validated_scenarios = self._validate_and_score_scenarios(ground_truth_scenarios)

        logger.info(
            f"✅ Generated {len(validated_scenarios)} realistic ground truth scenarios"
        )
        return validated_scenarios

    def _generate_realistic_same_source_scenarios(self) -> Dict:
        """Generate scenarios using semantically similar datasets from the same source"""
        scenarios = {}
        datasets_df = self.dataset_engine.datasets_df

        # Group by source
        source_groups = datasets_df.groupby("source")

        scenario_count = 1
        for source, group in source_groups:
            if len(group) >= 2:  # Need at least 2 datasets
                
                # Singapore Statistics - semantic clustering by topic
                if "Singapore Statistics" in source or "SingStat" in source:
                    # Create topic-based clusters
                    economic_stats = group[group["title"].str.contains("GDP|Economic|Employment|Income|Trade", case=False, na=False)]
                    demographic_stats = group[group["title"].str.contains("Population|Demographic|Resident|Birth|Death", case=False, na=False)]
                    
                    if len(economic_stats) >= 2:
                        primary_title = economic_stats.iloc[0]["title"]
                        # Generate semantic query instead of exact title
                        semantic_query = self._generate_semantic_query(primary_title)
                        scenarios[f"singapore_economic_stats_{scenario_count}"] = {
                            "primary": semantic_query,
                            "complementary": economic_stats.iloc[1:min(4, len(economic_stats))]["title"].tolist(),
                            "explanation": "Singapore economic statistics analysis",
                            "confidence": 0.95,
                            "source": "semantic_clustering",
                            "generation_method": "enhanced_semantic",
                        }
                        scenario_count += 1
                    
                    if len(demographic_stats) >= 2:
                        primary_title = demographic_stats.iloc[0]["title"]
                        semantic_query = self._generate_semantic_query(primary_title)
                        scenarios[f"singapore_demographic_stats_{scenario_count}"] = {
                            "primary": semantic_query,
                            "complementary": demographic_stats.iloc[1:min(4, len(demographic_stats))]["title"].tolist(),
                            "explanation": "Singapore demographic statistics analysis", 
                            "confidence": 0.9,
                            "source": "semantic_clustering",
                            "generation_method": "enhanced_semantic",
                        }
                        scenario_count += 1

                # LTA transport series
                elif (
                    "LTA" in source
                    or group["title"].str.contains("LTA", na=False).any()
                ):
                    lta_data = group[group["title"].str.contains("LTA", na=False)]
                    if len(lta_data) >= 2:
                        scenarios[f"transport_system_{scenario_count}"] = {
                            "primary": lta_data.iloc[0]["title"],
                            "complementary": lta_data.iloc[1 : min(4, len(lta_data))][
                                "title"
                            ].tolist(),
                            "explanation": "Singapore transport system analysis",
                            "confidence": 0.9,
                            "source": "same_source_analysis",
                            "generation_method": "realistic_clustering",
                        }
                        scenario_count += 1

                # World Bank series
                elif "World Bank" in source:
                    wb_data = group.head(4)  # Take first 4 from World Bank
                    if len(wb_data) >= 2:
                        scenarios[f"world_bank_indicators_{scenario_count}"] = {
                            "primary": wb_data.iloc[0]["title"],
                            "complementary": wb_data.iloc[1:]["title"].tolist(),
                            "explanation": "World Bank development indicators",
                            "confidence": 0.85,
                            "source": "same_source_analysis",
                            "generation_method": "realistic_clustering",
                        }
                        scenario_count += 1

        return scenarios

    def _generate_realistic_same_category_scenarios(self) -> Dict:
        """Generate scenarios using datasets from the same category"""
        scenarios = {}
        datasets_df = self.dataset_engine.datasets_df

        # Group by category
        category_groups = datasets_df.groupby("category")

        for category, group in category_groups:
            if len(group) >= 2:
                # Economic development category
                if category == "economic_development":
                    econ_data = group.head(4)
                    scenarios["economic_development_analysis"] = {
                        "primary": econ_data.iloc[0]["title"],
                        "complementary": econ_data.iloc[1:]["title"].tolist(),
                        "explanation": "Economic development indicators analysis",
                        "confidence": 0.8,
                        "source": "same_category_analysis",
                        "generation_method": "realistic_clustering",
                    }

                # Economic finance category
                elif category == "economic_finance":
                    finance_data = group.head(3)
                    if len(finance_data) >= 2:
                        scenarios["financial_analysis"] = {
                            "primary": finance_data.iloc[0]["title"],
                            "complementary": finance_data.iloc[1:]["title"].tolist(),
                            "explanation": "Financial and economic analysis",
                            "confidence": 0.8,
                            "source": "same_category_analysis",
                            "generation_method": "realistic_clustering",
                        }

                # Transport category
                elif category == "transport":
                    transport_data = group.head(4)
                    if len(transport_data) >= 2:
                        scenarios["transport_comprehensive"] = {
                            "primary": transport_data.iloc[0]["title"],
                            "complementary": transport_data.iloc[1:]["title"].tolist(),
                            "explanation": "Comprehensive transport analysis",
                            "confidence": 0.85,
                            "source": "same_category_analysis",
                            "generation_method": "realistic_clustering",
                        }

        return scenarios

    def _generate_realistic_cross_domain_scenarios(self) -> Dict:
        """Generate semantically coherent cross-domain scenarios"""
        scenarios = {}
        datasets_df = self.dataset_engine.datasets_df

        # Cross-domain scenario 1: Economic Development Analysis
        poverty_data = datasets_df[
            datasets_df["title"].str.contains("Poverty|poverty", case=False, na=False)
        ]
        econ_data = datasets_df[
            datasets_df["title"].str.contains("GDP|Economic|economic|Development", case=False, na=False)
        ]

        if len(poverty_data) >= 1 and len(econ_data) >= 1:
            # Create semantic query instead of using exact title
            semantic_query = "economic development poverty analysis indicators"
            
            # Find semantically related datasets
            complementary = []
            # Add poverty datasets
            complementary.extend(poverty_data.head(2)["title"].tolist())
            # Add economic datasets  
            complementary.extend(econ_data.head(2)["title"].tolist())
            
            # Add development indicators if available
            development_data = datasets_df[
                datasets_df["title"].str.contains("Development|development|Growth|growth", case=False, na=False)
            ]
            if len(development_data) > 0:
                complementary.extend(development_data.head(1)["title"].tolist())

            scenarios["economic_development_analysis"] = {
                "primary": semantic_query,
                "complementary": list(set(complementary))[:4],  # Remove duplicates, max 4
                "explanation": "Economic development and poverty analysis",
                "confidence": 0.85,
                "source": "semantic_cross_domain",
                "generation_method": "enhanced_semantic_crossdomain",
            }

        # Cross-domain scenario 2: Health Demographics Analysis  
        health_data = datasets_df[
            datasets_df["title"].str.contains(
                "health|Health|mortality|Mortality|medical|Medical", case=False, na=False
            )
        ]
        demo_data = datasets_df[
            datasets_df["title"].str.contains(
                "Population|population|demographic|Demographic|residents|Residents", case=False, na=False
            )
        ]

        if len(health_data) >= 1 and len(demo_data) >= 1:
            # Create semantic query for health demographics
            semantic_query = "health outcomes demographic population statistics"
            
            complementary = []
            complementary.extend(health_data.head(2)["title"].tolist())
            complementary.extend(demo_data.head(2)["title"].tolist())
            
            # Add mortality data if available
            mortality_data = datasets_df[
                datasets_df["title"].str.contains("mortality|Mortality|death|Death", case=False, na=False)
            ]
            if len(mortality_data) > 0:
                complementary.extend(mortality_data.head(1)["title"].tolist())

            scenarios["health_demographics_analysis"] = {
                "primary": semantic_query,
                "complementary": list(set(complementary))[:4],  # Remove duplicates, max 4
                "explanation": "Health outcomes by demographic analysis",
                "confidence": 0.8,
                "source": "semantic_cross_domain",
                "generation_method": "enhanced_semantic_crossdomain",
            }
            
        # Cross-domain scenario 3: Transport Urban Planning
        transport_data = datasets_df[
            datasets_df["title"].str.contains("transport|Transport|traffic|Traffic|LTA", case=False, na=False)
        ]
        urban_data = datasets_df[
            datasets_df["title"].str.contains("urban|Urban|planning|Planning|development|Development", case=False, na=False)
        ]
        
        if len(transport_data) >= 1 and len(urban_data) >= 1:
            semantic_query = "transportation infrastructure urban planning development"
            
            complementary = []
            complementary.extend(transport_data.head(2)["title"].tolist())
            complementary.extend(urban_data.head(2)["title"].tolist())
            
            scenarios["transport_urban_planning"] = {
                "primary": semantic_query,
                "complementary": list(set(complementary))[:4],
                "explanation": "Transportation and urban development analysis",
                "confidence": 0.75,
                "source": "semantic_cross_domain", 
                "generation_method": "enhanced_semantic_crossdomain",
            }

        return scenarios

    def _validate_and_score_scenarios(self, scenarios: Dict) -> Dict:
        """Validate scenarios against actual dataset availability"""
        validated = {}
        existing_titles = set()

        if self.dataset_engine.datasets_df is not None:
            existing_titles = set(self.dataset_engine.datasets_df["title"].tolist())

        for scenario_name, scenario_data in scenarios.items():
            # Validate complementary datasets exist
            valid_complementary = [
                title
                for title in scenario_data.get("complementary", [])
                if title in existing_titles
            ]

            # More lenient validation - accept scenarios with at least 1 valid dataset
            if len(valid_complementary) >= 1:
                scenario_data["complementary"] = valid_complementary
                scenario_data["validation_score"] = len(valid_complementary) / max(
                    len(scenario_data.get("complementary", [])), 1
                )
                scenario_data["validated"] = True

                # Less aggressive confidence penalty
                original_confidence = scenario_data.get("confidence", 0.5)
                validation_penalty = (
                    1 - scenario_data["validation_score"]
                ) * 0.1  # Reduced from 0.2
                scenario_data["confidence"] = max(
                    0.3, original_confidence - validation_penalty
                )

                validated[scenario_name] = scenario_data
                logger.info(
                    f"✅ Validated {scenario_name}: {len(valid_complementary)} complementary datasets"
                )
            else:
                # Still try to salvage the scenario with alternative datasets
                alternative_datasets = self._find_alternative_datasets(
                    scenario_data.get("primary", ""), 3
                )
                if alternative_datasets:
                    scenario_data["complementary"] = alternative_datasets
                    scenario_data["validation_score"] = 0.5
                    scenario_data["validated"] = True
                    scenario_data["confidence"] = max(
                        0.3, scenario_data.get("confidence", 0.5) * 0.7
                    )
                    validated[scenario_name] = scenario_data
                    logger.info(f"⚠️ Salvaged {scenario_name} with alternative datasets")
                else:
                    logger.warning(
                        f"❌ Rejected {scenario_name}: no valid datasets found"
                    )

        return validated

    def _find_alternative_datasets(
        self, primary_query: str, count: int = 3
    ) -> List[str]:
        """Find alternative datasets based on primary query keywords"""
        query_words = set(primary_query.lower().split())
        dataset_scores = []

        for dataset_id, profile in self.dataset_engine.keyword_profiles.items():
            title = profile.get("title", "").lower()
            keywords = profile.get("keywords", {})

            # Score based on keyword overlap
            title_words = set(title.split())
            overlap_score = len(query_words.intersection(title_words)) / max(
                len(query_words), 1
            )

            # Boost for high relevance
            relevance_boost = profile.get("relevance_score", 0)

            total_score = overlap_score + relevance_boost * 0.5

            if total_score > 0:
                dataset_scores.append({"title": profile["title"], "score": total_score})

        # Sort and return top matches
        dataset_scores.sort(key=lambda x: x["score"], reverse=True)
        return [d["title"] for d in dataset_scores[:count]]

    def _generate_semantic_query(self, dataset_title: str) -> str:
        """Generate semantic query from dataset title for better ML training"""
        import re

        # Extract key semantic concepts from title
        title_lower = dataset_title.lower()
        
        # Define semantic mappings for better queries
        semantic_mappings = {
            'gdp': 'economic growth indicators',
            'population': 'demographic statistics',
            'employment': 'labor market data',
            'income': 'economic indicators',
            'trade': 'international trade statistics',
            'poverty': 'poverty analysis indicators',
            'mortality': 'health outcome statistics',
            'literacy': 'education statistics',
            'singapore statistics': 'singapore official data',
            'world bank': 'development indicators',
            'transport': 'transportation infrastructure data',
            'housing': 'residential property data',
            'health': 'healthcare statistics'
        }
        
        # Find matching concepts
        query_parts = []
        for key, semantic_term in semantic_mappings.items():
            if key in title_lower:
                query_parts.append(semantic_term)
        
        # If no matches, extract meaningful words
        if not query_parts:
            # Remove common non-semantic words
            words = re.findall(r'\b\w{3,}\b', title_lower)
            meaningful_words = [w for w in words if w not in 
                             ['the', 'and', 'for', 'data', 'dataset', 'statistics', 'total', 'annual']]
            if meaningful_words:
                query_parts = meaningful_words[:3]  # Take first 3 meaningful words
        
        # Create semantic query
        if query_parts:
            return ' '.join(query_parts[:3])  # Max 3 concepts for focused search
        else:
            # Fallback to cleaned title
            cleaned = re.sub(r'[^\w\s]', ' ', title_lower)
            return ' '.join(cleaned.split()[:4])


def main():
    """Main execution for Phase 2: Deep Analysis with User Behavior Integration"""
    print("🧠 Phase 2: Deep Analysis with User Behavior Integration")
    print("=" * 70)

    # Load configuration
    with open("config/data_pipeline.yml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    user_analyzer = UserBehaviorAnalyzer(config)
    dataset_engine = DatasetIntelligenceEngine(config)

    # Load data from Phase 1
    datasets_df = dataset_engine.load_extracted_datasets("data/processed")
    if datasets_df.empty:
        print("❌ No datasets found. Run Phase 1 (data extraction) first!")
        return

    # Load and analyze user behavior
    behavior_file = config["phase_2_analysis"]["user_behavior_file"]
    behavior_df = user_analyzer.load_user_behavior_data(behavior_file)

    if not behavior_df.empty:
        user_segments = user_analyzer.analyze_user_segments(behavior_df)
        interaction_patterns = user_analyzer.extract_search_patterns(behavior_df)
        print(
            f"👥 User Analysis: {user_segments.get('total_sessions', 0)} sessions, {interaction_patterns.get('total_unique_sessions', 0)} unique users"
        )
    else:
        print("⚠️ No user behavior data found, proceeding with dataset analysis only")

    # Extract intelligent keywords
    keyword_profiles = dataset_engine.extract_intelligent_keywords()
    print(f"🔍 Keyword Analysis: {len(keyword_profiles)} dataset profiles generated")

    # Discover dataset relationships
    relationships = dataset_engine.discover_dataset_relationships()
    print(
        f"🔗 Relationship Discovery: {sum(len(v) for v in relationships.values())} relationships found"
    )

    # Generate intelligent ground truth
    ground_truth_gen = IntelligentGroundTruthGenerator(
        config, user_analyzer, dataset_engine
    )
    ground_truth = ground_truth_gen.generate_intelligent_ground_truth()
    print(f"🎯 Ground Truth Generation: {len(ground_truth)} scenarios created")

    # Save analysis results
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all analysis outputs
    with open(output_path / "keyword_profiles.json", "w") as f:
        json.dump(keyword_profiles, f, indent=2, default=str)

    with open(output_path / "dataset_relationships.json", "w") as f:
        json.dump(relationships, f, indent=2, default=str)

    with open(output_path / "intelligent_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2, default=str)

    # Save user behavior analysis if available
    if not behavior_df.empty:
        user_analysis = {
            "user_segments": user_analyzer.user_segments,
            "interaction_patterns": user_analyzer.interaction_patterns,
        }
        with open(output_path / "user_behavior_analysis.json", "w") as f:
            json.dump(user_analysis, f, indent=2, default=str)

    # Summary report
    print(f"\n📊 Phase 2 Analysis Summary:")
    print(f"   Datasets analyzed: {len(datasets_df)}")
    print(f"   Keyword profiles: {len(keyword_profiles)}")
    print(f"   Relationships discovered: {sum(len(v) for v in relationships.values())}")
    print(f"   Ground truth scenarios: {len(ground_truth)}")
    if not behavior_df.empty:
        print(f"   User sessions analyzed: {user_segments.get('total_sessions', 0)}")

    print(f"\n✅ Phase 2 Complete! Files saved to data/processed/")
    print(f"🔄 Next: Run Phase 3 (EDA & Reporting)")


if __name__ == "__main__":
    main()
