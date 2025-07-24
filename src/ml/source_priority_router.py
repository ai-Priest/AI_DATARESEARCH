"""
Source Priority Routing System
Implements domain-specific routing with Singapore-first strategy for Task 3.2
"""

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class SourcePriorityRouter:
    """Routes queries to appropriate data sources based on domain and context"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Load source definitions
        self.source_definitions = self._load_source_definitions()

        # Load domain routing rules
        self.domain_routing_rules = self._load_domain_routing_rules()

        # Load training mappings for exact matches
        self.training_mappings = {}
        self.mappings_path = self.config.get("mappings_path", "training_mappings.md")
        self._load_training_mappings()

        logger.info("🧭 SourcePriorityRouter initialized")
        logger.info(f"  Sources: {len(self.source_definitions)}")
        logger.info(f"  Domains: {len(self.domain_routing_rules)}")
        logger.info(f"  Training mappings: {len(self.training_mappings)}")

    def _load_source_definitions(self) -> Dict:
        """Load source definitions with metadata"""
        return {
            "kaggle": {
                "name": "Kaggle",
                "description": "Platform for data science competitions and datasets",
                "url": "https://www.kaggle.com/datasets",
                "api_available": True,
                "domains": ["machine_learning", "psychology", "general"],
                "quality_score": 0.85,
                "update_frequency": "daily",
                "singapore_priority": False,
            },
            "zenodo": {
                "name": "Zenodo",
                "description": "Open research repository for academic data",
                "url": "https://zenodo.org/",
                "api_available": True,
                "domains": ["psychology", "climate", "research"],
                "quality_score": 0.80,
                "update_frequency": "weekly",
                "singapore_priority": False,
            },
            "world_bank": {
                "name": "World Bank Data",
                "description": "Global development data and indicators",
                "url": "https://data.worldbank.org/",
                "api_available": True,
                "domains": ["economics", "climate", "health", "education"],
                "quality_score": 0.90,
                "update_frequency": "quarterly",
                "singapore_priority": False,
            },
            "data_gov_sg": {
                "name": "Data.gov.sg",
                "description": "Singapore government open data portal",
                "url": "https://data.gov.sg/",
                "api_available": True,
                "domains": ["singapore", "health", "education", "transportation"],
                "quality_score": 0.95,
                "update_frequency": "monthly",
                "singapore_priority": True,
            },
            "singstat": {
                "name": "Singapore Department of Statistics",
                "description": "Official statistics about Singapore",
                "url": "https://www.singstat.gov.sg/",
                "api_available": True,
                "domains": ["singapore", "economics", "population"],
                "quality_score": 0.95,
                "update_frequency": "monthly",
                "singapore_priority": True,
            },
            "lta_datamall": {
                "name": "LTA DataMall",
                "description": "Singapore Land Transport Authority data",
                "url": "https://datamall.lta.gov.sg/",
                "api_available": True,
                "domains": ["singapore", "transportation"],
                "quality_score": 0.90,
                "update_frequency": "daily",
                "singapore_priority": True,
            },
        }

    def _load_domain_routing_rules(self) -> Dict:
        """Load domain-specific routing rules based on training mappings"""
        return {
            "psychology": {
                "primary_sources": [
                    "kaggle",
                    "zenodo",
                ],  # Based on training mappings: kaggle (0.95), zenodo (0.90)
                "secondary_sources": [
                    "world_bank"
                ],  # Lower relevance (0.25) but included as fallback
                "singapore_sources": ["data_gov_sg", "singstat"],
                "quality_threshold": 0.7,
                "reasoning": "Psychology datasets are best found on research platforms (kaggle 0.95, zenodo 0.90)",
                "fallback_strategy": "research_platforms",
            },
            "machine_learning": {
                "primary_sources": ["kaggle"],  # Primary ML platform (0.98)
                "secondary_sources": ["zenodo"],  # Academic ML research (0.85)
                "singapore_sources": ["data_gov_sg"],
                "quality_threshold": 0.8,
                "reasoning": "ML datasets require high-quality, well-documented sources (kaggle 0.98)",
                "fallback_strategy": "competition_platforms",
            },
            "climate": {
                "primary_sources": ["world_bank"],  # Primary climate source (0.95)
                "secondary_sources": [
                    "zenodo",
                    "kaggle",
                ],  # zenodo (0.85), kaggle (0.82)
                "singapore_sources": ["data_gov_sg"],
                "quality_threshold": 0.75,
                "reasoning": "Climate data requires authoritative sources (world_bank 0.95)",
                "fallback_strategy": "official_sources",
            },
            "economics": {
                "primary_sources": ["world_bank"],  # Excellent economic data (0.98)
                "secondary_sources": ["kaggle"],  # Some economic datasets (0.75)
                "singapore_sources": ["singstat", "data_gov_sg"],
                "quality_threshold": 0.8,
                "reasoning": "Economic data requires official statistical sources (world_bank 0.98)",
                "fallback_strategy": "official_statistics",
            },
            "singapore": {
                "primary_sources": [
                    "data_gov_sg",
                    "singstat",
                    "lta_datamall",
                ],  # Official Singapore sources
                "secondary_sources": ["world_bank"],  # Some Singapore indicators (0.60)
                "singapore_sources": ["data_gov_sg", "singstat", "lta_datamall"],
                "quality_threshold": 0.9,
                "reasoning": "Singapore data requires official government sources (data_gov_sg 0.98)",
                "fallback_strategy": "government_sources",
            },
            "health": {
                "primary_sources": [
                    "world_bank",
                    "zenodo",
                ],  # world_bank (0.88), zenodo (0.90)
                "secondary_sources": ["kaggle"],  # Health datasets (0.82)
                "singapore_sources": ["data_gov_sg", "singstat"],
                "quality_threshold": 0.8,
                "reasoning": "Health data from global indicators and academic research",
                "fallback_strategy": "health_authorities",
            },
            "education": {
                "primary_sources": ["world_bank"],  # Global education indicators (0.92)
                "secondary_sources": [
                    "kaggle",
                    "zenodo",
                ],  # kaggle (0.85), zenodo (0.83)
                "singapore_sources": ["data_gov_sg", "singstat"],
                "quality_threshold": 0.8,
                "reasoning": "Education data from global education indicators",
                "fallback_strategy": "educational_institutions",
            },
            "transportation": {
                "primary_sources": ["lta_datamall", "data_gov_sg"],
                "secondary_sources": ["world_bank", "kaggle"],
                "singapore_sources": ["lta_datamall", "data_gov_sg"],
                "quality_threshold": 0.8,
                "reasoning": "Transportation data requires official transport authority sources",
                "fallback_strategy": "transport_authorities",
            },
            "general": {
                "primary_sources": ["kaggle", "zenodo", "world_bank"],
                "secondary_sources": ["data_gov_sg"],
                "singapore_sources": ["data_gov_sg", "singstat"],
                "quality_threshold": 0.6,
                "reasoning": "General queries benefit from diverse, high-quality sources",
                "fallback_strategy": "diverse_sources",
            },
        }

    def _load_training_mappings(self):
        """Load training mappings for exact matches"""
        try:
            mappings_path = Path(self.mappings_path)
            if mappings_path.exists():
                with open(mappings_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse mappings
                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if (
                        line.startswith("- ")
                        and "→" in line
                        and "(" in line
                        and ")" in line
                    ):
                        try:
                            # Parse: "- psychology → kaggle (0.95) - Best platform for psychology datasets"
                            parts = line[2:].split("→")
                            query = parts[0].strip()
                            rest = parts[1].strip()
                            source_part = rest.split("(")[0].strip()
                            score_part = rest.split("(")[1].split(")")[0].strip()
                            source = source_part
                            relevance_score = float(score_part)

                            self.training_mappings[query] = {
                                "source": source,
                                "score": relevance_score,
                            }
                        except Exception:
                            continue

                logger.info(
                    f"✅ Loaded {len(self.training_mappings)} training mappings"
                )
            else:
                logger.warning(f"⚠️ Training mappings not found at {mappings_path}")
        except Exception as e:
            logger.error(f"❌ Error loading training mappings: {e}")

    def route_query(
        self, query: str, domain: str, singapore_first: bool = False
    ) -> List[Dict]:
        """Route query to appropriate sources based on domain and Singapore-first strategy"""

        # Check for exact match in training mappings first
        exact_matches = self._get_exact_training_matches(query)
        if exact_matches:
            return exact_matches

        # Get domain routing rules
        domain_rules = self.domain_routing_rules.get(
            domain, self.domain_routing_rules["general"]
        )

        # Determine routing strategy based on domain and Singapore-first flag
        if singapore_first or domain == "singapore":
            sources = self._route_singapore_first(query, domain, domain_rules)
        else:
            sources = self._route_domain_first(query, domain, domain_rules)

        # Apply fallback routing if no high-quality sources found
        if (
            not sources
            or max(s["relevance_score"] for s in sources)
            < domain_rules["quality_threshold"]
        ):
            fallback_sources = self._apply_fallback_routing(query, domain, domain_rules)
            # Merge with existing sources, avoiding duplicates
            existing_source_names = {s["source"] for s in sources}
            for fallback in fallback_sources:
                if fallback["source"] not in existing_source_names:
                    sources.append(fallback)

        return sources[:5]  # Return top 5

    def _get_exact_training_matches(self, query: str) -> List[Dict]:
        """Get exact matches from training mappings"""
        matches = []
        query_lower = query.lower()

        # Check for exact query match
        if query_lower in self.training_mappings:
            exact_match = self.training_mappings[query_lower]
            matches.append(
                {
                    "source": exact_match["source"],
                    "relevance_score": exact_match["score"],
                    "routing_reason": "exact_training_match",
                    "priority": 1,
                    "source_info": self.source_definitions.get(
                        exact_match["source"], {}
                    ),
                }
            )

        # Check for partial matches (query contains training mapping key)
        for mapping_query, mapping_data in self.training_mappings.items():
            if mapping_query != query_lower and mapping_query in query_lower:
                matches.append(
                    {
                        "source": mapping_data["source"],
                        "relevance_score": mapping_data["score"]
                        * 0.9,  # Slightly lower for partial match
                        "routing_reason": "partial_training_match",
                        "priority": 2,
                        "source_info": self.source_definitions.get(
                            mapping_data["source"], {}
                        ),
                    }
                )

        # Sort by priority and relevance, remove duplicates
        seen_sources = set()
        unique_matches = []
        for match in sorted(
            matches, key=lambda x: (x["priority"], -x["relevance_score"])
        ):
            if match["source"] not in seen_sources:
                unique_matches.append(match)
                seen_sources.add(match["source"])

        return unique_matches[:3]  # Return top 3 training matches

    def _apply_fallback_routing(
        self, query: str, domain: str, rules: Dict
    ) -> List[Dict]:
        """Apply fallback routing for ambiguous or new query types"""
        fallback_sources = []
        fallback_strategy = rules.get("fallback_strategy", "diverse_sources")

        logger.info(f"🔄 Applying fallback routing strategy: {fallback_strategy}")

        if fallback_strategy == "research_platforms":
            # For psychology and research queries
            fallback_candidates = ["kaggle", "zenodo", "world_bank"]
        elif fallback_strategy == "competition_platforms":
            # For ML and data science queries
            fallback_candidates = ["kaggle", "zenodo"]
        elif fallback_strategy == "official_sources":
            # For climate, health, economics
            fallback_candidates = ["world_bank", "data_gov_sg", "singstat"]
        elif fallback_strategy == "official_statistics":
            # For economics and statistics
            fallback_candidates = ["world_bank", "singstat", "data_gov_sg"]
        elif fallback_strategy == "government_sources":
            # For Singapore and official data
            fallback_candidates = [
                "data_gov_sg",
                "singstat",
                "lta_datamall",
                "world_bank",
            ]
        elif fallback_strategy == "health_authorities":
            # For health data
            fallback_candidates = ["world_bank", "data_gov_sg", "zenodo"]
        elif fallback_strategy == "educational_institutions":
            # For education data
            fallback_candidates = ["world_bank", "zenodo", "data_gov_sg"]
        elif fallback_strategy == "transport_authorities":
            # For transportation data
            fallback_candidates = ["lta_datamall", "data_gov_sg", "world_bank"]
        else:  # diverse_sources
            # General fallback
            fallback_candidates = ["kaggle", "zenodo", "world_bank", "data_gov_sg"]

        # Create fallback source entries
        for source in fallback_candidates:
            if source in self.source_definitions:
                relevance_score = self._calculate_fallback_relevance(
                    query, source, domain
                )
                fallback_sources.append(
                    {
                        "source": source,
                        "relevance_score": relevance_score,
                        "routing_reason": f"fallback_{fallback_strategy}",
                        "priority": 4,  # Lower priority than primary routing
                        "source_info": self.source_definitions[source],
                    }
                )

        # Sort by relevance score
        fallback_sources.sort(key=lambda x: -x["relevance_score"])

        return fallback_sources[:3]  # Return top 3 fallback sources

    def _calculate_fallback_relevance(
        self, query: str, source: str, domain: str
    ) -> float:
        """Calculate relevance score for fallback routing"""
        source_info = self.source_definitions.get(source, {})

        # Start with base quality score
        base_score = source_info.get("quality_score", 0.5) * 0.6  # Reduced for fallback

        # Domain match bonus
        if domain in source_info.get("domains", []):
            base_score += 0.2

        # Singapore bonus for Singapore sources
        if source_info.get("singapore_priority", False):
            if self._should_include_singapore_sources(query):
                base_score += 0.15

        # General platform bonuses
        if source == "kaggle" and any(
            term in query.lower() for term in ["data", "dataset", "analysis"]
        ):
            base_score += 0.1
        elif source == "world_bank" and any(
            term in query.lower() for term in ["global", "country", "statistics"]
        ):
            base_score += 0.1
        elif source == "zenodo" and any(
            term in query.lower() for term in ["research", "academic", "study"]
        ):
            base_score += 0.1

        return min(base_score, 0.8)  # Cap fallback scores at 0.8

    def _route_singapore_first(
        self, query: str, domain: str, rules: Dict
    ) -> List[Dict]:
        """Route with Singapore sources prioritized"""
        sources = []

        # 1. Singapore sources first
        for source in rules["singapore_sources"]:
            if source in self.source_definitions:
                sources.append(
                    {
                        "source": source,
                        "relevance_score": self._calculate_relevance(
                            query, source, domain
                        ),
                        "routing_reason": "singapore_first_priority",
                        "priority": 1,
                        "source_info": self.source_definitions[source],
                    }
                )

        # 2. Primary domain sources
        for source in rules["primary_sources"]:
            if (
                source not in rules["singapore_sources"]
                and source in self.source_definitions
            ):
                sources.append(
                    {
                        "source": source,
                        "relevance_score": self._calculate_relevance(
                            query, source, domain
                        ),
                        "routing_reason": "domain_primary",
                        "priority": 2,
                        "source_info": self.source_definitions[source],
                    }
                )

        # 3. Secondary sources
        for source in rules["secondary_sources"]:
            if (
                source not in [s["source"] for s in sources]
                and source in self.source_definitions
            ):
                sources.append(
                    {
                        "source": source,
                        "relevance_score": self._calculate_relevance(
                            query, source, domain
                        ),
                        "routing_reason": "secondary",
                        "priority": 3,
                        "source_info": self.source_definitions[source],
                    }
                )

        # Sort by priority, then by relevance
        sources.sort(key=lambda x: (x["priority"], -x["relevance_score"]))

        return sources[:5]  # Return top 5

    def _route_domain_first(self, query: str, domain: str, rules: Dict) -> List[Dict]:
        """Route with domain-specific sources prioritized"""
        sources = []

        # 1. Primary domain sources
        for source in rules["primary_sources"]:
            if source in self.source_definitions:
                sources.append(
                    {
                        "source": source,
                        "relevance_score": self._calculate_relevance(
                            query, source, domain
                        ),
                        "routing_reason": "domain_primary",
                        "priority": 1,
                        "source_info": self.source_definitions[source],
                    }
                )

        # 2. Singapore sources if applicable
        if self._should_include_singapore_sources(query):
            for source in rules["singapore_sources"]:
                if (
                    source not in [s["source"] for s in sources]
                    and source in self.source_definitions
                ):
                    sources.append(
                        {
                            "source": source,
                            "relevance_score": self._calculate_relevance(
                                query, source, domain
                            ),
                            "routing_reason": "singapore_secondary",
                            "priority": 2,
                            "source_info": self.source_definitions[source],
                        }
                    )

        # 3. Secondary sources
        for source in rules["secondary_sources"]:
            if (
                source not in [s["source"] for s in sources]
                and source in self.source_definitions
            ):
                sources.append(
                    {
                        "source": source,
                        "relevance_score": self._calculate_relevance(
                            query, source, domain
                        ),
                        "routing_reason": "secondary",
                        "priority": 3,
                        "source_info": self.source_definitions[source],
                    }
                )

        # Sort by priority, then by relevance
        sources.sort(key=lambda x: (x["priority"], -x["relevance_score"]))

        return sources[:5]

    def _calculate_relevance(self, query: str, source: str, domain: str) -> float:
        """Calculate relevance score for query-source pair"""

        # Check training mappings first
        if query.lower() in self.training_mappings:
            mapping = self.training_mappings[query.lower()]
            if mapping["source"] == source:
                return mapping["score"]

        # Base relevance from source definition
        source_info = self.source_definitions.get(source, {})
        base_relevance = 0.5

        # Domain match bonus
        if domain in source_info.get("domains", []):
            base_relevance += 0.3

        # Singapore query bonus for Singapore sources
        if "singapore" in query.lower() and source_info.get(
            "singapore_priority", False
        ):
            base_relevance += 0.2

        # Quality score influence
        quality_score = source_info.get("quality_score", 0.5)
        base_relevance = 0.7 * base_relevance + 0.3 * quality_score

        return min(base_relevance, 1.0)  # Cap at 1.0

    def _should_include_singapore_sources(self, query: str) -> bool:
        """Determine if Singapore sources should be included"""
        query_lower = query.lower()

        # Explicit Singapore mentions
        if "singapore" in query_lower or "sg" in query_lower:
            return True

        # Singapore-specific terms
        singapore_terms = ["hdb", "mrt", "lta", "singstat", "cpf", "medisave"]
        if any(term in query_lower for term in singapore_terms):
            return True

        # Generic terms that could benefit from Singapore data
        generic_terms = ["housing", "transport", "population", "education", "health"]
        if any(term in query_lower for term in generic_terms):
            # Only if not explicitly global
            global_terms = ["global", "international", "worldwide", "world"]
            if not any(term in query_lower for term in global_terms):
                return True

        return False


def route_query_to_sources(
    query: str, domain: str, singapore_first: bool = False, config: Dict = None
) -> List[Dict]:
    """Route a query to appropriate sources (external interface)"""
    router = SourcePriorityRouter(config)
    return router.route_query(query, domain, singapore_first)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the source priority router
    print("🧪 Testing Source Priority Router\n")

    router = SourcePriorityRouter()

    test_cases = [
        # Singapore-first queries
        ("singapore housing prices", "singapore", True),
        ("mrt data singapore", "transportation", True),
        ("singapore population statistics", "singapore", True),
        # Domain-specific queries
        ("psychology research datasets", "psychology", False),
        ("climate change global indicators", "climate", False),
        ("machine learning benchmark datasets", "machine_learning", False),
        # Generic queries
        ("housing statistics", "general", False),
        ("transportation data", "transportation", False),
        ("economic indicators", "economics", False),
    ]

    for query, domain, singapore_first in test_cases:
        print(
            f"📝 Query: '{query}' (Domain: {domain}, Singapore-first: {singapore_first})"
        )

        # Get routing
        sources = router.route_query(query, domain, singapore_first)

        print("  Top sources:")

        for i, source in enumerate(sources[:3], 1):
            print(
                f"    {i}. {source['source']} (relevance: {source['relevance_score']:.2f})"
            )
            print(f"       Reason: {source['routing_reason']}")
            print(
                f"       Description: {source['source_info'].get('description', 'N/A')}"
            )

        print()

    print("✅ Source Priority Router testing complete!")
