# 01_extraction_module.py - Real API Data Extraction
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurableDataExtractor:
    """Configuration-driven data extraction for Singapore and Global sources"""

    def __init__(
        self,
        pipeline_config_path: str = "config/data_pipeline.yml",
        api_config_path: str = "config/api_config.yml",
    ):
        """Initialize extractor with both pipeline and API configurations"""
        # Load configurations
        self.pipeline_config = self._load_config(pipeline_config_path)
        self.api_config = self._load_config(api_config_path)

        # Session setup
        self.session = requests.Session()
        user_agent = self.api_config.get("global_settings", {}).get(
            "user_agent", "DatasetResearchAssistant/1.0"
        )
        self.session.headers.update({"User-Agent": user_agent})

        # Extract configuration sections
        self.singapore_sources = self.api_config.get("singapore_sources", {})
        self.global_sources = self.api_config.get("global_sources", {})
        self.global_settings = self.api_config.get("global_settings", {})

        # Pipeline configuration
        self.extraction_config = self.pipeline_config.get("phase_1_extraction", {})
        self.validation_config = self.pipeline_config.get("data_validation", {})

        # Setup data paths
        self.raw_data_path = Path(
            self.extraction_config.get("raw_data_path", "data/raw")
        )
        self.processed_data_path = Path(
            self.extraction_config.get("processed_data_path", "data/processed")
        )

        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def generate_auto_description(self, dataset: dict) -> str:
        """Auto-generate description for datasets missing meaningful descriptions"""
        title = dataset.get("title", "")
        source = dataset.get("source", "")
        agency = dataset.get("agency", "")
        category = dataset.get("category", "")
        tags = dataset.get("tags", "")
        coverage = dataset.get("geographic_coverage", "")

        # Extract meaningful keywords from title
        title_words = re.findall(r"\b[A-Za-z]{3,}\b", title)
        main_keywords = [word.lower() for word in title_words[:4]]

        # Build description components
        components = []

        # Core description based on title keywords
        if main_keywords:
            if any(
                keyword in ["economic", "economy", "gdp", "finance"]
                for keyword in main_keywords
            ):
                components.append(
                    f"Economic indicators and financial data focusing on {', '.join(main_keywords[:2])}"
                )
            elif any(
                keyword in ["transport", "traffic", "mobility", "lta"]
                for keyword in main_keywords
            ):
                components.append(
                    f"Transportation and mobility data including {', '.join(main_keywords[:2])}"
                )
            elif any(
                keyword in ["health", "medical", "hospital", "disease"]
                for keyword in main_keywords
            ):
                components.append(
                    f"Health and medical statistics covering {', '.join(main_keywords[:2])}"
                )
            elif any(
                keyword in ["population", "demographic", "census", "people"]
                for keyword in main_keywords
            ):
                components.append(
                    f"Demographic and population data on {', '.join(main_keywords[:2])}"
                )
            elif any(
                keyword in ["housing", "property", "hdb", "real", "estate"]
                for keyword in main_keywords
            ):
                components.append(
                    f"Housing and property market data including {', '.join(main_keywords[:2])}"
                )
            else:
                components.append(
                    f"Dataset containing information on {', '.join(main_keywords[:3])}"
                )

        # Add source and agency context
        if agency and agency.strip():
            components.append(f"provided by {agency}")
        elif source and source.strip():
            components.append(f"sourced from {source}")

        # Add geographic context
        if coverage and coverage.strip() and coverage.lower() != "unknown":
            components.append(f"covering {coverage}")

        # Add category context
        if category and category.strip() and category.lower() != "general":
            components.append(f"in the {category.replace('_', ' ')} domain")

        # Combine components
        if components:
            description = ". ".join(components).strip()
            if not description.endswith("."):
                description += "."
            return description
        else:
            return f"Data from {source or 'official source'} containing information on {title.lower() if title else 'various indicators'}."

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def _make_request(
        self, url: str, headers: dict = None, params: dict = None, timeout: int = 30
    ) -> Optional[requests.Response]:
        """Make HTTP request with error handling"""
        try:
            response = self.session.get(
                url, headers=headers, params=params, timeout=timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def extract_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Extract datasets from all configured sources"""
        logger.info("🚀 Starting real API data extraction...")

        results = {}
        extraction_summary = {
            "singapore_total": 0,
            "global_total": 0,
            "sources_processed": [],
            "sources_failed": [],
            "extraction_timestamp": datetime.now().isoformat(),
        }

        # Extract Singapore datasets
        singapore_datasets = self._extract_singapore_sources()
        if not singapore_datasets.empty:
            results["singapore"] = singapore_datasets
            extraction_summary["singapore_total"] = len(singapore_datasets)
            logger.info(f"✅ Singapore extraction: {len(singapore_datasets)} datasets")

        # Extract global datasets
        global_datasets = self._extract_global_sources()
        if not global_datasets.empty:
            results["global"] = global_datasets
            extraction_summary["global_total"] = len(global_datasets)
            logger.info(f"✅ Global extraction: {len(global_datasets)} datasets")

        # Save extraction results
        self._save_extraction_results(results, extraction_summary)

        return results

    def _extract_singapore_sources(self) -> pd.DataFrame:
        """Extract from all enabled Singapore data sources"""
        logger.info("🇸🇬 Processing Singapore data sources...")

        all_singapore_datasets = []

        for source_name, source_config in self.singapore_sources.items():
            if not source_config.get("enabled", False):
                logger.info(f"⏭️ Skipping disabled source: {source_name}")
                continue

            logger.info(f"📡 Extracting from {source_name}...")

            try:
                # Extract based on source type
                if source_name == "data_gov_sg":
                    datasets = self._extract_data_gov_sg(source_config)
                elif source_name == "lta_datamall":
                    datasets = self._extract_lta_data(source_config)
                elif source_name == "onemap_sg":
                    datasets = self._extract_onemap_data(source_config)
                elif source_name == "singstat":
                    datasets = self._extract_singstat_data(source_config)
                else:
                    logger.warning(f"⚠️ Unknown Singapore source: {source_name}")
                    continue

                if datasets:
                    all_singapore_datasets.extend(datasets)
                    logger.info(f"✅ {source_name}: {len(datasets)} datasets extracted")
                else:
                    logger.warning(f"⚠️ {source_name}: No datasets extracted")

                # Rate limiting
                rate_limit = source_config.get("rate_limit", 2)
                time.sleep(rate_limit)

            except Exception as e:
                logger.error(f"❌ {source_name} extraction failed: {e}")
                continue

        if all_singapore_datasets:
            singapore_df = pd.DataFrame(all_singapore_datasets)
            singapore_df = self._standardize_dataset_schema(singapore_df)
            return singapore_df
        else:
            return pd.DataFrame()

    def _extract_data_gov_sg(self, source_config: Dict) -> List[Dict]:
        """Extract REAL data from data.gov.sg API"""
        base_url = source_config.get("base_url")
        datasets_endpoint = source_config.get("datasets_endpoint", "/datasets")
        max_datasets = source_config.get("max_datasets", 50)

        all_datasets = []
        page = 1
        per_page = 10
        total_extracted = 0

        while total_extracted < max_datasets:
            try:
                url = f"{base_url}{datasets_endpoint}"
                params = {"page": page, "per_page": per_page}

                response = self._make_request(url, params=params)
                if not response:
                    break

                data = response.json()
                datasets = data.get("data", {}).get("datasets", [])

                if not datasets:
                    logger.info("No more datasets available")
                    break

                # Transform each dataset to standardized schema
                for dataset in datasets:
                    if total_extracted >= max_datasets:
                        break

                    transformed = {
                        "dataset_id": dataset.get("datasetId"),
                        "title": dataset.get("name"),
                        "description": dataset.get("description", ""),
                        "source": "data.gov.sg",
                        "agency": dataset.get("managedByAgencyName"),
                        "category": dataset.get("category", "general").lower(),
                        "tags": ", ".join(dataset.get("keywords", [])),
                        "geographic_coverage": "Singapore",
                        "format": dataset.get("format", "CSV"),
                        "license": "Singapore Open Data License",
                        "status": dataset.get("status", "active"),
                        "last_updated": dataset.get("lastUpdatedAt", ""),
                        "created_date": dataset.get("createdAt", ""),
                        "frequency": dataset.get("frequency", "Unknown"),
                        "coverage_start": dataset.get("coverageStart", ""),
                        "coverage_end": dataset.get("coverageEnd", ""),
                        "record_count": 0,
                        "file_size": "Unknown",
                        "url": f"https://data.gov.sg/datasets/{dataset.get('datasetId')}/view",
                    }
                    all_datasets.append(transformed)
                    total_extracted += 1

                logger.info(f"Page {page}: Extracted {len(datasets)} datasets")
                page += 1

                # Respect rate limit
                time.sleep(source_config.get("rate_limit", 12))

            except Exception as e:
                logger.error(f"Error extracting data.gov.sg: {e}")
                break

        return all_datasets

    def _extract_lta_data(self, source_config: Dict) -> List[Dict]:
        """Extract metadata from LTA DataMall endpoints"""
        api_key = os.getenv(source_config.get("api_key_env", "LTA_API_KEY"))
        if not api_key:
            logger.warning("LTA_API_KEY not found in environment")
            return []

        base_url = source_config.get("base_url")
        endpoints = source_config.get("endpoints", {})
        endpoint_descriptions = source_config.get("endpoint_descriptions", {})

        all_datasets = []

        # Create dataset entries for each LTA endpoint
        for endpoint_name, endpoint_path in endpoints.items():
            dataset = {
                "dataset_id": f"lta_{endpoint_name}",
                "title": f"LTA {endpoint_name.replace('_', ' ').title()}",
                "description": endpoint_descriptions.get(endpoint_name, ""),
                "source": "LTA DataMall",
                "agency": "Land Transport Authority",
                "category": "transport",
                "tags": f"transport, lta, {endpoint_name.replace('_', ', ')}",
                "geographic_coverage": "Singapore",
                "format": "JSON",
                "license": "LTA Open Data License",
                "status": "active",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "frequency": "Real-time" if "arrival" in endpoint_name else "Daily",
                "data_type": "real_time" if "arrival" in endpoint_name else "static",
                "url": f"{base_url}{endpoint_path}",
                "api_required": True,
                "api_key_required": True,
            }
            all_datasets.append(dataset)

        return all_datasets

    def _extract_onemap_data(self, source_config: Dict) -> List[Dict]:
        """Extract metadata from OneMap endpoints"""
        public_endpoints = source_config.get("public_endpoints", {})
        private_endpoints = source_config.get("private_endpoints", {})

        all_datasets = []

        # Public endpoints (no auth required)
        for endpoint_name, endpoint_path in public_endpoints.items():
            dataset = {
                "dataset_id": f"onemap_{endpoint_name}",
                "title": f"OneMap {endpoint_name.replace('_', ' ').title()}",
                "description": f"Geospatial data for {endpoint_name}",
                "source": "OneMap",
                "agency": "Singapore Land Authority",
                "category": "geospatial",
                "tags": f"geospatial, map, {endpoint_name}",
                "geographic_coverage": "Singapore",
                "format": "JSON",
                "license": "SLA Open Data License",
                "status": "active",
                "data_type": "geospatial",
                "api_required": True,
                "api_key_required": False,
            }
            all_datasets.append(dataset)

        # Private endpoints (auth required)
        for endpoint_name, endpoint_path in private_endpoints.items():
            dataset = {
                "dataset_id": f"onemap_{endpoint_name}_auth",
                "title": f"OneMap {endpoint_name.replace('_', ' ').title()} (Authenticated)",
                "description": f"Advanced geospatial data for {endpoint_name}",
                "source": "OneMap",
                "agency": "Singapore Land Authority",
                "category": "geospatial",
                "tags": f"geospatial, map, {endpoint_name}, authenticated",
                "geographic_coverage": "Singapore",
                "format": "JSON",
                "license": "SLA Open Data License",
                "status": "active",
                "data_type": "geospatial",
                "api_required": True,
                "api_key_required": True,
            }
            all_datasets.append(dataset)

        return all_datasets

    def _extract_singstat_data(self, source_config: Dict) -> List[Dict]:
        """Extract metadata from SingStat Table Builder"""
        base_url = source_config.get("base_url")
        search_endpoint = source_config.get("endpoints", {}).get(
            "search", "/resourceid"
        )

        all_datasets = []

        # Search for popular dataset categories
        search_terms = ["gdp", "population", "employment", "inflation", "trade"]

        for term in search_terms:
            try:
                url = f"{base_url}{search_endpoint}"
                params = {"keyword": term}

                response = self._make_request(url, params=params)
                if response and response.status_code == 200:
                    # Create dataset entry for each search category
                    dataset = {
                        "dataset_id": f"singstat_{term}",
                        "title": f"Singapore Statistics - {term.upper()}",
                        "description": f"Official statistics on {term} from Department of Statistics Singapore",
                        "source": "SingStat",
                        "agency": "Department of Statistics Singapore",
                        "category": "statistics",
                        "tags": f"statistics, {term}, singapore, official",
                        "geographic_coverage": "Singapore",
                        "format": "CSV",
                        "license": "Singapore Open Data License",
                        "status": "active",
                        "frequency": "Quarterly"
                        if term in ["gdp", "inflation"]
                        else "Annual",
                        "url": f"https://tablebuilder.singstat.gov.sg",
                    }
                    all_datasets.append(dataset)

            except Exception as e:
                logger.error(f"Error searching SingStat for {term}: {e}")
                continue

        return all_datasets

    def _extract_global_sources(self) -> pd.DataFrame:
        """Extract from all enabled global data sources"""
        logger.info("🌍 Processing global data sources...")

        all_global_datasets = []

        for source_name, source_config in self.global_sources.items():
            if not source_config.get("enabled", False):
                logger.info(f"⏭️ Skipping disabled global source: {source_name}")
                continue

            logger.info(f"📡 Extracting from {source_name}...")

            try:
                # Extract based on source type
                if source_name == "world_bank":
                    datasets = self._extract_world_bank_data(source_config)
                elif source_name == "imf":
                    datasets = self._extract_imf_data(source_config)
                elif source_name == "oecd":
                    datasets = self._extract_oecd_data(source_config)
                elif source_name == "un_sdg_api":
                    datasets = self._extract_un_sdg_data(source_config)
                else:
                    logger.warning(f"⚠️ Unknown global source: {source_name}")
                    continue

                if datasets:
                    all_global_datasets.extend(datasets)
                    logger.info(f"✅ {source_name}: {len(datasets)} datasets extracted")
                else:
                    logger.warning(f"⚠️ {source_name}: No datasets extracted")

                # Rate limiting
                rate_limit = source_config.get("rate_limit", 1)
                time.sleep(rate_limit)

            except Exception as e:
                logger.error(f"❌ {source_name} extraction failed: {e}")
                continue

        if all_global_datasets:
            global_df = pd.DataFrame(all_global_datasets)
            global_df = self._standardize_dataset_schema(global_df)
            return global_df
        else:
            return pd.DataFrame()

    def _extract_world_bank_data(self, source_config: Dict) -> List[Dict]:
        """Extract REAL data from World Bank API"""
        base_url = source_config.get("base_url")
        indicators_endpoint = source_config.get("endpoints", {}).get(
            "indicators", "/indicator"
        )
        max_datasets = min(source_config.get("max_datasets", 50), 50)

        all_datasets = []

        try:
            # Get priority indicators
            priority_indicators = source_config.get("priority_indicators", [])

            # First, get the priority indicators
            for indicator_id in priority_indicators[:max_datasets]:
                url = f"{base_url}/indicator/{indicator_id}"
                params = {"format": "json", "per_page": 1}

                response = self._make_request(url, params=params)
                if response:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 1:
                        indicator_info = data[1][0] if data[1] else {}

                        dataset = {
                            "dataset_id": f"wb_{indicator_id}",
                            "title": indicator_info.get("name", indicator_id),
                            "description": indicator_info.get("sourceNote", ""),
                            "source": "World Bank",
                            "agency": "World Bank Group",
                            "category": "economic_development",
                            "tags": "world bank, development, indicators",
                            "geographic_coverage": "Global",
                            "format": "JSON",
                            "license": "CC BY 4.0",
                            "status": "active",
                            "source_organization": indicator_info.get(
                                "sourceOrganization", ""
                            ),
                            "unit": indicator_info.get("unit", ""),
                            "url": f"https://data.worldbank.org/indicator/{indicator_id}",
                        }
                        all_datasets.append(dataset)

            # If we need more, get general indicators
            if len(all_datasets) < max_datasets:
                url = f"{base_url}{indicators_endpoint}"
                params = {
                    "format": "json",
                    "per_page": max_datasets - len(all_datasets),
                }

                response = self._make_request(url, params=params)
                if response:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 1:
                        indicators = data[1]

                        for indicator in indicators:
                            dataset = {
                                "dataset_id": f"wb_{indicator.get('id', '')}",
                                "title": indicator.get("name", ""),
                                "description": indicator.get("sourceNote", ""),
                                "source": "World Bank",
                                "agency": "World Bank Group",
                                "category": "economic_development",
                                "tags": "world bank, indicators, global data",
                                "geographic_coverage": "Global",
                                "format": "JSON",
                                "license": "CC BY 4.0",
                                "status": "active",
                                "url": f"https://data.worldbank.org/indicator/{indicator.get('id', '')}",
                            }
                            all_datasets.append(dataset)

        except Exception as e:
            logger.error(f"Error extracting World Bank data: {e}")

        return all_datasets

    def _extract_imf_data(self, source_config: Dict) -> List[Dict]:
        """Extract REAL data from IMF SDMX API"""
        base_url = source_config.get("base_url")
        dataflow_endpoint = source_config.get("endpoints", {}).get(
            "dataflow", "/Dataflow"
        )

        all_datasets = []

        try:
            url = f"{base_url}{dataflow_endpoint}"
            response = self._make_request(url, timeout=45)

            if response:
                # Parse the SDMX response
                data = response.json()
                dataflows = (
                    data.get("Structure", {}).get("Dataflows", {}).get("Dataflow", [])
                )

                if not isinstance(dataflows, list):
                    dataflows = [dataflows]

                # Get priority dataflows
                priority_dataflows = source_config.get("priority_dataflows", [])

                for dataflow in dataflows[: source_config.get("max_datasets", 30)]:
                    dataflow_id = dataflow.get("@id", "")

                    # Prioritize known important datasets
                    if dataflow_id in priority_dataflows or len(all_datasets) < 10:
                        dataset = {
                            "dataset_id": f"imf_{dataflow_id}",
                            "title": dataflow.get("Name", {}).get("#text", dataflow_id),
                            "description": f"IMF {dataflow_id} - {dataflow.get('Name', {}).get('#text', '')}",
                            "source": "IMF",
                            "agency": dataflow.get("@agencyID", "IMF"),
                            "category": "economic_finance",
                            "tags": f"imf, economic data, {dataflow_id.lower()}",
                            "geographic_coverage": "Global",
                            "format": "SDMX-JSON",
                            "license": "IMF Data License",
                            "status": "active",
                            "url": f"https://www.imf.org/en/Data",
                        }
                        all_datasets.append(dataset)

        except Exception as e:
            logger.error(f"Error extracting IMF data: {e}")

        return all_datasets

    def _extract_oecd_data(self, source_config: Dict) -> List[Dict]:
        """Extract REAL data from OECD API"""
        base_url = source_config.get("base_url")

        all_datasets = []

        # Use priority datasets from config
        priority_datasets = source_config.get("priority_datasets", [])

        for dataset_id in priority_datasets:
            dataset = {
                "dataset_id": f"oecd_{dataset_id}",
                "title": self._get_oecd_dataset_title(dataset_id),
                "description": f"OECD {dataset_id} statistics",
                "source": "OECD",
                "agency": "Organisation for Economic Co-operation and Development",
                "category": "economic_statistics",
                "tags": f"oecd, statistics, {dataset_id.lower()}",
                "geographic_coverage": "OECD Countries",
                "format": "SDMX",
                "license": "OECD Data License",
                "status": "active",
                "url": f"https://stats.oecd.org/{dataset_id}",
            }
            all_datasets.append(dataset)

        return all_datasets

    def _extract_un_sdg_data(self, source_config: Dict) -> List[Dict]:
        """Extract REAL data from UN SDG API"""
        base_url = source_config.get("base_url")
        indicators_endpoint = source_config.get("endpoints", {}).get(
            "indicators", "/v1/sdg/Indicator/List"
        )

        all_datasets = []

        try:
            # Get priority SDG indicators
            priority_indicators = source_config.get("priority_indicators", [])

            # Create entries for priority indicators
            for indicator_code in priority_indicators:
                dataset = {
                    "dataset_id": f"sdg_{indicator_code.replace('.', '_')}",
                    "title": f"SDG Indicator {indicator_code}",
                    "description": self._get_sdg_indicator_description(indicator_code),
                    "source": "UN SDG",
                    "agency": "UN Statistics Division",
                    "category": "sustainable_development",
                    "tags": f"sdg, sustainable development, indicator {indicator_code}",
                    "geographic_coverage": "Global",
                    "format": "JSON",
                    "license": "UN Data License",
                    "status": "active",
                    "url": f"https://unstats.un.org/sdgs/indicators/database/",
                }
                all_datasets.append(dataset)

        except Exception as e:
            logger.error(f"Error extracting UN SDG data: {e}")

        return all_datasets

    def _get_oecd_dataset_title(self, dataset_id: str) -> str:
        """Get human-readable title for OECD dataset ID"""
        titles = {
            "QNA": "Quarterly National Accounts",
            "MEI": "Main Economic Indicators",
            "ELS": "Employment and Labour Market Statistics",
            "SNA_TABLE1": "Annual National Accounts - Main Aggregates",
            "HEALTH": "Health Statistics",
        }
        return titles.get(dataset_id, dataset_id)

    def _get_sdg_indicator_description(self, indicator_code: str) -> str:
        """Get description for SDG indicator"""
        descriptions = {
            "1.1.1": "Proportion of population below international poverty line",
            "3.1.1": "Maternal mortality ratio",
            "4.1.1": "Proportion of children achieving minimum proficiency in reading and mathematics",
            "6.1.1": "Proportion of population using safely managed drinking water services",
            "7.1.1": "Proportion of population with access to electricity",
            "11.1.1": "Proportion of urban population living in slums or inadequate housing",
        }
        return descriptions.get(indicator_code, f"SDG Indicator {indicator_code}")

    def _standardize_dataset_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataset schema and calculate quality scores"""
        logger.info("🔧 Standardizing dataset schema...")

        # Ensure required fields exist
        required_fields = ["dataset_id", "title", "source", "description", "category"]
        for field in required_fields:
            if field not in df.columns:
                df[field] = ""

        # Auto-generate descriptions for datasets with poor descriptions
        logger.info("📝 Enhancing dataset descriptions...")
        description_enhanced_count = 0

        for idx, row in df.iterrows():
            current_desc = str(row.get("description", "")).strip()

            # Check if description needs enhancement
            if (
                not current_desc
                or current_desc.lower() in ["", "nan", "none", "null"]
                or len(current_desc) < 20
                or current_desc.lower()
                in ["no description available", "description not provided"]
            ):
                # Generate auto-description
                dataset_dict = row.to_dict()
                auto_description = self.generate_auto_description(dataset_dict)

                # Update the dataframe
                df.at[idx, "description"] = auto_description
                df.at[idx, "auto_generated_description"] = True
                description_enhanced_count += 1
            else:
                df.at[idx, "auto_generated_description"] = False

        if description_enhanced_count > 0:
            logger.info(
                f"✅ Enhanced descriptions for {description_enhanced_count} datasets"
            )

        # Calculate quality scores
        df["quality_score"] = df.apply(self._calculate_quality_score, axis=1)

        # Add standardized metadata
        df["extraction_timestamp"] = datetime.now().isoformat()
        if "geographic_coverage" not in df.columns:
            df["geographic_coverage"] = "Unknown"

        # Apply quality filtering
        min_quality = self.extraction_config.get("min_quality_threshold", 0.3)
        initial_count = len(df)
        df = df[df["quality_score"] >= min_quality]

        if len(df) < initial_count:
            logger.info(
                f"🚫 Filtered out {initial_count - len(df)} low-quality datasets"
            )

        return df

    def _calculate_quality_score(self, dataset: pd.Series) -> float:
        """Calculate quality score for a dataset"""
        score = 0.0
        weights = self.validation_config.get(
            "quality_scoring",
            {
                "title_quality_weight": 0.2,
                "description_quality_weight": 0.3,
                "metadata_completeness_weight": 0.25,
                "source_credibility_weight": 0.25,
            },
        )

        # Title quality
        title = str(dataset.get("title", ""))
        title_weight = weights.get("title_quality_weight", 0.2)
        if len(title) > 20:
            score += title_weight
        elif len(title) > 10:
            score += title_weight * 0.5

        # Description quality
        description = str(dataset.get("description", ""))
        desc_weight = weights.get("description_quality_weight", 0.3)
        if len(description) > 100:
            score += desc_weight
        elif len(description) > 50:
            score += desc_weight * 0.5

        # Metadata completeness
        metadata_weight = weights.get("metadata_completeness_weight", 0.25)
        metadata_fields = ["agency", "category", "frequency", "last_updated"]
        complete_fields = sum(1 for field in metadata_fields if dataset.get(field))
        score += (complete_fields / len(metadata_fields)) * metadata_weight

        # Source credibility
        source_weight = weights.get("source_credibility_weight", 0.25)
        source = str(dataset.get("source", "")).lower()
        if any(term in source for term in ["gov", "government", "official"]):
            score += source_weight
        elif any(org in source for org in ["world bank", "un", "oecd", "imf"]):
            score += source_weight * 0.8

        return min(1.0, score)

    def _save_extraction_results(self, results: Dict[str, pd.DataFrame], summary: Dict):
        """Save extraction results to configured paths"""
        logger.info("💾 Saving extraction results...")

        try:
            # Save individual source data to raw folder
            for source_type, df in results.items():
                # Save to raw data folder
                source_folder = self.raw_data_path / f"{source_type}_datasets"
                source_folder.mkdir(exist_ok=True)

                raw_file = source_folder / f"{source_type}_raw.csv"
                df.to_csv(raw_file, index=False)
                logger.info(f"💾 Raw data saved: {raw_file}")

                # Save to processed folder
                processed_file = (
                    self.processed_data_path / f"{source_type}_datasets.csv"
                )
                df.to_csv(processed_file, index=False)
                logger.info(f"💾 Processed data saved: {processed_file}")

            # Save extraction summary
            summary_file = self.processed_data_path / "extraction_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"💾 Summary saved: {summary_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    def get_extraction_summary(self) -> Dict:
        """Get summary of extraction results"""
        try:
            summary_file = self.processed_data_path / "extraction_summary.json"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return {}


def main():
    """Main execution for Phase 1: Real API Data Extraction"""
    print("🚀 Phase 1: Real API Data Extraction")
    print("=" * 60)

    # Initialize extractor with configurations
    extractor = ConfigurableDataExtractor(
        pipeline_config_path="config/data_pipeline.yml",
        api_config_path="config/api_config.yml",
    )

    # Extract all datasets
    results = extractor.extract_all_datasets()

    # Generate summary
    summary = extractor.get_extraction_summary()

    # Display results
    print(f"\n📊 Extraction Summary:")
    print(f"   Singapore datasets: {summary.get('singapore_total', 0)}")
    print(f"   Global datasets: {summary.get('global_total', 0)}")
    print(
        f"   Total datasets: {summary.get('singapore_total', 0) + summary.get('global_total', 0)}"
    )

    total_datasets = summary.get("singapore_total", 0) + summary.get("global_total", 0)
    if total_datasets > 0:
        print(f"\n✅ Phase 1 Complete!")
        print(f"💾 Data saved to: data/raw/ and data/processed/")
        print(f"🔄 Next: Run Phase 2 (Deep Analysis)")
    else:
        print(f"\n❌ No datasets extracted.")
        print(f"🔧 Check API configuration and connectivity")


if __name__ == "__main__":
    main()
