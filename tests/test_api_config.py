#!/usr/bin/env python3
"""
API Configuration Validator for Dataset Research Assistant

This script tests all API endpoints defined in config/api_config.yml
to ensure they are accessible and return expected data formats.

Usage:
    python tests/test_api_config.py [--source SOURCE_NAME] [--verbose]

Examples:
    python tests/test_api_config.py                    # Test all sources
    python tests/test_api_config.py --source data_gov_sg  # Test specific source
    python tests/test_api_config.py --verbose          # Detailed output
"""

import os
import sys
import yaml
import requests
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Environment setup
from dotenv import load_dotenv

load_dotenv()


class APIConfigValidator:
    """Validates API configuration and tests endpoint connectivity."""

    def __init__(self, config_path: str = "config/api_config.yml"):
        """Initialize validator with configuration file."""
        self.config_path = Path(project_root) / config_path
        self.config = self.load_config()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_sources": 0,
            "successful": 0,
            "failed": 0,
            "sources": {},
        }

    def load_config(self) -> dict:
        """Load and parse API configuration file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                print(f"✅ Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)

    def get_env_var(self, var_name: str, required: bool = False) -> Optional[str]:
        """Get environment variable with error handling."""
        value = os.getenv(var_name)
        if required and not value:
            print(f"⚠️  Required environment variable {var_name} not found")
            return None
        return value

    def make_request(
        self, url: str, headers: dict = None, timeout: int = 30
    ) -> Tuple[bool, dict]:
        """Make HTTP request with error handling."""
        try:
            # Default headers
            default_headers = {
                "User-Agent": "DatasetResearchAssistant/1.0 (Test)",
                "Accept": "application/json",
            }
            if headers:
                default_headers.update(headers)

            response = requests.get(url, headers=default_headers, timeout=timeout)

            # Check if response is successful
            if response.status_code == 200:
                try:
                    # Try to parse as JSON
                    data = response.json()
                    return True, {
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "data_sample": str(data)[:200] + "..."
                        if len(str(data)) > 200
                        else str(data),
                        "response_size": len(response.content),
                    }
                except json.JSONDecodeError:
                    # Non-JSON response (e.g., CSV, XML)
                    return True, {
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "content_sample": response.text[:200] + "..."
                        if len(response.text) > 200
                        else response.text,
                        "response_size": len(response.content),
                    }
            else:
                return False, {
                    "status_code": response.status_code,
                    "error": response.text[:200],
                    "headers": dict(response.headers),
                }

        except requests.exceptions.Timeout:
            return False, {"error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return False, {"error": "Connection error"}
        except Exception as e:
            return False, {"error": str(e)}

    def test_singapore_sources(self, verbose: bool = False) -> Dict[str, dict]:
        """Test all Singapore data sources."""
        print("\n🇸🇬 Testing Singapore Data Sources...")
        singapore_results = {}

        singapore_sources = self.config.get("singapore_sources", {})

        for source_name, source_config in singapore_sources.items():
            if not source_config.get("enabled", True):
                print(f"⏭️  Skipping {source_name} (disabled)")
                continue

            print(f"\n📊 Testing {source_name}...")
            singapore_results[source_name] = self.test_source(
                source_name, source_config, verbose
            )
            time.sleep(2)  # Rate limiting between tests

        return singapore_results

    def test_global_sources(self, verbose: bool = False) -> Dict[str, dict]:
        """Test all global data sources."""
        print("\n🌍 Testing Global Data Sources...")
        global_results = {}

        global_sources = self.config.get("global_sources", {})

        for source_name, source_config in global_sources.items():
            if not source_config.get("enabled", True):
                print(f"⏭️  Skipping {source_name} (disabled)")
                continue

            print(f"\n📈 Testing {source_name}...")
            global_results[source_name] = self.test_source(
                source_name, source_config, verbose
            )
            time.sleep(1)  # Rate limiting between tests

        return global_results

    def test_source(
        self, source_name: str, source_config: dict, verbose: bool = False
    ) -> dict:
        """Test individual data source."""
        result = {
            "source_name": source_name,
            "enabled": source_config.get("enabled", True),
            "tests": [],
            "overall_status": "unknown",
            "auth_required": False,
            "auth_status": "not_required",
        }

        base_url = source_config.get("base_url", "")
        if not base_url:
            result["overall_status"] = "failed"
            result["error"] = "No base_url configured"
            return result

        # Check authentication requirements
        auth_header = None
        api_key_env = source_config.get("api_key_env")

        if api_key_env:
            result["auth_required"] = True
            api_key = self.get_env_var(api_key_env)
            if api_key:
                auth_header_name = source_config.get("auth_header", "Authorization")
                auth_header = {auth_header_name: api_key}
                result["auth_status"] = "key_provided"
                print(f"🔑 Using API key from {api_key_env}")
            else:
                result["auth_status"] = "key_missing"
                print(
                    f"⚠️  API key {api_key_env} not found - testing public endpoints only"
                )

        # Test endpoints based on source type
        if source_name == "data_gov_sg":
            result["tests"] = self.test_data_gov_sg(source_config, auth_header, verbose)
        elif source_name == "lta_datamall":
            result["tests"] = self.test_lta_datamall(
                source_config, auth_header, verbose
            )
        elif source_name == "onemap_sg":
            result["tests"] = self.test_onemap_sg(source_config, verbose)
        elif source_name == "singstat":
            result["tests"] = self.test_singstat(source_config, verbose)
        elif source_name == "world_bank":
            result["tests"] = self.test_world_bank(source_config, verbose)
        elif source_name == "imf":
            result["tests"] = self.test_imf(source_config, verbose)
        elif source_name == "oecd":
            result["tests"] = self.test_oecd(source_config, verbose)
        elif source_name == "undata_api":
            result["tests"] = self.test_undata_api(source_config, auth_header, verbose)
        elif source_name == "un_sdg_api":
            result["tests"] = self.test_un_sdg_api(source_config, verbose)
        else:
            result["tests"] = [{"test": "unknown_source", "status": "skipped"}]

        # Determine overall status
        if not result["tests"]:
            result["overall_status"] = "no_tests"
        elif all(test.get("status") == "success" for test in result["tests"]):
            result["overall_status"] = "success"
            print(f"✅ {source_name}: All tests passed")
        elif any(test.get("status") == "success" for test in result["tests"]):
            result["overall_status"] = "partial"
            print(f"⚠️  {source_name}: Some tests passed")
        else:
            result["overall_status"] = "failed"
            print(f"❌ {source_name}: All tests failed")

        return result

    def test_data_gov_sg(
        self, config: dict, auth_header: dict, verbose: bool
    ) -> List[dict]:
        """Test data.gov.sg API endpoints."""
        tests = []
        base_url = config["base_url"]

        # Test datasets listing
        datasets_url = f"{base_url}/datasets?page=1&per_page=5"
        success, response = self.make_request(datasets_url, auth_header)

        test_result = {
            "test": "datasets_listing",
            "url": datasets_url,
            "status": "success" if success else "failed",
            "details": response,
        }

        if verbose and success:
            print(
                f"  📋 Datasets endpoint: {response.get('response_size', 0)} bytes received"
            )

        tests.append(test_result)

        # If datasets listing works, try to get metadata for first dataset
        if success and "data_sample" in response:
            try:
                # Try to extract a dataset ID from response
                sample_data = response["data_sample"]
                if "datasetId" in sample_data:
                    print(
                        "  🔍 Found dataset IDs in response - metadata endpoint available"
                    )
            except:
                pass

        return tests

    def test_lta_datamall(
        self, config: dict, auth_header: dict, verbose: bool
    ) -> List[dict]:
        """Test LTA DataMall API endpoints."""
        tests = []
        base_url = config["base_url"]

        if not auth_header:
            tests.append(
                {
                    "test": "authentication",
                    "status": "failed",
                    "error": "LTA_API_KEY required",
                }
            )
            return tests

        # Test bus arrival endpoint (most commonly used)
        endpoints = config.get("endpoints", {})
        bus_arrival_endpoint = endpoints.get("bus_arrival", "/v3/BusArrival")
        bus_url = f"{base_url}{bus_arrival_endpoint}?BusStopCode=83139"  # Test stop

        success, response = self.make_request(bus_url, auth_header)

        tests.append(
            {
                "test": "bus_arrival",
                "url": bus_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(f"  🚌 Bus arrival data: {response.get('response_size', 0)} bytes")

        return tests

    def test_onemap_sg(self, config: dict, verbose: bool) -> List[dict]:
        """Test OneMap Singapore API endpoints."""
        tests = []
        base_url = config["base_url"]

        # Test public search endpoint (no auth required)
        public_endpoints = config.get("public_endpoints", {})
        search_endpoint = public_endpoints.get("search", "/common/elastic/search")
        search_url = f"{base_url}{search_endpoint}?searchVal=raffles&returnGeom=Y&getAddrDetails=Y"

        success, response = self.make_request(search_url)

        tests.append(
            {
                "test": "public_search",
                "url": search_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(f"  🗺️  Search results: {response.get('response_size', 0)} bytes")

        # Test token generation if credentials provided
        email = self.get_env_var("ONEMAP_EMAIL")
        password = self.get_env_var("ONEMAP_PASSWORD")

        if email and password:
            auth_url = f"{base_url}/auth/post/getToken"
            auth_payload = {"email": email, "password": password}

            try:
                auth_response = requests.post(auth_url, json=auth_payload, timeout=30)
                if auth_response.status_code == 200:
                    tests.append(
                        {
                            "test": "token_generation",
                            "status": "success",
                            "details": {"token_obtained": True},
                        }
                    )
                    if verbose:
                        print("  🔐 Token generation: Success")
                else:
                    tests.append(
                        {
                            "test": "token_generation",
                            "status": "failed",
                            "details": {"status_code": auth_response.status_code},
                        }
                    )
            except Exception as e:
                tests.append(
                    {"test": "token_generation", "status": "failed", "error": str(e)}
                )

        return tests

    def test_singstat(self, config: dict, verbose: bool) -> List[dict]:
        """Test SingStat Table Builder API."""
        tests = []
        base_url = config["base_url"]

        # Test search endpoint
        search_url = f"{base_url}/resourceid?keyword=population"
        success, response = self.make_request(search_url)

        tests.append(
            {
                "test": "search_datasets",
                "url": search_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(f"  📊 Search results: {response.get('response_size', 0)} bytes")

        return tests

    def test_undata_api(
        self, config: dict, auth_header: dict, verbose: bool
    ) -> List[dict]:
        """Test UNdata API endpoints."""
        tests = []
        base_url = config["base_url"]

        # Test WHO databases endpoint (most reliable)
        who_databases_url = f"{base_url}/WHO/databases"

        # Add authentication if available
        params = {"format": "json"}
        app_id = self.get_env_var("UNDATA_APP_ID")
        app_key = self.get_env_var("UNDATA_APP_KEY")

        if app_id and app_key:
            params.update({"app_id": app_id, "app_key": app_key})
            auth_status = "authenticated"
        else:
            auth_status = "public_only"

        # Convert params to query string
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        test_url = f"{who_databases_url}?{query_string}"

        success, response = self.make_request(test_url)

        tests.append(
            {
                "test": "who_databases",
                "url": test_url,
                "status": "success" if success else "failed",
                "auth_status": auth_status,
                "details": response,
            }
        )

        if verbose and success:
            print(
                f"  🏥 UNdata WHO databases: {response.get('response_size', 0)} bytes"
            )

        return tests

    def test_un_sdg_api(self, config: dict, verbose: bool) -> List[dict]:
        """Test UN SDG API endpoints."""
        tests = []
        base_url = config["base_url"]

        # Test goals endpoint
        goals_endpoint = config.get("endpoints", {}).get("goals", "/v1/sdg/Goal/List")
        goals_url = f"{base_url}{goals_endpoint}"

        success, response = self.make_request(goals_url)

        tests.append(
            {
                "test": "sdg_goals",
                "url": goals_url,
                "status": "success" if success else "failed",
                "details": response,
                "note": "Test API - data for testing purposes",
            }
        )

        if verbose and success:
            print(f"  🎯 UN SDG Goals: {response.get('response_size', 0)} bytes")

        # Test alternative SDG6 API if main fails
        if not success:
            sdg6_config = config.get("sdg6_backup", {})
            if sdg6_config:
                sdg6_url = (
                    f"{sdg6_config['base_url']}/indicator/all?_format=json&per_page=5"
                )
                success_backup, response_backup = self.make_request(sdg6_url)

                tests.append(
                    {
                        "test": "sdg6_backup",
                        "url": sdg6_url,
                        "status": "success" if success_backup else "failed",
                        "details": response_backup,
                    }
                )

                if verbose and success_backup:
                    print(
                        f"  💧 SDG6 backup API: {response_backup.get('response_size', 0)} bytes"
                    )

        return tests

    def test_world_bank(self, config: dict, verbose: bool) -> List[dict]:
        """Test World Bank Open Data API."""
        tests = []
        base_url = config["base_url"]

        # Test indicators endpoint
        indicators_url = f"{base_url}/indicator?format=json&per_page=5"
        success, response = self.make_request(indicators_url)

        tests.append(
            {
                "test": "indicators_listing",
                "url": indicators_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(
                f"  🌍 World Bank indicators: {response.get('response_size', 0)} bytes"
            )

        # Test country data
        countries_url = f"{base_url}/country?format=json&per_page=5"
        success, response = self.make_request(countries_url)

        tests.append(
            {
                "test": "countries_listing",
                "url": countries_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        return tests

    def test_imf(self, config: dict, verbose: bool) -> List[dict]:
        """Test IMF SDMX API."""
        tests = []
        base_url = config["base_url"]

        # Test dataflow endpoint (list all available datasets)
        dataflow_url = f"{base_url}/Dataflow"
        success, response = self.make_request(dataflow_url)

        tests.append(
            {
                "test": "sdmx_dataflow",
                "url": dataflow_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(f"  💰 IMF SDMX dataflows: {response.get('response_size', 0)} bytes")

        # If dataflow works, test getting specific data
        if success:
            # Test IFS (International Financial Statistics) structure
            ifs_url = f"{base_url}/DataStructure/IFS"
            success_ifs, response_ifs = self.make_request(ifs_url)

            tests.append(
                {
                    "test": "sdmx_datastructure_ifs",
                    "url": ifs_url,
                    "status": "success" if success_ifs else "failed",
                    "details": response_ifs,
                }
            )

        return tests

    def test_oecd(self, config: dict, verbose: bool) -> List[dict]:
        """Test OECD SDMX API."""
        tests = []
        base_url = config["base_url"]

        # Test dataflows endpoint
        dataflows_url = f"{base_url}/dataflow/all"
        success, response = self.make_request(dataflows_url)

        tests.append(
            {
                "test": "dataflows_listing",
                "url": dataflows_url,
                "status": "success" if success else "failed",
                "details": response,
            }
        )

        if verbose and success:
            print(f"  📈 OECD dataflows: {response.get('response_size', 0)} bytes")

        return tests

    def test_un_data(self, config: dict, verbose: bool) -> List[dict]:
        """Test UN Data Portal API."""
        tests = []

        # UN Data API testing would require specific implementation
        # Note: UN Data Portal has complex API structure
        tests.append(
            {
                "test": "configuration",
                "status": "success",
                "details": {
                    "configured": True,
                    "note": "UN Data API structure varies by dataset",
                },
            }
        )

        return tests

    def run_all_tests(self, verbose: bool = False, source_filter: str = None) -> dict:
        """Run all API configuration tests."""
        print("🚀 Starting API Configuration Validation")
        print(f"📅 Timestamp: {self.results['timestamp']}")
        print("=" * 50)

        # Test Singapore sources
        if not source_filter or source_filter in self.config.get(
            "singapore_sources", {}
        ):
            singapore_results = self.test_singapore_sources(verbose)
            self.results["sources"].update(singapore_results)

        # Test Global sources
        if not source_filter or source_filter in self.config.get("global_sources", {}):
            global_results = self.test_global_sources(verbose)
            self.results["sources"].update(global_results)

        # Test specific source if requested
        if source_filter:
            if source_filter not in self.results["sources"]:
                print(f"❌ Source '{source_filter}' not found in configuration")
                return self.results

        # Calculate summary statistics
        self.results["total_sources"] = len(self.results["sources"])
        for source_result in self.results["sources"].values():
            if source_result["overall_status"] == "success":
                self.results["successful"] += 1
            else:
                self.results["failed"] += 1

        return self.results

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)

        print(f"Total Sources: {self.results['total_sources']}")
        print(f"✅ Successful: {self.results['successful']}")
        print(f"❌ Failed: {self.results['failed']}")

        if self.results["total_sources"] > 0:
            success_rate = (
                self.results["successful"] / self.results["total_sources"]
            ) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")

        print("\n📋 Detailed Results:")
        for source_name, result in self.results["sources"].items():
            status_emoji = {
                "success": "✅",
                "partial": "⚠️",
                "failed": "❌",
                "no_tests": "⏭️",
            }.get(result["overall_status"], "❓")

            auth_info = ""
            if result["auth_required"]:
                if result["auth_status"] == "key_provided":
                    auth_info = " 🔑"
                elif result["auth_status"] == "key_missing":
                    auth_info = " 🔒"

            print(f"  {status_emoji} {source_name}{auth_info}")

            # Show test details for failed sources
            if result["overall_status"] == "failed" and result.get("tests"):
                for test in result["tests"]:
                    if test.get("status") == "failed":
                        print(
                            f"    💥 {test.get('test', 'unknown')}: {test.get('error', 'Failed')}"
                        )

        print("\n🔧 Next Steps:")
        if self.results["failed"] > 0:
            print("1. Check failed sources above")
            print("2. Verify API keys in .env file")
            print("3. Check internet connectivity")
            print("4. Review API rate limits")
        else:
            print("✅ All sources configured correctly!")
            print("🚀 Ready to run data extraction pipeline")

    def save_results(self, output_file: str = "tests/api_test_results.json"):
        """Save test results to JSON file."""
        output_path = Path(project_root) / output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to {output_path}")


def main():
    """Main function to run API configuration tests."""
    parser = argparse.ArgumentParser(
        description="Test API configuration for Dataset Research Assistant"
    )
    parser.add_argument("--source", type=str, help="Test specific source only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize validator
    validator = APIConfigValidator()

    # Run tests
    results = validator.run_all_tests(verbose=args.verbose, source_filter=args.source)

    # Print summary
    validator.print_summary()

    # Save results if requested
    if args.save:
        validator.save_results()

    # Exit with appropriate code
    if results["failed"] == 0:
        print("\n🎉 All tests passed! Configuration is ready.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {results['failed']} source(s) failed. Check configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
