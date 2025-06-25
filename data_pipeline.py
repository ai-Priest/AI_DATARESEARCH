# data_pipeline.py - Configuration-Driven Data Pipeline Orchestrator
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigurableDataPipeline:
    """Configuration-driven three-phase data pipeline orchestrator"""

    def __init__(self, config_path: str = "config/data_pipeline.yml"):
        """Initialize pipeline with configuration"""
        self.config_path = config_path
        self.config = self._load_configuration()
        self.pipeline_config = self.config.get("pipeline", {})
        self.execution_summary = {
            "pipeline_start_time": time.time(),
            "phases_completed": [],
            "phases_failed": [],
            "total_datasets_processed": 0,
            "ml_readiness_achieved": False,
        }

    def _load_configuration(self) -> Dict:
        """Load and validate pipeline configuration"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def validate_environment(self) -> bool:
        """Validate pipeline environment and dependencies"""
        logger.info("🔧 Validating pipeline environment...")

        # Check required directories
        required_dirs = [Path("data/raw"), Path("data/processed"), Path("outputs/EDA")]

        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Directory ready: {dir_path}")

        # Check user behavior file
        behavior_file = self.config.get("phase_2_analysis", {}).get(
            "user_behavior_file", "data/raw/user_behaviour.csv"
        )
        if not Path(behavior_file).exists():
            logger.warning(f"⚠️ User behavior file not found: {behavior_file}")
            logger.info("   Phase 2 will proceed with dataset analysis only")

        # Check required environment variables (optional for API keys)
        env_config = self.config.get("environment", {})
        required_vars = env_config.get("required_env_vars", [])
        optional_vars = env_config.get("optional_env_vars", [])

        import os

        missing_required = [var for var in required_vars if not os.getenv(var)]
        missing_optional = [var for var in optional_vars if not os.getenv(var)]

        if missing_required:
            logger.warning(
                f"⚠️ Missing required environment variables: {missing_required}"
            )
            logger.info("   Some API extractions may fail")

        if missing_optional:
            logger.info(
                f"📝 Missing optional environment variables: {missing_optional}"
            )

        logger.info("✅ Environment validation complete")
        return True

    def execute_phase_1_extraction(self) -> bool:
        """Execute Phase 1: Configuration-Driven Data Extraction"""
        logger.info("🚀 PHASE 1: Configuration-Driven Data Extraction")
        logger.info("=" * 60)

        phase_start_time = time.time()

        try:
            # Import extraction module (handle filename with number prefix)
            import importlib.util
            import os
            import sys

            # Add current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            # Load extraction module dynamically
            extraction_file = os.path.join(
                current_dir, "src", "data", "01_extraction_module.py"
            )
            spec = importlib.util.spec_from_file_location(
                "extraction_module", extraction_file
            )
            extraction_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extraction_module)

            ConfigurableDataExtractor = extraction_module.ConfigurableDataExtractor

            extractor = ConfigurableDataExtractor(self.config_path)

            # Execute extraction
            extraction_results = extractor.extract_all_datasets()

            # Get summary
            summary = extractor.get_extraction_summary()

            # Update execution summary
            total_datasets = summary.get("singapore_total", 0) + summary.get(
                "global_total", 0
            )
            self.execution_summary["total_datasets_processed"] = total_datasets

            if total_datasets > 0:
                self.execution_summary["phases_completed"].append("phase_1_extraction")

                # Display results
                phase_duration = time.time() - phase_start_time
                logger.info(f"\n📊 Phase 1 Results:")
                logger.info(
                    f"   Singapore datasets: {summary.get('singapore_total', 0)}"
                )
                logger.info(f"   Global datasets: {summary.get('global_total', 0)}")
                logger.info(f"   Total datasets: {total_datasets}")
                logger.info(f"   Execution time: {phase_duration:.1f} seconds")

                logger.info(f"\n✅ Phase 1 Complete!")
                logger.info(f"💾 Data saved to: data/raw/ and data/processed/")
                return True
            else:
                logger.error("❌ Phase 1 Failed: No datasets extracted")
                self.execution_summary["phases_failed"].append("phase_1_extraction")
                return False

        except Exception as e:
            logger.error(f"❌ Phase 1 Error: {e}")
            self.execution_summary["phases_failed"].append("phase_1_extraction")
            return False

    def execute_phase_2_analysis(self) -> bool:
        """Execute Phase 2: Deep Analysis with User Behavior Integration"""
        logger.info("\n🧠 PHASE 2: Deep Analysis with User Behavior Integration")
        logger.info("=" * 60)

        phase_start_time = time.time()

        try:
            # Import analysis modules (handle filename with number prefix)
            import importlib.util
            import os
            import sys

            # Add current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            # Load analysis module dynamically
            analysis_file = os.path.join(
                current_dir, "src", "data", "02_analysis_module.py"
            )
            spec = importlib.util.spec_from_file_location(
                "analysis_module", analysis_file
            )
            analysis_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analysis_module)

            UserBehaviorAnalyzer = analysis_module.UserBehaviorAnalyzer
            DatasetIntelligenceEngine = analysis_module.DatasetIntelligenceEngine
            IntelligentGroundTruthGenerator = (
                analysis_module.IntelligentGroundTruthGenerator
            )

            # Initialize components
            user_analyzer = UserBehaviorAnalyzer(self.config)
            dataset_engine = DatasetIntelligenceEngine(self.config)

            # Load datasets from Phase 1
            datasets_df = dataset_engine.load_extracted_datasets("data/processed")

            if datasets_df.empty:
                logger.error("❌ Phase 2 Failed: No data from Phase 1")
                self.execution_summary["phases_failed"].append("phase_2_analysis")
                return False

            # Load and analyze user behavior (optional)
            behavior_file = self.config["phase_2_analysis"]["user_behavior_file"]
            behavior_df = user_analyzer.load_user_behavior_data(behavior_file)

            user_segments = {}
            interaction_patterns = {}

            if not behavior_df.empty:
                user_segments = user_analyzer.analyze_user_segments(behavior_df)
                interaction_patterns = user_analyzer.extract_search_patterns(
                    behavior_df
                )
                logger.info(
                    f"👥 User behavior analysis: {user_segments.get('total_sessions', 0)} sessions"
                )
            else:
                logger.info(
                    "⚠️ No user behavior data - proceeding with dataset analysis only"
                )

            # Extract intelligent keywords
            keyword_profiles = dataset_engine.extract_intelligent_keywords()

            # Discover dataset relationships
            relationships = dataset_engine.discover_dataset_relationships()

            # Generate intelligent ground truth
            ground_truth_gen = IntelligentGroundTruthGenerator(
                self.config, user_analyzer, dataset_engine
            )
            ground_truth = ground_truth_gen.generate_intelligent_ground_truth()

            # Save analysis results
            output_path = Path("data/processed")

            # Save core analysis outputs
            with open(output_path / "keyword_profiles.json", "w") as f:
                json.dump(keyword_profiles, f, indent=2, default=str)

            with open(output_path / "dataset_relationships.json", "w") as f:
                json.dump(relationships, f, indent=2, default=str)

            with open(output_path / "intelligent_ground_truth.json", "w") as f:
                json.dump(ground_truth, f, indent=2, default=str)

            # Save user behavior analysis if available
            if not behavior_df.empty:
                user_analysis = {
                    "user_segments": user_segments,
                    "interaction_patterns": interaction_patterns,
                }
                with open(output_path / "user_behavior_analysis.json", "w") as f:
                    json.dump(user_analysis, f, indent=2, default=str)

            # Update execution summary
            self.execution_summary["phases_completed"].append("phase_2_analysis")

            # Display results
            phase_duration = time.time() - phase_start_time
            logger.info(f"\n📊 Phase 2 Results:")
            logger.info(f"   Datasets analyzed: {len(datasets_df)}")
            logger.info(f"   Keyword profiles: {len(keyword_profiles)}")
            logger.info(
                f"   Relationships discovered: {sum(len(v) for v in relationships.values())}"
            )
            logger.info(f"   Ground truth scenarios: {len(ground_truth)}")
            if not behavior_df.empty:
                logger.info(
                    f"   User sessions analyzed: {user_segments.get('total_sessions', 0)}"
                )
            logger.info(f"   Execution time: {phase_duration:.1f} seconds")

            logger.info(f"\n✅ Phase 2 Complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 2 Error: {e}")
            self.execution_summary["phases_failed"].append("phase_2_analysis")
            return False

    def execute_phase_3_reporting(self) -> bool:
        """Execute Phase 3: Comprehensive EDA & Reporting"""
        logger.info("\n📊 PHASE 3: Comprehensive EDA & Validation Reporting")
        logger.info("=" * 60)

        phase_start_time = time.time()

        try:
            # Import reporting module (handle filename with number prefix)
            import importlib.util
            import os
            import sys

            # Add current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            # Load reporting module dynamically
            reporting_file = os.path.join(
                current_dir, "src", "data", "03_reporting_module.py"
            )
            spec = importlib.util.spec_from_file_location(
                "reporting_module", reporting_file
            )
            reporting_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reporting_module)

            ComprehensiveEDAReporter = reporting_module.ComprehensiveEDAReporter

            # Initialize EDA reporter
            reporter = ComprehensiveEDAReporter(self.config_path)

            # Load all analysis data from previous phases
            if not reporter.load_all_analysis_data():
                logger.error("❌ Phase 3 Failed: Cannot load analysis data")
                self.execution_summary["phases_failed"].append("phase_3_reporting")
                return False

            # Run comprehensive analysis suite
            logger.info("🔍 Running comprehensive analysis suite...")

            # 1. Dataset collection overview
            collection_analysis = reporter.analyze_dataset_collection_overview()

            # 2. Keyword intelligence analysis
            keyword_analysis = reporter.analyze_keyword_intelligence()

            # 3. Relationship discovery analysis
            relationship_analysis = reporter.analyze_relationship_discovery()

            # 4. Ground truth validation
            gt_analysis = reporter.validate_ground_truth_scenarios()

            # 5. Data quality assessment
            quality_analysis = reporter.identify_data_quality_issues()

            # Create comprehensive visualizations
            logger.info("📊 Creating comprehensive visualizations...")
            reporter.create_comprehensive_visualizations()

            # Generate comprehensive reports
            logger.info("📋 Generating comprehensive reports...")
            comprehensive_report = reporter.generate_comprehensive_reports()

            # Update execution summary with ML readiness
            ml_ready = gt_analysis.get("ml_readiness_assessment", {}).get(
                "ready_for_ml_training", False
            )
            self.execution_summary["ml_readiness_achieved"] = ml_ready
            self.execution_summary["phases_completed"].append("phase_3_reporting")

            # Display results
            phase_duration = time.time() - phase_start_time
            total_datasets = collection_analysis.get("collection_metadata", {}).get(
                "total_datasets", 0
            )
            high_quality_relationships = relationship_analysis.get(
                "relationship_summary", {}
            ).get("high_quality_relationships", 0)
            high_confidence_scenarios = (
                gt_analysis.get("quality_assessment", {})
                .get("confidence_distribution", {})
                .get("high_confidence", 0)
            )

            logger.info(f"\n📊 Phase 3 Results:")
            logger.info(f"   Total datasets analyzed: {total_datasets}")
            logger.info(f"   High-quality relationships: {high_quality_relationships}")
            logger.info(f"   High-confidence ground truth: {high_confidence_scenarios}")
            logger.info(
                f"   Data quality issues: {quality_analysis.get('issues_summary', {}).get('total_issues_identified', 0)}"
            )
            logger.info(f"   Execution time: {phase_duration:.1f} seconds")

            logger.info(f"\n📁 Generated Outputs:")
            logger.info(f"   📊 Visualizations: outputs/EDA/visualizations/")
            logger.info(f"   📋 Reports: outputs/EDA/reports/")

            logger.info(f"\n✅ Phase 3 Complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 3 Error: {e}")
            self.execution_summary["phases_failed"].append("phase_3_reporting")
            return False

    def assess_ml_readiness(self) -> Dict:
        """Assess overall pipeline readiness for ML model training"""
        logger.info("\n🎯 ML READINESS ASSESSMENT")
        logger.info("=" * 40)

        # Load ground truth analysis
        try:
            gt_file = Path("data/processed/intelligent_ground_truth.json")
            if gt_file.exists():
                with open(gt_file, "r") as f:
                    ground_truth = json.load(f)

                # Assess ground truth quality
                high_confidence_scenarios = sum(
                    1
                    for scenario in ground_truth.values()
                    if scenario.get("confidence", 0) >= 0.7
                )

                adequate_scenarios = sum(
                    1
                    for scenario in ground_truth.values()
                    if 0.5 <= scenario.get("confidence", 0) < 0.7
                )

                total_usable = high_confidence_scenarios + adequate_scenarios

                # Check ML readiness thresholds
                ml_thresholds = self.pipeline_config.get("ml_readiness_thresholds", {})
                min_scenarios = ml_thresholds.get("min_ground_truth_scenarios", 3)
                min_high_conf = ml_thresholds.get("min_high_confidence_scenarios", 2)
                min_datasets = ml_thresholds.get("min_total_datasets", 15)

                ml_ready = (
                    total_usable >= min_scenarios
                    and high_confidence_scenarios >= min_high_conf
                    and self.execution_summary["total_datasets_processed"]
                    >= min_datasets
                )

                assessment = {
                    "ready_for_ml_training": ml_ready,
                    "total_scenarios": len(ground_truth),
                    "high_confidence_scenarios": high_confidence_scenarios,
                    "adequate_scenarios": adequate_scenarios,
                    "total_usable_scenarios": total_usable,
                    "total_datasets": self.execution_summary[
                        "total_datasets_processed"
                    ],
                    "thresholds": {
                        "min_scenarios_required": min_scenarios,
                        "min_high_confidence_required": min_high_conf,
                        "min_datasets_required": min_datasets,
                    },
                }

                # Estimate performance
                if ml_ready:
                    if high_confidence_scenarios >= 5:
                        assessment["expected_performance"] = {
                            "f1_score": "0.75-0.85",
                            "confidence": "very_high",
                        }
                    elif high_confidence_scenarios >= 3:
                        assessment["expected_performance"] = {
                            "f1_score": "0.70-0.80",
                            "confidence": "high",
                        }
                    else:
                        assessment["expected_performance"] = {
                            "f1_score": "0.60-0.70",
                            "confidence": "medium",
                        }
                else:
                    assessment["expected_performance"] = {
                        "f1_score": "0.40-0.60",
                        "confidence": "low",
                    }

                return assessment

            else:
                return {
                    "ready_for_ml_training": False,
                    "error": "No ground truth data found",
                }

        except Exception as e:
            logger.error(f"❌ Cannot assess ML readiness: {e}")
            return {"ready_for_ml_training": False, "error": str(e)}

    def generate_execution_summary(self):
        """Generate comprehensive execution summary"""
        total_time = time.time() - self.execution_summary["pipeline_start_time"]

        logger.info(f"\n🎉 DATA PIPELINE EXECUTION SUMMARY")
        logger.info(f"=" * 50)
        logger.info(f"⏱️ Total execution time: {total_time:.1f} seconds")
        logger.info(
            f"✅ Phases completed: {len(self.execution_summary['phases_completed'])}/3"
        )
        logger.info(f"❌ Phases failed: {len(self.execution_summary['phases_failed'])}")
        logger.info(
            f"📊 Total datasets processed: {self.execution_summary['total_datasets_processed']}"
        )

        # ML Readiness Assessment
        ml_assessment = self.assess_ml_readiness()
        ml_ready = ml_assessment.get("ready_for_ml_training", False)

        logger.info(
            f"\n🎯 ML Training Readiness: {'READY' if ml_ready else 'NEEDS IMPROVEMENT'}"
        )

        if ml_ready:
            expected_perf = ml_assessment.get("expected_performance", {})
            logger.info(f"   Expected F1 Score: {expected_perf.get('f1_score', 'N/A')}")
            logger.info(
                f"   Confidence Level: {expected_perf.get('confidence', 'N/A').title()}"
            )
            logger.info(
                f"   High-confidence scenarios: {ml_assessment.get('high_confidence_scenarios', 0)}"
            )
            logger.info(
                f"   Total usable scenarios: {ml_assessment.get('total_usable_scenarios', 0)}"
            )
        else:
            logger.info(
                f"   High-confidence scenarios: {ml_assessment.get('high_confidence_scenarios', 0)} (need ≥{ml_assessment.get('thresholds', {}).get('min_high_confidence_required', 2)})"
            )
            logger.info(
                f"   Total scenarios: {ml_assessment.get('total_scenarios', 0)} (need ≥{ml_assessment.get('thresholds', {}).get('min_scenarios_required', 3)})"
            )
            logger.info(
                f"   Total datasets: {ml_assessment.get('total_datasets', 0)} (need ≥{ml_assessment.get('thresholds', {}).get('min_datasets_required', 15)})"
            )

        # Generated files summary
        logger.info(f"\n📁 Generated Files Summary:")

        file_checks = [
            ("data/processed/singapore_datasets.csv", "Singapore datasets"),
            ("data/processed/global_datasets.csv", "Global datasets"),
            ("data/processed/keyword_profiles.json", "Keyword profiles"),
            ("data/processed/dataset_relationships.json", "Dataset relationships"),
            ("data/processed/intelligent_ground_truth.json", "Ground truth scenarios"),
            (
                "outputs/EDA/visualizations/dataset_distribution_overview.png",
                "Dataset visualizations",
            ),
            ("outputs/EDA/reports/executive_summary.md", "Executive summary"),
            (
                "outputs/EDA/reports/comprehensive_analysis_report.json",
                "Detailed analysis report",
            ),
        ]

        for filepath, description in file_checks:
            if Path(filepath).exists():
                size = Path(filepath).stat().st_size
                size_str = self._format_file_size(size)
                logger.info(f"   ✅ {description}: {filepath} ({size_str})")
            else:
                logger.info(f"   ❌ {description}: {filepath} (not found)")

        # Next steps
        logger.info(f"\n🚀 NEXT STEPS:")

        if ml_ready:
            logger.info(f"   🎯 READY FOR ML PIPELINE!")
            logger.info(f"   📝 Next phase: ML Pipeline (train_models.py)")
            logger.info(
                f"   📈 Expected performance: {ml_assessment.get('expected_performance', {}).get('f1_score', 'N/A')}"
            )
            logger.info(f"   ⏱️ Estimated timeline: 1-2 weeks to ML results")
        else:
            logger.info(f"   🔄 IMPROVEMENT NEEDED:")
            logger.info(f"   📋 Review: outputs/EDA/reports/executive_summary.md")
            logger.info(f"   🔧 Action: Address data quality issues identified")
            logger.info(f"   ⏱️ Estimated timeline: 2-3 weeks with improvements")

        # Configuration-driven success message
        logger.info(f"\n🎯 Configuration-Driven Pipeline Complete!")
        logger.info(f"   🔧 Config file: {self.config_path}")
        logger.info(f"   📊 Total phases: 3 (Extraction → Analysis → Reporting)")
        logger.info(f"   🚀 Ready for: ML/DL/AI Enhancement Phases")

        # Save execution summary
        self._save_execution_summary(ml_assessment, total_time)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.1f} MB"
        else:
            return f"{size_bytes / 1024**3:.1f} GB"

    def _save_execution_summary(self, ml_assessment: Dict, total_time: float):
        """Save execution summary to file"""
        try:
            summary_data = {
                "pipeline_execution": {
                    "config_file": self.config_path,
                    "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_execution_time_seconds": total_time,
                    "phases_completed": self.execution_summary["phases_completed"],
                    "phases_failed": self.execution_summary["phases_failed"],
                    "total_datasets_processed": self.execution_summary[
                        "total_datasets_processed"
                    ],
                },
                "ml_readiness_assessment": ml_assessment,
                "next_steps": {
                    "ml_ready": ml_assessment.get("ready_for_ml_training", False),
                    "recommended_next_phase": "ML Pipeline"
                    if ml_assessment.get("ready_for_ml_training", False)
                    else "Data Collection Enhancement",
                    "estimated_timeline": "1-2 weeks"
                    if ml_assessment.get("ready_for_ml_training", False)
                    else "2-3 weeks",
                },
            }

            summary_file = Path("data/processed/pipeline_execution_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            logger.info(f"💾 Execution summary saved: {summary_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save execution summary: {e}")

    def run_complete_pipeline(self):
        """Execute the complete three-phase data pipeline"""
        logger.info("🤖 AI-Powered Dataset Research Assistant")
        logger.info("🔄 Configuration-Driven Three-Phase Data Pipeline")
        logger.info("=" * 70)

        # Validate environment
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return

        # Track pipeline success
        pipeline_success = True
        stop_on_failure = self.pipeline_config.get("stop_on_phase_failure", True)

        # Phase 1: Data Extraction
        phase1_success = self.execute_phase_1_extraction()
        if not phase1_success:
            pipeline_success = False
            if stop_on_failure:
                logger.error("\n❌ Pipeline stopped: Phase 1 failed")
                self.generate_execution_summary()
                return

        # Phase 2: Deep Analysis
        if phase1_success:
            phase2_success = self.execute_phase_2_analysis()
            if not phase2_success:
                pipeline_success = False
                if stop_on_failure:
                    logger.error("\n❌ Pipeline stopped: Phase 2 failed")
                    self.generate_execution_summary()
                    return

        # Phase 3: EDA & Reporting
        if phase1_success:  # Phase 3 can run if Phase 1 succeeded (Phase 2 optional)
            phase3_success = self.execute_phase_3_reporting()
            if not phase3_success:
                pipeline_success = False
                logger.warning("\n⚠️ Phase 3 failed, but continuing to summary...")

        # Generate comprehensive summary
        self.generate_execution_summary()

        if pipeline_success:
            logger.info(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.warning(f"\n⚠️ PIPELINE COMPLETED WITH ISSUES")
            logger.info(f"   Check logs above for details")


def main():
    """Main entry point for the data pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered Dataset Research Assistant - Data Pipeline"
    )
    parser.add_argument(
        "--config", default="config/data_pipeline.yml", help="Configuration file path"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Run specific phase or all phases",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't run pipeline",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ConfigurableDataPipeline(args.config)

    # Validate only mode
    if args.validate_only:
        pipeline.validate_environment()
        logger.info("✅ Environment validation complete")
        return

    # Run specific phase or complete pipeline
    if args.phase == "all":
        pipeline.run_complete_pipeline()
    elif args.phase == "1":
        pipeline.validate_environment()
        pipeline.execute_phase_1_extraction()
    elif args.phase == "2":
        pipeline.validate_environment()
        pipeline.execute_phase_2_analysis()
    elif args.phase == "3":
        pipeline.validate_environment()
        pipeline.execute_phase_3_reporting()


if __name__ == "__main__":
    main()
