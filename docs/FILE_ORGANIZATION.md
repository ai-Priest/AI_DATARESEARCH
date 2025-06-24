# File Organization Reference

This document outlines the organized file structure for the AI Data Research project.

## 📁 Root Directory Files (Core Pipeline)
Essential files that remain in the project root:

```
├── data_pipeline.py          # Main data extraction/analysis pipeline
├── ml_pipeline.py            # Machine learning training pipeline  
├── dl_pipeline.py            # Deep learning neural network pipeline
├── main.py                   # Project entry point
├── CLAUDE.md                 # Claude Code guidance and findings
├── Readme.md                 # Main project documentation
├── pyproject.toml           # Python project configuration
├── uv.lock                  # Dependency lock file
├── env_example.sh           # Environment setup example
└── .env                     # Environment variables (private)
```

## 📂 Organized Directory Structure

### `/src/` - Source Code Modules
```
src/
├── data/                    # Data pipeline modules
│   ├── 01_extraction_module.py
│   ├── 02_analysis_module.py
│   └── 03_reporting_module.py
├── ml/                      # Machine learning modules
│   ├── ml_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── enhanced_ml_pipeline.py
│   └── [other ML modules]
├── dl/                      # Deep learning modules
│   ├── neural_preprocessing.py
│   ├── model_architecture.py
│   ├── advanced_training.py
│   ├── deep_evaluation.py
│   ├── neural_inference.py
│   ├── improved_model_architecture.py
│   └── [other DL modules]
└── utils/                   # Utility functions (organized)
```

### `/scripts/` - Utility and Enhancement Scripts  
```
scripts/
├── enhancement/            # Data and model enhancement scripts
│   ├── enhance_ground_truth.py
│   ├── enhance_training_data.py
│   ├── enhance_with_graded_relevance.py  # 🎯 TARGET ACHIEVEMENT
│   ├── dl_pipeline_graded.py            # 🎯 Graded relevance pipeline
│   ├── improved_training_pipeline.py
│   └── quick_retrain_with_enhanced_data.py
├── evaluation/            # Evaluation and testing scripts
│   ├── quick_evaluation.py
│   ├── quick_eval.py
│   ├── quick_graded_improvement.py      # 🎯 TARGET ACHIEVEMENT
│   └── achieve_70_target.py             # 🎯 Final target validation
├── utils/                 # General utility scripts
│   └── verify_organization.py
└── pipeline_support/      # Pipeline support scripts
```

### `/tests/` - Testing Framework
```
tests/
├── test_api_config.py      # API configuration tests
├── test_api_debug.py       # API debugging tests
└── test_scripts/          # Additional test scripts
    ├── test_advanced_ensemble.py
    ├── test_dl_demo.py
    ├── test_enhanced_pipeline.py
    ├── test_ensemble.py
    └── test_improvements.py
```

### `/docs/` - Documentation and Reports
```
docs/
├── guides/                # Setup and usage guides
│   ├── PRODUCTION_DEPLOYMENT_GUIDE.md
│   ├── api_setup_guide.md
│   ├── ml_phase_prompt.md
│   └── readme_ml.txt
├── reports/              # Performance and analysis reports
│   ├── performance/      # Performance analysis reports
│   │   ├── ACTUAL_PERFORMANCE_SUMMARY.md
│   │   └── PERFORMANCE_IMPROVEMENT_PLAN.md
│   ├── dl/              # Deep learning reports
│   │   ├── DATA_FIX_RESULTS.md
│   │   └── DL_BREAKTHROUGH_DEPLOYMENT_REPORT.md
│   └── ml/              # Machine learning reports
└── summaries/           # Project summaries
```

### `/config/` - Configuration Files
```
config/
├── data_pipeline.yml      # Main data pipeline configuration
├── ml_config.yml         # ML model configuration
├── dl_config.yml         # DL neural network configuration
├── api_config.yml        # API endpoints configuration
└── api_config.txt        # API configuration backup
```

### `/data/` - Data Storage
```
data/
├── raw/                  # Raw data from APIs
│   ├── singapore_datasets/
│   ├── global_datasets/
│   └── user_behaviour.csv
├── processed/           # Processed ML-ready data
│   ├── enhanced_training_data.json
│   ├── graded_relevance_training.json      # 🎯 3,500 graded samples
│   ├── threshold_tuning_analysis.json      # 🎯 Optimal threshold data
│   ├── domain_enhanced_training_20250622.json
│   ├── intelligent_ground_truth.json
│   └── [other processed files]
└── feedback/           # User feedback data
    └── user_feedback.json
```

### `/models/` - Trained Models
```
models/
├── dl/                  # Deep learning models
│   ├── lightweight_cross_attention_best.pt  # 69.0% baseline model
│   ├── graded_relevance_best.pt             # 🎯 Graded relevance model
│   ├── siamese_transformer.pt
│   ├── graph_attention.pt
│   ├── improved_models/
│   └── checkpoints/
├── [ML model files]     # Traditional ML models
└── [metadata files]    # Model metadata and configs
```

### `/outputs/` - Pipeline Results
```
outputs/
├── EDA/                # Data analysis outputs
│   ├── reports/
│   └── visualizations/
├── ML/                 # ML pipeline outputs
│   ├── reports/
│   ├── evaluations/
│   └── visualizations/
└── DL/                 # DL pipeline outputs
    ├── reports/        # Comprehensive evaluation reports with 10+ visualizations
    ├── target_achievement_report_*.json  # 🎯 75.0% achievement validation
    ├── quick_graded_enhancement_*.json   # Quick enhancement results
    ├── evaluations/
    └── visualizations/
```

### `/logs/` - Execution Logs
```
logs/
├── dl_pipeline.log     # Deep learning pipeline logs
├── ml_pipeline.log     # ML pipeline logs
└── dl/                # Detailed DL training logs
    └── [tensorboard runs]
```

## 🚀 Usage After Organization

### Running Core Pipelines (from root)
```bash
# All main commands work from project root
python data_pipeline.py
python ml_pipeline.py  
python dl_pipeline.py
```

### Running Enhancement Scripts
```bash
# Enhancement scripts (from root)
python scripts/enhancement/improved_training_pipeline.py
python scripts/enhancement/enhance_training_data.py
python scripts/enhancement/enhance_with_graded_relevance.py  # 🎯 Generate graded data
python scripts/enhancement/dl_pipeline_graded.py            # 🎯 Graded training

# Evaluation scripts (from root)
python scripts/evaluation/quick_evaluation.py
python scripts/evaluation/quick_graded_improvement.py       # 🎯 Quick enhancement
python scripts/evaluation/achieve_70_target.py             # 🎯 Target validation
```

### Running Tests
```bash
# Core tests
python -m pytest tests/

# Specific test scripts
python tests/test_scripts/test_enhanced_pipeline.py
```

## 🔧 Import Path Updates

All imports in the organized files maintain compatibility:
- Core pipeline files remain in root with no import changes needed
- Enhanced scripts use relative imports from project root
- All functionality preserved with organized structure

## 🎯 Target Achievement Organization (June 2025)

### New Files Added for 75.0% NDCG@3 Achievement
```bash
# Enhancement Scripts (Advanced Training)
scripts/enhancement/enhance_with_graded_relevance.py  # Graded relevance data generation
scripts/enhancement/dl_pipeline_graded.py            # Full graded relevance pipeline

# Evaluation Scripts (Performance Validation)  
scripts/evaluation/quick_graded_improvement.py       # Quick enhancement application
scripts/evaluation/achieve_70_target.py             # Final target achievement validation

# Data Files (Enhanced Training Data)
data/processed/graded_relevance_training.json       # 3,500 graded samples  
data/processed/threshold_tuning_analysis.json       # Optimal threshold: 0.485

# Model Files (Target Achievement Models)
models/dl/graded_relevance_best.pt                  # Graded relevance trained model

# Results (Achievement Documentation)
outputs/DL/target_achievement_report_*.json         # 75.0% achievement validation
outputs/DL/quick_graded_enhancement_*.json          # Enhancement results
outputs/DL/reports/                                 # 10+ comprehensive visualizations
```

## 📋 Benefits of Organization

1. **Clear Structure**: Easy to find files by purpose
2. **Maintainability**: Related files grouped together  
3. **Scalability**: Room for growth in each category
4. **Documentation**: Reports and guides properly organized
5. **Testing**: All test files in dedicated test directory
6. **Core Accessibility**: Main pipelines remain easily accessible
7. **🎯 Target Achievement**: New files properly categorized by function

This organization maintains full functionality while providing a professional, scalable project structure with clear documentation of the 75.0% NDCG@3 target achievement.