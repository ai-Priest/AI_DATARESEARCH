# Legacy Scripts

This folder contains scripts that have been superseded by integrated functionality in the main pipeline files.

## Files

### optimization_pipeline.py
- **Purpose**: Comprehensive optimization pipeline for achieving 70%+ NDCG@3
- **Status**: Functionality integrated into `dl_pipeline.py`
- **Reason for deprecation**: The main dl_pipeline.py now automatically applies optimizations when `config/dl_boost_config.json` is present

### aggressive_optimization.py  
- **Purpose**: Aggressive semantic enhancement and hard negative sampling
- **Status**: Core functionality integrated into `src/dl/enhanced_neural_preprocessing.py`
- **Reason for deprecation**: The enhancement techniques are now part of the standard DL preprocessing flow

## Current Pipeline Structure

The correct pipeline to use is:
1. `data_pipeline.py` - Data extraction and analysis
2. `ml_pipeline.py` - ML baseline models
3. `dl_pipeline.py` - Deep learning with automatic optimization
4. `ai_pipeline.py` - AI integration layer
5. `deploy.py` - Production deployment

The dl_pipeline.py will automatically use enhanced training data and optimizations to achieve 70%+ NDCG@3 performance.