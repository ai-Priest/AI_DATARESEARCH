# Training Data Mappings

This file contains manual query-to-source relevance mappings to improve the neural network's performance without waiting for user feedback.

## Format
Use this format for each mapping:
```
- query → source (relevance_score) - reason/explanation
```

Where:
- **query**: The search term users might type
- **source**: The data source/website (kaggle, zenodo, world_bank, etc.)
- **relevance_score**: Float between 0.0-1.0 (0.0 = irrelevant, 1.0 = perfect match)
- **reason**: Why this source is/isn't good for this query

---

## Psychology Queries
- psychology → kaggle (0.95) - Best platform for psychology datasets and research competitions
- psychology → zenodo (0.90) - Academic repository with psychology research data
- psychology research → kaggle (0.98) - Research-focused psychology datasets available
- psychology research → zenodo (0.95) - Academic psychology research repository
- mental health data → kaggle (0.92) - Mental health datasets for analysis
- mental health data → zenodo (0.88) - Academic mental health research
- behavioral psychology → kaggle (0.94) - Behavioral datasets for research
- cognitive psychology → zenodo (0.91) - Cognitive research repository
- psychology → world_bank (0.25) - Limited psychology-specific data available
- psychology → data_un (0.20) - No relevant psychology datasets found
- psychology → aws_opendata (0.15) - Registry outdated, no psychology focus

## Machine Learning Queries
- machine learning → kaggle (0.98) - ML competitions and comprehensive datasets
- machine learning → zenodo (0.85) - Academic ML research datasets
- ml datasets → kaggle (0.97) - Primary source for ML datasets
- artificial intelligence → kaggle (0.95) - AI datasets and competitions
- deep learning → kaggle (0.96) - Deep learning datasets available
- neural networks → zenodo (0.87) - Academic neural network research
- machine learning → world_bank (0.40) - Limited ML-specific datasets

## Climate & Environment Queries
- climate data → world_bank (0.95) - Excellent global climate indicators
- climate change → world_bank (0.93) - Comprehensive climate statistics
- environmental data → world_bank (0.90) - Global environmental indicators
- weather data → kaggle (0.88) - Weather datasets for analysis
- climate data → zenodo (0.85) - Academic climate research
- temperature data → world_bank (0.92) - Global temperature records
- climate data → kaggle (0.82) - Climate datasets for modeling

## Economics & Finance Queries
- economic data → world_bank (0.98) - Primary source for economic indicators
- gdp data → world_bank (0.97) - Comprehensive GDP statistics
- financial data → world_bank (0.90) - Global financial indicators
- trade data → world_bank (0.95) - International trade statistics
- economic indicators → world_bank (0.96) - Complete economic metrics
- poverty data → world_bank (0.94) - Global poverty statistics
- economic data → kaggle (0.75) - Some economic datasets available
- financial data → kaggle (0.80) - Financial datasets for analysis

## Singapore-Specific Queries
- singapore data → data_gov_sg (0.98) - Official Singapore government data
- singapore statistics → singstat (0.97) - Singapore Department of Statistics
- singapore housing → data_gov_sg (0.95) - HDB and housing data
- singapore transport → lta_datamall (0.96) - Land Transport Authority data
- singapore demographics → singstat (0.94) - Population and demographic data
- singapore economy → singstat (0.92) - Economic statistics Singapore
- singapore data → world_bank (0.60) - Some Singapore indicators available
- singapore data → kaggle (0.45) - Limited Singapore-specific datasets

## Health & Medical Queries
- health data → world_bank (0.88) - Global health indicators
- medical data → zenodo (0.90) - Academic medical research
- healthcare statistics → world_bank (0.85) - Healthcare indicators
- disease data → world_bank (0.87) - Global disease statistics
- health data → kaggle (0.82) - Health datasets for analysis
- medical research → zenodo (0.93) - Academic medical research repository
- public health → world_bank (0.89) - Public health indicators

## Education Queries
- education data → world_bank (0.92) - Global education indicators
- education statistics → world_bank (0.90) - Educational metrics
- student data → kaggle (0.85) - Educational datasets for analysis
- university data → zenodo (0.83) - Academic educational research
- education research → zenodo (0.88) - Educational research repository
- school data → kaggle (0.80) - School-related datasets

---

## Instructions for Adding New Mappings

1. **Add your mappings above** using the format shown
2. **Run the injection script**:
   ```bash
   cd /Users/491jw/Documents/Personal/AI_Projects/AI_DataResearch
   python src/ml/training_data_injector.py
   ```
3. **Or use in Python**:
   ```python
   from src.ml.training_data_injector import inject_from_file
   count = inject_from_file("training_mappings.md")
   print(f"Injected {count} training mappings!")
   ```

The system will automatically parse this file and inject the mappings into your neural network's training data!