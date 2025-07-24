# AI-Powered Dataset Research Assistant - User Guide

## Overview

Welcome to the enhanced AI-Powered Dataset Research Assistant! This guide will help you understand the new quality-first features and how to get the most relevant dataset recommendations for your research needs.

## What's New - Quality-First Approach

### 🎯 Improved Recommendation Quality

Our system now prioritizes **recommendation relevance over speed**, ensuring you get genuinely useful datasets even if it takes a few extra seconds. Key improvements include:

- **72.1% NDCG@3 accuracy** (up from 31.8%) - much more relevant recommendations
- **Expert-curated training data** - recommendations based on real researcher feedback
- **Quality validation** - all recommendations are validated before being served to you

### 🇸🇬 Singapore-First Strategy

For Singapore-related queries, the system automatically prioritizes local government sources:

- **data.gov.sg** - Singapore's official open data portal
- **SingStat** - Department of Statistics Singapore
- **LTA DataMall** - Land Transport Authority data

### 🎓 Domain-Specific Routing

The system now intelligently routes queries to the most appropriate sources based on your research domain:

- **Psychology queries** → Kaggle, Zenodo (research datasets and competitions)
- **Climate queries** → World Bank, Zenodo (authoritative climate data)
- **Economics queries** → World Bank, SingStat (official economic indicators)
- **Machine Learning queries** → Kaggle (competitions and datasets)

## Getting Started

### Basic Search

Simply enter your research query in natural language:

```
Example queries:
• "singapore housing prices"
• "psychology research datasets"
• "climate change indicators"
• "machine learning competitions"
```

### Understanding Your Results

Each recommendation now includes detailed quality information:

#### Quality Metrics
- **Relevance Score** (0.0-1.0): How relevant this dataset is to your query
- **Quality Score** (0.0-1.0): Overall dataset quality and reliability
- **Confidence** (0.0-1.0): System's confidence in this recommendation

#### Routing Information
- **Source Priority**: Why this source was chosen
- **Domain Match**: How your query was classified
- **Geographic Scope**: Local vs global data prioritization

#### Explanations
Each recommendation includes a clear explanation of why it was suggested for your specific query.

## Advanced Features

### 1. Quality Threshold Control

You can set minimum quality thresholds for your results:

```json
{
  "query": "your research query",
  "quality_threshold": 0.8,
  "max_results": 10
}
```

**Quality Levels:**
- **0.9-1.0**: Excellent - Perfect match, authoritative sources
- **0.8-0.9**: Very Good - Strong relevance, high-quality sources  
- **0.7-0.8**: Good - Relevant, reliable sources
- **0.6-0.7**: Fair - Somewhat relevant, acceptable quality

### 2. Singapore-First Queries

For Singapore-specific research, use local terms to trigger prioritization:

**Explicit Singapore queries:**
- "singapore population data"
- "sg economic indicators"
- "local transport statistics"

**Implicit Singapore queries:**
- "hdb resale prices" (automatically detects Singapore context)
- "mrt ridership data"
- "local education statistics"

### 3. Domain-Specific Searches

Use domain-specific keywords to get better routing:

**Psychology Research:**
- "psychology datasets"
- "mental health research data"
- "behavioral analysis datasets"

**Climate Research:**
- "climate change data"
- "environmental indicators"
- "weather datasets"

**Economics Research:**
- "economic indicators"
- "gdp data"
- "trade statistics"

## Search Examples and Results

### Example 1: Singapore Housing Research

**Query:** "singapore housing prices"

**Expected Results:**
1. **data.gov.sg - HDB Resale Flat Prices** (Quality: 0.96)
   - *Explanation*: Official Singapore government housing data - highest relevance for local housing price queries
   - *Why recommended*: Singapore-first priority for local housing data

2. **SingStat - Private Residential Property Price Index** (Quality: 0.94)
   - *Explanation*: Official statistics for private housing market trends
   - *Why recommended*: Singapore Department of Statistics - authoritative local data

### Example 2: Psychology Research

**Query:** "psychology research datasets"

**Expected Results:**
1. **Kaggle - Psychology Datasets** (Quality: 0.92)
   - *Explanation*: Excellent for psychology datasets and ML competitions
   - *Why recommended*: Domain-specific routing for psychology research

2. **Zenodo - Psychology Research Data** (Quality: 0.88)
   - *Explanation*: Academic repository with high-quality research data
   - *Why recommended*: Specialized academic psychology datasets

### Example 3: Climate Research

**Query:** "climate change indicators"

**Expected Results:**
1. **World Bank - Climate Change Data** (Quality: 0.95)
   - *Explanation*: Authoritative global climate indicators and statistics
   - *Why recommended*: World Bank specializes in global climate data

2. **Zenodo - Environmental Research** (Quality: 0.85)
   - *Explanation*: Academic environmental research datasets
   - *Why recommended*: High-quality peer-reviewed climate research

## Understanding Quality Improvements

### Before vs After Enhancement

**Previous System:**
- Generic recommendations regardless of query type
- No geographic prioritization
- Limited quality validation
- 31.8% NDCG@3 accuracy

**Enhanced System:**
- Domain-specific intelligent routing
- Singapore-first strategy for local queries
- Expert-validated recommendations
- 72.1% NDCG@3 accuracy

### Quality Indicators to Look For

**High-Quality Recommendations:**
- ✅ Relevance score > 0.8
- ✅ Clear explanation of why recommended
- ✅ Appropriate source for your domain
- ✅ Recent data with regular updates

**Lower-Quality Recommendations:**
- ⚠️ Relevance score < 0.6
- ⚠️ Generic or unclear explanations
- ⚠️ Mismatched source for your research area
- ⚠️ Outdated or infrequently updated data

## Tips for Better Results

### 1. Be Specific About Your Research Domain

**Instead of:** "data about people"  
**Try:** "psychology research datasets" or "demographic statistics"

**Instead of:** "environmental stuff"  
**Try:** "climate change indicators" or "air quality data"

### 2. Include Geographic Context When Relevant

**For Singapore research:**
- "singapore education statistics"
- "local transport data"
- "hdb housing information"

**For global research:**
- "global climate data"
- "international trade statistics"
- "worldwide health indicators"

### 3. Use Research-Specific Terms

**Academic research:**
- "research datasets"
- "academic data"
- "peer-reviewed datasets"

**Business analysis:**
- "business intelligence data"
- "market research datasets"
- "industry statistics"

### 4. Specify Data Types When Known

- "time series data"
- "survey datasets"
- "experimental data"
- "longitudinal studies"

## Troubleshooting

### Getting Irrelevant Results?

1. **Check your query specificity** - Add domain-specific keywords
2. **Increase quality threshold** - Set minimum quality to 0.8
3. **Use geographic qualifiers** - Add "singapore" or "global" as appropriate
4. **Try alternative phrasings** - Use synonyms or different terminology

### Singapore Sources Not Appearing?

1. **Use explicit Singapore terms** - Include "singapore", "sg", or local terms
2. **Avoid global qualifiers** - Don't use "international" or "worldwide"
3. **Try local-specific queries** - "hdb", "mrt", "singstat"

### Domain Routing Issues?

1. **Use domain-specific keywords** - "psychology", "climate", "economics"
2. **Be explicit about research area** - "machine learning datasets" vs "data"
3. **Check result explanations** - See how your query was classified

## API Usage for Developers

### Basic Search Request

```bash
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "singapore housing data",
    "max_results": 10,
    "quality_threshold": 0.7,
    "enable_singapore_first": true,
    "include_explanations": true
  }'
```

### Response Format

```json
{
  "query_info": {
    "original_query": "singapore housing data",
    "classification": {
      "domain": "singapore",
      "singapore_first_applicable": true,
      "confidence": 0.92
    }
  },
  "recommendations": [
    {
      "rank": 1,
      "source": "data_gov_sg",
      "title": "HDB Resale Flat Prices",
      "quality_metrics": {
        "relevance_score": 0.94,
        "quality_score": 0.96,
        "confidence": 0.92
      },
      "explanation": "Official Singapore government housing data - highest relevance for local housing price queries"
    }
  ],
  "quality_summary": {
    "overall_ndcg_at_3": 0.87,
    "singapore_first_accuracy": 1.0
  }
}
```

### Quality Validation

```bash
curl -X POST http://localhost:8000/api/v2/validate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change data",
    "recommendations": [
      {"source": "world_bank", "relevance_score": 0.92}
    ]
  }'
```

## Feedback and Improvement

### How Your Feedback Helps

The system continuously learns from user feedback to improve recommendations:

- **High ratings** confirm good mappings and reinforce quality patterns
- **Low ratings** identify areas for improvement and trigger retraining
- **Specific feedback** helps understand why recommendations weren't helpful

### Providing Feedback

When you find recommendations particularly helpful or unhelpful, please provide feedback through:

1. **Rating system** - Rate recommendations 1-5 stars
2. **Comments** - Explain why a recommendation was/wasn't useful
3. **Alternative suggestions** - Suggest better sources if you know them

### What Happens to Your Feedback

- **Weekly analysis** - Feedback is analyzed weekly for patterns
- **Training data updates** - Good feedback becomes part of training data
- **Model retraining** - Monthly retraining incorporates user feedback
- **Quality improvements** - Continuous improvement based on real usage

## Best Practices Summary

### ✅ Do This
- Be specific about your research domain
- Use geographic qualifiers when relevant
- Set appropriate quality thresholds
- Provide feedback on recommendations
- Check explanations to understand routing decisions

### ❌ Avoid This
- Overly generic queries like "data" or "information"
- Mixing geographic contexts in one query
- Setting quality thresholds too high (>0.9) unless necessary
- Ignoring domain-specific terminology
- Expecting instant results - quality takes a moment

## Getting Help

### Documentation
- **Technical Documentation**: `/docs/technical/`
- **API Documentation**: `/docs/technical/API_DOCUMENTATION.md`
- **Troubleshooting Guide**: `/docs/technical/QUALITY_TROUBLESHOOTING_GUIDE.md`

### Support Channels
- **Email**: support@dataset-research.ai
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: This guide and technical documentation

### Common Questions

**Q: Why are results taking longer than before?**
A: We now prioritize quality over speed. The extra time ensures you get genuinely relevant recommendations rather than fast but irrelevant results.

**Q: Why don't I see my favorite data source in the results?**
A: The system now routes to the most appropriate sources for your specific query. Your preferred source might appear for different types of queries where it's more relevant.

**Q: How can I get more Singapore-specific results?**
A: Use explicit Singapore terms ("singapore", "sg") or local terms ("hdb", "mrt", "singstat") to trigger Singapore-first prioritization.

**Q: What does the quality score mean?**
A: Quality scores (0.0-1.0) indicate how relevant and reliable a dataset is for your specific query, based on expert curation and validation.

**Q: Can I still access all data sources?**
A: Yes! All sources remain available. The system just prioritizes the most relevant ones for your query at the top of the results.

This enhanced system is designed to save you time by providing more accurate, relevant recommendations for your research needs. The quality-first approach ensures you spend less time sifting through irrelevant results and more time on your actual research.