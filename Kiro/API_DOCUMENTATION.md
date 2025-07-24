# API Documentation - Enhanced Dataset Research Assistant

## Overview

This document provides comprehensive API documentation for the enhanced AI-Powered Dataset Research Assistant with search quality improvements. The system now includes conversational query processing, real-time URL validation, intelligent source routing, and dynamic performance metrics.

## New Features (Search Quality Improvements)

### 🗣️ Conversational Query Processing
- Intelligent intent detection for dataset vs non-dataset queries
- Query normalization for external source searches
- Singapore context detection and domain classification
- Inappropriate content filtering

### 🔗 URL Validation and Correction
- Real-time URL validation for all search results
- Automatic correction of broken external source URLs
- Source-specific URL pattern fixes (Kaggle, World Bank, AWS, etc.)
- Fallback URL strategies for failed validations

### 🎯 Enhanced Source Routing
- Minimum source coverage requirements (3+ sources when possible)
- Singapore-first strategy for local queries
- Domain-specific routing (psychology→Kaggle, climate→World Bank)
- Graceful handling of source failures

### 📊 Dynamic Performance Metrics
- Real-time NDCG@3 scores from actual model performance
- Actual response time measurements and cache hit rates
- System health monitoring integration
- No more hardcoded performance values

## Base URL

```
Production: https://api.dataset-research.ai
Development: http://localhost:8000
```

## Authentication

Currently, the API uses simple API key authentication:

```bash
curl -H "X-API-Key: your-api-key" https://api.dataset-research.ai/search
```

## Core Endpoints

### 1. Enhanced Search Endpoint

**Endpoint:** `POST /api/search`

**Description:** Primary search endpoint with conversational processing, URL validation, and intelligent routing.

**Request:**
```json
{
  "query": "I need Singapore housing data",
  "max_results": 10,
  "enable_conversational_processing": true,
  "enable_url_validation": true,
  "enable_singapore_first": true,
  "include_explanations": true,
  "filters": {
    "quality_threshold": 0.7,
    "source": ["data.gov.sg", "singstat"]
  }
}
```

**Response:**
```json
{
  "query_processing": {
    "original_query": "I need Singapore housing data",
    "is_dataset_request": true,
    "extracted_terms": ["Singapore", "housing", "data"],
    "confidence": 0.92,
    "detected_domain": "housing",
    "singapore_context": true,
    "processing_time_ms": 245
  },
  "recommendations": [
    {
      "rank": 1,
      "source": "data_gov_sg",
      "title": "HDB Resale Flat Prices",
      "description": "Historical resale prices of HDB flats from 1990 onwards",
      "url": "https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view",
      "quality_metrics": {
        "relevance_score": 0.94,
        "quality_score": 0.96,
        "ndcg_contribution": 0.94,
        "confidence": 0.92
      },
      "url_validation": {
        "status": "verified",
        "status_code": 200,
        "validation_timestamp": "2025-07-23T10:30:00Z",
        "response_time_ms": 156
      },
      "routing_info": {
        "source_priority": 1,
        "singapore_source": true,
        "domain_match": "housing",
        "routing_reason": "Singapore-first priority for local housing data"
      },
      "explanation": "Official Singapore government housing data - highest relevance for local housing price queries",
      "metadata": {
        "last_updated": "2024-01-15",
        "format": "CSV",
        "size": "45MB",
        "update_frequency": "Monthly"
      }
    },
    {
      "rank": 2,
      "source": "singstat",
      "title": "Private Residential Property Price Index",
      "description": "Price indices for private residential properties in Singapore",
      "url": "https://www.singstat.gov.sg/find-data/search-by-theme/economy/prices-and-price-indices",
      "quality_metrics": {
        "relevance_score": 0.89,
        "quality_score": 0.94,
        "ndcg_contribution": 0.81,
        "confidence": 0.88
      },
      "routing_info": {
        "source_priority": 2,
        "singapore_source": true,
        "domain_match": "singapore",
        "routing_reason": "Singapore Department of Statistics - authoritative local data"
      },
      "explanation": "Official statistics for private housing market trends and price indices",
      "metadata": {
        "last_updated": "2024-01-10",
        "format": "Excel",
        "size": "12MB",
        "update_frequency": "Quarterly"
      }
    }
  ],
  "quality_summary": {
    "overall_ndcg_at_3": 0.87,
    "average_relevance_score": 0.91,
    "singapore_first_accuracy": 1.0,
    "domain_routing_accuracy": 1.0,
    "quality_threshold_met": true,
    "cached_result": false
  },
  "routing_summary": {
    "singapore_first_applied": true,
    "domain_detected": "singapore",
    "sources_considered": ["data_gov_sg", "singstat", "lta_datamall", "world_bank"],
    "sources_filtered": 0,
    "quality_filtered": 0
  },
  "performance_info": {
    "total_time_ms": 1247,
    "neural_inference_ms": 245,
    "web_search_ms": 892,
    "llm_enhancement_ms": 110,
    "cache_lookup_ms": 12,
    "quality_validation_ms": 88
  }
}
```

**Quality Metrics Explanation:**
- **relevance_score**: Neural model's relevance prediction (0.0-1.0)
- **quality_score**: Overall quality assessment based on source reliability and content freshness
- **ndcg_contribution**: Contribution to overall NDCG@3 score
- **confidence**: Model's confidence in the recommendation

### 2. Query Classification Endpoint

**Endpoint:** `POST /api/v2/classify`

**Description:** Classify query intent, domain, and routing strategy without performing full search.

**Request:**
```json
{
  "query": "psychology research datasets",
  "include_routing_info": true
}
```

**Response:**
```json
{
  "classification": {
    "original_query": "psychology research datasets",
    "domain": "psychology",
    "geographic_scope": "global",
    "intent": "research",
    "confidence": 0.89,
    "singapore_first_applicable": false,
    "recommended_sources": ["kaggle", "zenodo", "world_bank"],
    "explanation": "Classified as psychology query; Recommending Kaggle for psychology datasets and competitions"
  },
  "routing_strategy": {
    "primary_sources": ["kaggle", "zenodo"],
    "fallback_sources": ["world_bank", "aws_opendata"],
    "singapore_priority": false,
    "domain_specific_routing": true,
    "expected_quality_score": 0.85
  },
  "processing_time_ms": 45
}
```

### 3. Quality Validation Endpoint

**Endpoint:** `POST /api/v2/validate`

**Description:** Validate recommendations against training mappings and quality thresholds.

**Request:**
```json
{
  "query": "climate change data",
  "recommendations": [
    {
      "source": "world_bank",
      "relevance_score": 0.92
    },
    {
      "source": "kaggle", 
      "relevance_score": 0.78
    }
  ]
}
```

**Response:**
```json
{
  "validation_result": {
    "overall_quality": "high",
    "ndcg_at_3": 0.85,
    "training_mapping_compliance": true,
    "quality_threshold_met": true,
    "recommendations_validated": 2
  },
  "individual_validation": [
    {
      "source": "world_bank",
      "validation_status": "excellent",
      "expected_relevance": 0.95,
      "actual_relevance": 0.92,
      "variance": -0.03,
      "training_mapping_match": true
    },
    {
      "source": "kaggle",
      "validation_status": "good", 
      "expected_relevance": 0.75,
      "actual_relevance": 0.78,
      "variance": 0.03,
      "training_mapping_match": true
    }
  ],
  "quality_feedback": {
    "strengths": ["Correct domain routing to World Bank for climate data"],
    "improvements": ["Consider adding Zenodo for academic climate research"],
    "compliance_score": 0.92
  }
}
```

### 4. Cache Management Endpoints

#### Get Cache Statistics

**Endpoint:** `GET /api/v2/cache/stats`

**Response:**
```json
{
  "cache_statistics": {
    "hit_rate": 0.847,
    "total_entries": 1247,
    "memory_entries": 892,
    "avg_quality_score": 0.823,
    "avg_ndcg_at_3": 0.789,
    "quality_threshold": 0.7,
    "quality_distribution": {
      "excellent": 456,
      "good": 623,
      "fair": 168,
      "poor": 0,
      "total": 1247
    },
    "training_mappings_loaded": 89
  },
  "performance_impact": {
    "avg_response_time_cached": "156ms",
    "avg_response_time_uncached": "1247ms",
    "cache_efficiency": "87.5%"
  }
}
```

#### Invalidate Low Quality Cache

**Endpoint:** `POST /api/v2/cache/invalidate`

**Request:**
```json
{
  "quality_threshold": 0.65,
  "force_refresh": false
}
```

**Response:**
```json
{
  "invalidation_result": {
    "entries_invalidated": 23,
    "quality_threshold_used": 0.65,
    "cache_size_before": 1247,
    "cache_size_after": 1224,
    "avg_quality_improvement": 0.034
  }
}
```

### 5. Quality Monitoring Endpoints

#### Get Quality Metrics

**Endpoint:** `GET /api/v2/quality/metrics`

**Response:**
```json
{
  "current_metrics": {
    "overall_ndcg_at_3": 0.721,
    "relevance_accuracy": 0.783,
    "domain_routing_accuracy": 0.852,
    "singapore_first_accuracy": 0.917,
    "user_satisfaction_score": 0.756,
    "recommendation_diversity": 0.634
  },
  "trend_analysis": {
    "ndcg_trend_7d": "+0.023",
    "relevance_trend_7d": "+0.015",
    "quality_improvement": "steady",
    "alert_status": "normal"
  },
  "training_mapping_compliance": {
    "total_mappings": 89,
    "compliant_predictions": 78,
    "compliance_rate": 0.876,
    "last_validation": "2024-01-15T10:30:00Z"
  }
}
```

#### Generate Quality Report

**Endpoint:** `POST /api/v2/quality/report`

**Request:**
```json
{
  "time_range": "7d",
  "include_details": true,
  "format": "json"
}
```

**Response:**
```json
{
  "report_summary": {
    "period": "2024-01-08 to 2024-01-15",
    "total_queries": 2847,
    "avg_quality_score": 0.789,
    "quality_threshold_compliance": 0.923
  },
  "domain_performance": {
    "psychology": {
      "queries": 234,
      "avg_ndcg": 0.856,
      "top_sources": ["kaggle", "zenodo"],
      "routing_accuracy": 0.897
    },
    "singapore": {
      "queries": 567,
      "avg_ndcg": 0.912,
      "top_sources": ["data_gov_sg", "singstat"],
      "singapore_first_accuracy": 0.945
    },
    "climate": {
      "queries": 189,
      "avg_ndcg": 0.834,
      "top_sources": ["world_bank", "zenodo"],
      "routing_accuracy": 0.823
    }
  },
  "quality_improvements": [
    {
      "improvement": "Singapore-first routing accuracy increased by 4.2%",
      "impact": "Better local data prioritization"
    },
    {
      "improvement": "Psychology domain routing improved by 2.8%", 
      "impact": "More accurate Kaggle/Zenodo recommendations"
    }
  ]
}
```

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "QUALITY_THRESHOLD_NOT_MET",
    "message": "No recommendations meet the specified quality threshold of 0.8",
    "details": {
      "query": "machine learning datasets",
      "quality_threshold": 0.8,
      "best_quality_found": 0.73,
      "suggestions": [
        "Lower quality threshold to 0.7",
        "Refine query with more specific terms",
        "Check if training mappings exist for this query type"
      ]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `QUALITY_THRESHOLD_NOT_MET` | No results meet quality threshold | 200 |
| `INVALID_QUERY` | Query is empty or malformed | 400 |
| `NEURAL_MODEL_ERROR` | Neural model inference failed | 500 |
| `CACHE_ERROR` | Cache system error | 500 |
| `TRAINING_MAPPING_ERROR` | Training mappings validation failed | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

## Quality-First Features

### 1. Singapore-First Strategy

The API automatically detects queries that would benefit from Singapore government data prioritization:

**Trigger Conditions:**
- Explicit mention of "Singapore" or "SG"
- Singapore-specific terms (HDB, MRT, LTA, etc.)
- Generic terms with local context (housing, transport, education, etc.)

**Source Prioritization:**
1. `data_gov_sg` - Singapore Open Data Portal
2. `singstat` - Department of Statistics Singapore  
3. `lta_datamall` - Land Transport Authority DataMall
4. Other sources based on domain relevance

### 2. Domain-Specific Routing

**Psychology Queries** → Kaggle, Zenodo
- Keywords: psychology, mental health, behavioral, cognitive
- Rationale: High-quality research datasets and competitions

**Climate Queries** → World Bank, Zenodo
- Keywords: climate, weather, environmental, temperature
- Rationale: Authoritative global climate data and research

**Economics Queries** → World Bank, SingStat
- Keywords: economic, GDP, financial, trade, poverty
- Rationale: Official economic indicators and statistics

### 3. Quality Validation

All recommendations are validated against:
- **Training Mappings**: Expert-curated query-source relevance scores
- **Quality Thresholds**: Minimum acceptable relevance scores
- **Domain Compliance**: Correct routing for specialized domains
- **Geographic Accuracy**: Appropriate local vs global source selection

## SDK and Client Libraries

### Python SDK

```python
from dataset_research_client import DatasetResearchClient

client = DatasetResearchClient(
    api_key="your-api-key",
    base_url="https://api.dataset-research.ai",
    quality_threshold=0.7
)

# Enhanced search with quality metrics
results = client.search(
    query="singapore housing prices",
    enable_singapore_first=True,
    include_explanations=True
)

print(f"NDCG@3: {results.quality_summary.overall_ndcg_at_3}")
print(f"Singapore-first applied: {results.routing_summary.singapore_first_applied}")

for rec in results.recommendations:
    print(f"{rec.rank}. {rec.title} (Quality: {rec.quality_metrics.quality_score})")
    print(f"   Explanation: {rec.explanation}")
```

### JavaScript SDK

```javascript
import { DatasetResearchClient } from '@dataset-research/client';

const client = new DatasetResearchClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.dataset-research.ai',
  qualityThreshold: 0.7
});

// Enhanced search with quality validation
const results = await client.search({
  query: 'psychology research datasets',
  enableDomainRouting: true,
  includeExplanations: true
});

console.log(`Domain detected: ${results.queryInfo.classification.domain}`);
console.log(`Quality score: ${results.qualitySummary.overallNdcgAt3}`);

results.recommendations.forEach(rec => {
  console.log(`${rec.rank}. ${rec.title}`);
  console.log(`   Quality: ${rec.qualityMetrics.qualityScore}`);
  console.log(`   Routing: ${rec.routingInfo.routingReason}`);
});
```

## Rate Limits

| Endpoint | Rate Limit | Burst Limit |
|----------|------------|-------------|
| `/api/v2/search` | 100/hour | 10/minute |
| `/api/v2/classify` | 200/hour | 20/minute |
| `/api/v2/validate` | 50/hour | 5/minute |
| `/api/v2/quality/*` | 20/hour | 2/minute |

## Monitoring and Observability

### Health Check

**Endpoint:** `GET /api/v2/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "components": {
    "neural_model": "healthy",
    "cache_system": "healthy", 
    "quality_validator": "healthy",
    "training_mappings": "healthy"
  },
  "quality_metrics": {
    "current_ndcg": 0.721,
    "quality_threshold": 0.7,
    "status": "above_threshold"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Metrics Endpoint

**Endpoint:** `GET /api/v2/metrics`

Returns Prometheus-compatible metrics for monitoring:

```
# HELP dataset_search_requests_total Total number of search requests
# TYPE dataset_search_requests_total counter
dataset_search_requests_total{status="success"} 12847
dataset_search_requests_total{status="error"} 23

# HELP dataset_search_quality_score Current quality score
# TYPE dataset_search_quality_score gauge  
dataset_search_quality_score{metric="ndcg_at_3"} 0.721
dataset_search_quality_score{metric="relevance_accuracy"} 0.783

# HELP dataset_cache_hit_rate Cache hit rate
# TYPE dataset_cache_hit_rate gauge
dataset_cache_hit_rate 0.847
```

This API documentation provides comprehensive coverage of the quality-first features and enhanced capabilities of the dataset research assistant, enabling developers to build high-quality applications that leverage the improved recommendation system.