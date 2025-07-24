# Enhanced Search API Documentation

## Overview

The Enhanced Search API provides intelligent dataset discovery with conversational query processing, URL validation, source routing, and performance metrics. This documentation covers the new search quality improvements including Singapore-first strategy, domain-specific routing, and real-time URL validation.

## Base Configuration

### Server Startup
The API server includes enhanced startup with port conflict resolution:

```bash
# Automatic port fallback (8000 → 8001 → 8002 → 8003)
python start_server.py

# Or via main launcher
python main.py --backend
```

### Health Check
Verify API status and performance metrics:

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 1456.03,
  "performance_stats": {
    "total_requests": 42,
    "avg_response_time": 0.78,
    "cache_hits": 28,
    "cache_misses": 14
  },
  "component_status": {
    "neural_model": "loaded (76.0% NDCG@3)",
    "search_engine": "healthy",
    "cache_manager": "healthy (66.67% hit rate)",
    "conversational_processor": "healthy"
  }
}
```

## Core Search Endpoint

### POST /api/search

Enhanced search with conversational processing and intelligent routing.

#### Request Format

```json
{
  "query": "I need Singapore housing data",
  "max_results": 10,
  "enable_conversational_processing": true,
  "enable_url_validation": true,
  "include_explanations": true,
  "filters": {
    "source": ["data.gov.sg", "singstat"],
    "quality_threshold": 0.7,
    "singapore_first": true
  }
}
```

#### Response Format

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
  "results": [
    {
      "rank": 1,
      "dataset_id": "hdb_resale_2024",
      "title": "HDB Resale Flat Prices",
      "description": "Monthly resale flat transactions...",
      "source": "data.gov.sg",
      "url": "https://data.gov.sg/datasets/hdb-resale-flat-prices",
      "quality_metrics": {
        "relevance_score": 0.94,
        "quality_score": 0.96,
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
        "routing_reason": "Singapore-first priority for local housing data"
      },
      "explanation": "Official Singapore government housing data - highest relevance for local housing price queries"
    }
  ],
  "routing_summary": {
    "singapore_first_applied": true,
    "sources_considered": ["data.gov.sg", "singstat", "lta_datamall"],
    "sources_filtered": 0,
    "minimum_sources_met": true
  },
  "performance": {
    "total_time_ms": 1247,
    "conversational_processing_ms": 245,
    "web_search_ms": 892,
    "url_validation_ms": 110,
    "cache_status": "miss"
  }
}
```

## Conversational Processing

### Intent Detection

The API automatically determines if input is a legitimate dataset request:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hello, how are you?",
    "enable_conversational_processing": true
  }'
```

Response for non-dataset queries:
```json
{
  "query_processing": {
    "original_query": "Hello, how are you?",
    "is_dataset_request": false,
    "confidence": 0.95,
    "suggested_response": "Hello! I specialize in helping find datasets. What type of data are you looking for?"
  },
  "results": [],
  "message": "Query identified as conversational, not dataset-related"
}
```

### Query Normalization

Conversational input is cleaned for external source searches:

```json
{
  "query": "I really need some HDB housing data for my research project",
  "query_processing": {
    "original_query": "I really need some HDB housing data for my research project",
    "extracted_terms": ["HDB", "housing", "data", "research"],
    "normalized_query": "HDB housing data research",
    "conversational_words_removed": ["I", "really", "need", "some", "for", "my", "project"]
  }
}
```

## URL Validation and Correction

### Real-time Validation

All returned URLs are validated in real-time:

```json
{
  "url_validation": {
    "status": "verified",           // verified, corrected, fallback, failed
    "status_code": 200,
    "validation_timestamp": "2025-07-23T10:30:00Z",
    "response_time_ms": 156,
    "original_url": "https://original-url.com",  // Only if corrected
    "correction_applied": false
  }
}
```

### URL Correction

Failed URLs are automatically corrected using source-specific patterns:

#### Kaggle URL Correction
```json
{
  "original_url": "https://kaggle.com/datasets/I need psychology data",
  "corrected_url": "https://www.kaggle.com/datasets?search=psychology+data",
  "correction_reason": "Removed conversational language and fixed search format"
}
```

#### World Bank URL Correction
```json
{
  "original_url": "https://worldbank.org/climate change data",
  "corrected_url": "https://data.worldbank.org/indicator?tab=all&q=climate+change+data",
  "correction_reason": "Fixed domain and search parameter format"
}
```

#### AWS Open Data URL Correction
```json
{
  "original_url": "https://opendata.aws/machine learning",
  "corrected_url": "https://registry.opendata.aws/search?q=machine+learning",
  "correction_reason": "Fixed domain and added search parameters"
}
```

## Source Coverage and Routing

### Minimum Source Requirements

The API ensures comprehensive source coverage:

```json
{
  "routing_summary": {
    "minimum_sources_required": 3,
    "sources_returned": 5,
    "sources_considered": ["data.gov.sg", "singstat", "kaggle", "world_bank", "zenodo"],
    "sources_filtered": 0,
    "minimum_sources_met": true
  }
}
```

### Singapore-First Strategy

For Singapore-related queries, local sources are prioritized:

```json
{
  "query": "housing prices",
  "routing_info": {
    "singapore_first_applied": true,
    "singapore_sources_prioritized": [
      {"source": "data.gov.sg", "priority": 1},
      {"source": "singstat", "priority": 2},
      {"source": "lta_datamall", "priority": 3}
    ],
    "global_sources": [
      {"source": "world_bank", "priority": 4},
      {"source": "kaggle", "priority": 5}
    ]
  }
}
```

### Domain-Specific Routing

Queries are routed to appropriate sources based on detected domain:

#### Psychology Queries → Kaggle, Zenodo
```json
{
  "query": "psychology research datasets",
  "routing_info": {
    "detected_domain": "psychology",
    "primary_sources": ["kaggle", "zenodo"],
    "routing_reason": "Psychology domain - prioritizing research datasets and competitions"
  }
}
```

#### Climate Queries → World Bank, Zenodo
```json
{
  "query": "climate change indicators",
  "routing_info": {
    "detected_domain": "climate",
    "primary_sources": ["world_bank", "zenodo"],
    "routing_reason": "Climate domain - prioritizing authoritative global data"
  }
}
```

## Error Handling

### Port Conflict Resolution

The server automatically handles port conflicts:

```bash
# Server startup log
INFO: Port 8000 is already in use
INFO: Trying port 8001...
INFO: Server started successfully on port 8001
INFO: API available at http://localhost:8001
```

### Graceful Degradation

When components fail, the system provides fallbacks:

```json
{
  "error": {
    "component": "url_validator",
    "message": "URL validation temporarily unavailable",
    "fallback_used": true,
    "impact": "URLs returned without real-time validation"
  },
  "results": [
    {
      "url_validation": {
        "status": "not_validated",
        "reason": "Validation service unavailable",
        "fallback_applied": true
      }
    }
  ]
}
```

### Query Processing Errors

```json
{
  "error": {
    "type": "conversational_processing_error",
    "message": "Unable to determine query intent",
    "suggested_action": "Please specify what type of dataset you're looking for",
    "fallback_processing": true
  }
}
```

## Performance Metrics

### Real-time Performance Data

The API provides actual performance metrics:

```bash
curl http://localhost:8000/api/metrics
```

Response:
```json
{
  "neural_performance": {
    "ndcg_at_3": 76.0,
    "singapore_accuracy": 100.0,
    "domain_accuracy": 100.0,
    "last_updated": "2025-07-23T10:30:00Z"
  },
  "response_time": {
    "average_response_time": 4.75,
    "improvement_percentage": 84.0,
    "samples_count": 150
  },
  "cache_performance": {
    "overall_hit_rate": 66.67,
    "search_cache_hit_rate": 70.2,
    "neural_cache_hit_rate": 65.8,
    "quality_cache_hit_rate": 64.1
  },
  "system_health": {
    "system_status": "healthy",
    "components_operational": 5,
    "components_total": 5
  }
}
```

### Performance Monitoring

Enable detailed performance tracking:

```json
{
  "query": "singapore transport data",
  "enable_performance_tracking": true,
  "performance": {
    "total_time_ms": 1247,
    "breakdown": {
      "conversational_processing_ms": 245,
      "neural_inference_ms": 156,
      "web_search_ms": 892,
      "url_validation_ms": 110,
      "cache_lookup_ms": 12,
      "response_formatting_ms": 32
    },
    "cache_performance": {
      "cache_hits": 3,
      "cache_misses": 2,
      "cache_efficiency": "60%"
    }
  }
}
```

## Advanced Features

### Batch URL Validation

Validate multiple URLs simultaneously:

```bash
curl -X POST http://localhost:8000/api/validate-urls \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://data.gov.sg/datasets/hdb-resale-flat-prices",
      "https://www.kaggle.com/datasets/psychology-data",
      "https://data.worldbank.org/indicator/climate-data"
    ]
  }'
```

Response:
```json
{
  "validation_results": [
    {
      "url": "https://data.gov.sg/datasets/hdb-resale-flat-prices",
      "status": "verified",
      "status_code": 200,
      "response_time_ms": 156
    },
    {
      "url": "https://www.kaggle.com/datasets/psychology-data",
      "status": "corrected",
      "status_code": 200,
      "corrected_url": "https://www.kaggle.com/datasets?search=psychology+data",
      "response_time_ms": 234
    }
  ]
}
```

### Source Coverage Analysis

Analyze source coverage for specific domains:

```bash
curl -X POST http://localhost:8000/api/analyze-coverage \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "psychology",
    "query_samples": ["psychology datasets", "mental health data", "behavioral research"]
  }'
```

Response:
```json
{
  "domain": "psychology",
  "coverage_analysis": {
    "recommended_sources": ["kaggle", "zenodo", "huggingface"],
    "source_relevance": {
      "kaggle": 0.92,
      "zenodo": 0.88,
      "huggingface": 0.75,
      "world_bank": 0.23
    },
    "coverage_score": 0.85,
    "recommendations": [
      "Kaggle provides excellent psychology datasets and competitions",
      "Zenodo offers high-quality academic psychology research data",
      "Consider adding PsychData.org for specialized psychology datasets"
    ]
  }
}
```

## Client Integration Examples

### Python Client

```python
import requests
import asyncio

class EnhancedSearchClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def search_datasets(self, query, **options):
        """Search with enhanced features"""
        payload = {
            "query": query,
            "enable_conversational_processing": True,
            "enable_url_validation": True,
            "include_explanations": True,
            **options
        }
        
        response = requests.post(f"{self.base_url}/api/search", json=payload)
        return response.json()
    
    async def validate_intent(self, query):
        """Check if query is dataset-related"""
        result = await self.search_datasets(query, max_results=0)
        return result["query_processing"]["is_dataset_request"]

# Usage
client = EnhancedSearchClient()
results = await client.search_datasets("I need Singapore housing data")

if results["query_processing"]["is_dataset_request"]:
    for result in results["results"]:
        print(f"✅ {result['title']} - {result['url_validation']['status']}")
else:
    print("❌ Not a dataset request")
```

### JavaScript Client

```javascript
class EnhancedSearchAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async searchDatasets(query, options = {}) {
        const payload = {
            query,
            enable_conversational_processing: true,
            enable_url_validation: true,
            include_explanations: true,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
    
    async validateUrls(urls) {
        const response = await fetch(`${this.baseUrl}/api/validate-urls`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls })
        });
        
        return await response.json();
    }
}

// Usage
const api = new EnhancedSearchAPI();

const results = await api.searchDatasets("psychology research datasets");
console.log(`Intent confidence: ${results.query_processing.confidence}`);
console.log(`Sources found: ${results.results.length}`);

// Validate URLs
const urls = results.results.map(r => r.url);
const validation = await api.validateUrls(urls);
console.log(`URLs validated: ${validation.validation_results.length}`);
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check which ports are in use
   lsof -i :8000
   
   # Server will automatically try 8001, 8002, 8003
   # Check logs for actual port used
   ```

2. **URL Validation Failures**
   ```json
   {
     "url_validation": {
       "status": "failed",
       "error": "Connection timeout",
       "fallback_applied": true,
       "fallback_url": "https://source-homepage.com"
     }
   }
   ```

3. **Low Confidence Scores**
   ```json
   {
     "query_processing": {
       "confidence": 0.45,
       "suggested_clarification": "Could you specify what type of data you're looking for?"
     }
   }
   ```

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python start_server.py
```

### Health Diagnostics

```bash
curl http://localhost:8000/api/health?detailed=true
```

This enhanced API provides comprehensive dataset discovery with intelligent query processing, real-time URL validation, and performance monitoring, ensuring users get high-quality, accessible dataset recommendations.