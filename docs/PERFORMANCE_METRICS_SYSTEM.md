# Performance Metrics System Documentation

## Overview

The Performance Metrics System dynamically collects and reports actual system performance metrics, replacing hardcoded values with real-time data from neural models, cache systems, and monitoring infrastructure. This system provides transparency into actual system performance and enables data-driven optimization.

## Architecture

### Core Component: PerformanceMetricsCollector

Located in `src/ai/performance_metrics_collector.py`, this component integrates with existing monitoring systems to provide comprehensive performance insights.

```python
from src.ai.performance_metrics_collector import PerformanceMetricsCollector

# Initialize with system configuration
collector = PerformanceMetricsCollector(config)

# Get all performance metrics
metrics = await collector.get_all_metrics()
```

### System Integration

```
PerformanceMetricsCollector
├── Neural Performance (models/dl/quality_first/)
├── Cache Performance (cache/*/cache_metadata.db)
├── Response Time Metrics (API monitoring)
├── System Health (deployment/health_monitor.py)
└── Metrics Database (cache/performance_metrics.db)
```

## Key Features

### 1. Neural Performance Tracking

Retrieves actual NDCG@3 scores from trained models:

```python
neural_metrics = await collector.get_current_neural_performance()
# Returns:
{
    'ndcg_at_3': 76.0,                    # Actual model performance
    'final_test_ndcg': 76.0,
    'singapore_accuracy': 100.0,
    'domain_accuracy': 100.0,
    'training_time_minutes': 45.2,
    'model_parameters': 125000,
    'last_updated': '2025-07-23T10:30:00Z'
}
```

### 2. Real-Time Response Metrics

Collects actual response times from API calls:

```python
response_metrics = await collector.get_response_time_metrics()
# Returns:
{
    'average_response_time': 4.75,        # Actual measured time
    'min_response_time': 1.2,
    'max_response_time': 8.3,
    'samples_count': 150,
    'improvement_percentage': 84.0        # vs baseline
}
```

### 3. Cache Performance Monitoring

Tracks cache hit rates across all cache systems:

```python
cache_metrics = await collector.get_cache_performance()
# Returns:
{
    'overall_hit_rate': 66.67,           # Actual cache performance
    'search_cache_hit_rate': 70.2,
    'neural_cache_hit_rate': 65.8,
    'quality_cache_hit_rate': 64.1,
    'cache_entries': 1247
}
```

### 4. System Health Monitoring

Integrates with deployment health monitoring:

```python
health_metrics = await collector.get_system_health_metrics()
# Returns:
{
    'system_status': 'healthy',
    'response_time': 0.78,
    'cpu_usage': 45.2,
    'memory_usage': 67.8,
    'error_rate': 0.02,
    'requests_per_minute': 42
}
```

## Data Sources

### Neural Model Performance

**Source**: `models/dl/quality_first/training_summary.json`

```json
{
  "best_ndcg_at_3": 0.760,
  "final_test_ndcg": 0.760,
  "test_metrics": {
    "singapore_accuracy": 1.0,
    "domain_accuracy": 1.0
  },
  "training_time_minutes": 45.2,
  "model_parameters": 125000
}
```

### Cache Statistics

**Sources**: 
- `cache/*/cache_metadata.db` (SQLite databases)
- `src/ai/intelligent_cache.py` (CacheManager)
- `src/ai/quality_aware_cache.py` (QualityAwareCacheManager)

### API Performance

**Sources**:
- Health monitoring endpoint (`/api/health`)
- Response time measurements
- Error rate tracking

### System Metrics

**Sources**:
- `src/deployment/health_monitor.py`
- System resource monitoring
- Database performance logs

## Metrics Database

### Schema

The system maintains a SQLite database at `cache/performance_metrics.db`:

```sql
-- Performance metrics table
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    metric_type TEXT,
    metric_name TEXT,
    metric_value REAL,
    metadata TEXT
);

-- System health table
CREATE TABLE system_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    status TEXT,
    response_time REAL,
    cpu_usage REAL,
    memory_usage REAL,
    cache_hit_rate REAL,
    error_rate REAL,
    requests_per_minute INTEGER
);
```

### Data Retention

- **Performance Metrics**: 30 days
- **System Health**: 7 days
- **Automatic Cleanup**: Daily at 2 AM

## API Integration

### Startup Banner Integration

The system integrates with `main.py` to display actual metrics:

```python
# In main.py
async def print_banner():
    collector = PerformanceMetricsCollector(config)
    metrics = await collector.get_all_metrics()
    formatted = collector.format_metrics_for_display(metrics)
    
    print(f"""
🧠 Neural Performance: {formatted['ndcg_at_3']} {formatted['ndcg_status']}
⚡ Response Time: {formatted['response_time']}
💾 Cache Hit Rate: {formatted['cache_hit_rate']}
🏥 System Status: {formatted['system_status']}
📊 Last Updated: {formatted['last_updated']}
    """)
```

### Health Endpoint Integration

```python
# GET /api/health response includes real metrics
{
  "status": "healthy",
  "performance_stats": {
    "neural_ndcg_at_3": 76.0,
    "avg_response_time": 4.75,
    "cache_hit_rate": 66.67,
    "total_requests": 1247
  },
  "component_status": {
    "neural_model": "loaded (76.0% NDCG@3)",
    "cache_manager": "healthy (66.67% hit rate)",
    "performance_collector": "active"
  }
}
```

## Configuration

### Basic Configuration

```yaml
# config/ai_config.yml
performance_metrics:
  enable_collection: true
  collection_interval: 60  # seconds
  database_path: "cache/performance_metrics.db"
  retention_days: 30
  
  # Component integrations
  neural_model_path: "models/dl/quality_first/"
  cache_directories: ["cache/llm", "cache/neural", "cache/quality_aware"]
  health_monitor_url: "http://localhost:8000/api/health"
```

### Advanced Configuration

```yaml
performance_metrics:
  # Monitoring integrations
  enable_neural_monitoring: true
  enable_cache_monitoring: true
  enable_health_monitoring: true
  enable_response_tracking: true
  
  # Performance thresholds
  response_time_threshold: 5.0  # seconds
  cache_hit_rate_threshold: 60.0  # percentage
  error_rate_threshold: 5.0  # percentage
  
  # Database settings
  database_cleanup_hour: 2  # 2 AM daily cleanup
  max_database_size_mb: 100
  
  # Fallback values (when real data unavailable)
  fallback_ndcg: 70.0
  fallback_response_time: 5.0
  fallback_cache_hit_rate: 60.0
```

## Usage Examples

### Basic Metrics Collection

```python
from src.ai.performance_metrics_collector import PerformanceMetricsCollector

# Initialize collector
collector = PerformanceMetricsCollector(config)

# Get all metrics
all_metrics = await collector.get_all_metrics()
print(f"Neural NDCG@3: {all_metrics['neural_performance']['ndcg_at_3']:.1f}%")
print(f"Cache hit rate: {all_metrics['cache_performance']['overall_hit_rate']:.1f}%")

# Format for display
formatted = collector.format_metrics_for_display(all_metrics)
print(formatted['ndcg_at_3'])  # "76.0% NDCG@3 (TARGET EXCEEDED)"
```

### Response Time Tracking

```python
# Track individual response times
collector.add_response_time(2.34, 'search_query')
collector.add_response_time(1.87, 'dataset_details')

# Get aggregated metrics
response_metrics = await collector.get_response_time_metrics()
print(f"Average: {response_metrics['average_response_time']:.2f}s")
```

### Performance Trends

```python
# Get performance trends over time
neural_trends = collector.get_performance_trends('neural_performance', hours=24)
for trend in neural_trends:
    print(f"{trend['timestamp']}: {trend['metric_name']} = {trend['metric_value']}")

# Get system health history
health_history = collector.get_system_health_history(hours=6)
for health in health_history:
    print(f"Status: {health['status']}, Response: {health['response_time']:.2f}s")
```

### Custom Metrics

```python
# Log custom performance metrics
collector.log_performance_metric(
    metric_type='custom_feature',
    metric_name='query_complexity_score',
    metric_value=0.85,
    metadata={'query_length': 45, 'domain': 'psychology'}
)

# Log system health events
collector.log_system_health({
    'status': 'healthy',
    'response_time': 1.23,
    'cpu_usage': 45.2,
    'memory_usage': 67.8,
    'cache_hit_rate': 70.5,
    'error_rate': 0.01,
    'requests_per_minute': 42
})
```

## Monitoring Integration

### Health Monitor Integration

```python
# Connect to existing health monitor
from src.deployment.health_monitor import HealthMonitor

collector = PerformanceMetricsCollector(config)
collector.start_monitoring_integration()

# Health monitor automatically feeds data to collector
```

### Cache Manager Integration

```python
# Automatic integration with cache managers
from src.ai.intelligent_cache import CacheManager
from src.ai.quality_aware_cache import QualityAwareCacheManager

# Collector automatically discovers and connects to cache systems
cache_stats = await collector.get_cache_performance()
```

## Fallback Handling

### Graceful Degradation

When real metrics are unavailable, the system provides meaningful fallbacks:

```python
# Neural model not available
{
    'ndcg_at_3': 'Calculating...',
    'note': 'Neural model loading'
}

# Cache system unavailable
{
    'cache_hit_rate': 'Not Available',
    'note': 'Cache system initializing'
}

# Health monitor unavailable
{
    'system_status': 'Checking...',
    'note': 'Health monitor starting'
}
```

### Display Formatting

```python
formatted = collector.format_metrics_for_display(metrics)

# Handles missing data gracefully
print(formatted['ndcg_at_3'])        # "76.0% NDCG@3" or "Calculating..."
print(formatted['response_time'])    # "4.75s average" or "Measuring..."
print(formatted['cache_hit_rate'])   # "66.67% hit rate" or "Not Available"
```

## Performance Impact

### Collection Overhead

- **Metrics Collection**: < 100ms per collection cycle
- **Database Operations**: < 50ms per write
- **Memory Usage**: < 10MB for metrics storage
- **CPU Impact**: < 1% additional CPU usage

### Optimization Features

```python
# Async collection to avoid blocking
metrics = await collector.get_all_metrics()

# Caching of expensive operations
@lru_cache(maxsize=128)
def get_cached_neural_metrics():
    return load_neural_performance()

# Batch database operations
collector.batch_log_metrics(metrics_list)
```

## Troubleshooting

### Common Issues

1. **Metrics Database Locked**
   ```bash
   # Remove lock files
   rm cache/performance_metrics.db-wal
   rm cache/performance_metrics.db-shm
   ```

2. **Neural Model Metrics Missing**
   ```bash
   # Check if training summary exists
   ls -la models/dl/quality_first/training_summary.json
   
   # Retrain model if needed
   python dl_pipeline.py
   ```

3. **Cache Statistics Unavailable**
   ```python
   # Check cache manager connections
   collector = PerformanceMetricsCollector(config)
   print(f"Cache managers: {len(collector.cache_managers)}")
   ```

### Debug Mode

```python
import logging
logging.getLogger('src.ai.performance_metrics_collector').setLevel(logging.DEBUG)

# Detailed logging of metrics collection
collector = PerformanceMetricsCollector(config)
metrics = await collector.get_all_metrics()
```

### Health Checks

```python
# Check collector status
status = collector.get_monitoring_status()
print(f"Health monitor active: {status['health_monitor_active']}")
print(f"Cache managers connected: {status['cache_managers_connected']}")
print(f"Database available: {status['metrics_database_available']}")
```

## Best Practices

### 1. Regular Monitoring

```python
# Set up periodic metrics collection
import asyncio

async def periodic_collection():
    collector = PerformanceMetricsCollector(config)
    while True:
        metrics = await collector.get_all_metrics()
        # Process metrics...
        await asyncio.sleep(60)  # Collect every minute
```

### 2. Alerting Integration

```python
# Set up performance alerts
async def check_performance_alerts():
    metrics = await collector.get_all_metrics()
    
    # Alert on low NDCG@3
    ndcg = metrics['neural_performance'].get('ndcg_at_3', 0)
    if ndcg < 70:
        send_alert(f"Neural performance below threshold: {ndcg}%")
    
    # Alert on high response times
    response_time = metrics['response_time'].get('average_response_time', 0)
    if response_time > 5:
        send_alert(f"Response time above threshold: {response_time}s")
```

### 3. Performance Optimization

```python
# Use metrics for optimization decisions
metrics = await collector.get_all_metrics()

# Adjust cache size based on hit rate
hit_rate = metrics['cache_performance'].get('overall_hit_rate', 0)
if hit_rate < 60:
    increase_cache_size()

# Scale resources based on response time
response_time = metrics['response_time'].get('average_response_time', 0)
if response_time > 3:
    scale_up_resources()
```

The Performance Metrics System provides comprehensive, real-time insights into system performance, enabling data-driven optimization and transparent performance reporting. By replacing hardcoded values with actual measurements, it ensures users have accurate information about system capabilities and performance.