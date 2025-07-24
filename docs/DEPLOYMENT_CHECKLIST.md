# Deployment Checklist - Search Quality Improvements

## Overview

This checklist ensures successful deployment of the search quality improvements including conversational query processing, URL validation, source routing enhancements, and performance metrics system.

## Pre-Deployment Checklist

### ✅ Environment Setup

- [ ] **Python Version**: Verify Python 3.8+ is installed
- [ ] **Dependencies**: Install all requirements from `requirements.txt`
- [ ] **Virtual Environment**: Set up isolated Python environment
- [ ] **Environment Variables**: Configure required API keys and settings

```bash
# Environment setup commands
python --version  # Should be 3.8+
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Required environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export LOG_LEVEL="INFO"  # or DEBUG for detailed logging
```

### ✅ Configuration Files

- [ ] **AI Configuration**: Verify `config/ai_config.yml` exists and is valid
- [ ] **API Configuration**: Check `config/api_config.yml` (optional)
- [ ] **Neural Model Config**: Ensure `config/dl_config.yml` is configured
- [ ] **Conversational Settings**: Validate conversational query processing config

```yaml
# config/ai_config.yml - Key sections to verify
conversational_query:
  confidence_threshold: 0.7
  max_processing_time: 3.0
  enable_singapore_detection: true

performance_metrics:
  enable_collection: true
  collection_interval: 60
  database_path: "cache/performance_metrics.db"

url_validation:
  timeout: 10
  max_retries: 2
  enable_real_time_validation: true
```

### ✅ Neural Model Setup

- [ ] **Model Files**: Verify neural model files exist in `models/dl/quality_first/`
- [ ] **Training Summary**: Check `training_summary.json` for performance metrics
- [ ] **Model Loading**: Test model loading without errors

```bash
# Check neural model files
ls -la models/dl/quality_first/
# Should contain:
# - best_quality_model.pt
# - training_summary.json
# - training_history.json

# Test model loading
python -c "
from src.dl.quality_first_neural_model import QualityFirstNeuralModel
model = QualityFirstNeuralModel()
print('Neural model loaded successfully')
"
```

### ✅ Database Setup

- [ ] **Cache Directories**: Ensure cache directories exist with proper permissions
- [ ] **Performance Database**: Verify metrics database can be created
- [ ] **Database Permissions**: Check read/write permissions for cache directories

```bash
# Create cache directories
mkdir -p cache/{llm,neural,quality_aware,search}
mkdir -p cache/test_{integration,quality_aware}

# Test database creation
python -c "
from src.ai.performance_metrics_collector import PerformanceMetricsCollector
collector = PerformanceMetricsCollector()
print('Performance metrics database initialized')
"
```

### ✅ Component Testing

- [ ] **Conversational Processor**: Test intent detection and query normalization
- [ ] **URL Validator**: Test URL validation and correction functionality
- [ ] **Web Search Engine**: Test external source URL generation
- [ ] **Performance Collector**: Test metrics collection from all sources

```bash
# Run component tests
python -m pytest tests/test_conversational_query_processor.py -v
python -m pytest tests/test_url_generation_validation_integration.py -v
python -m pytest tests/test_system_source_coverage_performance.py -v
```

## Deployment Steps

### 1. Server Startup Testing

- [ ] **Port Availability**: Test automatic port fallback functionality
- [ ] **Health Check**: Verify health endpoint responds correctly
- [ ] **Component Status**: Check all components load successfully

```bash
# Test server startup
python start_server.py

# In another terminal, test health endpoint
curl http://localhost:8000/api/health

# Expected response should include:
# - status: "healthy"
# - component_status with all components healthy
# - performance_stats with actual metrics
```

### 2. API Functionality Testing

- [ ] **Basic Search**: Test basic dataset search functionality
- [ ] **Conversational Processing**: Test intent detection for various inputs
- [ ] **URL Validation**: Verify URLs are validated and corrected
- [ ] **Source Routing**: Test Singapore-first and domain-specific routing

```bash
# Test basic search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "singapore housing data", "max_results": 5}'

# Test conversational processing
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?", "enable_conversational_processing": true}'

# Test URL validation
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "psychology datasets", "enable_url_validation": true}'
```

### 3. Performance Metrics Validation

- [ ] **Neural Metrics**: Verify actual NDCG@3 scores are displayed
- [ ] **Response Times**: Check real response time measurements
- [ ] **Cache Performance**: Validate cache hit rate reporting
- [ ] **System Health**: Test health monitoring integration

```bash
# Test performance metrics endpoint
curl http://localhost:8000/api/metrics

# Check startup banner shows real metrics (not hardcoded values)
python main.py | grep -E "(NDCG|response|cache|System)"

# Verify metrics database is being populated
sqlite3 cache/performance_metrics.db "SELECT COUNT(*) FROM performance_metrics;"
```

### 4. Error Handling Testing

- [ ] **Port Conflicts**: Test behavior when default port is occupied
- [ ] **Component Failures**: Test graceful degradation when components fail
- [ ] **Invalid Queries**: Test handling of malformed or inappropriate queries
- [ ] **Network Issues**: Test behavior with network connectivity problems

```bash
# Test port conflict handling
# Start server on port 8000 in one terminal
python start_server.py

# Start another instance (should use port 8001)
python start_server.py

# Test invalid queries
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "", "max_results": 5}'

# Test inappropriate queries
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "inappropriate content here", "enable_conversational_processing": true}'
```

## Production Deployment

### ✅ Production Configuration

- [ ] **Logging Level**: Set appropriate logging level for production
- [ ] **Performance Settings**: Configure production performance settings
- [ ] **Security Settings**: Enable security features and rate limiting
- [ ] **Monitoring**: Set up production monitoring and alerting

```yaml
# Production configuration adjustments
logging:
  level: "INFO"  # Not DEBUG in production
  file: "logs/production_api.log"

performance_metrics:
  collection_interval: 300  # 5 minutes in production
  retention_days: 30

security:
  enable_rate_limiting: true
  max_requests_per_minute: 60
  enable_cors: true
  allowed_origins: ["https://yourdomain.com"]
```

### ✅ Production Testing

- [ ] **Load Testing**: Test system under expected production load
- [ ] **Concurrent Users**: Verify system handles multiple simultaneous users
- [ ] **Memory Usage**: Monitor memory consumption under load
- [ ] **Response Times**: Validate response times meet SLA requirements

```bash
# Load testing with multiple concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "test query '$i'", "max_results": 10}' &
done
wait

# Monitor system resources
top -p $(pgrep -f "start_server.py")
```

### ✅ Monitoring Setup

- [ ] **Health Monitoring**: Set up automated health checks
- [ ] **Performance Alerts**: Configure alerts for performance degradation
- [ ] **Error Monitoring**: Set up error tracking and notification
- [ ] **Log Monitoring**: Configure log aggregation and analysis

```bash
# Set up health monitoring script
cat > monitor_health.sh << 'EOF'
#!/bin/bash
while true; do
  response=$(curl -s http://localhost:8000/api/health)
  status=$(echo $response | jq -r '.status')
  if [ "$status" != "healthy" ]; then
    echo "ALERT: System unhealthy at $(date)"
    # Send alert notification
  fi
  sleep 60
done
EOF

chmod +x monitor_health.sh
nohup ./monitor_health.sh &
```

## Post-Deployment Validation

### ✅ Functionality Verification

- [ ] **Search Quality**: Verify search results are relevant and high-quality
- [ ] **URL Accessibility**: Check that returned URLs are accessible
- [ ] **Performance Metrics**: Confirm metrics are being collected and displayed
- [ ] **Error Handling**: Test error scenarios work as expected

### ✅ Performance Validation

- [ ] **Response Times**: Measure actual response times vs targets
- [ ] **Cache Efficiency**: Monitor cache hit rates and effectiveness
- [ ] **Neural Performance**: Verify NDCG@3 scores meet requirements
- [ ] **System Resources**: Monitor CPU, memory, and disk usage

```bash
# Performance validation script
cat > validate_performance.sh << 'EOF'
#!/bin/bash

echo "=== Performance Validation ==="

# Test response times
start_time=$(date +%s%N)
curl -s -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "singapore housing data", "max_results": 10}' > /dev/null
end_time=$(date +%s%N)
response_time=$(( (end_time - start_time) / 1000000 ))

echo "Response time: ${response_time}ms"

# Check metrics
metrics=$(curl -s http://localhost:8000/api/metrics)
ndcg=$(echo $metrics | jq -r '.neural_performance.ndcg_at_3')
cache_hit_rate=$(echo $metrics | jq -r '.cache_performance.overall_hit_rate')

echo "NDCG@3: ${ndcg}%"
echo "Cache hit rate: ${cache_hit_rate}%"

# Validate targets
if (( $(echo "$ndcg >= 70" | bc -l) )); then
  echo "✅ NDCG@3 target met"
else
  echo "❌ NDCG@3 below target (70%)"
fi

if (( $(echo "$cache_hit_rate >= 60" | bc -l) )); then
  echo "✅ Cache hit rate target met"
else
  echo "❌ Cache hit rate below target (60%)"
fi
EOF

chmod +x validate_performance.sh
./validate_performance.sh
```

### ✅ User Acceptance Testing

- [ ] **Query Variety**: Test with diverse query types and formats
- [ ] **Singapore Queries**: Verify Singapore-first strategy works correctly
- [ ] **Domain Routing**: Test domain-specific routing accuracy
- [ ] **Conversational Inputs**: Test various conversational input styles

```bash
# User acceptance test queries
test_queries=(
  "singapore housing prices"
  "I need HDB data for my research"
  "psychology research datasets"
  "climate change indicators"
  "Hello, how are you?"
  "transport statistics"
  "machine learning competitions"
)

for query in "${test_queries[@]}"; do
  echo "Testing: $query"
  response=$(curl -s -X POST http://localhost:8000/api/search \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"enable_conversational_processing\": true}")
  
  is_dataset=$(echo $response | jq -r '.query_processing.is_dataset_request')
  confidence=$(echo $response | jq -r '.query_processing.confidence')
  results_count=$(echo $response | jq -r '.results | length')
  
  echo "  Dataset request: $is_dataset, Confidence: $confidence, Results: $results_count"
  echo ""
done
```

## Rollback Plan

### ✅ Rollback Preparation

- [ ] **Backup Current Version**: Create backup of working system
- [ ] **Rollback Scripts**: Prepare automated rollback procedures
- [ ] **Database Backup**: Backup current database state
- [ ] **Configuration Backup**: Save current configuration files

```bash
# Create deployment backup
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $backup_dir

# Backup key files
cp -r src/ $backup_dir/
cp -r config/ $backup_dir/
cp -r models/ $backup_dir/
cp -r cache/ $backup_dir/
cp requirements.txt $backup_dir/

echo "Backup created in $backup_dir"
```

### ✅ Rollback Triggers

Initiate rollback if:
- [ ] **Response Times**: Average response time > 10 seconds
- [ ] **Error Rate**: Error rate > 10%
- [ ] **Neural Performance**: NDCG@3 drops below 60%
- [ ] **System Stability**: Frequent crashes or memory issues

### ✅ Rollback Procedure

```bash
# Automated rollback script
cat > rollback.sh << 'EOF'
#!/bin/bash

echo "=== INITIATING ROLLBACK ==="

# Stop current server
pkill -f "start_server.py"

# Restore from backup
BACKUP_DIR=$1
if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: ./rollback.sh <backup_directory>"
  exit 1
fi

# Restore files
cp -r $BACKUP_DIR/src/ ./
cp -r $BACKUP_DIR/config/ ./
cp -r $BACKUP_DIR/models/ ./
cp $BACKUP_DIR/requirements.txt ./

# Reinstall dependencies
pip install -r requirements.txt

# Restart server
python start_server.py &

echo "=== ROLLBACK COMPLETE ==="
EOF

chmod +x rollback.sh
```

## Success Criteria

### ✅ Deployment Success Indicators

- [ ] **Server Startup**: Server starts successfully with automatic port fallback
- [ ] **Health Check**: `/api/health` returns "healthy" status
- [ ] **Performance Metrics**: Real metrics displayed (not hardcoded values)
- [ ] **Conversational Processing**: Intent detection works for various inputs
- [ ] **URL Validation**: URLs are validated and corrected as needed
- [ ] **Source Routing**: Singapore-first and domain routing work correctly
- [ ] **Response Times**: Average response time < 5 seconds
- [ ] **Error Rate**: Error rate < 5%
- [ ] **Neural Performance**: NDCG@3 ≥ 70%
- [ ] **Cache Performance**: Cache hit rate ≥ 60%

### ✅ Quality Assurance

- [ ] **Search Relevance**: Search results are relevant to queries
- [ ] **URL Accessibility**: Returned URLs are accessible and working
- [ ] **Error Handling**: System handles errors gracefully
- [ ] **Performance Stability**: Performance remains stable under load
- [ ] **Monitoring**: All monitoring systems are operational

## Maintenance

### ✅ Ongoing Maintenance Tasks

- [ ] **Log Monitoring**: Regular review of application logs
- [ ] **Performance Monitoring**: Track performance metrics trends
- [ ] **Database Maintenance**: Regular cleanup of metrics database
- [ ] **Model Updates**: Periodic retraining of neural models
- [ ] **Dependency Updates**: Regular security and feature updates

### ✅ Scheduled Tasks

```bash
# Daily maintenance script
cat > daily_maintenance.sh << 'EOF'
#!/bin/bash

# Clean old logs (keep 7 days)
find logs/ -name "*.log" -mtime +7 -delete

# Clean old metrics (keep 30 days)
sqlite3 cache/performance_metrics.db "DELETE FROM performance_metrics WHERE timestamp < $(date -d '30 days ago' +%s);"

# Backup database
cp cache/performance_metrics.db backups/metrics_$(date +%Y%m%d).db

# Check system health
curl -s http://localhost:8000/api/health | jq '.status'
EOF

chmod +x daily_maintenance.sh

# Add to crontab for daily execution
echo "0 2 * * * /path/to/daily_maintenance.sh" | crontab -
```

This deployment checklist ensures a smooth and successful deployment of the search quality improvements with proper validation, monitoring, and rollback procedures.