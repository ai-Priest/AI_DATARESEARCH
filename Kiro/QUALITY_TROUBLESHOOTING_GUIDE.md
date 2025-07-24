# Quality Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for quality-related issues in the AI-Powered Dataset Research Assistant. It covers common problems, diagnostic procedures, and step-by-step solutions for maintaining optimal recommendation quality.

## Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```bash
# 1. Check system health
curl -X GET http://localhost:8000/api/v2/health

# 2. Verify quality metrics
curl -X GET http://localhost:8000/api/v2/quality/metrics

# 3. Test basic search functionality
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "quality_threshold": 0.7}'

# 4. Check cache statistics
curl -X GET http://localhost:8000/api/v2/cache/stats
```

## Common Quality Issues

### 1. Low NDCG@3 Scores

**Symptoms:**
- NDCG@3 below 0.65 threshold
- Poor recommendation relevance
- User complaints about irrelevant results
- Quality metrics showing degradation

**Diagnostic Steps:**

```bash
# Check current quality metrics
python -c "
from src.ai.quality_monitoring_system import QualityMonitoringSystem
monitor = QualityMonitoringSystem()
metrics = monitor.get_current_quality_metrics()
print(f'Current NDCG@3: {metrics.ndcg_at_3}')
print(f'Relevance Accuracy: {metrics.relevance_accuracy}')
print(f'Training Mapping Compliance: {metrics.training_mapping_compliance}')
"

# Validate against training mappings
python -c "
from src.ai.quality_aware_cache import QualityAwareCacheManager
cache_manager = QualityAwareCacheManager()
stats = cache_manager.get_quality_cache_statistics()
print(f'Average Quality Score: {stats[\"avg_quality_score\"]}')
print(f'Training Mappings Loaded: {stats[\"training_mappings_loaded\"]}')
"

# Test specific problematic queries
python -c "
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

test_queries = [
    'psychology research data',
    'singapore housing prices', 
    'climate change datasets'
]

for query in test_queries:
    classification = router.classify_query(query)
    print(f'Query: {query}')
    print(f'  Domain: {classification.domain}')
    print(f'  Confidence: {classification.confidence}')
    print(f'  Singapore-first: {classification.singapore_first_applicable}')
    print(f'  Sources: {classification.recommended_sources[:3]}')
    print()
"
```

**Solutions:**

#### Solution 1: Update Training Mappings

```bash
# 1. Check training mappings file
ls -la training_mappings.md

# 2. Validate current mappings
python -c "
import re
with open('training_mappings.md', 'r') as f:
    content = f.read()

# Count mappings
mappings = re.findall(r'→.*\([\d.]+\)', content)
print(f'Total mappings found: {len(mappings)}')

# Check for low-quality mappings
low_quality = re.findall(r'→.*\(0\.[0-4]\d*\)', content)
print(f'Low quality mappings (< 0.5): {len(low_quality)}')

# Check domain coverage
domains = ['psychology', 'climate', 'singapore', 'economics']
for domain in domains:
    domain_count = content.lower().count(domain)
    print(f'{domain} mentions: {domain_count}')
"

# 3. Add missing high-quality mappings
# Edit training_mappings.md to add more expert-curated examples
```

#### Solution 2: Retrain Neural Model

```bash
# 1. Backup current model
cp models/dl/quality_first/best_quality_model.pt models/dl/quality_first/backup_model.pt

# 2. Retrain with updated mappings
python src/dl/quality_first_trainer.py \
  --training_data data/processed/enhanced_training_mappings.json \
  --quality_threshold 0.7 \
  --curriculum_learning true \
  --max_epochs 100

# 3. Validate new model
python -c "
from src.dl.quality_first_neural_model import QualityAwareRankingModel
import torch

# Load new model
model = QualityAwareRankingModel({
    'embedding_dim': 256,
    'hidden_dim': 128,
    'num_domains': 8,
    'num_sources': 10,
    'vocab_size': 10000
})

checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Test predictions
test_cases = [
    ('psychology research data', 'kaggle'),
    ('singapore housing prices', 'data_gov_sg'),
    ('climate change data', 'world_bank')
]

for query, source in test_cases:
    relevance = model.predict_relevance(query, source)
    print(f'{query} → {source}: {relevance:.3f}')
"
```

#### Solution 3: Adjust Quality Thresholds

```bash
# Temporarily lower threshold while improving
python -c "
from src.ai.quality_aware_cache import QualityAwareCacheManager
cache_manager = QualityAwareCacheManager(quality_threshold=0.6)  # Lower from 0.7

# Test with lower threshold
stats = cache_manager.get_quality_cache_statistics()
print(f'Cache entries with threshold 0.6: {stats[\"total_entries\"]}')
"

# Update configuration
cat > config/quality_config.yml << EOF
quality_settings:
  ndcg_threshold: 0.6  # Temporarily lowered
  relevance_threshold: 0.6
  cache_quality_threshold: 0.6
  alert_threshold: 0.55
  
retraining_triggers:
  ndcg_below: 0.55
  relevance_below: 0.55
  compliance_below: 0.7
EOF
```

### 2. Singapore-First Strategy Not Working

**Symptoms:**
- Local queries returning global sources first
- Singapore government sources ranked too low
- Geographic routing accuracy below 85%

**Diagnostic Steps:**

```python
# Test Singapore-first detection
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

singapore_test_queries = [
    "singapore housing prices",
    "local transport data", 
    "sg population statistics",
    "hdb resale prices",
    "mrt ridership data",
    "singapore education statistics"
]

print("Singapore-First Detection Test:")
print("=" * 50)

for query in singapore_test_queries:
    classification = router.classify_query(query)
    print(f"Query: {query}")
    print(f"  Singapore-first: {classification.singapore_first_applicable}")
    print(f"  Domain: {classification.domain}")
    print(f"  Top sources: {classification.recommended_sources[:3]}")
    print(f"  Explanation: {classification.explanation}")
    print()
```

**Solutions:**

#### Solution 1: Update Singapore Keywords

```python
# Check current Singapore detection logic
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

# Current Singapore-first terms
current_terms = router.singapore_first_terms
print("Current Singapore-first terms:")
for term in sorted(current_terms):
    print(f"  - {term}")

# Add missing terms
additional_terms = {
    'hdb', 'bto', 'coe', 'erp', 'gst', 'cpf', 'medisave',
    'singpass', 'nric', 'pr', 'citizen', 'resident',
    'void deck', 'hawker', 'kopitiam', 'mrt', 'lrt', 'bus',
    'taxi', 'grab', 'gojek', 'foodpanda', 'deliveroo'
}

# Update the router configuration
updated_terms = current_terms.union(additional_terms)
print(f"\nAdding {len(additional_terms)} new Singapore-specific terms")
```

#### Solution 2: Fix Source Prioritization

```python
# Check source priority logic
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

# Test source prioritization
test_query = "singapore housing data"
classification = router.classify_query(test_query)

print(f"Query: {test_query}")
print(f"Singapore-first applicable: {classification.singapore_first_applicable}")
print(f"Recommended sources: {classification.recommended_sources}")

# Check if Singapore sources are prioritized
singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
top_3_sources = classification.recommended_sources[:3]

singapore_in_top_3 = any(source in singapore_sources for source in top_3_sources)
print(f"Singapore source in top 3: {singapore_in_top_3}")

if not singapore_in_top_3:
    print("❌ Singapore sources not properly prioritized!")
    print("Fix needed in _get_recommended_sources method")
```

#### Solution 3: Retrain Singapore Classifier

```bash
# Create Singapore-specific training data
python -c "
import json

singapore_training_examples = [
    {'query': 'singapore housing prices', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'hdb resale prices', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'mrt ridership data', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'sg population statistics', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'local transport data', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'singapore education statistics', 'singapore_first': True, 'domain': 'singapore'},
    {'query': 'global climate data', 'singapore_first': False, 'domain': 'climate'},
    {'query': 'international economics', 'singapore_first': False, 'domain': 'economics'},
    {'query': 'worldwide psychology research', 'singapore_first': False, 'domain': 'psychology'}
]

with open('data/processed/singapore_training_examples.json', 'w') as f:
    json.dump(singapore_training_examples, f, indent=2)

print('Created Singapore-specific training examples')
"

# Retrain with focus on Singapore classification
python src/dl/quality_first_trainer.py \
  --focus_singapore_classification true \
  --singapore_weight 0.3 \
  --additional_data data/processed/singapore_training_examples.json
```

### 3. Domain Routing Errors

**Symptoms:**
- Psychology queries not routing to Kaggle/Zenodo
- Climate queries not prioritizing World Bank
- Domain classification accuracy below 80%

**Diagnostic Steps:**

```python
# Test domain classification accuracy
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

domain_test_cases = [
    # Psychology domain
    ("psychology research data", "psychology", ["kaggle", "zenodo"]),
    ("mental health datasets", "psychology", ["kaggle", "zenodo"]),
    ("behavioral analysis data", "psychology", ["kaggle", "zenodo"]),
    
    # Climate domain  
    ("climate change datasets", "climate", ["world_bank", "zenodo"]),
    ("weather data", "climate", ["world_bank"]),
    ("environmental indicators", "climate", ["world_bank", "zenodo"]),
    
    # Machine learning domain
    ("machine learning datasets", "machine_learning", ["kaggle"]),
    ("ml competitions", "machine_learning", ["kaggle"]),
    ("deep learning data", "machine_learning", ["kaggle", "zenodo"]),
    
    # Economics domain
    ("economic indicators", "economics", ["world_bank"]),
    ("gdp data", "economics", ["world_bank", "singstat"]),
    ("trade statistics", "economics", ["world_bank"])
]

print("Domain Classification Test:")
print("=" * 60)

correct_predictions = 0
total_predictions = len(domain_test_cases)

for query, expected_domain, expected_sources in domain_test_cases:
    classification = router.classify_query(query)
    
    domain_correct = classification.domain == expected_domain
    source_overlap = len(set(classification.recommended_sources[:2]) & set(expected_sources)) > 0
    
    if domain_correct:
        correct_predictions += 1
    
    status = "✅" if domain_correct else "❌"
    print(f"{status} Query: {query}")
    print(f"    Expected domain: {expected_domain}")
    print(f"    Actual domain: {classification.domain}")
    print(f"    Expected sources: {expected_sources}")
    print(f"    Actual sources: {classification.recommended_sources[:3]}")
    print(f"    Source overlap: {source_overlap}")
    print()

accuracy = correct_predictions / total_predictions
print(f"Domain Classification Accuracy: {accuracy:.2%}")

if accuracy < 0.8:
    print("❌ Domain classification accuracy below threshold!")
```

**Solutions:**

#### Solution 1: Expand Domain Keywords

```python
# Update domain definitions with more comprehensive keywords
domain_updates = {
    'psychology': {
        'additional_keywords': [
            'behavioral', 'cognitive', 'neuroscience', 'psychiatry',
            'therapy', 'counseling', 'mental health', 'wellbeing',
            'personality', 'social psychology', 'developmental',
            'clinical psychology', 'psychometric', 'survey data'
        ]
    },
    'climate': {
        'additional_keywords': [
            'greenhouse gas', 'carbon emissions', 'temperature',
            'precipitation', 'weather patterns', 'global warming',
            'sea level', 'ice caps', 'deforestation', 'renewable energy',
            'sustainability', 'environmental impact', 'carbon footprint'
        ]
    },
    'machine_learning': {
        'additional_keywords': [
            'neural networks', 'deep learning', 'computer vision',
            'natural language processing', 'nlp', 'classification',
            'regression', 'clustering', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'ai',
            'artificial intelligence', 'data science', 'predictive modeling'
        ]
    }
}

# Apply updates to router configuration
print("Updating domain keyword definitions...")
for domain, updates in domain_updates.items():
    print(f"  {domain}: +{len(updates['additional_keywords'])} keywords")
```

#### Solution 2: Improve Training Data Coverage

```bash
# Generate domain-specific training examples
python -c "
import json

domain_training_examples = []

# Psychology examples
psychology_examples = [
    {'query': 'psychology research datasets', 'domain': 'psychology', 'positive_source': 'kaggle', 'relevance_score': 0.9},
    {'query': 'mental health survey data', 'domain': 'psychology', 'positive_source': 'zenodo', 'relevance_score': 0.85},
    {'query': 'behavioral analysis datasets', 'domain': 'psychology', 'positive_source': 'kaggle', 'relevance_score': 0.88},
    {'query': 'cognitive psychology data', 'domain': 'psychology', 'positive_source': 'zenodo', 'relevance_score': 0.82},
]

# Climate examples
climate_examples = [
    {'query': 'climate change indicators', 'domain': 'climate', 'positive_source': 'world_bank', 'relevance_score': 0.95},
    {'query': 'global temperature data', 'domain': 'climate', 'positive_source': 'world_bank', 'relevance_score': 0.92},
    {'query': 'environmental datasets', 'domain': 'climate', 'positive_source': 'zenodo', 'relevance_score': 0.85},
    {'query': 'carbon emissions data', 'domain': 'climate', 'positive_source': 'world_bank', 'relevance_score': 0.93},
]

# Machine learning examples
ml_examples = [
    {'query': 'machine learning datasets', 'domain': 'machine_learning', 'positive_source': 'kaggle', 'relevance_score': 0.95},
    {'query': 'deep learning competitions', 'domain': 'machine_learning', 'positive_source': 'kaggle', 'relevance_score': 0.92},
    {'query': 'computer vision datasets', 'domain': 'machine_learning', 'positive_source': 'kaggle', 'relevance_score': 0.90},
    {'query': 'nlp datasets', 'domain': 'machine_learning', 'positive_source': 'kaggle', 'relevance_score': 0.88},
]

domain_training_examples.extend(psychology_examples)
domain_training_examples.extend(climate_examples)
domain_training_examples.extend(ml_examples)

with open('data/processed/domain_specific_training.json', 'w') as f:
    json.dump(domain_training_examples, f, indent=2)

print(f'Created {len(domain_training_examples)} domain-specific training examples')
"

# Retrain with domain focus
python src/dl/quality_first_trainer.py \
  --focus_domain_classification true \
  --domain_weight 0.3 \
  --additional_data data/processed/domain_specific_training.json
```

### 4. Model Performance Degradation

**Symptoms:**
- Increasing inference time
- Memory usage growth
- Quality metrics declining over time
- Cache hit rate decreasing

**Diagnostic Steps:**

```bash
# Performance profiling
python -c "
import time
import psutil
import torch
from src.dl.quality_first_neural_model import QualityAwareRankingModel

# Memory usage before
memory_before = psutil.Process().memory_info().rss / 1024 / 1024

# Load model
model = QualityAwareRankingModel({
    'embedding_dim': 256,
    'hidden_dim': 128,
    'num_domains': 8,
    'num_sources': 10,
    'vocab_size': 10000
})

checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Memory usage after loading
memory_after = psutil.Process().memory_info().rss / 1024 / 1024

print(f'Model loading memory usage: {memory_after - memory_before:.1f} MB')

# Inference timing
test_queries = ['test query'] * 100
start_time = time.time()

for query in test_queries:
    relevance = model.predict_relevance(query, 'kaggle')

end_time = time.time()
avg_inference_time = (end_time - start_time) / len(test_queries) * 1000

print(f'Average inference time: {avg_inference_time:.1f} ms')
print(f'Model parameters: {model.count_parameters():,}')
"

# Cache analysis
python -c "
from src.ai.quality_aware_cache import QualityAwareCacheManager
cache_manager = QualityAwareCacheManager()

stats = cache_manager.get_quality_cache_statistics()
print('Cache Statistics:')
print(f'  Hit rate: {stats[\"hit_rate\"]:.3f}')
print(f'  Total entries: {stats[\"total_entries\"]}')
print(f'  Average quality: {stats[\"avg_quality_score\"]:.3f}')
print(f'  Quality distribution: {stats[\"quality_distribution\"]}')
"
```

**Solutions:**

#### Solution 1: Model Optimization

```bash
# Model quantization
python -c "
import torch
from src.dl.quality_first_neural_model import QualityAwareRankingModel

# Load model
model = QualityAwareRankingModel({
    'embedding_dim': 256,
    'hidden_dim': 128,
    'num_domains': 8,
    'num_sources': 10,
    'vocab_size': 10000
})

checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save({
    'model_state_dict': quantized_model.state_dict(),
    'config': checkpoint['config']
}, 'models/dl/quality_first/quantized_model.pt')

print('Model quantized and saved')
"

# Model pruning (if needed)
python -c "
import torch
import torch.nn.utils.prune as prune
from src.dl.quality_first_neural_model import QualityAwareRankingModel

# Load model
model = QualityAwareRankingModel({
    'embedding_dim': 256,
    'hidden_dim': 128,
    'num_domains': 8,
    'num_sources': 10,
    'vocab_size': 10000
})

checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Apply structured pruning to linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)

# Save pruned model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': checkpoint['config']
}, 'models/dl/quality_first/pruned_model.pt')

print('Model pruned and saved')
"
```

#### Solution 2: Cache Optimization

```bash
# Clear low-quality cache entries
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"quality_threshold": 0.7, "force_refresh": true}'

# Optimize cache warming
python -c "
from src.ai.intelligent_cache_warming import IntelligentCacheWarming
from src.ai.quality_aware_cache import QualityAwareCacheManager

cache_manager = QualityAwareCacheManager()
cache_warmer = IntelligentCacheWarming(cache_manager)

# Warm cache with popular high-quality queries
popular_queries = [
    'singapore housing data',
    'psychology research datasets', 
    'climate change indicators',
    'machine learning competitions',
    'economic indicators singapore'
]

cache_warmer.warm_cache_with_popular_queries(popular_queries)
print('Cache warmed with popular queries')
"
```

#### Solution 3: Memory Management

```bash
# Implement garbage collection
python -c "
import gc
import torch

# Clear PyTorch cache
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Force garbage collection
gc.collect()

print('Memory cleanup completed')
"

# Monitor memory usage
python -c "
import psutil
import time

def monitor_memory(duration=60):
    start_time = time.time()
    max_memory = 0
    
    while time.time() - start_time < duration:
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        max_memory = max(max_memory, current_memory)
        time.sleep(1)
    
    print(f'Maximum memory usage over {duration}s: {max_memory:.1f} MB')

monitor_memory(30)
"
```

## Quality Monitoring Automation

### Automated Quality Checks

Create automated quality monitoring scripts:

```bash
# Create quality monitoring script
cat > scripts/quality_monitor.sh << 'EOF'
#!/bin/bash

# Quality monitoring script
LOG_FILE="/var/log/quality_monitor.log"
ALERT_THRESHOLD=0.65

echo "$(date): Starting quality monitoring check" >> $LOG_FILE

# Get current quality metrics
QUALITY_RESPONSE=$(curl -s http://localhost:8000/api/v2/quality/metrics)
NDCG_SCORE=$(echo $QUALITY_RESPONSE | jq -r '.current_metrics.overall_ndcg_at_3')

echo "$(date): Current NDCG@3: $NDCG_SCORE" >> $LOG_FILE

# Check if quality is below threshold
if (( $(echo "$NDCG_SCORE < $ALERT_THRESHOLD" | bc -l) )); then
    echo "$(date): ALERT - Quality below threshold!" >> $LOG_FILE
    
    # Send alert (customize as needed)
    echo "Quality Alert: NDCG@3 is $NDCG_SCORE (below $ALERT_THRESHOLD)" | \
        mail -s "Dataset Research Assistant Quality Alert" admin@example.com
    
    # Trigger automatic remediation
    echo "$(date): Triggering automatic remediation" >> $LOG_FILE
    
    # Clear low-quality cache
    curl -X POST http://localhost:8000/api/v2/cache/invalidate \
        -H "Content-Type: application/json" \
        -d '{"quality_threshold": 0.7}'
    
    # Generate quality report
    curl -X POST http://localhost:8000/api/v2/quality/report \
        -H "Content-Type: application/json" \
        -d '{"time_range": "24h", "include_details": true}' \
        > /tmp/quality_report_$(date +%Y%m%d_%H%M%S).json
fi

echo "$(date): Quality monitoring check completed" >> $LOG_FILE
EOF

chmod +x scripts/quality_monitor.sh

# Set up cron job for regular monitoring
echo "*/15 * * * * /path/to/scripts/quality_monitor.sh" | crontab -
```

### Quality Dashboard Setup

```bash
# Create quality dashboard script
cat > scripts/quality_dashboard.py << 'EOF'
#!/usr/bin/env python3

import requests
import json
import time
from datetime import datetime

def get_quality_metrics():
    """Get current quality metrics"""
    try:
        response = requests.get('http://localhost:8000/api/v2/quality/metrics')
        return response.json()
    except Exception as e:
        print(f"Error getting quality metrics: {e}")
        return None

def display_quality_dashboard():
    """Display quality dashboard"""
    metrics = get_quality_metrics()
    if not metrics:
        return
    
    current = metrics['current_metrics']
    trend = metrics['trend_analysis']
    
    print("\n" + "="*60)
    print("QUALITY DASHBOARD")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("CURRENT METRICS:")
    print(f"  NDCG@3:                    {current['overall_ndcg_at_3']:.3f}")
    print(f"  Relevance Accuracy:        {current['relevance_accuracy']:.3f}")
    print(f"  Domain Routing Accuracy:   {current['domain_routing_accuracy']:.3f}")
    print(f"  Singapore-First Accuracy:  {current['singapore_first_accuracy']:.3f}")
    print(f"  User Satisfaction:         {current['user_satisfaction_score']:.3f}")
    print()
    
    print("7-DAY TRENDS:")
    print(f"  NDCG Trend:               {trend['ndcg_trend_7d']}")
    print(f"  Relevance Trend:          {trend['relevance_trend_7d']}")
    print(f"  Quality Status:           {trend['quality_improvement']}")
    print(f"  Alert Status:             {trend['alert_status']}")
    print()
    
    # Quality status indicators
    ndcg = current['overall_ndcg_at_3']
    if ndcg >= 0.8:
        status = "🟢 EXCELLENT"
    elif ndcg >= 0.7:
        status = "🟡 GOOD"
    elif ndcg >= 0.6:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"
    
    print(f"OVERALL STATUS: {status}")
    print("="*60)

if __name__ == "__main__":
    while True:
        display_quality_dashboard()
        time.sleep(30)  # Update every 30 seconds
EOF

chmod +x scripts/quality_dashboard.py
```

## Emergency Procedures

### Quality Emergency Response

If quality drops critically (NDCG@3 < 0.5):

```bash
# 1. Immediate fallback to rule-based routing
cat > config/emergency_config.yml << EOF
emergency_mode: true
use_neural_model: false
fallback_to_rules: true
quality_threshold: 0.4  # Temporarily lowered
EOF

# 2. Restart services with emergency config
sudo systemctl restart dataset-research-assistant

# 3. Investigate root cause
python -c "
from src.ai.quality_monitoring_system import QualityMonitoringSystem
monitor = QualityMonitoringSystem()

# Generate emergency report
report = monitor.generate_emergency_quality_report()
print('Emergency Quality Report:')
print(json.dumps(report, indent=2))
"

# 4. Restore from backup if needed
cp models/dl/quality_first/backup_model.pt models/dl/quality_first/best_quality_model.pt

# 5. Clear all cache to force fresh results
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"quality_threshold": 0.0, "force_refresh": true}'
```

### Recovery Checklist

After resolving quality issues:

- [ ] Verify NDCG@3 > 0.7
- [ ] Test Singapore-first strategy with local queries
- [ ] Validate domain routing for psychology, climate, ML queries
- [ ] Check cache hit rate > 80%
- [ ] Confirm training mapping compliance > 85%
- [ ] Run full integration tests
- [ ] Update monitoring thresholds if needed
- [ ] Document lessons learned

This troubleshooting guide provides comprehensive procedures for maintaining optimal recommendation quality in the dataset research assistant system.