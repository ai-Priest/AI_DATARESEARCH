# Quality Metrics Interpretation Guide

## Overview

This guide provides comprehensive information on interpreting quality metrics and alerts in the AI-Powered Dataset Research Assistant. Understanding these metrics is crucial for maintaining optimal system performance and making informed operational decisions.

## Core Quality Metrics

### 1. NDCG@3 (Normalized Discounted Cumulative Gain at Rank 3)

**Definition:** Measures the quality of ranking by considering both relevance and position of recommendations in the top 3 results.

**Formula:**
```
NDCG@3 = DCG@3 / IDCG@3

DCG@3 = Σ(i=1 to 3) (2^relevance_i - 1) / log2(i + 1)
IDCG@3 = DCG@3 with perfect ranking
```

**Interpretation:**

| NDCG@3 Range | Quality Level | Interpretation | Action Required |
|--------------|---------------|----------------|-----------------|
| 0.90 - 1.00 | Excellent | Perfect or near-perfect ranking | Monitor and maintain |
| 0.80 - 0.89 | Very Good | High-quality recommendations | Continue current approach |
| 0.70 - 0.79 | Good | Acceptable quality, room for improvement | Minor optimizations |
| 0.60 - 0.69 | Fair | Below optimal, needs attention | Review and improve |
| 0.50 - 0.59 | Poor | Significant issues | Immediate investigation |
| < 0.50 | Critical | System failure | Emergency response |

**Example Calculation:**
```python
# Example: Query "singapore housing data"
# Recommendations with relevance scores:
# 1. data_gov_sg (relevance: 0.95)
# 2. singstat (relevance: 0.85) 
# 3. world_bank (relevance: 0.60)

DCG@3 = (2^0.95 - 1)/log2(2) + (2^0.85 - 1)/log2(3) + (2^0.60 - 1)/log2(4)
      = 1.93/1.0 + 1.80/1.58 + 0.52/2.0
      = 1.93 + 1.14 + 0.26 = 3.33

# Perfect ranking would be: 0.95, 0.85, 0.60 (same order)
IDCG@3 = 3.33

NDCG@3 = 3.33 / 3.33 = 1.00 (Perfect!)
```

**Monitoring Commands:**
```bash
# Get current NDCG@3
curl -s http://localhost:8000/api/v2/quality/metrics | jq '.current_metrics.overall_ndcg_at_3'

# Get NDCG@3 trend
curl -s http://localhost:8000/api/v2/quality/metrics | jq '.trend_analysis.ndcg_trend_7d'

# Test specific query NDCG@3
curl -X POST http://localhost:8000/api/v2/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}' | jq '.quality_summary.overall_ndcg_at_3'
```

### 2. Relevance Accuracy

**Definition:** Measures how well the system's relevance scores align with expert-curated training mappings.

**Calculation:**
```
Relevance Accuracy = (Correct Predictions / Total Predictions) × 100%

Where "Correct" means |predicted_score - expected_score| ≤ 0.2
```

**Interpretation:**

| Accuracy Range | Quality Level | Interpretation | Action Required |
|----------------|---------------|----------------|-----------------|
| 90% - 100% | Excellent | Perfect alignment with training data | Maintain current model |
| 80% - 89% | Very Good | Strong alignment | Minor fine-tuning |
| 70% - 79% | Good | Acceptable alignment | Review training data |
| 60% - 69% | Fair | Moderate misalignment | Update training mappings |
| 50% - 59% | Poor | Significant misalignment | Retrain model |
| < 50% | Critical | Model not learning properly | Emergency retraining |

**Diagnostic Commands:**
```bash
# Check relevance accuracy
curl -s http://localhost:8000/api/v2/quality/metrics | jq '.current_metrics.relevance_accuracy'

# Validate against training mappings
python3 << 'EOF'
from src.ai.quality_monitoring_system import QualityMonitoringSystem
monitor = QualityMonitoringSystem()
compliance = monitor.validate_training_mapping_compliance()
print(f"Training mapping compliance: {compliance:.1%}")
EOF
```

### 3. Domain Routing Accuracy

**Definition:** Measures how accurately the system classifies queries into correct domains and routes to appropriate sources.

**Domains Tracked:**
- Psychology → Kaggle, Zenodo
- Climate → World Bank, Zenodo  
- Singapore → data.gov.sg, SingStat, LTA DataMall
- Economics → World Bank, SingStat
- Machine Learning → Kaggle
- Health → World Bank, Zenodo, data.gov.sg
- Education → World Bank, Zenodo, data.gov.sg

**Interpretation:**

| Accuracy Range | Quality Level | Interpretation | Action Required |
|----------------|---------------|----------------|-----------------|
| 95% - 100% | Excellent | Perfect domain routing | Monitor patterns |
| 85% - 94% | Very Good | Strong domain classification | Minor adjustments |
| 75% - 84% | Good | Acceptable routing | Review domain definitions |
| 65% - 74% | Fair | Moderate routing issues | Update domain keywords |
| 55% - 64% | Poor | Significant routing problems | Retrain domain classifier |
| < 55% | Critical | Domain routing failure | Emergency intervention |

**Testing Commands:**
```bash
# Test domain routing
python3 << 'EOF'
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

test_cases = [
    ("psychology research data", "psychology", ["kaggle", "zenodo"]),
    ("singapore housing prices", "singapore", ["data_gov_sg", "singstat"]),
    ("climate change indicators", "climate", ["world_bank", "zenodo"]),
    ("machine learning datasets", "machine_learning", ["kaggle"])
]

correct = 0
for query, expected_domain, expected_sources in test_cases:
    classification = router.classify_query(query)
    domain_correct = classification.domain == expected_domain
    source_overlap = len(set(classification.recommended_sources[:2]) & set(expected_sources)) > 0
    
    if domain_correct and source_overlap:
        correct += 1
    
    print(f"Query: {query}")
    print(f"  Expected: {expected_domain} → {expected_sources}")
    print(f"  Got: {classification.domain} → {classification.recommended_sources[:2]}")
    print(f"  Correct: {domain_correct and source_overlap}")
    print()

accuracy = correct / len(test_cases)
print(f"Domain Routing Accuracy: {accuracy:.1%}")
EOF
```

### 4. Singapore-First Accuracy

**Definition:** Measures how effectively the system applies Singapore-first strategy for local queries.

**Singapore-First Triggers:**
- Explicit mention: "singapore", "sg"
- Local terms: "hdb", "mrt", "lta", "singstat"
- Generic local context: housing, transport, education (without global qualifiers)

**Expected Behavior:**
- Singapore sources prioritized: data.gov.sg, SingStat, LTA DataMall
- Local relevance over global alternatives
- Appropriate geographic scope detection

**Interpretation:**

| Accuracy Range | Quality Level | Interpretation | Action Required |
|----------------|---------------|----------------|-----------------|
| 95% - 100% | Excellent | Perfect local prioritization | Monitor edge cases |
| 90% - 94% | Very Good | Strong Singapore-first strategy | Fine-tune keywords |
| 80% - 89% | Good | Generally correct prioritization | Review local terms |
| 70% - 79% | Fair | Inconsistent local prioritization | Update detection logic |
| 60% - 69% | Poor | Frequent Singapore-first failures | Retrain classifier |
| < 60% | Critical | Singapore-first not working | Emergency fix |

**Testing Commands:**
```bash
# Test Singapore-first strategy
python3 << 'EOF'
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

singapore_queries = [
    "singapore housing data",
    "hdb resale prices", 
    "mrt ridership statistics",
    "local population data",
    "sg education statistics"
]

singapore_sources = ['data_gov_sg', 'singstat', 'lta_datamall']
correct = 0

print("Singapore-First Strategy Test:")
for query in singapore_queries:
    classification = router.classify_query(query)
    singapore_first_applied = classification.singapore_first_applicable
    top_source = classification.recommended_sources[0] if classification.recommended_sources else ""
    singapore_source_first = any(sg_source in top_source.lower() for sg_source in singapore_sources)
    
    is_correct = singapore_first_applied and singapore_source_first
    if is_correct:
        correct += 1
    
    print(f"  Query: {query}")
    print(f"    Singapore-first: {singapore_first_applied}")
    print(f"    Top source: {top_source}")
    print(f"    Correct: {is_correct}")

accuracy = correct / len(singapore_queries)
print(f"\nSingapore-First Accuracy: {accuracy:.1%}")
EOF
```

### 5. Cache Hit Rate

**Definition:** Percentage of queries served from cache vs. requiring fresh computation.

**Formula:**
```
Cache Hit Rate = (Cache Hits / Total Requests) × 100%
```

**Interpretation:**

| Hit Rate Range | Performance Level | Interpretation | Action Required |
|----------------|-------------------|----------------|-----------------|
| 90% - 100% | Excellent | Optimal caching | Monitor cache quality |
| 80% - 89% | Very Good | Good caching efficiency | Minor optimizations |
| 70% - 79% | Good | Acceptable caching | Review cache strategy |
| 60% - 69% | Fair | Suboptimal caching | Improve cache warming |
| 50% - 59% | Poor | Poor caching efficiency | Redesign cache strategy |
| < 50% | Critical | Cache not effective | Emergency cache review |

**Quality-Aware Cache Metrics:**
```bash
# Get cache statistics
curl -s http://localhost:8000/api/v2/cache/stats | jq '{
  hit_rate: .cache_statistics.hit_rate,
  avg_quality: .cache_statistics.avg_quality_score,
  quality_distribution: .cache_statistics.quality_distribution
}'

# Check cache quality breakdown
python3 << 'EOF'
from src.ai.quality_aware_cache import QualityAwareCacheManager
cache_manager = QualityAwareCacheManager()
stats = cache_manager.get_quality_cache_statistics()

print(f"Cache Hit Rate: {stats['hit_rate']:.1%}")
print(f"Average Quality: {stats['avg_quality_score']:.3f}")
print(f"High Quality Entries: {stats['quality_distribution']['excellent'] + stats['quality_distribution']['good']}")
print(f"Low Quality Entries: {stats['quality_distribution']['poor']}")

if stats['quality_distribution']['poor'] > stats['quality_distribution']['total'] * 0.2:
    print("⚠️  Too many low-quality cached entries - cleanup recommended")
EOF
```

## Alert Thresholds and Responses

### Critical Alerts (P0)

**Triggers:**
- NDCG@3 < 0.50
- System availability < 95%
- Neural model inference failures > 10%

**Response Time:** 5 minutes  
**Actions:**
1. Immediate escalation to on-call engineer
2. Activate emergency fallback to rule-based routing
3. Clear all cache to force fresh results
4. Restore from last known good model backup

```bash
# Emergency response script
#!/bin/bash
echo "🚨 CRITICAL ALERT RESPONSE"

# 1. Switch to emergency mode
cat > config/emergency_config.yml << EOF
emergency_mode: true
use_neural_model: false
fallback_to_rules: true
quality_threshold: 0.4
EOF

# 2. Restart services
sudo systemctl restart dataset-research-assistant

# 3. Clear cache
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"quality_threshold": 0.0, "force_refresh": true}'

# 4. Restore backup model
cp models/dl/quality_first/backup_model.pt models/dl/quality_first/best_quality_model.pt

echo "✅ Emergency response completed"
```

### High Priority Alerts (P1)

**Triggers:**
- NDCG@3 < 0.65
- Relevance accuracy < 0.70
- Domain routing accuracy < 0.75
- Singapore-first accuracy < 0.80

**Response Time:** 15 minutes  
**Actions:**
1. Investigate root cause
2. Apply automatic remediation
3. Monitor for improvement
4. Escalate if no improvement in 1 hour

```bash
# High priority alert response
#!/bin/bash
echo "⚠️  HIGH PRIORITY ALERT RESPONSE"

# 1. Get current metrics
METRICS=$(curl -s http://localhost:8000/api/v2/quality/metrics)
NDCG=$(echo $METRICS | jq -r '.current_metrics.overall_ndcg_at_3')
RELEVANCE=$(echo $METRICS | jq -r '.current_metrics.relevance_accuracy')

echo "Current NDCG@3: $NDCG"
echo "Current Relevance Accuracy: $RELEVANCE"

# 2. Clear low-quality cache
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"quality_threshold": 0.7}'

# 3. Warm cache with high-quality queries
python3 << 'EOF'
import requests
high_quality_queries = [
    "singapore housing data",
    "psychology research datasets",
    "climate change indicators"
]

for query in high_quality_queries:
    requests.post('http://localhost:8000/api/v2/search', 
                 json={"query": query, "quality_threshold": 0.8})
EOF

# 4. Verify improvement
sleep 30
NEW_METRICS=$(curl -s http://localhost:8000/api/v2/quality/metrics)
NEW_NDCG=$(echo $NEW_METRICS | jq -r '.current_metrics.overall_ndcg_at_3')

echo "New NDCG@3: $NEW_NDCG"

if (( $(echo "$NEW_NDCG > 0.65" | bc -l) )); then
    echo "✅ Alert resolved"
else
    echo "❌ Alert persists - escalating"
fi
```

### Medium Priority Alerts (P2)

**Triggers:**
- NDCG@3 < 0.75
- Cache hit rate < 0.70
- Response time > 2 seconds average
- Training mapping compliance < 0.80

**Response Time:** 1 hour  
**Actions:**
1. Log and monitor
2. Schedule optimization during maintenance window
3. Update training mappings if needed

### Low Priority Alerts (P3)

**Triggers:**
- NDCG@3 < 0.80
- Minor performance degradation
- Non-critical feature issues

**Response Time:** 24 hours  
**Actions:**
1. Log for trend analysis
2. Include in weekly review
3. Plan improvements in next sprint

## Trend Analysis

### Weekly Trend Interpretation

```bash
# Generate weekly trend report
python3 << 'EOF'
from src.ai.quality_monitoring_system import QualityMonitoringSystem
import json
from datetime import datetime, timedelta

monitor = QualityMonitoringSystem()

# Get 7-day trends
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

trends = monitor.get_quality_trends(start_date, end_date)

print("Weekly Quality Trends:")
print("=" * 30)

for metric, data in trends.items():
    current = data['current']
    previous = data['previous']
    change = data['change']
    trend = data['trend']  # 'improving', 'stable', 'declining'
    
    trend_emoji = {
        'improving': '📈',
        'stable': '➡️',
        'declining': '📉'
    }
    
    print(f"{trend_emoji.get(trend, '❓')} {metric.replace('_', ' ').title()}")
    print(f"  Current: {current:.3f}")
    print(f"  Previous: {previous:.3f}")
    print(f"  Change: {change:+.3f}")
    print(f"  Trend: {trend}")
    print()

# Recommendations based on trends
recommendations = []

if trends.get('ndcg_at_3', {}).get('trend') == 'declining':
    recommendations.append("NDCG@3 declining - review training mappings")

if trends.get('cache_hit_rate', {}).get('trend') == 'declining':
    recommendations.append("Cache performance declining - optimize cache strategy")

if trends.get('relevance_accuracy', {}).get('trend') == 'declining':
    recommendations.append("Relevance accuracy declining - retrain model")

if recommendations:
    print("Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
else:
    print("✅ All trends stable or improving")
EOF
```

### Monthly Performance Review

```bash
# Generate monthly performance review
python3 << 'EOF'
from src.ai.quality_monitoring_system import QualityMonitoringSystem
import json
from datetime import datetime, timedelta

monitor = QualityMonitoringSystem()

# Get 30-day performance data
performance_data = monitor.get_monthly_performance_summary()

print("Monthly Performance Review:")
print("=" * 40)

# Overall performance
overall = performance_data['overall']
print(f"Average NDCG@3: {overall['avg_ndcg']:.3f}")
print(f"Average Relevance Accuracy: {overall['avg_relevance']:.3f}")
print(f"Total Queries Processed: {overall['total_queries']:,}")
print(f"Quality Threshold Compliance: {overall['compliance_rate']:.1%}")

# Domain performance
print("\nDomain Performance:")
for domain, perf in performance_data['domains'].items():
    print(f"  {domain.title()}:")
    print(f"    Queries: {perf['query_count']:,}")
    print(f"    Avg NDCG@3: {perf['avg_ndcg']:.3f}")
    print(f"    Routing Accuracy: {perf['routing_accuracy']:.1%}")

# Quality improvements
improvements = performance_data.get('improvements', [])
if improvements:
    print("\nQuality Improvements:")
    for improvement in improvements:
        print(f"  • {improvement}")

# Issues identified
issues = performance_data.get('issues', [])
if issues:
    print("\nIssues Identified:")
    for issue in issues:
        print(f"  ⚠️  {issue}")

# Recommendations
recommendations = performance_data.get('recommendations', [])
if recommendations:
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  📋 {rec}")
EOF
```

## Troubleshooting Common Metric Issues

### NDCG@3 Suddenly Drops

**Possible Causes:**
1. Training mappings file corrupted or missing
2. Neural model weights corrupted
3. New query patterns not covered in training data
4. Cache serving stale low-quality results

**Diagnostic Steps:**
```bash
# 1. Check training mappings
ls -la training_mappings.md
wc -l training_mappings.md

# 2. Test neural model
python3 -c "
from src.dl.quality_first_neural_model import QualityAwareRankingModel
import torch
try:
    model = QualityAwareRankingModel({'embedding_dim': 256, 'hidden_dim': 128, 'num_domains': 8, 'num_sources': 10, 'vocab_size': 10000})
    checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    relevance = model.predict_relevance('test query', 'kaggle')
    print(f'Model test: {relevance:.3f}')
except Exception as e:
    print(f'Model error: {e}')
"

# 3. Clear cache and retest
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"quality_threshold": 0.0, "force_refresh": true}'
```

### Singapore-First Strategy Not Working

**Possible Causes:**
1. Singapore keyword detection logic broken
2. Source prioritization not working
3. Training data insufficient for Singapore queries

**Diagnostic Steps:**
```bash
# Test Singapore detection
python3 -c "
from src.ai.enhanced_query_router import EnhancedQueryRouter
router = EnhancedQueryRouter()

test_queries = ['singapore housing', 'hdb prices', 'local transport']
for query in test_queries:
    classification = router.classify_query(query)
    print(f'{query}: Singapore-first={classification.singapore_first_applicable}, Sources={classification.recommended_sources[:3]}')
"
```

### Cache Hit Rate Declining

**Possible Causes:**
1. Query patterns changing
2. Cache TTL too short
3. Quality threshold too high
4. Cache storage issues

**Diagnostic Steps:**
```bash
# Analyze cache performance
python3 -c "
from src.ai.quality_aware_cache import QualityAwareCacheManager
cache_manager = QualityAwareCacheManager()
stats = cache_manager.get_quality_cache_statistics()
print(f'Hit rate: {stats[\"hit_rate\"]:.1%}')
print(f'Total entries: {stats[\"total_entries\"]}')
print(f'Quality distribution: {stats[\"quality_distribution\"]}')
"
```

This comprehensive guide enables operators to effectively interpret quality metrics and respond appropriately to maintain optimal system performance.