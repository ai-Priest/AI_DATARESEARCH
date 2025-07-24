# Quality Monitoring and Maintenance Runbook

## Overview

This runbook provides step-by-step operational procedures for monitoring and maintaining recommendation quality in the AI-Powered Dataset Research Assistant. It covers daily operations, quality monitoring, incident response, and maintenance procedures.

## Daily Operations

### Morning Quality Check (9:00 AM)

**Frequency:** Daily  
**Duration:** 10 minutes  
**Responsibility:** Operations Team

```bash
#!/bin/bash
# Daily morning quality check script

echo "=== DAILY QUALITY CHECK - $(date) ==="

# 1. Check system health
echo "1. System Health Check:"
curl -s http://localhost:8000/api/v2/health | jq '.status, .components, .quality_metrics'

# 2. Get overnight quality metrics
echo -e "\n2. Quality Metrics:"
curl -s http://localhost:8000/api/v2/quality/metrics | jq '.current_metrics'

# 3. Check cache performance
echo -e "\n3. Cache Performance:"
curl -s http://localhost:8000/api/v2/cache/stats | jq '.cache_statistics.hit_rate, .cache_statistics.avg_quality_score'

# 4. Test key functionality
echo -e "\n4. Functionality Test:"
test_queries=("singapore housing data" "psychology research" "climate change")

for query in "${test_queries[@]}"; do
    echo "Testing: $query"
    response=$(curl -s -X POST http://localhost:8000/api/v2/search \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"max_results\": 3}")
    
    ndcg=$(echo $response | jq -r '.quality_summary.overall_ndcg_at_3 // "N/A"')
    singapore_first=$(echo $response | jq -r '.routing_summary.singapore_first_applied // false')
    
    echo "  NDCG@3: $ndcg, Singapore-first: $singapore_first"
done

echo -e "\n=== DAILY CHECK COMPLETE ==="
```

**Expected Results:**
- System status: "healthy"
- NDCG@3: > 0.70
- Cache hit rate: > 0.80
- All test queries return relevant results

**Escalation:** If any metric is below threshold, follow [Quality Degradation Response](#quality-degradation-response)

### Hourly Automated Monitoring

**Frequency:** Every hour  
**Automation:** Cron job  
**Alert Threshold:** NDCG@3 < 0.65

```bash
# Cron job: 0 * * * * /opt/dataset-research/scripts/hourly_quality_check.sh

#!/bin/bash
# Hourly quality monitoring

LOG_FILE="/var/log/quality_monitor.log"
ALERT_THRESHOLD=0.65
WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Get current metrics
METRICS=$(curl -s http://localhost:8000/api/v2/quality/metrics)
NDCG=$(echo $METRICS | jq -r '.current_metrics.overall_ndcg_at_3')
RELEVANCE=$(echo $METRICS | jq -r '.current_metrics.relevance_accuracy')

# Log metrics
echo "$(date): NDCG@3=$NDCG, Relevance=$RELEVANCE" >> $LOG_FILE

# Check thresholds
if (( $(echo "$NDCG < $ALERT_THRESHOLD" | bc -l) )); then
    # Send alert
    curl -X POST $WEBHOOK_URL \
        -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 Quality Alert: NDCG@3 is $NDCG (below $ALERT_THRESHOLD)\"}"
    
    # Log alert
    echo "$(date): ALERT - Quality degradation detected" >> $LOG_FILE
    
    # Trigger automatic remediation
    /opt/dataset-research/scripts/auto_remediation.sh
fi
```

## Quality Monitoring Procedures

### 1. Real-Time Quality Dashboard

**Access:** http://localhost:3000/quality-dashboard  
**Update Frequency:** Every 30 seconds  
**Key Metrics:**

- **NDCG@3**: Primary quality indicator (target: > 0.70)
- **Relevance Accuracy**: Agreement with training mappings (target: > 0.75)
- **Domain Routing Accuracy**: Correct domain classification (target: > 0.80)
- **Singapore-First Accuracy**: Local query routing (target: > 0.85)
- **Cache Hit Rate**: Performance indicator (target: > 0.80)

**Dashboard Setup:**
```bash
# Install dashboard dependencies
npm install -g pm2
cd /opt/dataset-research/dashboard
npm install

# Start dashboard
pm2 start dashboard.js --name quality-dashboard
pm2 startup
pm2 save
```

### 2. Weekly Quality Report Generation

**Schedule:** Every Monday 8:00 AM  
**Recipients:** Product Team, Engineering Team  
**Automation:** Cron job

```bash
# Cron job: 0 8 * * 1 /opt/dataset-research/scripts/weekly_quality_report.sh

#!/bin/bash
# Weekly quality report generation

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/opt/dataset-research/reports/quality_report_$REPORT_DATE.json"

# Generate comprehensive quality report
curl -X POST http://localhost:8000/api/v2/quality/report \
    -H "Content-Type: application/json" \
    -d '{"time_range": "7d", "include_details": true, "format": "json"}' \
    > $REPORT_FILE

# Generate summary email
python3 << EOF
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load report
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)

# Create email
msg = MIMEMultipart()
msg['From'] = 'quality-monitor@dataset-research.ai'
msg['To'] = 'team@dataset-research.ai'
msg['Subject'] = f'Weekly Quality Report - {report["report_summary"]["period"]}'

# Email body
body = f"""
Weekly Quality Report Summary
============================

Period: {report["report_summary"]["period"]}
Total Queries: {report["report_summary"]["total_queries"]:,}
Average Quality Score: {report["report_summary"]["avg_quality_score"]:.3f}
Quality Threshold Compliance: {report["report_summary"]["quality_threshold_compliance"]:.1%}

Domain Performance:
"""

for domain, perf in report["domain_performance"].items():
    body += f"""
{domain.title()}:
  - Queries: {perf["queries"]:,}
  - Avg NDCG@3: {perf["avg_ndcg"]:.3f}
  - Top Sources: {", ".join(perf["top_sources"])}
"""

body += f"""

Quality Improvements:
"""
for improvement in report["quality_improvements"]:
    body += f"- {improvement['improvement']}: {improvement['impact']}\n"

body += f"""

Full report available at: /opt/dataset-research/reports/quality_report_{REPORT_DATE}.json
"""

msg.attach(MIMEText(body, 'plain'))

# Send email
server = smtplib.SMTP('localhost')
server.send_message(msg)
server.quit()

print(f"Weekly quality report sent for {REPORT_DATE}")
EOF
```

### 3. Training Mapping Compliance Monitoring

**Purpose:** Ensure recommendations align with expert-curated mappings  
**Frequency:** Continuous monitoring with daily reports

```bash
#!/bin/bash
# Training mapping compliance check

echo "=== TRAINING MAPPING COMPLIANCE CHECK ==="

# Test key training mappings
python3 << 'EOF'
import requests
import json

# Key training mappings to test
test_mappings = [
    {
        "query": "psychology research data",
        "expected_sources": ["kaggle", "zenodo"],
        "expected_relevance": 0.85
    },
    {
        "query": "singapore housing prices", 
        "expected_sources": ["data_gov_sg", "singstat"],
        "expected_relevance": 0.90
    },
    {
        "query": "climate change indicators",
        "expected_sources": ["world_bank", "zenodo"],
        "expected_relevance": 0.88
    },
    {
        "query": "machine learning datasets",
        "expected_sources": ["kaggle"],
        "expected_relevance": 0.92
    }
]

compliance_results = []

for mapping in test_mappings:
    # Test search
    response = requests.post('http://localhost:8000/api/v2/search', json={
        "query": mapping["query"],
        "max_results": 5
    })
    
    if response.status_code == 200:
        data = response.json()
        recommendations = data.get('recommendations', [])
        
        # Check if expected sources are in top 3
        top_sources = [rec['source'] for rec in recommendations[:3]]
        expected_found = any(source in top_sources for source in mapping['expected_sources'])
        
        # Check quality
        avg_relevance = sum(rec['quality_metrics']['relevance_score'] for rec in recommendations[:3]) / 3
        quality_met = avg_relevance >= mapping['expected_relevance'] - 0.1  # 0.1 tolerance
        
        compliance_results.append({
            "query": mapping["query"],
            "expected_sources_found": expected_found,
            "quality_threshold_met": quality_met,
            "top_sources": top_sources,
            "avg_relevance": avg_relevance,
            "expected_relevance": mapping["expected_relevance"]
        })
        
        status = "✅" if expected_found and quality_met else "❌"
        print(f"{status} {mapping['query']}")
        print(f"    Expected: {mapping['expected_sources']}")
        print(f"    Got: {top_sources}")
        print(f"    Relevance: {avg_relevance:.3f} (expected: {mapping['expected_relevance']:.3f})")
        print()

# Calculate overall compliance
total_tests = len(compliance_results)
passed_tests = sum(1 for r in compliance_results if r['expected_sources_found'] and r['quality_threshold_met'])
compliance_rate = passed_tests / total_tests

print(f"Overall Compliance Rate: {compliance_rate:.1%} ({passed_tests}/{total_tests})")

if compliance_rate < 0.8:
    print("⚠️  Compliance below 80% - Investigation required")
    
    # Log non-compliant cases
    for result in compliance_results:
        if not (result['expected_sources_found'] and result['quality_threshold_met']):
            print(f"Non-compliant: {result['query']}")
            print(f"  Sources issue: {not result['expected_sources_found']}")
            print(f"  Quality issue: {not result['quality_threshold_met']}")
EOF
```

## Incident Response Procedures

### Quality Degradation Response

**Trigger:** NDCG@3 < 0.65 or Relevance Accuracy < 0.70  
**Response Time:** 15 minutes  
**Escalation:** After 1 hour if unresolved

#### Step 1: Immediate Assessment (0-5 minutes)

```bash
#!/bin/bash
# Immediate quality degradation assessment

echo "=== QUALITY DEGRADATION RESPONSE ==="
echo "Incident started at: $(date)"

# 1. Check system health
echo "1. System Health:"
HEALTH=$(curl -s http://localhost:8000/api/v2/health)
echo $HEALTH | jq '.status, .components'

# 2. Get detailed quality metrics
echo -e "\n2. Quality Metrics:"
METRICS=$(curl -s http://localhost:8000/api/v2/quality/metrics)
echo $METRICS | jq '.current_metrics'

# 3. Check recent errors
echo -e "\n3. Recent Errors:"
tail -50 /var/log/dataset-research/error.log | grep -i error

# 4. Check resource usage
echo -e "\n4. Resource Usage:"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"

# 5. Check neural model status
echo -e "\n5. Neural Model Status:"
python3 -c "
try:
    from src.dl.quality_first_neural_model import QualityAwareRankingModel
    import torch
    
    model = QualityAwareRankingModel({
        'embedding_dim': 256, 'hidden_dim': 128,
        'num_domains': 8, 'num_sources': 10, 'vocab_size': 10000
    })
    
    checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test prediction
    relevance = model.predict_relevance('test query', 'kaggle')
    print(f'Model test successful: {relevance:.3f}')
    
except Exception as e:
    print(f'Model error: {e}')
"
```

#### Step 2: Automatic Remediation (5-10 minutes)

```bash
#!/bin/bash
# Automatic remediation steps

echo "=== AUTOMATIC REMEDIATION ==="

# 1. Clear low-quality cache
echo "1. Clearing low-quality cache..."
curl -X POST http://localhost:8000/api/v2/cache/invalidate \
    -H "Content-Type: application/json" \
    -d '{"quality_threshold": 0.7, "force_refresh": true}'

# 2. Restart neural model service
echo "2. Restarting neural model service..."
sudo systemctl restart dataset-research-neural-service

# 3. Warm cache with high-quality queries
echo "3. Warming cache with high-quality queries..."
python3 << 'EOF'
import requests
import time

high_quality_queries = [
    "singapore housing data",
    "psychology research datasets", 
    "climate change indicators",
    "machine learning competitions",
    "world bank economic data"
]

for query in high_quality_queries:
    try:
        response = requests.post('http://localhost:8000/api/v2/search', 
            json={"query": query, "quality_threshold": 0.8}, 
            timeout=10)
        if response.status_code == 200:
            print(f"✅ Warmed cache for: {query}")
        else:
            print(f"❌ Failed to warm cache for: {query}")
    except Exception as e:
        print(f"❌ Error warming cache for {query}: {e}")
    
    time.sleep(2)  # Rate limiting
EOF

# 4. Verify remediation
echo "4. Verifying remediation..."
sleep 30  # Wait for services to stabilize

METRICS_AFTER=$(curl -s http://localhost:8000/api/v2/quality/metrics)
NDCG_AFTER=$(echo $METRICS_AFTER | jq -r '.current_metrics.overall_ndcg_at_3')

echo "NDCG@3 after remediation: $NDCG_AFTER"

if (( $(echo "$NDCG_AFTER > 0.65" | bc -l) )); then
    echo "✅ Automatic remediation successful"
else
    echo "❌ Automatic remediation failed - Manual intervention required"
fi
```

#### Step 3: Manual Investigation (10-30 minutes)

If automatic remediation fails:

```bash
#!/bin/bash
# Manual investigation procedures

echo "=== MANUAL INVESTIGATION ==="

# 1. Deep dive into training mappings
echo "1. Training Mappings Analysis:"
python3 << 'EOF'
import re

# Check training mappings file
try:
    with open('training_mappings.md', 'r') as f:
        content = f.read()
    
    # Count mappings by quality
    high_quality = len(re.findall(r'→.*\(0\.[8-9]\d*\)', content))
    medium_quality = len(re.findall(r'→.*\(0\.[5-7]\d*\)', content))
    low_quality = len(re.findall(r'→.*\(0\.[0-4]\d*\)', content))
    
    print(f"High quality mappings (0.8+): {high_quality}")
    print(f"Medium quality mappings (0.5-0.7): {medium_quality}")
    print(f"Low quality mappings (<0.5): {low_quality}")
    
    # Check for recent changes
    import os
    import time
    
    mtime = os.path.getmtime('training_mappings.md')
    hours_since_modified = (time.time() - mtime) / 3600
    
    print(f"Training mappings last modified: {hours_since_modified:.1f} hours ago")
    
except Exception as e:
    print(f"Error analyzing training mappings: {e}")
EOF

# 2. Model performance analysis
echo -e "\n2. Model Performance Analysis:"
python3 << 'EOF'
import torch
import time
import psutil

try:
    from src.dl.quality_first_neural_model import QualityAwareRankingModel
    
    # Memory before loading
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Load and test model
    model = QualityAwareRankingModel({
        'embedding_dim': 256, 'hidden_dim': 128,
        'num_domains': 8, 'num_sources': 10, 'vocab_size': 10000
    })
    
    checkpoint = torch.load('models/dl/quality_first/best_quality_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Memory after loading
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"Model memory usage: {memory_after - memory_before:.1f} MB")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test inference speed
    test_queries = ['test query'] * 10
    start_time = time.time()
    
    for query in test_queries:
        relevance = model.predict_relevance(query, 'kaggle')
    
    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_queries) * 1000
    
    print(f"Average inference time: {avg_time:.1f} ms")
    
    # Test specific problematic cases
    test_cases = [
        ('psychology research data', 'kaggle'),
        ('singapore housing prices', 'data_gov_sg'),
        ('climate change data', 'world_bank')
    ]
    
    print("\nTest case predictions:")
    for query, source in test_cases:
        relevance = model.predict_relevance(query, source)
        print(f"  {query} → {source}: {relevance:.3f}")
        
except Exception as e:
    print(f"Model analysis error: {e}")
EOF

# 3. Check for data corruption
echo -e "\n3. Data Integrity Check:"
python3 << 'EOF'
import json
import os

# Check training data files
data_files = [
    'data/processed/enhanced_training_mappings.json',
    'data/processed/enhanced_training_data.json',
    'training_mappings.md'
]

for file_path in data_files:
    if os.path.exists(file_path):
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ {file_path}: Valid JSON, {len(data)} entries")
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
                print(f"✅ {file_path}: {len(content)} characters")
        except Exception as e:
            print(f"❌ {file_path}: Error - {e}")
    else:
        print(f"❌ {file_path}: File not found")
EOF
```

### Cache Performance Issues

**Trigger:** Cache hit rate < 0.70  
**Impact:** Increased response times, higher resource usage

#### Diagnosis and Resolution:

```bash
#!/bin/bash
# Cache performance investigation

echo "=== CACHE PERFORMANCE INVESTIGATION ==="

# 1. Get detailed cache statistics
echo "1. Cache Statistics:"
curl -s http://localhost:8000/api/v2/cache/stats | jq '.'

# 2. Analyze cache quality distribution
echo -e "\n2. Cache Quality Analysis:"
python3 << 'EOF'
from src.ai.quality_aware_cache import QualityAwareCacheManager
import sqlite3

cache_manager = QualityAwareCacheManager()
stats = cache_manager.get_quality_cache_statistics()

print(f"Total cache entries: {stats['total_entries']}")
print(f"Hit rate: {stats['hit_rate']:.3f}")
print(f"Average quality: {stats['avg_quality_score']:.3f}")

# Quality distribution
dist = stats['quality_distribution']
print(f"\nQuality Distribution:")
print(f"  Excellent (0.9+): {dist['excellent']}")
print(f"  Good (0.7-0.9): {dist['good']}")
print(f"  Fair (0.5-0.7): {dist['fair']}")
print(f"  Poor (<0.5): {dist['poor']}")

# Recommendations
if stats['hit_rate'] < 0.7:
    print("\n🔧 Recommendations:")
    if dist['poor'] > dist['total'] * 0.2:
        print("  - Clear low-quality cache entries")
    if dist['excellent'] < dist['total'] * 0.3:
        print("  - Warm cache with high-quality queries")
    if stats['avg_quality_score'] < 0.7:
        print("  - Increase quality threshold for caching")
EOF

# 3. Cache cleanup if needed
echo -e "\n3. Cache Cleanup:"
POOR_ENTRIES=$(curl -s http://localhost:8000/api/v2/cache/stats | jq '.cache_statistics.quality_distribution.poor')

if [ "$POOR_ENTRIES" -gt 100 ]; then
    echo "Clearing poor quality cache entries..."
    curl -X POST http://localhost:8000/api/v2/cache/invalidate \
        -H "Content-Type: application/json" \
        -d '{"quality_threshold": 0.6}'
    
    echo "Cache cleanup completed"
fi
```

## Maintenance Procedures

### Weekly Training Mapping Updates

**Schedule:** Every Friday 6:00 PM  
**Duration:** 30 minutes  
**Responsibility:** ML Engineering Team

```bash
#!/bin/bash
# Weekly training mapping update procedure

echo "=== WEEKLY TRAINING MAPPING UPDATE ==="

# 1. Backup current mappings
cp training_mappings.md training_mappings_backup_$(date +%Y%m%d).md

# 2. Collect user feedback for the week
echo "1. Collecting user feedback..."
python3 << 'EOF'
import json
import sqlite3
from datetime import datetime, timedelta

# Connect to feedback database
conn = sqlite3.connect('data/feedback/user_feedback.db')

# Get feedback from last week
week_ago = datetime.now() - timedelta(days=7)
cursor = conn.execute('''
    SELECT query, recommended_source, user_rating, feedback_text
    FROM user_feedback 
    WHERE timestamp > ? AND user_rating IS NOT NULL
''', (week_ago.isoformat(),))

feedback_data = cursor.fetchall()
conn.close()

print(f"Collected {len(feedback_data)} feedback entries from last week")

# Analyze feedback for potential mapping updates
low_rated = [f for f in feedback_data if f[2] < 3]  # Rating < 3
high_rated = [f for f in feedback_data if f[2] >= 4]  # Rating >= 4

print(f"Low-rated recommendations: {len(low_rated)}")
print(f"High-rated recommendations: {len(high_rated)}")

# Save feedback summary
with open('data/feedback/weekly_feedback_summary.json', 'w') as f:
    json.dump({
        'total_feedback': len(feedback_data),
        'low_rated': len(low_rated),
        'high_rated': len(high_rated),
        'low_rated_examples': low_rated[:10],  # Top 10 examples
        'high_rated_examples': high_rated[:10]
    }, f, indent=2)

print("Feedback analysis saved to weekly_feedback_summary.json")
EOF

# 3. Review and update mappings
echo -e "\n2. Review current mapping performance..."
python3 << 'EOF'
from src.ai.quality_monitoring_system import QualityMonitoringSystem

monitor = QualityMonitoringSystem()

# Get queries with consistently low performance
low_performance_queries = monitor.get_low_performance_queries(threshold=0.6, days=7)

print("Queries with consistently low performance:")
for query_data in low_performance_queries:
    print(f"  - {query_data['query']}: NDCG@3 = {query_data['avg_ndcg']:.3f}")

# Suggest mapping updates
print("\nSuggested mapping updates:")
for query_data in low_performance_queries:
    print(f"  Query: {query_data['query']}")
    print(f"    Current top source: {query_data['top_source']}")
    print(f"    Suggested alternatives: {', '.join(query_data['alternative_sources'])}")
    print()
EOF

# 4. Apply approved updates
echo -e "\n3. Applying approved mapping updates..."
# This would typically involve manual review and approval
# For automation, you could implement an approval workflow

# 5. Retrain model with updated mappings
echo -e "\n4. Retraining model with updated mappings..."
python src/ml/enhanced_training_integrator.py \
    --input_file training_mappings.md \
    --output_file data/processed/enhanced_training_mappings.json

python src/dl/quality_first_trainer.py \
    --training_data data/processed/enhanced_training_mappings.json \
    --quick_retrain true \
    --max_epochs 20

# 6. Validate updated model
echo -e "\n5. Validating updated model..."
python3 << 'EOF'
from src.ai.enhanced_query_router import EnhancedQueryRouter

router = EnhancedQueryRouter()

# Test key queries
test_queries = [
    "psychology research data",
    "singapore housing prices", 
    "climate change indicators",
    "machine learning datasets"
]

print("Validation Results:")
for query in test_queries:
    classification = router.classify_query(query)
    print(f"  {query}:")
    print(f"    Domain: {classification.domain}")
    print(f"    Confidence: {classification.confidence:.3f}")
    print(f"    Top sources: {classification.recommended_sources[:3]}")
EOF

echo -e "\n=== WEEKLY UPDATE COMPLETE ==="
```

### Monthly Model Retraining

**Schedule:** First Saturday of each month  
**Duration:** 4-6 hours  
**Responsibility:** ML Engineering Team

```bash
#!/bin/bash
# Monthly comprehensive model retraining

echo "=== MONTHLY MODEL RETRAINING ==="

# 1. Backup current model
echo "1. Backing up current model..."
cp models/dl/quality_first/best_quality_model.pt \
   models/dl/quality_first/backup_$(date +%Y%m%d).pt

# 2. Prepare comprehensive training data
echo "2. Preparing training data..."
python src/ml/enhanced_training_integrator.py \
    --input_file training_mappings.md \
    --output_file data/processed/enhanced_training_mappings.json \
    --augment_data true \
    --generate_negatives true

# 3. Full model retraining
echo "3. Starting full model retraining..."
python src/dl/quality_first_trainer.py \
    --training_data data/processed/enhanced_training_mappings.json \
    --curriculum_learning true \
    --max_epochs 100 \
    --early_stopping true \
    --save_best_model true

# 4. Comprehensive validation
echo "4. Comprehensive model validation..."
python src/dl/quality_first_trainer.py \
    --mode validate \
    --model_path models/dl/quality_first/best_quality_model.pt \
    --test_data data/processed/enhanced_training_mappings.json

# 5. A/B testing setup
echo "5. Setting up A/B testing..."
cp models/dl/quality_first/best_quality_model.pt \
   models/dl/quality_first/candidate_model.pt

# Configure A/B testing (50/50 split)
cat > config/ab_test_config.yml << EOF
ab_testing:
  enabled: true
  traffic_split: 0.5
  model_a: models/dl/quality_first/backup_$(date +%Y%m%d).pt
  model_b: models/dl/quality_first/candidate_model.pt
  metrics_to_track:
    - ndcg_at_3
    - relevance_accuracy
    - user_satisfaction
  duration_days: 7
EOF

echo "6. A/B testing configured for 7 days"
echo "   Monitor results at: http://localhost:3000/ab-testing-dashboard"

echo -e "\n=== MONTHLY RETRAINING COMPLETE ==="
```

### Quarterly System Review

**Schedule:** End of each quarter  
**Duration:** 1 day  
**Responsibility:** Full Engineering Team

```bash
#!/bin/bash
# Quarterly comprehensive system review

echo "=== QUARTERLY SYSTEM REVIEW ==="

# 1. Generate comprehensive performance report
echo "1. Generating quarterly performance report..."
curl -X POST http://localhost:8000/api/v2/quality/report \
    -H "Content-Type: application/json" \
    -d '{"time_range": "90d", "include_details": true, "format": "json"}' \
    > reports/quarterly_report_$(date +%Y_Q$((($(date +%m)-1)/3+1))).json

# 2. System health assessment
echo "2. System health assessment..."
python3 << 'EOF'
import psutil
import subprocess
import json
from datetime import datetime

health_report = {
    "timestamp": datetime.now().isoformat(),
    "system_resources": {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    },
    "service_status": {},
    "log_analysis": {}
}

# Check service status
services = ['dataset-research-assistant', 'nginx', 'redis-server']
for service in services:
    try:
        result = subprocess.run(['systemctl', 'is-active', service], 
                              capture_output=True, text=True)
        health_report["service_status"][service] = result.stdout.strip()
    except Exception as e:
        health_report["service_status"][service] = f"error: {e}"

# Analyze error logs
try:
    result = subprocess.run(['grep', '-c', 'ERROR', '/var/log/dataset-research/error.log'], 
                          capture_output=True, text=True)
    health_report["log_analysis"]["error_count_90d"] = int(result.stdout.strip())
except:
    health_report["log_analysis"]["error_count_90d"] = "unknown"

with open(f'reports/system_health_{datetime.now().strftime("%Y_Q%m")}.json', 'w') as f:
    json.dump(health_report, f, indent=2)

print("System health report generated")
EOF

# 3. Performance optimization recommendations
echo "3. Generating optimization recommendations..."
python3 << 'EOF'
from src.ai.quality_monitoring_system import QualityMonitoringSystem
from src.ai.quality_aware_cache import QualityAwareCacheManager

monitor = QualityMonitoringSystem()
cache_manager = QualityAwareCacheManager()

# Get 90-day performance trends
trends = monitor.get_performance_trends(days=90)
cache_stats = cache_manager.get_quality_cache_statistics()

recommendations = []

# Quality recommendations
if trends['avg_ndcg'] < 0.75:
    recommendations.append({
        "category": "quality",
        "issue": "NDCG@3 below optimal threshold",
        "recommendation": "Increase training mapping coverage",
        "priority": "high"
    })

# Performance recommendations
if cache_stats['hit_rate'] < 0.8:
    recommendations.append({
        "category": "performance", 
        "issue": "Cache hit rate below optimal",
        "recommendation": "Optimize cache warming strategy",
        "priority": "medium"
    })

# Resource recommendations
import psutil
if psutil.virtual_memory().percent > 80:
    recommendations.append({
        "category": "resources",
        "issue": "High memory usage",
        "recommendation": "Consider model quantization or scaling",
        "priority": "high"
    })

print("Quarterly Optimization Recommendations:")
print("=" * 50)
for rec in recommendations:
    print(f"[{rec['priority'].upper()}] {rec['category'].title()}: {rec['issue']}")
    print(f"  Recommendation: {rec['recommendation']}")
    print()

# Save recommendations
import json
with open(f'reports/optimization_recommendations_{datetime.now().strftime("%Y_Q%m")}.json', 'w') as f:
    json.dump(recommendations, f, indent=2)
EOF

echo -e "\n=== QUARTERLY REVIEW COMPLETE ==="
echo "Reports generated in: reports/"
echo "Next steps: Review recommendations and plan implementation"
```

## Emergency Contacts and Escalation

### Contact Information

| Role | Primary Contact | Backup Contact | Response Time |
|------|----------------|----------------|---------------|
| On-Call Engineer | +1-555-0101 | +1-555-0102 | 15 minutes |
| ML Engineering Lead | +1-555-0201 | +1-555-0202 | 30 minutes |
| DevOps Lead | +1-555-0301 | +1-555-0302 | 15 minutes |
| Product Manager | +1-555-0401 | +1-555-0402 | 1 hour |

### Escalation Matrix

| Severity | Criteria | Response Time | Escalation |
|----------|----------|---------------|------------|
| P0 - Critical | NDCG@3 < 0.5, System down | 5 minutes | Immediate to On-Call + ML Lead |
| P1 - High | NDCG@3 < 0.6, Major functionality broken | 15 minutes | On-Call Engineer |
| P2 - Medium | NDCG@3 < 0.7, Performance degradation | 1 hour | Standard support |
| P3 - Low | Minor issues, Feature requests | 24 hours | Standard support |

### Communication Channels

- **Slack**: #dataset-research-alerts (automated alerts)
- **Slack**: #dataset-research-ops (operational discussions)
- **Email**: ops-team@dataset-research.ai
- **PagerDuty**: dataset-research-assistant service

This runbook provides comprehensive operational procedures for maintaining optimal quality in the dataset research assistant system.