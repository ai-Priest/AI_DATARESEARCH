# Training Mappings Maintenance Guide

## Overview

This guide provides comprehensive procedures for maintaining and updating the `training_mappings.md` file, which serves as the ground truth for the AI-Powered Dataset Research Assistant's quality-first neural model. Proper maintenance of training mappings is crucial for maintaining high recommendation quality and system performance.

## Understanding Training Mappings

### File Structure

The `training_mappings.md` file contains expert-curated query-source mappings in the following format:

```markdown
# Training Mappings for Quality-First Neural Model

## Psychology Domain
- psychology research data → kaggle (0.92) - Excellent for psychology datasets and ML competitions
- mental health datasets → zenodo (0.85) - Academic repository with quality research data
- behavioral analysis data → kaggle (0.88) - Strong community and diverse behavioral datasets

## Singapore-First Strategy
- singapore housing prices → data_gov_sg (0.96) - Official government housing data
- hdb resale prices → data_gov_sg (0.94) - Comprehensive HDB transaction data
- singapore population data → singstat (0.95) - Authoritative demographic statistics

## Climate Domain  
- climate change data → world_bank (0.95) - Authoritative global climate indicators
- environmental datasets → zenodo (0.82) - Academic environmental research
- weather data → world_bank (0.88) - Comprehensive meteorological data
```

### Mapping Components

Each mapping consists of:
1. **Query**: User search query or intent
2. **Source**: Recommended data source
3. **Relevance Score**: 0.0-1.0 scale indicating recommendation quality
4. **Explanation**: Rationale for the mapping

### Relevance Score Guidelines

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.90-1.00 | Excellent | Perfect match, authoritative source |
| 0.80-0.89 | Very Good | Strong relevance, high-quality source |
| 0.70-0.79 | Good | Relevant, reliable source |
| 0.60-0.69 | Fair | Somewhat relevant, acceptable quality |
| 0.50-0.59 | Poor | Limited relevance, low priority |
| 0.00-0.49 | Very Poor | Irrelevant or unreliable |

## Maintenance Procedures

### Daily Monitoring

**Frequency:** Daily  
**Duration:** 10 minutes  
**Responsibility:** ML Operations Team

```bash
#!/bin/bash
# Daily training mappings monitoring

echo "=== DAILY TRAINING MAPPINGS CHECK ==="

# 1. Check file integrity
if [ -f "training_mappings.md" ]; then
    echo "✅ Training mappings file exists"
    
    # Count total mappings
    TOTAL_MAPPINGS=$(grep -c "→" training_mappings.md)
    echo "📊 Total mappings: $TOTAL_MAPPINGS"
    
    # Check for recent changes
    LAST_MODIFIED=$(stat -c %Y training_mappings.md)
    CURRENT_TIME=$(date +%s)
    HOURS_SINCE_MODIFIED=$(( (CURRENT_TIME - LAST_MODIFIED) / 3600 ))
    
    echo "🕒 Last modified: $HOURS_SINCE_MODIFIED hours ago"
    
    if [ $HOURS_SINCE_MODIFIED -gt 168 ]; then  # 1 week
        echo "⚠️  Training mappings not updated in over a week"
    fi
    
else
    echo "❌ Training mappings file missing!"
    exit 1
fi

# 2. Validate mapping format
echo -e "\n2. Validating mapping format..."
python3 << 'EOF'
import re

with open('training_mappings.md', 'r') as f:
    content = f.read()

# Check for proper format: query → source (score) - explanation
valid_mappings = re.findall(r'^- .+ → .+ \(0\.\d+\) - .+$', content, re.MULTILINE)
all_mappings = re.findall(r'^- .+ → .+', content, re.MULTILINE)

print(f"Valid format mappings: {len(valid_mappings)}")
print(f"Total mapping lines: {len(all_mappings)}")

if len(valid_mappings) != len(all_mappings):
    print("⚠️  Some mappings have invalid format")
    
    # Find invalid mappings
    for i, mapping in enumerate(all_mappings):
        if mapping not in [vm.split(' - ')[0] for vm in valid_mappings]:
            print(f"  Invalid: {mapping}")

# Check score distribution
scores = re.findall(r'\(0\.(\d+)\)', content)
if scores:
    avg_score = sum(int(s) for s in scores) / len(scores) / 100
    print(f"Average relevance score: {avg_score:.3f}")
    
    high_quality = sum(1 for s in scores if int(s) >= 80)
    print(f"High quality mappings (≥0.8): {high_quality}/{len(scores)} ({high_quality/len(scores)*100:.1f}%)")
EOF

# 3. Check domain coverage
echo -e "\n3. Domain coverage analysis..."
python3 << 'EOF'
import re

with open('training_mappings.md', 'r') as f:
    content = f.read().lower()

domains = {
    'psychology': ['psychology', 'mental health', 'behavioral', 'cognitive'],
    'singapore': ['singapore', 'sg', 'hdb', 'mrt', 'singstat'],
    'climate': ['climate', 'weather', 'environmental', 'temperature'],
    'economics': ['economic', 'gdp', 'financial', 'trade'],
    'machine_learning': ['machine learning', 'ml', 'ai', 'neural'],
    'health': ['health', 'medical', 'healthcare', 'disease'],
    'education': ['education', 'student', 'university', 'school']
}

print("Domain coverage:")
for domain, keywords in domains.items():
    count = sum(content.count(keyword) for keyword in keywords)
    print(f"  {domain}: {count} mentions")
    
    if count < 3:
        print(f"    ⚠️  Low coverage for {domain} domain")
EOF

echo -e "\n=== DAILY CHECK COMPLETE ==="
```

### Weekly Quality Assessment

**Frequency:** Every Monday  
**Duration:** 30 minutes  
**Responsibility:** ML Engineering Team

```bash
#!/bin/bash
# Weekly training mappings quality assessment

echo "=== WEEKLY QUALITY ASSESSMENT ==="

# 1. Test mappings against current system
echo "1. Testing mappings against current system..."
python3 << 'EOF'
import re
import requests
import json

# Parse training mappings
with open('training_mappings.md', 'r') as f:
    content = f.read()

# Extract mappings
mapping_pattern = r'- (.+) → (.+) \((\d\.\d+)\) - (.+)'
mappings = re.findall(mapping_pattern, content)

print(f"Testing {len(mappings)} mappings...")

test_results = []
for query, expected_source, expected_score, explanation in mappings[:20]:  # Test first 20
    try:
        # Test search
        response = requests.post('http://localhost:8000/api/v2/search', 
            json={"query": query, "max_results": 5}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            
            # Check if expected source is in top 3
            top_sources = [rec['source'] for rec in recommendations[:3]]
            source_found = expected_source.lower() in [s.lower() for s in top_sources]
            
            # Get actual relevance score for expected source
            actual_score = 0.0
            for rec in recommendations:
                if expected_source.lower() in rec['source'].lower():
                    actual_score = rec['quality_metrics']['relevance_score']
                    break
            
            test_results.append({
                'query': query,
                'expected_source': expected_source,
                'expected_score': float(expected_score),
                'source_found': source_found,
                'actual_score': actual_score,
                'top_sources': top_sources
            })
            
        else:
            print(f"❌ API error for query: {query}")
            
    except Exception as e:
        print(f"❌ Error testing query '{query}': {e}")

# Analyze results
if test_results:
    correct_predictions = sum(1 for r in test_results if r['source_found'])
    accuracy = correct_predictions / len(test_results)
    
    score_differences = [abs(r['actual_score'] - r['expected_score']) 
                        for r in test_results if r['actual_score'] > 0]
    avg_score_diff = sum(score_differences) / len(score_differences) if score_differences else 0
    
    print(f"\nResults:")
    print(f"  Mapping accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_results)})")
    print(f"  Average score difference: {avg_score_diff:.3f}")
    
    # Identify problematic mappings
    problematic = [r for r in test_results if not r['source_found'] or 
                   abs(r['actual_score'] - r['expected_score']) > 0.2]
    
    if problematic:
        print(f"\nProblematic mappings ({len(problematic)}):")
        for r in problematic[:5]:  # Show top 5
            print(f"  Query: {r['query']}")
            print(f"    Expected: {r['expected_source']} ({r['expected_score']:.2f})")
            print(f"    Got: {r['top_sources'][0] if r['top_sources'] else 'None'} ({r['actual_score']:.2f})")
            print()
    
    # Save results for trending
    with open(f'data/quality/mapping_test_results_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
        json.dump(test_results, f, indent=2)
EOF

# 2. Analyze user feedback for mapping updates
echo -e "\n2. Analyzing user feedback..."
python3 << 'EOF'
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

try:
    # Connect to feedback database
    conn = sqlite3.connect('data/feedback/user_feedback.db')
    
    # Get feedback from last week
    week_ago = datetime.now() - timedelta(days=7)
    cursor = conn.execute('''
        SELECT query, recommended_source, user_rating, feedback_text
        FROM user_feedback 
        WHERE timestamp > ? AND user_rating IS NOT NULL
        ORDER BY user_rating ASC
    ''', (week_ago.isoformat(),))
    
    feedback_data = cursor.fetchall()
    conn.close()
    
    if feedback_data:
        print(f"Analyzed {len(feedback_data)} feedback entries")
        
        # Group by rating
        by_rating = defaultdict(list)
        for query, source, rating, text in feedback_data:
            by_rating[rating].append((query, source, text))
        
        # Low-rated recommendations (potential mapping updates needed)
        low_rated = by_rating[1] + by_rating[2]  # Ratings 1 and 2
        if low_rated:
            print(f"\nLow-rated recommendations ({len(low_rated)}):")
            for query, source, text in low_rated[:5]:
                print(f"  Query: {query}")
                print(f"  Source: {source}")
                print(f"  Feedback: {text}")
                print()
        
        # High-rated recommendations (confirm good mappings)
        high_rated = by_rating[4] + by_rating[5]  # Ratings 4 and 5
        print(f"\nHigh-rated recommendations: {len(high_rated)}")
        
        # Suggest mapping updates
        print("\nSuggested mapping updates:")
        for query, source, text in low_rated[:3]:
            print(f"  Consider updating: {query} → {source}")
            print(f"  Reason: {text}")
            print()
    
    else:
        print("No user feedback available for analysis")
        
except Exception as e:
    print(f"Error analyzing feedback: {e}")
EOF

echo -e "\n=== WEEKLY ASSESSMENT COMPLETE ==="
```

### Monthly Comprehensive Review

**Frequency:** First Friday of each month  
**Duration:** 2 hours  
**Responsibility:** ML Engineering Team + Domain Experts

```bash
#!/bin/bash
# Monthly comprehensive training mappings review

echo "=== MONTHLY COMPREHENSIVE REVIEW ==="

# 1. Backup current mappings
BACKUP_FILE="training_mappings_backup_$(date +%Y%m%d).md"
cp training_mappings.md "$BACKUP_FILE"
echo "✅ Backup created: $BACKUP_FILE"

# 2. Comprehensive quality analysis
echo -e "\n1. Comprehensive quality analysis..."
python3 << 'EOF'
import re
import json
from collections import defaultdict, Counter

with open('training_mappings.md', 'r') as f:
    content = f.read()

# Parse all mappings
mapping_pattern = r'- (.+) → (.+) \((\d\.\d+)\) - (.+)'
mappings = re.findall(mapping_pattern, content)

print(f"Total mappings: {len(mappings)}")

# Analyze by domain
domain_mappings = defaultdict(list)
for query, source, score, explanation in mappings:
    query_lower = query.lower()
    
    # Classify domain
    if any(term in query_lower for term in ['psychology', 'mental health', 'behavioral']):
        domain = 'psychology'
    elif any(term in query_lower for term in ['singapore', 'sg', 'hdb', 'mrt']):
        domain = 'singapore'
    elif any(term in query_lower for term in ['climate', 'weather', 'environmental']):
        domain = 'climate'
    elif any(term in query_lower for term in ['economic', 'gdp', 'financial']):
        domain = 'economics'
    elif any(term in query_lower for term in ['machine learning', 'ml', 'ai']):
        domain = 'machine_learning'
    else:
        domain = 'general'
    
    domain_mappings[domain].append((query, source, float(score), explanation))

# Domain analysis
print("\nDomain Analysis:")
for domain, domain_maps in domain_mappings.items():
    avg_score = sum(score for _, _, score, _ in domain_maps) / len(domain_maps)
    source_counts = Counter(source for _, source, _, _ in domain_maps)
    
    print(f"  {domain.title()}:")
    print(f"    Mappings: {len(domain_maps)}")
    print(f"    Avg Score: {avg_score:.3f}")
    print(f"    Top Sources: {dict(source_counts.most_common(3))}")
    print()

# Quality distribution
scores = [float(score) for _, _, score, _ in mappings]
quality_dist = {
    'excellent': sum(1 for s in scores if s >= 0.9),
    'very_good': sum(1 for s in scores if 0.8 <= s < 0.9),
    'good': sum(1 for s in scores if 0.7 <= s < 0.8),
    'fair': sum(1 for s in scores if 0.6 <= s < 0.7),
    'poor': sum(1 for s in scores if s < 0.6)
}

print("Quality Distribution:")
for quality, count in quality_dist.items():
    percentage = count / len(scores) * 100
    print(f"  {quality.title()}: {count} ({percentage:.1f}%)")

# Identify gaps
print("\nIdentified Gaps:")
if quality_dist['poor'] > len(scores) * 0.1:
    print(f"  - Too many poor quality mappings ({quality_dist['poor']})")

if len(domain_mappings.get('singapore', [])) < 10:
    print("  - Insufficient Singapore-specific mappings")

if len(domain_mappings.get('psychology', [])) < 5:
    print("  - Insufficient psychology domain mappings")

# Source analysis
all_sources = [source for _, source, _, _ in mappings]
source_counts = Counter(all_sources)

print(f"\nSource Distribution:")
for source, count in source_counts.most_common(10):
    percentage = count / len(all_sources) * 100
    print(f"  {source}: {count} ({percentage:.1f}%)")

# Save analysis
analysis_report = {
    'total_mappings': len(mappings),
    'domain_analysis': {domain: {
        'count': len(maps),
        'avg_score': sum(score for _, _, score, _ in maps) / len(maps),
        'top_sources': dict(Counter(source for _, source, _, _ in maps).most_common(3))
    } for domain, maps in domain_mappings.items()},
    'quality_distribution': quality_dist,
    'source_distribution': dict(source_counts.most_common(10))
}

with open(f'data/quality/mapping_analysis_{datetime.now().strftime("%Y%m")}.json', 'w') as f:
    json.dump(analysis_report, f, indent=2)

print("\nAnalysis saved to mapping_analysis file")
EOF

# 3. Generate update recommendations
echo -e "\n2. Generating update recommendations..."
python3 << 'EOF'
import json
import requests
from datetime import datetime, timedelta

# Load recent performance data
try:
    with open('data/quality/mapping_test_results_latest.json', 'r') as f:
        test_results = json.load(f)
except:
    test_results = []

# Load user feedback
try:
    with open('data/feedback/weekly_feedback_summary.json', 'r') as f:
        feedback_summary = json.load(f)
except:
    feedback_summary = {'low_rated': [], 'high_rated': []}

recommendations = []

# Based on test results
if test_results:
    low_accuracy_mappings = [r for r in test_results if not r['source_found']]
    for mapping in low_accuracy_mappings[:5]:
        recommendations.append({
            'type': 'update_mapping',
            'query': mapping['query'],
            'current_source': mapping['expected_source'],
            'suggested_source': mapping['top_sources'][0] if mapping['top_sources'] else 'unknown',
            'reason': 'Low accuracy in testing',
            'priority': 'high'
        })

# Based on user feedback
for feedback in feedback_summary.get('low_rated', [])[:3]:
    query, source, rating, text = feedback
    recommendations.append({
        'type': 'review_mapping',
        'query': query,
        'current_source': source,
        'user_rating': rating,
        'user_feedback': text,
        'reason': 'Low user rating',
        'priority': 'medium'
    })

# Domain gap recommendations
domain_gaps = {
    'singapore': ['singapore transport data', 'sg education statistics', 'local healthcare data'],
    'psychology': ['personality assessment data', 'therapy outcomes data', 'psychological surveys'],
    'climate': ['carbon footprint data', 'renewable energy statistics', 'biodiversity datasets']
}

for domain, suggested_queries in domain_gaps.items():
    for query in suggested_queries:
        recommendations.append({
            'type': 'add_mapping',
            'query': query,
            'domain': domain,
            'reason': f'Fill {domain} domain gap',
            'priority': 'low'
        })

print("Update Recommendations:")
print("=" * 50)

for i, rec in enumerate(recommendations[:10], 1):
    print(f"{i}. [{rec['priority'].upper()}] {rec['type'].replace('_', ' ').title()}")
    print(f"   Query: {rec['query']}")
    if 'current_source' in rec:
        print(f"   Current Source: {rec['current_source']}")
    if 'suggested_source' in rec:
        print(f"   Suggested Source: {rec['suggested_source']}")
    print(f"   Reason: {rec['reason']}")
    print()

# Save recommendations
with open(f'data/quality/mapping_recommendations_{datetime.now().strftime("%Y%m")}.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print(f"Generated {len(recommendations)} recommendations")
print("Saved to mapping_recommendations file")
EOF

echo -e "\n=== MONTHLY REVIEW COMPLETE ==="
```

## Adding New Mappings

### Process for Adding New Mappings

1. **Identify Need**
   - User feedback indicating poor recommendations
   - New domain or use case
   - Performance gaps in specific areas

2. **Research and Validate**
   - Test query with current system
   - Research best data sources for the query
   - Validate source quality and relevance

3. **Create Mapping**
   - Follow standard format
   - Assign appropriate relevance score
   - Provide clear explanation

4. **Test and Validate**
   - Test mapping with system
   - Verify improvement in recommendations
   - Check for conflicts with existing mappings

### Adding Mapping Template

```bash
#!/bin/bash
# Template for adding new training mapping

# Function to add new mapping
add_training_mapping() {
    local query="$1"
    local source="$2"
    local score="$3"
    local explanation="$4"
    local domain="$5"
    
    echo "Adding new training mapping:"
    echo "  Query: $query"
    echo "  Source: $source"
    echo "  Score: $score"
    echo "  Domain: $domain"
    
    # Validate inputs
    if [[ ! "$score" =~ ^0\.[0-9]+$ ]]; then
        echo "❌ Invalid score format. Use 0.XX format"
        return 1
    fi
    
    if (( $(echo "$score < 0.5" | bc -l) )); then
        echo "⚠️  Score below 0.5 - consider if this mapping is necessary"
    fi
    
    # Test current system performance for this query
    echo "Testing current system performance..."
    python3 << EOF
import requests
import json

try:
    response = requests.post('http://localhost:8000/api/v2/search', 
        json={"query": "$query", "max_results": 5}, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        recommendations = data.get('recommendations', [])
        
        print("Current top 3 recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['source']} (score: {rec['quality_metrics']['relevance_score']:.3f})")
        
        # Check if proposed source is already in top 3
        top_sources = [rec['source'] for rec in recommendations[:3]]
        if "$source" in top_sources:
            print("✅ Proposed source already in top 3")
        else:
            print("📝 Proposed source not in top 3 - mapping will improve recommendations")
    else:
        print("❌ Error testing current system")
        
except Exception as e:
    print(f"❌ Error: {e}")
EOF
    
    # Add to training mappings file
    echo "Adding to training_mappings.md..."
    
    # Find appropriate section or create new one
    if grep -q "## $domain" training_mappings.md; then
        # Add to existing domain section
        sed -i "/## $domain/a\\- $query → $source ($score) - $explanation" training_mappings.md
    else
        # Create new domain section
        echo -e "\n## $domain\n- $query → $source ($score) - $explanation" >> training_mappings.md
    fi
    
    echo "✅ Mapping added successfully"
    
    # Validate file format
    echo "Validating file format..."
    if grep -q "- $query → $source ($score) - $explanation" training_mappings.md; then
        echo "✅ Mapping found in file"
    else
        echo "❌ Mapping not found - check file manually"
    fi
    
    # Suggest next steps
    echo -e "\nNext steps:"
    echo "1. Test the updated mappings with the system"
    echo "2. Retrain the model if significant changes were made"
    echo "3. Monitor performance improvements"
}

# Example usage
# add_training_mapping "singapore startup data" "data_gov_sg" "0.88" "Official government data on Singapore startups and business registrations" "singapore"
```

### Bulk Import Process

For adding multiple mappings from external sources:

```python
#!/usr/bin/env python3
# Bulk import training mappings

import csv
import json
import re
from datetime import datetime

def bulk_import_mappings(csv_file, domain):
    """Import mappings from CSV file"""
    
    mappings_to_add = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            query = row['query'].strip()
            source = row['source'].strip()
            score = float(row['score'])
            explanation = row['explanation'].strip()
            
            # Validate
            if not (0.0 <= score <= 1.0):
                print(f"❌ Invalid score for query '{query}': {score}")
                continue
                
            if score < 0.5:
                print(f"⚠️  Low score for query '{query}': {score}")
            
            mappings_to_add.append({
                'query': query,
                'source': source,
                'score': score,
                'explanation': explanation,
                'domain': domain
            })
    
    print(f"Prepared {len(mappings_to_add)} mappings for import")
    
    # Add to training_mappings.md
    with open('training_mappings.md', 'a') as f:
        f.write(f"\n## {domain.title()} (Imported {datetime.now().strftime('%Y-%m-%d')})\n")
        
        for mapping in mappings_to_add:
            line = f"- {mapping['query']} → {mapping['source']} ({mapping['score']:.2f}) - {mapping['explanation']}\n"
            f.write(line)
    
    print(f"✅ Imported {len(mappings_to_add)} mappings to training_mappings.md")
    
    # Generate summary
    source_counts = {}
    score_sum = 0
    
    for mapping in mappings_to_add:
        source = mapping['source']
        source_counts[source] = source_counts.get(source, 0) + 1
        score_sum += mapping['score']
    
    avg_score = score_sum / len(mappings_to_add)
    
    print(f"\nImport Summary:")
    print(f"  Domain: {domain}")
    print(f"  Total mappings: {len(mappings_to_add)}")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Sources used: {list(source_counts.keys())}")
    
    return mappings_to_add

# Example CSV format:
# query,source,score,explanation
# "singapore education data","data_gov_sg",0.92,"Official education statistics and school data"
# "local university rankings","singstat",0.85,"Comprehensive higher education statistics"

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python bulk_import.py <csv_file> <domain>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    domain = sys.argv[2]
    
    try:
        mappings = bulk_import_mappings(csv_file, domain)
        print(f"✅ Bulk import completed successfully")
    except Exception as e:
        print(f"❌ Bulk import failed: {e}")
```

## Quality Assurance

### Mapping Quality Checklist

Before adding or updating mappings, verify:

- [ ] **Query Clarity**: Query is clear and represents real user intent
- [ ] **Source Accuracy**: Source actually contains relevant data for the query
- [ ] **Score Justification**: Relevance score is justified and consistent with guidelines
- [ ] **Explanation Quality**: Explanation clearly states why this source is recommended
- [ ] **Domain Classification**: Mapping is placed in correct domain section
- [ ] **No Duplicates**: No duplicate or conflicting mappings exist
- [ ] **Format Compliance**: Follows exact format specification
- [ ] **Testing**: Mapping improves system recommendations when tested

### Validation Scripts

```bash
#!/bin/bash
# Comprehensive mapping validation

echo "=== TRAINING MAPPINGS VALIDATION ==="

# 1. Format validation
echo "1. Format Validation:"
python3 << 'EOF'
import re

with open('training_mappings.md', 'r') as f:
    content = f.read()

# Check format compliance
lines = content.split('\n')
mapping_lines = [line for line in lines if line.strip().startswith('- ') and '→' in line]

format_errors = []
for i, line in enumerate(mapping_lines, 1):
    # Check format: - query → source (score) - explanation
    if not re.match(r'^- .+ → .+ \(\d\.\d+\) - .+$', line):
        format_errors.append((i, line))

if format_errors:
    print(f"❌ {len(format_errors)} format errors found:")
    for line_num, line in format_errors[:5]:
        print(f"  Line {line_num}: {line}")
else:
    print("✅ All mappings follow correct format")

# Check for duplicate queries
queries = re.findall(r'^- (.+) →', content, re.MULTILINE)
duplicate_queries = [q for q in set(queries) if queries.count(q) > 1]

if duplicate_queries:
    print(f"⚠️  {len(duplicate_queries)} duplicate queries found:")
    for query in duplicate_queries[:3]:
        print(f"  - {query}")
else:
    print("✅ No duplicate queries found")
EOF

# 2. Score validation
echo -e "\n2. Score Validation:"
python3 << 'EOF'
import re

with open('training_mappings.md', 'r') as f:
    content = f.read()

# Extract all scores
scores = re.findall(r'\((\d\.\d+)\)', content)
scores = [float(s) for s in scores]

if scores:
    print(f"Total scores: {len(scores)}")
    print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    
    # Check for unrealistic scores
    very_low = [s for s in scores if s < 0.3]
    very_high = [s for s in scores if s > 0.98]
    
    if very_low:
        print(f"⚠️  {len(very_low)} very low scores (<0.3) - review needed")
    
    if very_high:
        print(f"⚠️  {len(very_high)} very high scores (>0.98) - verify accuracy")
    
    # Score distribution
    excellent = sum(1 for s in scores if s >= 0.9)
    good = sum(1 for s in scores if 0.7 <= s < 0.9)
    fair = sum(1 for s in scores if 0.5 <= s < 0.7)
    poor = sum(1 for s in scores if s < 0.5)
    
    print(f"\nScore Distribution:")
    print(f"  Excellent (≥0.9): {excellent} ({excellent/len(scores)*100:.1f}%)")
    print(f"  Good (0.7-0.9): {good} ({good/len(scores)*100:.1f}%)")
    print(f"  Fair (0.5-0.7): {fair} ({fair/len(scores)*100:.1f}%)")
    print(f"  Poor (<0.5): {poor} ({poor/len(scores)*100:.1f}%)")
    
    if poor > len(scores) * 0.1:
        print("⚠️  Too many poor quality mappings - review recommended")
EOF

# 3. Domain balance validation
echo -e "\n3. Domain Balance Validation:"
python3 << 'EOF'
import re
from collections import defaultdict

with open('training_mappings.md', 'r') as f:
    content = f.read()

# Extract domain sections
domain_sections = re.findall(r'## ([^#\n]+)', content)
print(f"Domains found: {len(domain_sections)}")

# Count mappings per domain
domain_mappings = defaultdict(int)
current_domain = "unknown"

for line in content.split('\n'):
    if line.startswith('## '):
        current_domain = line[3:].strip().lower()
    elif line.strip().startswith('- ') and '→' in line:
        domain_mappings[current_domain] += 1

print("\nMappings per domain:")
for domain, count in sorted(domain_mappings.items()):
    print(f"  {domain}: {count}")
    
    if count < 3:
        print(f"    ⚠️  Low coverage - consider adding more mappings")

# Check for essential domains
essential_domains = ['singapore', 'psychology', 'climate', 'economics']
missing_domains = [d for d in essential_domains if d not in [k.lower() for k in domain_mappings.keys()]]

if missing_domains:
    print(f"\n⚠️  Missing essential domains: {missing_domains}")
EOF

echo -e "\n=== VALIDATION COMPLETE ==="
```

This comprehensive maintenance guide ensures that training mappings remain high-quality and effective for the neural model's performance.