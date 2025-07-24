# Troubleshooting Guide - Search Quality Improvements

## Overview

This guide covers common issues and solutions related to the search quality improvements, including conversational query processing, URL validation, source routing, and server startup problems.

## Server Startup Issues

### Port Conflicts

**Problem**: Server fails to start with "Port already in use" error

**Symptoms**:
```bash
ERROR: Port 8000 is already in use
OSError: [Errno 48] Address already in use
```

**Solutions**:

1. **Automatic Port Fallback** (Recommended)
   ```bash
   python start_server.py
   # Server will automatically try ports 8001, 8002, 8003
   ```

2. **Check Port Usage**
   ```bash
   # Find what's using port 8000
   lsof -i :8000
   
   # Kill the process if needed
   kill -9 <PID>
   ```

3. **Manual Port Selection**
   ```bash
   # Set preferred port via environment variable
   export PREFERRED_PORT=8001
   python start_server.py
   ```

4. **Configuration Override**
   ```python
   # In config/api_config.yml
   server:
     port: 8001
     fallback_ports: [8002, 8003, 8004]
   ```

**Expected Behavior**:
```bash
INFO: Port 8000 is already in use
INFO: Trying port 8001...
INFO: Server started successfully on port 8001
INFO: API available at http://localhost:8001
```

### Configuration Errors

**Problem**: Server fails to start due to missing or invalid configuration

**Symptoms**:
```bash
FileNotFoundError: config/ai_config.yml not found
KeyError: 'llm_config' not found in configuration
```

**Solutions**:

1. **Check Configuration Files**
   ```bash
   # Verify config files exist
   ls -la config/
   
   # Required files:
   # - ai_config.yml
   # - api_config.yml (optional)
   # - dl_config.yml (for neural models)
   ```

2. **Validate Configuration Format**
   ```python
   import yaml
   
   # Test config loading
   with open('config/ai_config.yml', 'r') as f:
       config = yaml.safe_load(f)
       print("Configuration loaded successfully")
   ```

3. **Use Default Configuration**
   ```bash
   # Copy example configuration
   cp config/ai_config.yml.example config/ai_config.yml
   ```

### Dependency Issues

**Problem**: Missing dependencies or version conflicts

**Symptoms**:
```bash
ModuleNotFoundError: No module named 'aiohttp'
ImportError: cannot import name 'LLMManager'
```

**Solutions**:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check Python Version**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Conversational Query Processing Issues

### Low Confidence Scores

**Problem**: System has low confidence in determining query intent

**Symptoms**:
```json
{
  "query_processing": {
    "confidence": 0.45,
    "is_dataset_request": false,
    "suggested_clarification": "Could you specify what type of data you're looking for?"
  }
}
```

**Diagnosis**:
```python
# Check query processing details
result = await processor.process_input("your query here")
print(f"Confidence: {result.confidence}")
print(f"Extracted terms: {result.extracted_terms}")
print(f"Detected domain: {result.detected_domain}")
```

**Solutions**:

1. **Adjust Confidence Threshold**
   ```yaml
   # config/ai_config.yml
   conversational_query:
     confidence_threshold: 0.6  # Lower from 0.7
   ```

2. **Add Domain-Specific Keywords**
   ```python
   # Update singapore_keywords in conversational_query_processor.py
   singapore_keywords = {
       'your_domain': ['keyword1', 'keyword2', 'keyword3']
   }
   ```

3. **Improve Query Phrasing**
   ```bash
   # Instead of: "I need some information"
   # Try: "I need housing data" or "population statistics"
   ```

### Incorrect Intent Detection

**Problem**: System incorrectly identifies dataset vs non-dataset queries

**Symptoms**:
```json
{
  "query": "Hello, how are you?",
  "query_processing": {
    "is_dataset_request": true  // Should be false
  }
}
```

**Diagnosis**:
```python
# Enable debug logging
import logging
logging.getLogger('src.ai.conversational_query_processor').setLevel(logging.DEBUG)

# Check rule-based vs LLM results
quick_result = processor._quick_intent_check(query)
llm_result = await processor._llm_intent_detection(query)
```

**Solutions**:

1. **Update Non-Dataset Patterns**
   ```python
   # In conversational_query_processor.py
   non_dataset_patterns = [
       r'\b(hello|hi|hey|good morning)\b',
       r'\b(how are you|what\'s up)\b',
       r'\b(thank you|thanks|bye)\b',
       # Add more patterns as needed
   ]
   ```

2. **Improve LLM Prompts**
   ```python
   # Update the prompt in _llm_intent_detection method
   prompt = f"""Enhanced prompt with better examples..."""
   ```

3. **Adjust Dataset Keywords**
   ```python
   # Add or remove keywords in dataset_keywords list
   dataset_keywords = [
       'data', 'dataset', 'statistics', 'information',
       # Add domain-specific terms
   ]
   ```

### Query Normalization Issues

**Problem**: Extracted search terms are not clean or relevant

**Symptoms**:
```json
{
  "extracted_terms": ["I", "need", "some", "data", "please"]  // Too many stop words
}
```

**Solutions**:

1. **Update Stop Words List**
   ```python
   # In _extract_basic_terms method
   stop_words = {
       'i', 'need', 'want', 'looking', 'for', 'find', 'get', 'show', 'me',
       'can', 'you', 'please', 'help', 'with', 'about', 'some', 'any'
       # Add more stop words as needed
   }
   ```

2. **Improve Term Filtering**
   ```python
   # Filter terms by length and relevance
   terms = [word for word in words 
            if word not in stop_words 
            and len(word) > 2 
            and word.isalpha()]
   ```

## URL Validation and Correction Issues

### URL Validation Failures

**Problem**: URLs are incorrectly marked as invalid or validation times out

**Symptoms**:
```json
{
  "url_validation": {
    "status": "failed",
    "status_code": 0,
    "error": "Connection timeout"
  }
}
```

**Diagnosis**:
```python
# Test URL validation manually
from src.ai.url_validator import URLValidator

validator = URLValidator()
is_valid, status_code = await validator.validate_url("https://example.com")
print(f"Valid: {is_valid}, Status: {status_code}")
```

**Solutions**:

1. **Increase Timeout**
   ```python
   # In URLValidator.__init__
   self.timeout = 15  # Increase from 10 seconds
   ```

2. **Check Network Connectivity**
   ```bash
   # Test connectivity to common sources
   curl -I https://data.gov.sg
   curl -I https://www.kaggle.com
   curl -I https://data.worldbank.org
   ```

3. **Disable Validation Temporarily**
   ```json
   {
     "query": "your query",
     "enable_url_validation": false
   }
   ```

### URL Correction Not Working

**Problem**: Broken URLs are not being corrected properly

**Symptoms**:
```json
{
  "original_url": "https://kaggle.com/I need psychology data",
  "corrected_url": "https://kaggle.com/I need psychology data",  // No correction
  "correction_applied": false
}
```

**Diagnosis**:
```python
# Test URL correction manually
validator = URLValidator()
corrected = validator.correct_external_source_url(
    source="kaggle", 
    query="psychology data", 
    current_url="broken_url"
)
print(f"Corrected URL: {corrected}")
```

**Solutions**:

1. **Update URL Patterns**
   ```python
   # In URLValidator._initialize_external_patterns
   'kaggle': {
       'search_pattern': 'https://www.kaggle.com/datasets?search={query}',
       'fallback_url': 'https://www.kaggle.com/datasets'
   }
   ```

2. **Add Source-Specific Fixes**
   ```python
   # Add new source correction method
   def _fix_new_source_url(self, url: str, query: str) -> str:
       # Custom correction logic
       return corrected_url
   ```

3. **Check Query Cleaning**
   ```python
   # Verify query cleaning works
   clean_query = validator._clean_query_for_url("I need some data")
   print(f"Clean query: {clean_query}")
   ```

## Source Coverage and Routing Issues

### Insufficient Source Coverage

**Problem**: Queries return fewer than 3 sources

**Symptoms**:
```json
{
  "routing_summary": {
    "sources_returned": 1,
    "minimum_sources_met": false
  }
}
```

**Solutions**:

1. **Check Source Availability**
   ```python
   # Test individual sources
   from src.ai.web_search_engine import WebSearchEngine
   
   engine = WebSearchEngine(config)
   results = await engine._search_kaggle_datasets("test query")
   print(f"Kaggle results: {len(results)}")
   ```

2. **Add Fallback Sources**
   ```python
   # In source routing logic
   fallback_sources = ['zenodo', 'huggingface', 'aws_open_data']
   if len(primary_results) < 3:
       # Try fallback sources
   ```

3. **Lower Quality Thresholds**
   ```json
   {
     "query": "your query",
     "filters": {
       "quality_threshold": 0.5  // Lower from 0.7
     }
   }
   ```

### Singapore-First Not Working

**Problem**: Singapore sources not being prioritized for local queries

**Symptoms**:
```json
{
  "query": "HDB housing data",
  "routing_summary": {
    "singapore_first_applied": false
  }
}
```

**Diagnosis**:
```python
# Check Singapore context detection
result = await processor.process_input("HDB housing data")
print(f"Singapore context: {result.requires_singapore_context}")
print(f"Detected domain: {result.detected_domain}")
```

**Solutions**:

1. **Update Singapore Keywords**
   ```python
   # Add more Singapore-specific terms
   singapore_keywords = {
       'housing': ['hdb', 'bto', 'condo', 'ec', 'landed'],
       'transport': ['mrt', 'lrt', 'bus', 'taxi', 'grab', 'gojek']
   }
   ```

2. **Check Query Processing**
   ```python
   # Ensure Singapore context is detected
   if any(sg_term in query.lower() for sg_term in ['hdb', 'mrt', 'singapore']):
       apply_singapore_first = True
   ```

### Domain Routing Issues

**Problem**: Queries not routed to appropriate domain-specific sources

**Symptoms**:
```json
{
  "query": "psychology research datasets",
  "routing_info": {
    "detected_domain": null,
    "primary_sources": ["data.gov.sg"]  // Should be kaggle, zenodo
  }
}
```

**Solutions**:

1. **Update Domain Detection**
   ```python
   # Add domain-specific keywords
   domain_keywords = {
       'psychology': ['psychology', 'mental', 'behavioral', 'cognitive'],
       'climate': ['climate', 'weather', 'environmental', 'temperature'],
       'economics': ['economic', 'gdp', 'financial', 'trade']
   }
   ```

2. **Improve Source Mapping**
   ```python
   # Map domains to appropriate sources
   domain_source_mapping = {
       'psychology': ['kaggle', 'zenodo', 'huggingface'],
       'climate': ['world_bank', 'zenodo', 'aws_open_data'],
       'economics': ['world_bank', 'oecd', 'imf']
   }
   ```

## Performance Issues

### Slow Response Times

**Problem**: API responses are taking too long

**Symptoms**:
```json
{
  "performance": {
    "total_time_ms": 8500,  // > 5000ms is slow
    "conversational_processing_ms": 3200
  }
}
```

**Diagnosis**:
```bash
# Check component performance
curl http://localhost:8000/api/metrics
```

**Solutions**:

1. **Reduce LLM Timeout**
   ```yaml
   # config/ai_config.yml
   conversational_query:
     max_processing_time: 2.0  # Reduce from 3.0
   ```

2. **Enable Caching**
   ```python
   # Ensure caching is enabled
   cache_config = {
       'enable_llm_cache': True,
       'enable_search_cache': True,
       'cache_ttl': 3600
   }
   ```

3. **Optimize URL Validation**
   ```python
   # Reduce validation timeout
   self.timeout = 5  # Reduce from 10 seconds
   
   # Use HEAD requests instead of GET
   async with session.head(url) as response:
       return response.status < 400
   ```

### High Memory Usage

**Problem**: System consuming too much memory

**Solutions**:

1. **Limit Cache Size**
   ```yaml
   # config/ai_config.yml
   cache:
     max_memory_size: 500  # Reduce from 1000
     cleanup_threshold: 0.8
   ```

2. **Reduce Concurrent Validations**
   ```python
   # Limit concurrent URL validations
   semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
   ```

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging

# Enable debug for specific components
logging.getLogger('src.ai.conversational_query_processor').setLevel(logging.DEBUG)
logging.getLogger('src.ai.url_validator').setLevel(logging.DEBUG)
logging.getLogger('src.ai.web_search_engine').setLevel(logging.DEBUG)
```

### Performance Monitoring

```bash
# Check system health
curl http://localhost:8000/api/health?detailed=true

# Get performance metrics
curl http://localhost:8000/api/metrics

# Monitor response times
tail -f logs/production_api.log | grep "response_time"
```

### Component Testing

```python
# Test individual components
from src.ai.conversational_query_processor import ConversationalQueryProcessor
from src.ai.url_validator import URLValidator
from src.ai.web_search_engine import WebSearchEngine

# Test conversational processing
processor = ConversationalQueryProcessor(llm_manager, config)
result = await processor.process_input("test query")

# Test URL validation
validator = URLValidator()
is_valid, status = await validator.validate_url("https://example.com")

# Test web search
engine = WebSearchEngine(config)
results = await engine.search_datasets("test query")
```

## Common Error Messages

### "Neural model not available"
```bash
# Check if model file exists
ls -la models/dl/quality_first/best_quality_model.pt

# Retrain model if missing
python dl_pipeline.py
```

### "LLM client timeout"
```bash
# Check API keys and connectivity
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Test LLM connectivity
python -c "from src.ai.llm_clients import LLMManager; print('LLM OK')"
```

### "Cache database locked"
```bash
# Remove lock files
rm cache/*/cache_metadata.db-wal
rm cache/*/cache_metadata.db-shm

# Restart server
python start_server.py
```

## Getting Help

### Log Analysis
```bash
# Check recent errors
tail -100 logs/production_api.log | grep ERROR

# Monitor real-time logs
tail -f logs/production_api.log
```

### System Information
```bash
# Get system status
curl http://localhost:8000/api/health

# Check component versions
python -c "import sys; print(f'Python: {sys.version}')"
pip list | grep -E "(aiohttp|fastapi|torch)"
```

### Support Channels
- **Documentation**: Check `docs/` directory for detailed guides
- **GitHub Issues**: Report bugs and feature requests
- **Logs**: Always include relevant log excerpts when reporting issues

This troubleshooting guide covers the most common issues encountered with the search quality improvements. For additional help, check the component-specific documentation and enable debug logging for detailed diagnostics.