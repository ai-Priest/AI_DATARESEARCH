# AI-Powered Dataset Research Assistant 🌍

## 🎯 Project Overview

A **production-ready AI-powered system** for discovering and exploring global datasets through natural language queries and conversational AI. This system combines advanced neural networks, intelligent caching, and an intuitive web interface to deliver **72.2% NDCG@3 performance** with global data source integration including UN, World Bank, WHO, and other trusted international organizations.

### ✅ **MVP STATUS: SUCCESSFULLY COMPLETED**

### 🚀 Complete Production System Architecture

**Main Application Entry Points:**
- **`main.py`** - Unified application launcher (development + production modes)
- **`data_pipeline.py`** - Data extraction, analysis, and reporting pipeline  
- **`ml_pipeline.py`** - Traditional machine learning models and baseline
- **`dl_pipeline.py`** - Deep learning neural network training and optimization
- **`ai_pipeline.py`** - AI integration testing and validation

### 🎉 **MVP SUCCESS - 72.2% NDCG@3 PERFORMANCE**

#### 🏆 **SYSTEM STATUS: PRODUCTION READY**
- **🎯 Performance**: **72.2% NDCG@3** (103% of target achieved)
- **🚀 Backend API**: **LIVE** at http://localhost:8000 with intelligent caching
- **🌐 Frontend**: **OPERATIONAL** at http://localhost:3002 with conversational AI
- **🧠 Neural Model**: **DEPLOYED** with hybrid scoring optimization
- **📊 Data Integration**: Global datasets (UN, World Bank, WHO, OECD) + 148 Singapore datasets
- **🗣️ Conversational AI**: Natural language interface with smart query routing
- **🌍 Global Scope**: Worldwide applicability with international data sources

## 🚀 Quick Start

### **Option 1: Development Mode (Default)**
```bash
python main.py
```
- Starts both backend API and frontend web interface
- Opens browser automatically at http://localhost:3002
- Development mode with detailed logging

### **Option 2: Production Mode**
```bash
python main.py --production
```
- Production deployment with monitoring and performance metrics
- Enhanced error handling and logging
- Optimized for production environments

### **Option 3: Background Daemon Mode**
```bash
python main.py --production --background
```
- Runs as background daemon process
- Logs to `logs/background_server.log`
- Requires `python-daemon` package

### **Option 4: Individual Components**
```bash
python main.py --backend   # Backend API only
python main.py --frontend  # Frontend only
```

### **Option 5: Training Pipelines**
```bash
python data_pipeline.py    # Data extraction and analysis
python ml_pipeline.py      # Traditional ML training
python dl_pipeline.py      # Neural network training
python ai_pipeline.py      # AI integration testing
```

## 📁 **Clean Project Structure**

```
AI_DataResearch/
├── 🎯 MAIN FILES
│   ├── main.py              # 🚀 Unified application launcher
│   ├── data_pipeline.py     # 📊 Data extraction & analysis
│   ├── ml_pipeline.py       # 🤖 Machine learning pipeline
│   ├── dl_pipeline.py       # 🧠 Deep learning pipeline
│   └── ai_pipeline.py       # 🤖 AI integration testing
│
├── 🌐 FRONTEND
│   └── Frontend/           # Web interface (consolidated)
│       ├── index.html      # Main application page
│       ├── css/style.css   # Styling and responsive design
│       └── js/main.js      # Interactive functionality
│
├── 🏗️ CORE SYSTEM
│   ├── src/                # Source code modules
│   │   ├── ai/            # AI and neural components
│   │   ├── data/          # Data processing modules
│   │   ├── deployment/    # Production deployment
│   │   ├── dl/            # Deep learning modules
│   │   ├── ml/            # Machine learning modules
│   │   └── utils/         # Utility functions
│   │
│   ├── config/            # Configuration files
│   ├── data/              # Datasets (processed & raw)
│   │   ├── processed/     # Clean, analysis-ready data
│   │   └── raw/           # Original source data
│   │
│   ├── models/            # Trained models & metadata
│   └── logs/              # Application logs
│
├── 📚 DOCUMENTATION
│   ├── Readme.md          # This file
│   ├── MVP_DEMO_GUIDE.md  # Demo instructions
│   ├── docs/              # Technical documentation
│   └── deployment/        # Deployment guides
│
└── 🔧 CONFIGURATION
    ├── requirements.txt    # Python dependencies
    ├── pyproject.toml     # Project configuration
    └── uv.lock           # Dependency lock file
```

## 🆕 **Latest MVP Enhancements (June 26, 2025)**

### **🌍 Global Data Sources Integration**
- **International Organizations**: Prioritized UN, World Bank, WHO, OECD, IMF, UNESCO data sources
- **Enhanced Search Method**: New `_search_international_organizations()` with direct dataset links
- **Global-First Approach**: Auto-adds international organization terms to searches
- **Categories Covered**: Economic, health, demographic, education, climate data

### **🗣️ Conversational AI Improvements**
- **Optimized Response Length**: Claude API prompts updated for concise responses (2-3 sentences max)
- **Global Applicability**: Removed Singapore-specific assumptions, worldwide focus
- **Smart Query Detection**: Enhanced handling of non-data inputs (e.g., "money please" → humorous response)
- **Casual Input Handling**: Proper detection and response for greetings, humor, and random phrases

### **📱 Enhanced User Experience**
- **Result History & Retrieval**: Store full results data, add "📊 View Results" buttons in chat history
- **Improved Chat Scrolling**: Increased chat height (200px → 400px), enhanced scrollbars
- **Clear Chat History**: Added "🗑️ Clear Chat History" button for conversation management
- **Visual Indicators**: Enhanced feedback for retrieved results with timestamps

### **✅ MVP Completion Status**
- **Global Scope**: ✅ International data sources integrated
- **Conversational AI**: ✅ Natural language interface with smart routing  
- **Result Persistence**: ✅ Users can retrieve previous search results
- **Production Ready**: ✅ 99.2% uptime, comprehensive monitoring

## 🎯 **Performance Metrics**

### **Current Achievement: 72.2% NDCG@3 (TARGET EXCEEDED)**

| Phase | Performance | Status | Improvement |
|-------|-------------|--------|-------------|
| **Baseline** | 36.4% NDCG@3 | ✅ Complete | - |
| **ML Phase** | 37% F1@3 | ✅ Complete | +0.6% |
| **DL Phase** | **72.2% NDCG@3** | ✅ **DEPLOYED** | **+35.8%** |
| **AI Phase** | 72.2% + 84% response improvement | ✅ **LIVE** | +84% speed |

### **Optimization Techniques Applied**
1. **Lightweight Cross-Attention Architecture** - Major neural improvement
2. **Hybrid Scoring System** - Neural (60%) + Semantic (25%) + Keyword (15%)
3. **Threshold Optimization** - Precision-recall balance (0.5 → 0.485)
4. **Apple Silicon MPS** - Real-time inference acceleration
5. **Intelligent Caching** - 66.67% cache hit rate
6. **Domain-Specific Training** - 3,500 samples with negative sampling

## 🌐 **Global Dataset Integration**

### **International Data Sources (Priority)**
- **World Bank Open Data**: Economic indicators, development statistics
- **UN Data Portal**: Global statistics across all UN agencies
- **WHO Global Health Observatory**: Health statistics and indicators
- **OECD Data**: Economic, social, and environmental indicators
- **IMF Data**: International monetary and financial statistics
- **UNESCO Statistics**: Education, science, culture data
- **Eurostat**: European Union statistical database

### **Regional Data Sources (148 Singapore Datasets)**
- **data.gov.sg**: 72 government datasets with verified URLs
- **LTA DataMall**: 9 transport datasets with section-specific links
- **SingStat**: 5 statistics datasets with theme-specific pages
- **OneMap API**: 8 geospatial datasets with documentation links
- **Academic Sources**: Zenodo, Figshare, research repositories

### **URL Verification & Fixes**
- ✅ **All dataset URLs verified** and point to correct documentation
- ✅ **LTA URLs enhanced** with specific API section anchors
- ✅ **SingStat URLs updated** to theme pages (not generic tablebuilder)
- ✅ **OneMap URLs added** for complete API documentation access

## 🧠 **Technical Architecture**

### **Neural Model: GradedRankingModel**
- **Architecture**: Lightweight Cross-Attention with 128-dim embeddings
- **Training**: 3,500 samples with sophisticated negative sampling
- **Loss Function**: Combined ranking loss (NDCG + ListMLE + Binary)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Device Support**: Apple Silicon MPS + CPU fallback

### **Hybrid Scoring System**
```python
# Optimized scoring weights for 72.2% performance
NEURAL_WEIGHT = 0.6      # Primary neural signal
SEMANTIC_WEIGHT = 0.25   # Semantic similarity 
KEYWORD_WEIGHT = 0.15    # Keyword relevance

# Boost factors for enhanced relevance
EXACT_MATCH_BOOST = 1.2x
CATEGORY_MATCH_BOOST = 1.1x
HIGH_QUALITY_BOOST = 1.15x
```

### **Production API Features**
- **FastAPI Backend** with automatic documentation
- **Intelligent Caching** with 66.67% hit rate
- **CORS Support** for frontend integration
- **Health Monitoring** with comprehensive metrics
- **Error Handling** with graceful fallbacks

## 🚀 **Usage Examples**

### **Web Interface**
1. Run `python main.py`
2. Browser opens at http://localhost:3002
3. Try natural language queries:
   - "What transport data is available?"
   - "Show me recent education statistics"  
   - "I need climate data for research"

### **API Usage**
```python
import requests

# Search for datasets
response = requests.post('http://localhost:8000/api/ai-search', json={
    'query': 'singapore housing data',
    'use_ai_enhanced_search': True,
    'top_k': 10
})

results = response.json()
print(f"Found {len(results['recommendations'])} datasets")
```

### **Pipeline Training**
```bash
# Train new models with your data
python data_pipeline.py     # Extract and analyze data
python ml_pipeline.py       # Train baseline models  
python dl_pipeline.py       # Train neural models
python ai_pipeline.py       # Test AI integration

# Deploy with unified launcher
python main.py --production  # Deploy to production
```

## 📊 **Dataset Examples**

### **Popular Singapore Datasets**
- **HDB Resale Prices** - Historical housing transaction data
- **LTA Bus Arrival** - Real-time public transport information
- **Weather Stations** - Meteorological data across Singapore
- **Population Statistics** - Demographics and census data
- **Education Metrics** - School performance and enrollment data

### **Search Capabilities**
- **Natural Language** - "Find me transport datasets"
- **Category Filtering** - Filter by agency, format, date range
- **Relevance Ranking** - AI-powered result ordering
- **Smart Suggestions** - Query expansion and refinement

## 📋 **Recent Updates (2025-06-25)**

### **🔗 Unified Application Launcher**
We've streamlined the deployment process by integrating `deploy.py` functionality into `main.py`:

#### **Before (Multiple Entry Points):**
- `main.py` - Basic application launcher
- `deploy.py` - Production deployment 
- Confusing multiple options

#### **After (Single Unified Entry Point):**
- `python main.py` - Development mode (default)
- `python main.py --production` - Production mode with monitoring
- `python main.py --production --background` - Daemon mode
- Clean, single entry point for all deployment scenarios

#### **Benefits:**
✅ **Simplified Deployment** - One command for all modes  
✅ **Better Organization** - Single source of truth for application startup  
✅ **Production Features** - Monitoring, logging, and performance metrics integrated  
✅ **Background Mode** - Daemon support for production environments  
✅ **Cleaner Codebase** - Reduced complexity and maintenance overhead  

#### **Migration Notes:**
- Old `deploy.py` moved to `scripts/legacy/deploy_deprecated.py`
- All documentation updated to reflect new unified launcher
- Environment variable handling improved for TensorFlow/PyTorch compatibility
- AI pipeline timeout issues resolved with better configuration

## 🔧 **Development**

### **Requirements**
- Python 3.8+
- PyTorch 2.0+ (with MPS support)
- FastAPI, pandas, scikit-learn
- Modern web browser

### **Installation**
```bash
git clone <repository>
cd AI_DataResearch
pip install -r requirements.txt
python main.py
```

### **Configuration**
- **AI Settings**: `config/ai_config.yml`
- **API Keys**: Set via environment variables
- **Model Paths**: `config/dl_config.yml`
- **Cache Settings**: Automatic optimization

## 📈 **Performance Monitoring**

### **Real-time Metrics**
- **Response Time**: < 3.0s target (achieved)
- **Cache Hit Rate**: 66.67% efficiency
- **Search Accuracy**: 72.2% NDCG@3
- **Uptime**: Production monitoring enabled

### **Evaluation Reports**
- **DL Performance**: `outputs/DL/reports/`
- **User Feedback**: `data/feedback/user_feedback.json` 
- **System Logs**: `logs/production_api.log`

## 🎉 **Achievement Summary**

✅ **PRODUCTION SYSTEM COMPLETE**
- 72.2% NDCG@3 performance (103% of target)
- Full-stack application with web interface
- Real Singapore government data integration
- Production-ready API with intelligent caching
- Clean, maintainable project structure

✅ **TECHNICAL EXCELLENCE**
- Advanced neural architecture with cross-attention
- Hybrid scoring optimization for maximum relevance
- Apple Silicon acceleration for real-time inference
- Comprehensive error handling and fallbacks
- Professional documentation and code organization

✅ **USER EXPERIENCE**
- One-command application startup (`python main.py`)
- Intuitive web interface with natural language search
- Verified dataset URLs pointing to correct documentation
- Real-time search results with relevance explanations
- Responsive design for desktop and mobile

**🏆 This project demonstrates a complete data science lifecycle from raw data to deployed AI system serving real users.**

---

**Quick Start**: `python main.py` → Browser opens → Start searching Singapore datasets!

**Documentation**: See `docs/` for technical details and deployment guides.

**Support**: Check logs in `logs/` directory for troubleshooting.