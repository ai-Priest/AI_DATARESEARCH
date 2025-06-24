# Installation Guide

This guide provides comprehensive instructions for installing the AI Data Research Assistant with different dependency configurations.

## 🚀 Quick Start (Basic Installation)

### Option 1: Using uv (Recommended)
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd ai-dataresearch

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Option 2: Using pip with requirements.txt
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using pyproject.toml
```bash
# Install the package in development mode
pip install -e .

# Or install specific optional dependencies
pip install -e ".[dev]"              # Development tools
pip install -e ".[production]"       # Production deployment
pip install -e ".[all]"              # All features
```

## 📦 Dependency Categories

### Core Dependencies (Always Installed)
- **Data Processing**: numpy, pandas, scikit-learn, scipy
- **Deep Learning**: torch, transformers, sentence-transformers
- **Visualization**: matplotlib, seaborn, plotly
- **Configuration**: pyyaml, python-dotenv
- **API Framework**: fastapi, uvicorn

### Optional Dependency Groups

#### Development (`requirements-dev.txt`)
```bash
pip install -r requirements-dev.txt
```
Includes:
- Testing: pytest, pytest-cov
- Code Quality: black, isort, flake8, ruff, mypy
- Documentation: sphinx, jupyter
- Debugging: ipython, debugpy

#### Production (`requirements-prod.txt`)
```bash
pip install -r requirements-prod.txt  
```
Includes:
- Web Server: gunicorn, uvicorn
- Database: redis, sqlalchemy, psycopg2
- Monitoring: sentry-sdk, prometheus-client
- Security: cryptography, passlib

#### Optional Features (via pyproject.toml)
```bash
# Cloud storage support
pip install -e ".[cloud]"

# Advanced ML algorithms
pip install -e ".[advanced-ml]"

# NLP processing
pip install -e ".[nlp]"

# Monitoring & logging
pip install -e ".[monitoring]"

# Vector databases
pip install -e ".[vector-db]"

# Everything
pip install -e ".[all]"
```

## 🛠️ Environment Setup

### 1. Create Environment File
```bash
cp env_example.sh .env
```

### 2. Configure API Keys (Optional)
Edit `.env` and add your API keys:
```bash
# Singapore Data APIs (optional)
LTA_API_KEY=your_lta_key
ONEMAP_API_KEY=your_onemap_key
URA_API_KEY=your_ura_key

# AI/LLM APIs (for future AI integration)
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

### 3. Verify Installation
```bash
# Test core pipelines
python data_pipeline.py --validate-only
python ml_pipeline.py --validate-only
python dl_pipeline.py --validate-only

# Run verification script
python scripts/utils/verify_organization.py
```

## 🐳 Docker Installation (Alternative)

### Option 1: Development Container
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Option 2: Production Container
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app"]
```

## 🔧 Development Setup

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Setup Pre-commit Hooks
```bash
pre-commit install
```

### 3. Configure Development Tools
```bash
# Format code
black .
isort .

# Lint code
flake8 .
ruff check .

# Type checking
mypy src/

# Run tests
pytest tests/
```

## 🚀 Production Deployment

### 1. Install Production Dependencies
```bash
pip install -r requirements-prod.txt
```

### 2. Configure Production Environment
```bash
# Set production environment variables
export ENVIRONMENT=production
export SECRET_KEY=your_secret_key
export DATABASE_URL=postgresql://user:pass@localhost/db
export REDIS_URL=redis://localhost:6379
```

### 3. Start Production Server
```bash
# Using Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 main:app

# Using Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Performance Optimization

### GPU Support (Optional)
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (M1/M2)
# PyTorch MPS is automatically available on macOS 12.3+
```

### Memory Optimization
```bash
# For large dataset processing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
```

## 🔍 Troubleshooting

### Common Issues

#### 1. ImportError for transformers/torch
```bash
# Update to latest versions
pip install --upgrade torch transformers
```

#### 2. MPS (Apple Silicon) Issues
```bash
# If MPS errors occur, fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 3. Memory Issues
```bash
# Reduce batch size in config files
# config/dl_config.yml: batch_size: 16 -> 8
# config/ml_config.yml: batch_size: 32 -> 16
```

#### 4. API Connection Issues
```bash
# Verify network connectivity
python data_pipeline.py --validate-only
```

### Getting Help
1. Check logs in `logs/` directory
2. Review configuration files in `config/`
3. Run verification script: `python scripts/utils/verify_organization.py`
4. Check documentation in `docs/` directory

## 📋 Installation Verification Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (basic or specific configuration)
- [ ] Environment variables configured (optional)
- [ ] Core pipelines validate successfully
- [ ] File organization verified
- [ ] Tests pass (if development setup)

## 🎯 Next Steps

After successful installation:

1. **Data Pipeline**: `python data_pipeline.py --validate-only`
2. **ML Training**: `python ml_pipeline.py`
3. **DL Breakthrough**: `python dl_pipeline.py`
4. **Enhanced Training**: `python scripts/enhancement/improved_training_pipeline.py`

See `Readme.md` for detailed usage instructions and `CLAUDE.md` for development guidance.