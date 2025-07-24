# Project Organization Plan

## 🎯 Goal: Clean up files while keeping system running

## 📁 Safe to Move/Organize:
1. **Log files** → `logs/archive/`
2. **Old test files** → `tests/archive/`
3. **Temporary screenshots** → `docs/screenshots/`
4. **Output files** → Better organization in `outputs/`
5. **Development scripts** → `scripts/development/`

## ⚠️ DO NOT TOUCH (Keep Running System):
- `Frontend/` (frontend server running)
- `src/` (backend code)
- `start_server.py` (backend running)
- `config/` (active configuration)
- `data/processed/` (active data)

## 📋 Organization Structure:
```
├── src/                    # Core application (KEEP AS-IS)
├── Frontend/               # Web interface (KEEP AS-IS)
├── config/                 # Configuration (KEEP AS-IS)
├── data/                   # Data files (KEEP AS-IS)
├── logs/
│   ├── archive/           # OLD LOG FILES → HERE
│   └── current/           # Keep active logs
├── tests/
│   ├── archive/           # OLD TEST FILES → HERE
│   └── current/           # Keep active tests
├── docs/
│   ├── screenshots/       # SCREENSHOTS → HERE
│   └── current/           # Keep current docs
├── scripts/
│   ├── development/       # DEV SCRIPTS → HERE
│   └── production/        # Keep production scripts
└── outputs/               # Clean organization
```