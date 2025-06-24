# 🚀 API Configuration Setup Guide

## Quick Start (5 minutes)

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Get Required API Keys

| Source | Required | Registration | Time | Cost |
|--------|----------|--------------|------|------|
| **LTA DataMall** | ✅ Required | [datamall.lta.gov.sg](https://datamall.lta.gov.sg/) | 2 min | Free |
| **OneMap SG** | ✅ Required | [onemap.gov.sg](https://www.onemap.gov.sg/) | 3 min | Free |
| **UNdata API** | 🔮 Optional | [undata-api.org](https://www.undata-api.org/) | 5 min | Free |
| **Claude API** | 🔮 Future | [console.anthropic.com](https://console.anthropic.com/) | 2 min | Paid |

### 3. Fill in API Keys
Edit `.env` file:
```bash
LTA_API_KEY=your_actual_lta_key_here
ONEMAP_EMAIL=your_email@example.com
ONEMAP_PASSWORD=your_onemap_password
```

### 4. Test Configuration
```bash
python tests/test_api_config.py
```

### 5. Run Data Pipeline
```bash
python data_pipeline.py
```

---

## 📋 Detailed Registration Steps

### LTA DataMall (Required - 2 minutes)

**What it provides:** Real-time transport data (bus arrivals, traffic, parking)

1. **Visit:** [https://datamall.lta.gov.sg/](https://datamall.lta.gov.sg/)
2. **Click:** "Request for API Access" 
3. **Fill:** Basic info (name, email, purpose: "Academic Research")
4. **Receive:** Instant email with API key
5. **Copy:** API key to `LTA_API_KEY` in `.env`

**✅ Success indicator:** Key format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

---

### OneMap Singapore (Required - 3 minutes)

**What it provides:** Geospatial data, mapping, address search

1. **Visit:** [https://www.onemap.gov.sg/](https://www.onemap.gov.sg/)
2. **Click:** "Sign Up" (top right)
3. **Create:** Account with email verification
4. **Login:** Verify email and complete profile
5. **Copy:** Email and password to `.env` file

**Note:** Some OneMap APIs work without authentication, but advanced features require account.

**✅ Success indicator:** Can login to OneMap dashboard

---

### UNdata API (Optional - 5 minutes)

**What it provides:** Enhanced UN statistics from WHO, FAO, UNESCO, and other organizations

1. **Visit:** [https://www.undata-api.org/](https://www.undata-api.org/)
2. **Click:** "Sign Up" to create account
3. **Verify:** Email verification required
4. **Get Keys:** Copy App ID and App Key from dashboard
5. **Add to .env:** Both `UNDATA_APP_ID` and `UNDATA_APP_KEY`

**Note:** UNdata API is optional - basic UN SDG data works without registration.

**✅ Success indicator:** Receives app_id and app_key in dashboard

---

## 🌍 Public APIs (No Registration)

These sources work immediately without any setup:
- **data.gov.sg** - Singapore government datasets
- **SingStat** - Singapore statistics
- **World Bank** - Global development indicators  
- **IMF DataMapper** - Economic and financial data
- **OECD** - Economic statistics
- **UN SDG API** - Sustainable Development Goals (test version)
- **UNdata** - Works without keys but limited to public datasets

---

## 🧪 Testing Your Setup

### Quick Test (30 seconds)
```bash
python tests/test_api_config.py --verbose
```

### Test Specific Source
```bash
python tests/test_api_config.py --source data_gov_sg
python tests/test_api_config.py --source lta_datamall
python tests/test_api_config.py --source undata_api
```

### Expected Output
```
🚀 Starting API Configuration Validation
📅 Timestamp: 2025-06-20T10:30:00

🇸🇬 Testing Singapore Data Sources...
📊 Testing data_gov_sg...
✅ data_gov_sg: All tests passed

📊 Testing lta_datamall...
🔑 Using API key from LTA_API_KEY
✅ lta_datamall: All tests passed

📊 TEST RESULTS SUMMARY
Total Sources: 11
✅ Successful: 11
❌ Failed: 0
📈 Success Rate: 100.0%
```

---

## ⚠️ Common Issues & Solutions

### "LTA_API_KEY not found"
```bash
# Check if .env file exists and contains the key
cat .env | grep LTA_API_KEY

# If empty, copy your actual key:
echo "LTA_API_KEY=your_actual_key_here" >> .env
```

### "OneMap authentication failed" 
- Verify email/password combination
- Check if account is email-verified
- Try logging into OneMap website first

### "Request timeout" errors
- Check internet connection
- APIs might be temporarily down
- Try again in a few minutes

### "403 Forbidden" errors
- Check API key is correct
- Verify account is active
- Some endpoints require additional permissions

---

## 📊 What Each Source Provides

| Source | Categories | Update Frequency | Use Cases |
|--------|------------|------------------|-----------|
| **data.gov.sg** | Government data, demographics, economics | Monthly/Quarterly | Policy research, demographics |
| **LTA DataMall** | Transport, traffic, public transit | Real-time | Transport analysis, smart city |
| **OneMap** | Geospatial, addresses, planning areas | Real-time | Location analysis, mapping |
| **SingStat** | Official statistics, census, economy | Monthly/Annual | Academic research, economics |
| **World Bank** | Global development, economics, poverty | Annual | International comparisons |
| **IMF** | Financial, monetary, economic indicators | Quarterly | Economic analysis, policy |
| **OECD** | Economic statistics, education, health | Quarterly/Annual | Policy research, benchmarking |
| **UNdata API** | WHO, FAO, UNESCO organizational data | Varies | Health, agriculture, education research |
| **UN SDG API** | Sustainable development goals, progress | Annual | Development research, SDG tracking |

---

## 🔒 Security Best Practices

### Environment Variables
```bash
# ✅ Good: Use environment variables
LTA_API_KEY=${LTA_API_KEY}

# ❌ Bad: Hardcode in scripts
api_key = "abc123-your-key-here"
```

### Git Ignore
Add to `.gitignore`:
```
.env
*.env
config/secrets.yml
```

### Production Deployment
- Use secure secret management (AWS Secrets, Azure Key Vault)
- Rotate API keys regularly
- Monitor API usage for suspicious activity
- Use separate keys for dev/staging/production

---

## 🚀 Next Steps

Once your API configuration is working:

1. **Run Data Pipeline:** `python data_pipeline.py`
2. **Check Results:** Review extracted datasets in `data/processed/`
3. **Train ML Models:** `python train_models.py` 
4. **Build AI Assistant:** Configure Claude API for natural language features

---

## 📞 Getting Help

### API-Specific Support
- **LTA DataMall:** [Support Portal](https://datamall.lta.gov.sg/content/datamall/en/support.html)
- **OneMap:** [API Documentation](https://www.onemap.gov.sg/apidocs/)
- **URA:** Contact through official website
- **Global APIs:** Check respective documentation

### Configuration Issues
1. **Check logs:** `python tests/test_api_config.py --verbose`
2. **Validate YAML:** Use online YAML validator for `api_config.yml`
3. **Test individual sources:** `--source source_name` flag
4. **Review documentation:** Check `README.md` and API documentation links

### Common Solutions
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Reset environment
rm .env
cp .env.example .env
# Fill in your keys again

# Test connectivity
curl -I https://api-production.data.gov.sg/v2/public/api/datasets
```

**Ready to start extracting real data? Run `python data_pipeline.py`** 🎉