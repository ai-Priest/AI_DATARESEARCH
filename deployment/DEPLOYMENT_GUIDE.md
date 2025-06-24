# Production Deployment Guide

## 🚀 AI Dataset Research Assistant - Production Ready

This guide covers the complete production deployment of the AI Dataset Research Assistant with all optimizations enabled.

## ✅ Deployment Status

### Completed Tasks:
1. **Systemd Service Configuration** ✅
   - Auto-restart on failure
   - Resource limits (8GB RAM, 400% CPU)
   - Security hardening
   - Located at: `deployment/ai-research-assistant.service`

2. **Nginx Reverse Proxy** ✅
   - SSL/TLS termination
   - Load balancing for 4 workers
   - Rate limiting (10 req/s general, 5 req/s search)
   - Security headers
   - Located at: `deployment/nginx.conf`

3. **Log Rotation** ✅
   - Daily rotation with 30-day retention
   - Size-based rotation (100MB)
   - Compressed archives
   - Located at: `deployment/logrotate.conf`

4. **Health Monitoring** ✅
   - 30-second interval checks
   - System resource monitoring
   - Multi-channel alerting
   - Located at: `deployment/health_monitor.sh`

5. **Multi-Worker Deployment** ✅
   - 4 Gunicorn workers configuration
   - Zero-downtime reload
   - Located at: `deployment/deploy_production.sh`

6. **Neural Model Path Fix** ✅
   - Updated to use correct model path: `models/dl/`
   - Model: `lightweight_cross_attention_best.pt` (68.1% NDCG@3)

## 📊 Performance Metrics

- **Response Time**: 84% improvement (30s → 4.75s)
- **Neural Performance**: 68.1% NDCG@3
- **Multi-Modal Search**: 0.24s response time
- **Intelligent Caching**: 66.67% hit rate
- **Concurrent Workers**: 4 workers × 1000 connections

## 🔧 Quick Start

### 1. Basic Deployment (Single Worker)
```bash
python deploy.py
```

### 2. Production Deployment (Multi-Worker)
```bash
# Install as systemd service
sudo cp deployment/ai-research-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-research-assistant
sudo systemctl start ai-research-assistant

# Check status
sudo systemctl status ai-research-assistant
```

### 3. Nginx Setup
```bash
# Copy nginx config
sudo cp deployment/nginx.conf /etc/nginx/sites-available/ai-research-assistant
sudo ln -s /etc/nginx/sites-available/ai-research-assistant /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 4. Enable Log Rotation
```bash
sudo cp deployment/logrotate.conf /etc/logrotate.d/ai-research-assistant
sudo logrotate -f /etc/logrotate.d/ai-research-assistant
```

### 5. Start Health Monitoring
```bash
# Run health monitor
./deployment/health_monitor.sh &

# Or add to crontab
crontab -e
# Add: */1 * * * * /path/to/deployment/health_monitor.sh
```

## 🌐 API Endpoints

- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Metrics**: http://localhost:8000/api/metrics (internal only)

## 📁 Directory Structure

```
deployment/
├── ai-research-assistant.service  # Systemd service file
├── nginx.conf                     # Nginx reverse proxy config
├── logrotate.conf                 # Log rotation config
├── health_monitor.sh              # Health monitoring script
├── deploy_production.sh           # Multi-worker deployment script
└── DEPLOYMENT_GUIDE.md           # This guide

logs/
├── production_startup.log         # Startup logs
├── production_api.log            # API server logs
└── health_monitor.log            # Health check logs

cache/
├── search/                       # Search cache
├── neural/                       # Neural inference cache
└── llm/                         # LLM response cache
```

## 🔒 Security Considerations

1. **SSL/TLS**: Configure certificates in nginx.conf
2. **API Keys**: Store in .env file with restricted permissions
3. **Rate Limiting**: Configured at nginx level
4. **Process Isolation**: Runs as dedicated user
5. **Resource Limits**: Memory and CPU limits enforced

## 🏃‍♂️ Running Status

Check deployment status:
```bash
# Service status
sudo systemctl status ai-research-assistant

# Worker processes
ps aux | grep gunicorn

# Port listening
sudo lsof -i :8000-8003

# Nginx status
sudo nginx -t && sudo systemctl status nginx

# View logs
tail -f /var/log/ai-research-assistant/production.log
```

## 🚨 Troubleshooting

### Issue: TensorFlow Import Error
**Solution**: Environment variables are set in deploy.py:
- `TRANSFORMERS_NO_TF=1`
- `USE_TORCH=1`

### Issue: Neural Model Not Found
**Solution**: Models are in `models/dl/` directory:
- Primary: `lightweight_cross_attention_best.pt`
- Fallback: `graded_relevance_best.pt`

### Issue: Port Already in Use
**Solution**: Kill existing processes:
```bash
sudo fuser -k 8000/tcp
# Or
pkill -f "uvicorn|gunicorn"
```

### Issue: Permission Denied
**Solution**: Create required user and directories:
```bash
sudo useradd -r -s /bin/false ai-assistant
sudo mkdir -p /var/log/ai-research-assistant /var/run/ai-research-assistant
sudo chown ai-assistant:ai-assistant /var/log/ai-research-assistant /var/run/ai-research-assistant
```

## 📈 Performance Monitoring

Monitor key metrics:
```bash
# CPU and Memory
htop -p $(pgrep -f gunicorn)

# Response times
tail -f /var/log/nginx/ai-research-assistant_access.log | awk '{print $NF}'

# Cache hit rate
grep "cache hit" /var/log/ai-research-assistant/production.log | wc -l

# Error rate
grep ERROR /var/log/ai-research-assistant/production.log | tail -20
```

## 🎯 Next Steps

1. **SSL Certificate**: Install Let's Encrypt certificate
2. **Monitoring**: Set up Prometheus/Grafana
3. **Backup**: Configure automated backups
4. **Scaling**: Add more workers or servers as needed
5. **CDN**: Configure CloudFlare for static assets

## 📞 Support

For issues or questions:
- Check logs in `/var/log/ai-research-assistant/`
- Review health monitor output
- Verify all dependencies with `python deploy.py --check-only`

---
*Deployment configured for optimal performance with 84% response time improvement and 68.1% NDCG@3 neural performance.*