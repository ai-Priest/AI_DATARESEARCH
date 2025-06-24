#!/bin/bash

# AI Research Assistant Health Monitoring Script
# Performs continuous health checks and alerts on issues

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/ai-research-assistant"
HEALTH_LOG="$LOG_DIR/health-monitor.log"
ALERT_EMAIL="ops@yourdomain.com"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Health check configuration
HEALTH_CHECK_INTERVAL=30  # seconds
FAILURE_THRESHOLD=3       # consecutive failures before alert
RECOVERY_THRESHOLD=2      # consecutive successes before recovery alert

# API endpoints to check
API_BASE_URL="http://localhost:8000"
HEALTH_ENDPOINTS=(
    "/api/health"
    "/api/metrics"
)

# Performance thresholds
MAX_RESPONSE_TIME=5000    # milliseconds
MAX_CPU_USAGE=80         # percentage
MAX_MEMORY_USAGE=80      # percentage
MIN_DISK_SPACE=10        # GB

# State tracking
FAILURE_COUNT=0
LAST_ALERT_TIME=0
ALERT_COOLDOWN=300       # 5 minutes between alerts

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$HEALTH_LOG"
}

# Error handling
error() {
    log "ERROR: $*"
    send_alert "ERROR" "$*"
    exit 1
}

# Send alert via multiple channels
send_alert() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check cooldown
    local current_time=$(date +%s)
    if [[ $((current_time - LAST_ALERT_TIME)) -lt $ALERT_COOLDOWN ]]; then
        log "Alert suppressed due to cooldown"
        return
    fi
    
    LAST_ALERT_TIME=$current_time
    
    # Email alert
    if command -v mail >/dev/null 2>&1; then
        echo -e "Subject: AI Research Assistant Health Alert - $level\n\nTime: $timestamp\nLevel: $level\nMessage: $message\n\nHost: $(hostname)\nService: AI Research Assistant" | \
            mail -s "AI Research Assistant Alert: $level" "$ALERT_EMAIL" || log "Failed to send email alert"
    fi
    
    # Slack alert
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 *AI Research Assistant Alert*\n*Level:* $level\n*Message:* $message\n*Host:* $(hostname)\n*Time:* $timestamp\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || log "Failed to send Slack alert"
    fi
    
    # Log to syslog
    logger -p user.warning -t "ai-research-health" "$level: $message"
}

# Check API endpoint health
check_api_health() {
    local endpoint="$1"
    local url="${API_BASE_URL}${endpoint}"
    local start_time=$(date +%s%N)
    
    # Make request with timeout
    local response=$(curl -s -w "\n%{http_code}\n%{time_total}" -o /tmp/health_response.txt "$url" --max-time 10 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n2 | head -n1)
    local response_time=$(echo "$response" | tail -n1)
    
    # Calculate response time in milliseconds
    local end_time=$(date +%s%N)
    local duration=$((($end_time - $start_time) / 1000000))
    
    # Check HTTP status code
    if [[ "$http_code" != "200" ]]; then
        log "FAIL: $endpoint returned HTTP $http_code"
        return 1
    fi
    
    # Check response time
    if [[ $duration -gt $MAX_RESPONSE_TIME ]]; then
        log "WARN: $endpoint slow response: ${duration}ms"
    fi
    
    # Parse health check response for /api/health
    if [[ "$endpoint" == "/api/health" ]] && [[ -f /tmp/health_response.txt ]]; then
        local status=$(grep -o '"status":"[^"]*"' /tmp/health_response.txt | cut -d'"' -f4)
        if [[ "$status" != "healthy" ]]; then
            log "FAIL: $endpoint status is $status"
            return 1
        fi
    fi
    
    log "OK: $endpoint (${duration}ms)"
    return 0
}

# Check system resources
check_system_resources() {
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)
    if [[ $cpu_usage -gt $MAX_CPU_USAGE ]]; then
        log "WARN: High CPU usage: ${cpu_usage}%"
        send_alert "WARNING" "High CPU usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    local mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [[ $mem_usage -gt $MAX_MEMORY_USAGE ]]; then
        log "WARN: High memory usage: ${mem_usage}%"
        send_alert "WARNING" "High memory usage: ${mem_usage}%"
    fi
    
    # Disk space
    local disk_free=$(df -BG /opt/ai-research-assistant | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $disk_free -lt $MIN_DISK_SPACE ]]; then
        log "WARN: Low disk space: ${disk_free}GB free"
        send_alert "WARNING" "Low disk space: ${disk_free}GB free"
    fi
    
    # Check process count
    local worker_count=$(pgrep -f "uvicorn.*ai-research" | wc -l)
    if [[ $worker_count -lt 4 ]]; then
        log "WARN: Only $worker_count workers running (expected 4)"
        send_alert "WARNING" "Only $worker_count workers running"
    fi
}

# Check log files for errors
check_error_logs() {
    local error_count=$(tail -n 1000 "$APP_DIR/logs/errors/error.log" 2>/dev/null | grep -c "ERROR" || echo 0)
    if [[ $error_count -gt 10 ]]; then
        log "WARN: Found $error_count errors in last 1000 log lines"
        local recent_errors=$(tail -n 10 "$APP_DIR/logs/errors/error.log" | head -n 3)
        send_alert "WARNING" "High error rate detected: $error_count errors. Recent: $recent_errors"
    fi
}

# Check model inference performance
check_inference_performance() {
    if [[ -f "$APP_DIR/logs/performance/inference_metrics.json" ]]; then
        local avg_time=$(jq -r '.average_inference_time' "$APP_DIR/logs/performance/inference_metrics.json" 2>/dev/null || echo "0")
        if (( $(echo "$avg_time > 1.0" | bc -l) )); then
            log "WARN: Slow inference time: ${avg_time}s"
            send_alert "WARNING" "Slow model inference: ${avg_time}s average"
        fi
    fi
}

# Main health check loop
run_health_checks() {
    local all_passed=true
    
    # Check each API endpoint
    for endpoint in "${HEALTH_ENDPOINTS[@]}"; do
        if ! check_api_health "$endpoint"; then
            all_passed=false
        fi
    done
    
    # Check system resources
    check_system_resources
    
    # Check logs for errors
    check_error_logs
    
    # Check inference performance
    check_inference_performance
    
    # Update failure count
    if [[ "$all_passed" == "true" ]]; then
        if [[ $FAILURE_COUNT -ge $FAILURE_THRESHOLD ]]; then
            log "Service recovered after $FAILURE_COUNT failures"
            send_alert "RECOVERY" "Service has recovered and is healthy"
        fi
        FAILURE_COUNT=0
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        if [[ $FAILURE_COUNT -ge $FAILURE_THRESHOLD ]]; then
            send_alert "CRITICAL" "Service has failed $FAILURE_COUNT consecutive health checks"
            
            # Attempt automatic recovery
            log "Attempting automatic recovery..."
            if systemctl is-active --quiet ai-research-assistant; then
                systemctl restart ai-research-assistant
                log "Service restarted"
            fi
        fi
    fi
}

# Signal handlers
trap 'log "Health monitor shutting down..."; exit 0' SIGTERM SIGINT

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Main loop
log "Health monitor starting (PID: $$)"
log "Monitoring endpoints: ${HEALTH_ENDPOINTS[*]}"
log "Check interval: ${HEALTH_CHECK_INTERVAL}s"

while true; do
    run_health_checks
    sleep "$HEALTH_CHECK_INTERVAL"
done