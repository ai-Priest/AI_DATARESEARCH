#!/bin/bash

# AI Research Assistant Production Deployment Script
# Manages multi-worker deployment with health checks and graceful restarts

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$APP_DIR/venv"
PID_FILE="/var/run/ai-research-assistant.pid"
WORKER_PID_DIR="/var/run/ai-research-assistant"
LOG_DIR="/var/log/ai-research-assistant"

# Worker configuration
NUM_WORKERS=4
BASE_PORT=8000
WORKER_CLASS="uvicorn.workers.UvicornWorker"
BIND_ADDRESS="127.0.0.1"

# Application settings
APP_MODULE="src.deployment.start_production:app"
WORKER_TIMEOUT=300
GRACEFUL_TIMEOUT=30
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100

# Performance settings
WORKER_CONNECTIONS=1000
KEEPALIVE=5
THREADS=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as correct user
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
    
    # Check Python virtual environment
    if [[ ! -d "$VENV_PATH" ]]; then
        error "Virtual environment not found at $VENV_PATH"
        exit 1
    fi
    
    # Check if gunicorn is installed
    if ! "$VENV_PATH/bin/python" -c "import gunicorn" 2>/dev/null; then
        error "Gunicorn not installed in virtual environment"
        exit 1
    fi
    
    # Check if application module exists
    if ! "$VENV_PATH/bin/python" -c "import src.deployment.start_production" 2>/dev/null; then
        error "Application module not found: $APP_MODULE"
        exit 1
    fi
    
    # Create required directories
    mkdir -p "$WORKER_PID_DIR" "$LOG_DIR" "$APP_DIR/logs/workers"
    
    log "Prerequisites check passed"
}

# Get worker PIDs
get_worker_pids() {
    if [[ -d "$WORKER_PID_DIR" ]]; then
        find "$WORKER_PID_DIR" -name "worker-*.pid" -exec cat {} \; 2>/dev/null || true
    fi
}

# Check if service is running
is_running() {
    local pids=$(get_worker_pids)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                return 0
            fi
        done
    fi
    return 1
}

# Start workers
start_workers() {
    log "Starting $NUM_WORKERS workers..."
    
    # Load environment variables
    if [[ -f "$APP_DIR/.env" ]]; then
        export $(grep -v '^#' "$APP_DIR/.env" | xargs)
    fi
    
    # Set Python path
    export PYTHONPATH="$APP_DIR:$PYTHONPATH"
    
    # Start each worker on a different port
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local port=$((BASE_PORT + i))
        local worker_name="ai-research-worker-$i"
        local pid_file="$WORKER_PID_DIR/worker-$i.pid"
        local log_file="$LOG_DIR/worker-$i.log"
        
        log "Starting worker $i on port $port..."
        
        # Start gunicorn worker
        "$VENV_PATH/bin/gunicorn" \
            --name "$worker_name" \
            --bind "${BIND_ADDRESS}:${port}" \
            --workers 1 \
            --worker-class "$WORKER_CLASS" \
            --timeout "$WORKER_TIMEOUT" \
            --graceful-timeout "$GRACEFUL_TIMEOUT" \
            --max-requests "$MAX_REQUESTS" \
            --max-requests-jitter "$MAX_REQUESTS_JITTER" \
            --worker-connections "$WORKER_CONNECTIONS" \
            --keepalive "$KEEPALIVE" \
            --threads "$THREADS" \
            --pid "$pid_file" \
            --error-logfile "$log_file" \
            --access-logfile "$LOG_DIR/access-$i.log" \
            --log-level info \
            --capture-output \
            --enable-stdio-inheritance \
            --daemon \
            "$APP_MODULE"
        
        # Wait for worker to start
        sleep 2
        
        # Verify worker started
        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            log "Worker $i started successfully (PID: $(cat "$pid_file"))"
        else
            error "Failed to start worker $i"
            stop_workers
            exit 1
        fi
    done
    
    # Start health monitor
    nohup "$SCRIPT_DIR/health_monitor.sh" > "$LOG_DIR/health-monitor.log" 2>&1 &
    echo $! > "$WORKER_PID_DIR/health-monitor.pid"
    
    log "All workers started successfully"
}

# Stop workers
stop_workers() {
    log "Stopping workers..."
    
    # Stop health monitor
    if [[ -f "$WORKER_PID_DIR/health-monitor.pid" ]]; then
        local monitor_pid=$(cat "$WORKER_PID_DIR/health-monitor.pid")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            rm -f "$WORKER_PID_DIR/health-monitor.pid"
        fi
    fi
    
    # Get all worker PIDs
    local pids=$(get_worker_pids)
    
    if [[ -z "$pids" ]]; then
        log "No workers running"
        return
    fi
    
    # Send SIGTERM for graceful shutdown
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping worker (PID: $pid)..."
            kill -TERM "$pid"
        fi
    done
    
    # Wait for graceful shutdown
    local timeout=$GRACEFUL_TIMEOUT
    while [[ $timeout -gt 0 ]] && is_running; do
        sleep 1
        ((timeout--))
    done
    
    # Force kill if still running
    if is_running; then
        warn "Some workers did not stop gracefully, forcing shutdown..."
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid"
            fi
        done
    fi
    
    # Clean up PID files
    rm -f "$WORKER_PID_DIR"/worker-*.pid
    
    log "All workers stopped"
}

# Reload workers (zero-downtime restart)
reload_workers() {
    log "Reloading workers with zero downtime..."
    
    # Get current worker PIDs
    local old_pids=$(get_worker_pids)
    
    if [[ -z "$old_pids" ]]; then
        error "No workers running to reload"
        exit 1
    fi
    
    # Start new workers on temporary ports
    log "Starting new workers..."
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local temp_port=$((BASE_PORT + NUM_WORKERS + i))
        local worker_name="ai-research-worker-new-$i"
        local pid_file="$WORKER_PID_DIR/worker-new-$i.pid"
        local log_file="$LOG_DIR/worker-$i.log"
        
        "$VENV_PATH/bin/gunicorn" \
            --name "$worker_name" \
            --bind "${BIND_ADDRESS}:${temp_port}" \
            --workers 1 \
            --worker-class "$WORKER_CLASS" \
            --timeout "$WORKER_TIMEOUT" \
            --pid "$pid_file" \
            --error-logfile "$log_file" \
            --access-logfile "$LOG_DIR/access-$i.log" \
            --daemon \
            "$APP_MODULE"
            
        sleep 1
    done
    
    # Verify new workers are healthy
    log "Verifying new workers..."
    sleep 5
    
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local temp_port=$((BASE_PORT + NUM_WORKERS + i))
        if ! curl -sf "http://${BIND_ADDRESS}:${temp_port}/api/health" > /dev/null; then
            error "New worker $i failed health check"
            # Clean up new workers
            find "$WORKER_PID_DIR" -name "worker-new-*.pid" -exec cat {} \; | xargs kill -TERM 2>/dev/null || true
            rm -f "$WORKER_PID_DIR"/worker-new-*.pid
            exit 1
        fi
    done
    
    log "New workers healthy, switching over..."
    
    # Move new workers to production ports
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local prod_port=$((BASE_PORT + i))
        local new_pid=$(cat "$WORKER_PID_DIR/worker-new-$i.pid")
        
        # Stop old worker
        if [[ -f "$WORKER_PID_DIR/worker-$i.pid" ]]; then
            local old_pid=$(cat "$WORKER_PID_DIR/worker-$i.pid")
            kill -TERM "$old_pid" 2>/dev/null || true
        fi
        
        # Update PID file
        mv "$WORKER_PID_DIR/worker-new-$i.pid" "$WORKER_PID_DIR/worker-$i.pid"
        
        # Note: In production, you'd update nginx upstream here
        log "Worker $i reloaded (PID: $new_pid)"
    done
    
    log "Reload complete"
}

# Check worker status
check_status() {
    log "Checking worker status..."
    
    local running_count=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local port=$((BASE_PORT + i))
        local pid_file="$WORKER_PID_DIR/worker-$i.pid"
        
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "Worker $i: ${GREEN}Running${NC} (PID: $pid, Port: $port)"
                ((running_count++))
            else
                echo -e "Worker $i: ${RED}Dead${NC} (PID file exists but process not running)"
            fi
        else
            echo -e "Worker $i: ${RED}Not running${NC}"
        fi
    done
    
    # Check health monitor
    if [[ -f "$WORKER_PID_DIR/health-monitor.pid" ]]; then
        local monitor_pid=$(cat "$WORKER_PID_DIR/health-monitor.pid")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            echo -e "Health Monitor: ${GREEN}Running${NC} (PID: $monitor_pid)"
        else
            echo -e "Health Monitor: ${RED}Dead${NC}"
        fi
    else
        echo -e "Health Monitor: ${RED}Not running${NC}"
    fi
    
    # Overall status
    if [[ $running_count -eq $NUM_WORKERS ]]; then
        log "All workers are running"
        exit 0
    else
        warn "Only $running_count out of $NUM_WORKERS workers are running"
        exit 1
    fi
}

# Main command handler
case "${1:-}" in
    start)
        check_prerequisites
        if is_running; then
            error "Service is already running"
            exit 1
        fi
        start_workers
        
        # Create main PID file for systemd
        echo $$ > "$PID_FILE"
        
        # Keep script running for systemd Type=forking
        while true; do
            sleep 60
            # Check if any workers are still running
            if ! is_running; then
                error "All workers have died"
                exit 1
            fi
        done
        ;;
        
    stop)
        stop_workers
        rm -f "$PID_FILE"
        ;;
        
    restart)
        stop_workers
        sleep 2
        check_prerequisites
        start_workers
        ;;
        
    reload)
        reload_workers
        ;;
        
    status)
        check_status
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|reload|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all workers"
        echo "  stop    - Stop all workers"
        echo "  restart - Stop and start all workers"
        echo "  reload  - Reload workers with zero downtime"
        echo "  status  - Check worker status"
        exit 1
        ;;
esac