"""
Health Monitoring and Performance Tracking
Real-time monitoring for production deployment
"""
import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import requests

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics data structure."""
    timestamp: str
    status: str
    response_time: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    cache_hit_rate: float
    error_rate: float
    requests_per_minute: int


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: str
    level: str  # INFO, WARNING, CRITICAL
    metric: str
    value: float
    threshold: float
    message: str


class HealthMonitor:
    """
    Comprehensive health monitoring for production deployment.
    Tracks performance metrics and alerts on issues.
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 check_interval: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        
        self.api_url = api_url
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "response_time": 5.0,  # seconds
            "cpu_usage": 80.0,     # percentage
            "memory_usage": 85.0,  # percentage
            "disk_usage": 90.0,    # percentage
            "error_rate": 5.0,     # percentage
            "cache_hit_rate": 50.0 # minimum percentage
        }
        
        # Metrics storage
        self.metrics_history: List[HealthMetrics] = []
        self.alerts_history: List[PerformanceAlert] = []
        self.max_history = 1440  # 24 hours at 1-minute intervals
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.last_minute_requests = []
        self.last_minute_errors = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup health monitoring logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Health monitor specific logger
        self.health_logger = logging.getLogger("health_monitor")
        handler = logging.FileHandler(log_dir / "health_monitor.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.health_logger.addHandler(handler)
        self.health_logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start the health monitoring in a separate thread."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"🩺 Health monitoring started (interval: {self.check_interval}s)")
        self.health_logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Health monitoring stopped")
        self.health_logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                if metrics:
                    # Store metrics
                    self.metrics_history.append(metrics)
                    
                    # Trim history if needed
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history = self.metrics_history[-self.max_history:]
                    
                    # Check for alerts
                    self._check_alerts(metrics)
                    
                    # Log metrics
                    self._log_metrics(metrics)
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> Optional[HealthMetrics]:
        """Collect comprehensive health metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # API health check
            api_status, response_time = self._check_api_health()
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections (approximate)
            connections = len(psutil.net_connections(kind='inet'))
            
            # Calculate rates
            current_time = time.time()
            
            # Clean old entries (older than 1 minute)
            self.last_minute_requests = [t for t in self.last_minute_requests if current_time - t < 60]
            self.last_minute_errors = [t for t in self.last_minute_errors if current_time - t < 60]
            
            requests_per_minute = len(self.last_minute_requests)
            errors_per_minute = len(self.last_minute_errors)
            error_rate = (errors_per_minute / max(requests_per_minute, 1)) * 100
            
            # Get cache metrics if available
            cache_hit_rate = self._get_cache_hit_rate()
            
            return HealthMetrics(
                timestamp=timestamp,
                status=api_status,
                response_time=response_time,
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                active_connections=connections,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                requests_per_minute=requests_per_minute
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def _check_api_health(self) -> tuple[str, float]:
        """Check API health and response time."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/api/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return "healthy", response_time
            else:
                return "unhealthy", response_time
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"API health check failed: {e}")
            return "unavailable", 0.0
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate from API metrics."""
        try:
            response = requests.get(f"{self.api_url}/api/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()
                cache_stats = data.get("cache_statistics", {})
                search_cache = cache_stats.get("search_cache", {})
                return search_cache.get("hit_rate", 0.0) * 100
                
        except Exception as e:
            logger.debug(f"Could not get cache metrics: {e}")
        
        return 0.0
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Response time alert
        if metrics.response_time > self.alert_thresholds["response_time"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="WARNING",
                metric="response_time",
                value=metrics.response_time,
                threshold=self.alert_thresholds["response_time"],
                message=f"High response time: {metrics.response_time:.2f}s"
            ))
        
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            level = "CRITICAL" if metrics.cpu_usage > 95 else "WARNING"
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level=level,
                metric="cpu_usage",
                value=metrics.cpu_usage,
                threshold=self.alert_thresholds["cpu_usage"],
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%"
            ))
        
        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            level = "CRITICAL" if metrics.memory_usage > 95 else "WARNING"
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level=level,
                metric="memory_usage",
                value=metrics.memory_usage,
                threshold=self.alert_thresholds["memory_usage"],
                message=f"High memory usage: {metrics.memory_usage:.1f}%"
            ))
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            level = "CRITICAL" if metrics.error_rate > 20 else "WARNING"
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level=level,
                metric="error_rate",
                value=metrics.error_rate,
                threshold=self.alert_thresholds["error_rate"],
                message=f"High error rate: {metrics.error_rate:.1f}%"
            ))
        
        # Cache hit rate alert (low hit rate)
        if metrics.cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="WARNING",
                metric="cache_hit_rate",
                value=metrics.cache_hit_rate,
                threshold=self.alert_thresholds["cache_hit_rate"],
                message=f"Low cache hit rate: {metrics.cache_hit_rate:.1f}%"
            ))
        
        # Status alert
        if metrics.status != "healthy":
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="CRITICAL",
                metric="api_status",
                value=0.0,
                threshold=1.0,
                message=f"API status: {metrics.status}"
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process and log an alert."""
        self.alerts_history.append(alert)
        
        # Trim alert history
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-100:]
        
        # Log alert
        log_method = {
            "INFO": self.health_logger.info,
            "WARNING": self.health_logger.warning,
            "CRITICAL": self.health_logger.error
        }.get(alert.level, self.health_logger.info)
        
        log_method(f"ALERT [{alert.level}] {alert.metric}: {alert.message}")
        
        # Console output for critical alerts
        if alert.level == "CRITICAL":
            logger.error(f"🚨 CRITICAL ALERT: {alert.message}")
    
    def _log_metrics(self, metrics: HealthMetrics):
        """Log metrics to file."""
        self.health_logger.info(
            f"Status: {metrics.status}, "
            f"Response: {metrics.response_time:.3f}s, "
            f"CPU: {metrics.cpu_usage:.1f}%, "
            f"Memory: {metrics.memory_usage:.1f}%, "
            f"Cache: {metrics.cache_hit_rate:.1f}%, "
            f"Requests/min: {metrics.requests_per_minute}, "
            f"Error rate: {metrics.error_rate:.1f}%"
        )
    
    def record_request(self):
        """Record a request for metrics calculation."""
        self.request_count += 1
        self.last_minute_requests.append(time.time())
    
    def record_error(self):
        """Record an error for metrics calculation."""
        self.error_count += 1
        self.last_minute_errors.append(time.time())
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-60:]  # Last hour
        
        if not recent_metrics:
            return {}
        
        return {
            "period": "last_hour",
            "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            "total_requests": sum(m.requests_per_minute for m in recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "uptime_percentage": (len([m for m in recent_metrics if m.status == "healthy"]) / len(recent_metrics)) * 100
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[PerformanceAlert]:
        """Get recent alerts."""
        return self.alerts_history[-count:] if self.alerts_history else []
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_count": len(self.metrics_history),
            "alerts_count": len(self.alerts_history),
            "metrics": [asdict(m) for m in self.metrics_history],
            "alerts": [asdict(a) for a in self.alerts_history],
            "summary": self.get_metrics_summary()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Metrics exported to {file_path}")


def create_health_monitor(api_url: str = "http://localhost:8000", 
                         check_interval: int = 30) -> HealthMonitor:
    """Create and configure a health monitor."""
    return HealthMonitor(api_url=api_url, check_interval=check_interval)


if __name__ == "__main__":
    # Demo health monitoring
    monitor = create_health_monitor()
    monitor.start_monitoring()
    
    try:
        logger.info("Health monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            current = monitor.get_current_metrics()
            if current:
                print(f"Status: {current.status}, Response: {current.response_time:.3f}s, "
                      f"CPU: {current.cpu_usage:.1f}%, Memory: {current.memory_usage:.1f}%")
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logger.info("Health monitoring stopped")