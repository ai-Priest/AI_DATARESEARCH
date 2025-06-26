"""
Production Deployment Configuration
Centralized configuration management for production deployment
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    access_log: bool = True
    timeout_keep_alive: int = 5
    max_request_size: int = 16777216  # 16MB


@dataclass 
class PerformanceConfig:
    """Performance optimization settings."""
    enable_caching: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600
    target_response_time: float = 5.0  # seconds
    max_concurrent_requests: int = 100
    request_timeout: int = 30


@dataclass
class SecurityConfig:
    """Security and authentication settings."""
    enable_cors: bool = True
    allowed_origins: list = field(default_factory=lambda: ["*"])
    allowed_methods: list = field(default_factory=lambda: ["*"])
    allowed_headers: list = field(default_factory=lambda: ["*"])
    enable_api_key_auth: bool = False
    api_key_header: str = "X-API-Key"


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enable_metrics: bool = True
    log_file: str = "logs/production_api.log"
    metrics_file: str = "logs/metrics.log"
    health_check_interval: int = 30  # seconds
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True


@dataclass
class AIConfig:
    """AI component configuration."""
    enable_neural_search: bool = True
    enable_multimodal_search: bool = True
    enable_intelligent_caching: bool = True
    enable_llm_integration: bool = True
    fallback_to_basic_search: bool = True
    neural_model_path: str = "models/dl/"
    max_search_results: int = 50


class DeploymentConfig:
    """Centralized deployment configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/deployment.yml"
        
        # Initialize with defaults
        self.server = ServerConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.ai = AIConfig()
        
        # Load configuration from file if exists
        self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self):
        """Load configuration from YAML file."""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations
                if 'server' in config_data:
                    self._update_config(self.server, config_data['server'])
                
                if 'performance' in config_data:
                    self._update_config(self.performance, config_data['performance'])
                
                if 'security' in config_data:
                    self._update_config(self.security, config_data['security'])
                
                if 'monitoring' in config_data:
                    self._update_config(self.monitoring, config_data['monitoring'])
                
                if 'ai' in config_data:
                    self._update_config(self.ai, config_data['ai'])
                    
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Server configuration
        self.server.host = os.getenv("SERVER_HOST", self.server.host)
        self.server.port = int(os.getenv("SERVER_PORT", self.server.port))
        self.server.workers = int(os.getenv("SERVER_WORKERS", self.server.workers))
        self.server.log_level = os.getenv("LOG_LEVEL", self.server.log_level)
        
        # Performance configuration
        self.performance.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.performance.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", self.performance.cache_max_size))
        self.performance.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", self.performance.target_response_time))
        
        # Security configuration
        self.security.enable_cors = os.getenv("ENABLE_CORS", "true").lower() == "true"
        self.security.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
        
        # AI configuration
        self.ai.enable_neural_search = os.getenv("ENABLE_NEURAL_SEARCH", "true").lower() == "true"
        self.ai.enable_multimodal_search = os.getenv("ENABLE_MULTIMODAL_SEARCH", "true").lower() == "true"
        self.ai.enable_llm_integration = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true"
    
    def _update_config(self, config_obj, config_dict):
        """Update configuration object with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "server": self.server.__dict__,
            "performance": self.performance.__dict__,
            "security": self.security.__dict__,
            "monitoring": self.monitoring.__dict__,
            "ai": self.ai.__dict__
        }
    
    def save_to_file(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = Path(file_path or self.config_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate server config
        if self.server.port < 1 or self.server.port > 65535:
            errors.append(f"Invalid port: {self.server.port}")
        
        if self.server.workers < 1:
            errors.append(f"Invalid workers count: {self.server.workers}")
        
        # Validate performance config
        if self.performance.target_response_time <= 0:
            errors.append(f"Invalid target response time: {self.performance.target_response_time}")
        
        if self.performance.cache_max_size < 1:
            errors.append(f"Invalid cache max size: {self.performance.cache_max_size}")
        
        # Validate paths
        if self.ai.enable_neural_search:
            neural_path = Path(self.ai.neural_model_path)
            if not neural_path.exists():
                errors.append(f"Neural model path does not exist: {neural_path}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get configuration for uvicorn server."""
        return {
            "host": self.server.host,
            "port": self.server.port,
            "workers": self.server.workers,
            "reload": self.server.reload,
            "log_level": self.server.log_level,
            "access_log": self.server.access_log,
            "timeout_keep_alive": self.server.timeout_keep_alive,
        }
    
    def get_fastapi_config(self) -> Dict[str, Any]:
        """Get configuration for FastAPI application."""
        return {
            "title": "AI-Powered Dataset Research Assistant",
            "description": "Production API with 84% response time improvement and 75% NDCG@3 performance",
            "version": "2.0.0",
            "docs_url": "/docs" if not self.security.enable_api_key_auth else None,
            "redoc_url": "/redoc" if not self.security.enable_api_key_auth else None,
        }


# Create default deployment configuration
default_config = DeploymentConfig()


def create_deployment_config_file():
    """Create a default deployment configuration file."""
    config = DeploymentConfig()
    
    # Create default configuration
    config_data = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "log_level": "info",
            "access_log": True
        },
        "performance": {
            "enable_caching": True,
            "cache_max_size": 1000,
            "cache_ttl": 3600,
            "target_response_time": 5.0,
            "max_concurrent_requests": 100
        },
        "security": {
            "enable_cors": True,
            "allowed_origins": ["*"],
            "enable_api_key_auth": False
        },
        "monitoring": {
            "enable_metrics": True,
            "log_file": "logs/production_api.log",
            "health_check_interval": 30,
            "enable_performance_tracking": True
        },
        "ai": {
            "enable_neural_search": True,
            "enable_multimodal_search": True,
            "enable_intelligent_caching": True,
            "enable_llm_integration": True,
            "neural_model_path": "models/dl/",
            "max_search_results": 50
        }
    }
    
    # Save to file
    config_path = Path("config/deployment.yml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created deployment configuration: {config_path}")
    return str(config_path)


if __name__ == "__main__":
    # Create default configuration file
    create_deployment_config_file()