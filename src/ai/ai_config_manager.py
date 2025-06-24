"""
AI Configuration Manager
Handles loading, validation, and management of AI pipeline configuration
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AIConfigManager:
    """
    Manages AI pipeline configuration with validation and hot-reloading
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or 'config/ai_config.yml')
        self.config: Dict[str, Any] = {}
        self.last_modified: Optional[float] = None
        self.validation_errors: List[str] = []
        
        # Load initial configuration
        self.load_config()
        
    def load_config(self) -> bool:
        """
        Load configuration from YAML file
        
        Returns:
            Success status
        """
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate configuration
            if not self.validate_config():
                logger.error(f"Configuration validation failed: {self.validation_errors}")
                return False
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Set last modified time
            self.last_modified = os.path.getmtime(self.config_path)
            
            logger.info("AI configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def validate_config(self) -> bool:
        """
        Validate configuration structure and values
        
        Returns:
            Validation success status
        """
        self.validation_errors = []
        
        # Check required sections
        required_sections = ['ai_pipeline', 'llm_providers', 'neural_integration']
        for section in required_sections:
            if section not in self.config:
                self.validation_errors.append(f"Missing required section: {section}")
        
        # Validate LLM providers
        if 'llm_providers' in self.config:
            self._validate_llm_providers()
        
        # Validate neural integration
        if 'neural_integration' in self.config:
            self._validate_neural_integration()
        
        # Validate API server config
        if 'api_server' in self.config:
            self._validate_api_server()
        
        return len(self.validation_errors) == 0
    
    def _validate_llm_providers(self):
        """Validate LLM provider configurations"""
        providers = self.config.get('llm_providers', {})
        
        for provider_name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                self.validation_errors.append(f"Invalid configuration for provider: {provider_name}")
                continue
            
            # Check required fields
            required_fields = ['enabled', 'model', 'api_base']
            for field in required_fields:
                if field not in provider_config:
                    self.validation_errors.append(f"Missing field '{field}' for provider: {provider_name}")
            
            # Validate priority
            if 'priority' in provider_config:
                if not isinstance(provider_config['priority'], int) or provider_config['priority'] < 1:
                    self.validation_errors.append(f"Invalid priority for provider: {provider_name}")
    
    def _validate_neural_integration(self):
        """Validate neural model integration settings"""
        neural_config = self.config.get('neural_integration', {})
        
        # Check model path
        if 'model_path' not in neural_config:
            self.validation_errors.append("Missing neural model path")
        
        # Validate inference config
        inference_config = neural_config.get('inference_config', {})
        if 'confidence_threshold' in inference_config:
            threshold = inference_config['confidence_threshold']
            if not (0 <= threshold <= 1):
                self.validation_errors.append("Confidence threshold must be between 0 and 1")
    
    def _validate_api_server(self):
        """Validate API server configuration"""
        api_config = self.config.get('api_server', {})
        
        # Validate port
        if 'port' in api_config:
            port = api_config['port']
            if not isinstance(port, int) or port < 1 or port > 65535:
                self.validation_errors.append("Invalid API server port")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration"""
        # LLM API keys from environment
        env_mappings = {
            'MINIMAX_API_KEY': ('llm_providers', 'minimax', 'api_key'),
            'MISTRAL_API_KEY': ('llm_providers', 'mistral', 'api_key'),
            'CLAUDE_API_KEY': ('llm_providers', 'claude', 'api_key'),
            'OPENAI_API_KEY': ('llm_providers', 'openai', 'api_key'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self.config, config_path, value)
        
        # Other environment overrides
        if os.getenv('AI_API_PORT'):
            self._set_nested_value(
                self.config,
                ('api_server', 'port'),
                int(os.getenv('AI_API_PORT'))
            )
    
    def _set_nested_value(self, config: Dict, path: tuple, value: Any):
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def reload_if_modified(self) -> bool:
        """
        Reload configuration if file has been modified
        
        Returns:
            True if reloaded, False otherwise
        """
        try:
            current_modified = os.path.getmtime(self.config_path)
            
            if current_modified > self.last_modified:
                logger.info("Configuration file modified, reloading...")
                return self.load_config()
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking configuration modification: {str(e)}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'llm_providers.minimax.model')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_llm_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific LLM provider"""
        return self.get(f'llm_providers.{provider}')
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled LLM providers sorted by priority"""
        providers = []
        
        for name, config in self.config.get('llm_providers', {}).items():
            if config.get('enabled', False):
                providers.append((name, config.get('priority', 999)))
        
        # Sort by priority
        providers.sort(key=lambda x: x[1])
        
        return [name for name, _ in providers]
    
    def get_neural_config(self) -> Dict[str, Any]:
        """Get neural model integration configuration"""
        return self.config.get('neural_integration', {})
    
    def get_response_config(self) -> Dict[str, Any]:
        """Get response settings configuration"""
        return self.config.get('response_settings', {})
    
    def get_research_config(self) -> Dict[str, Any]:
        """Get research settings configuration"""
        return self.config.get('research_settings', {})
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration and save to file
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            Success status
        """
        try:
            # Apply updates
            self._deep_update(self.config, updates)
            
            # Validate updated configuration
            if not self.validate_config():
                logger.error("Updated configuration is invalid")
                # Reload original configuration
                self.load_config()
                return False
            
            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            self.last_modified = os.path.getmtime(self.config_path)
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return False
    
    def _deep_update(self, base: Dict, updates: Dict):
        """Deep update dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def export_config(self, include_sensitive: bool = False) -> str:
        """
        Export configuration as JSON
        
        Args:
            include_sensitive: Whether to include sensitive data (API keys)
            
        Returns:
            JSON string of configuration
        """
        export_config = self.config.copy()
        
        if not include_sensitive:
            # Remove sensitive data
            self._remove_sensitive_data(export_config)
        
        return json.dumps(export_config, indent=2, default=str)
    
    def _remove_sensitive_data(self, config: Dict):
        """Remove sensitive data from configuration"""
        # Remove API keys
        if 'llm_providers' in config:
            for provider in config['llm_providers'].values():
                if 'api_key' in provider:
                    provider['api_key'] = '***REDACTED***'
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information and status"""
        return {
            "config_path": str(self.config_path),
            "last_modified": datetime.fromtimestamp(self.last_modified).isoformat() if self.last_modified else None,
            "validation_errors": self.validation_errors,
            "is_valid": len(self.validation_errors) == 0,
            "enabled_providers": self.get_enabled_providers(),
            "neural_model": self.get_neural_config().get('model_type', 'unknown'),
            "api_port": self.get('api_server.port', 8000)
        }