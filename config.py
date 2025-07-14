"""
Configuration module for AI Fitness Assistant
Centralized configuration management with environment variable support

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Exercise database settings
    exercise_db_version: str = "1.0"
    custom_exercises_file: Optional[str] = None
    cache_timeout: int = 3600  # 1 hour in seconds
    
    # Data validation settings
    min_exercise_duration: int = 5  # minutes
    max_exercise_duration: int = 180  # minutes
    min_calories_per_min: float = 1.0
    max_calories_per_min: float = 20.0

@dataclass
class ModelConfig:
    """ML Model configuration settings."""
    
    # MediaPipe settings
    pose_model_complexity: int = 1
    pose_detection_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5
    pose_segmentation_enabled: bool = False
    
    # Body fat calculation settings
    navy_method_enabled: bool = True
    bmi_method_fallback: bool = True
    measurement_tolerance: float = 0.1  # cm
    
    # Image processing settings
    image_resize_max_width: int = 1024
    image_resize_max_height: int = 1024
    image_quality: int = 85
    supported_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

@dataclass
class SecurityConfig:
    """Security and validation configuration."""
    
    # File upload limits
    max_image_size_mb: int = 10
    max_video_size_mb: int = 50
    allowed_origins: List[str] = field(default_factory=lambda: ['localhost', '127.0.0.1'])
    
    # Input validation
    max_input_length: int = 1000
    sanitize_inputs: bool = True
    
    # Session settings
    session_timeout: int = 3600  # 1 hour
    max_sessions_per_user: int = 5

@dataclass
class UIConfig:
    """User interface configuration."""
    
    # Styling
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    error_color: str = "#dc3545"
    
    # Layout
    sidebar_width: int = 300
    max_content_width: int = 1200
    
    # Features
    enable_dark_mode: bool = True
    enable_animations: bool = True
    show_debug_info: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/fitness_app.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Performance logging
    log_performance: bool = True
    slow_query_threshold: float = 1.0  # seconds

@dataclass
class CacheConfig:
    """Caching configuration settings."""
    
    # Streamlit caching
    enable_streamlit_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_entries: int = 1000
    
    # Memory cache settings
    enable_memory_cache: bool = True
    memory_cache_size: int = 128  # MB
    
    # Disk cache settings
    enable_disk_cache: bool = False
    disk_cache_path: str = "cache/"
    disk_cache_size: int = 1024  # MB

@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Basic app info
    app_title: str = "AI Fitness Assistant"
    app_icon: str = "🏋️‍♀️"
    version: str = "2.0.0"
    description: str = "Your Personal AI Fitness Coach"
    repository_url: str = "https://github.com/alphareum/apt-proof-of-concept"
    issues_url: str = "https://github.com/alphareum/apt-proof-of-concept/issues"
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Feature flags
    enable_body_fat_analysis: bool = True
    enable_exercise_recommendations: bool = True
    enable_pose_analysis: bool = True
    enable_progress_tracking: bool = True
    enable_social_features: bool = False
    enable_premium_features: bool = False
    
    # Physical constraints
    min_age: int = 18
    max_age: int = 100
    min_weight: float = 30.0  # kg
    max_weight: float = 300.0  # kg
    min_height: float = 100.0  # cm
    max_height: float = 250.0  # cm
    
    # Nested configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if they exist
        config.environment = Environment(os.getenv('APP_ENVIRONMENT', config.environment.value))
        config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
        config.version = os.getenv('APP_VERSION', config.version)
        
        # Security settings from environment
        config.security.max_image_size_mb = int(os.getenv('MAX_IMAGE_SIZE_MB', config.security.max_image_size_mb))
        config.security.session_timeout = int(os.getenv('SESSION_TIMEOUT', config.security.session_timeout))
        
        # Model settings from environment
        config.models.pose_detection_confidence = float(os.getenv('POSE_DETECTION_CONFIDENCE', config.models.pose_detection_confidence))
        config.models.pose_tracking_confidence = float(os.getenv('POSE_TRACKING_CONFIDENCE', config.models.pose_tracking_confidence))
        
        # Cache settings from environment
        config.cache.cache_ttl = int(os.getenv('CACHE_TTL', config.cache.cache_ttl))
        config.cache.enable_disk_cache = os.getenv('ENABLE_DISK_CACHE', str(config.cache.enable_disk_cache)).lower() == 'true'
        
        # Logging settings from environment
        config.logging.level = os.getenv('LOG_LEVEL', config.logging.level)
        config.logging.file_enabled = os.getenv('LOG_TO_FILE', str(config.logging.file_enabled)).lower() == 'true'
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'AppConfig':
        """Load configuration from JSON file."""
        config_path = Path(file_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create base config and update with file data
        config = cls()
        
        # Update basic settings
        for key, value in config_data.items():
            if hasattr(config, key) and not key.startswith('_'):
                if isinstance(getattr(config, key), (DatabaseConfig, ModelConfig, SecurityConfig, UIConfig, LoggingConfig, CacheConfig)):
                    # Handle nested configurations
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        result = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, (DatabaseConfig, ModelConfig, SecurityConfig, UIConfig, LoggingConfig, CacheConfig)):
                result[key] = value.__dict__
            elif isinstance(value, Environment):
                result[key] = value.value
            else:
                result[key] = value
        
        return result
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        config_path = Path(file_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate basic constraints
        if self.min_age >= self.max_age:
            errors.append("min_age must be less than max_age")
        
        if self.min_weight >= self.max_weight:
            errors.append("min_weight must be less than max_weight")
        
        if self.min_height >= self.max_height:
            errors.append("min_height must be less than max_height")
        
        # Validate model settings
        if not 0 <= self.models.pose_detection_confidence <= 1:
            errors.append("pose_detection_confidence must be between 0 and 1")
        
        if not 0 <= self.models.pose_tracking_confidence <= 1:
            errors.append("pose_tracking_confidence must be between 0 and 1")
        
        # Validate security settings
        if self.security.max_image_size_mb <= 0:
            errors.append("max_image_size_mb must be positive")
        
        if self.security.session_timeout <= 0:
            errors.append("session_timeout must be positive")
        
        # Validate cache settings
        if self.cache.cache_ttl <= 0:
            errors.append("cache_ttl must be positive")
        
        return errors
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        feature_map = {
            'body_fat_analysis': self.enable_body_fat_analysis,
            'exercise_recommendations': self.enable_exercise_recommendations,
            'pose_analysis': self.enable_pose_analysis,
            'progress_tracking': self.enable_progress_tracking,
            'social_features': self.enable_social_features,
            'premium_features': self.enable_premium_features,
        }
        
        return feature_map.get(feature, False)
    
    def get_cache_key(self, prefix: str, *args) -> str:
        """Generate a consistent cache key."""
        key_parts = [prefix, self.version] + [str(arg) for arg in args]
        return "_".join(key_parts)

# Global configuration instance
_config_instance = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        # Try to load from file first, then environment, then defaults
        config_file = os.getenv('CONFIG_FILE', 'config.json')
        
        if os.path.exists(config_file):
            _config_instance = AppConfig.from_file(config_file)
        else:
            _config_instance = AppConfig.from_env()
        
        # Validate configuration
        errors = _config_instance.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    return _config_instance

def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config_instance
    
    # Validate before setting
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    _config_instance = config

def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None

# Example configuration files for different environments
DEVELOPMENT_CONFIG = {
    "environment": "development",
    "debug": True,
    "logging": {
        "level": "DEBUG",
        "log_performance": True,
        "show_debug_info": True
    },
    "cache": {
        "cache_ttl": 300,  # 5 minutes for development
        "enable_disk_cache": False
    },
    "security": {
        "max_image_size_mb": 5,  # Smaller for development
        "session_timeout": 7200  # 2 hours for development
    }
}

PRODUCTION_CONFIG = {
    "environment": "production",
    "debug": False,
    "logging": {
        "level": "INFO",
        "log_performance": False,
        "show_debug_info": False
    },
    "cache": {
        "cache_ttl": 3600,  # 1 hour for production
        "enable_disk_cache": True
    },
    "security": {
        "max_image_size_mb": 10,
        "session_timeout": 3600  # 1 hour for production
    }
}

if __name__ == "__main__":
    # Example usage and testing
    config = get_config()
    print(f"App: {config.app_title} v{config.version}")
    print(f"Environment: {config.environment.value}")
    print(f"Debug: {config.debug}")
    
    # Save example configurations
    dev_config = AppConfig.from_env()
    dev_config.__dict__.update(DEVELOPMENT_CONFIG)
    dev_config.save_to_file("config_development.json")
    
    prod_config = AppConfig.from_env()
    prod_config.__dict__.update(PRODUCTION_CONFIG)
    prod_config.save_to_file("config_production.json")
    
    print("Example configuration files created!")