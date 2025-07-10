import os
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    uri: str = os.getenv("DATABASE_URL", "sqlite:///apt.db")
    track_modifications: bool = False
    echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"

@dataclass 
class AIConfig:
    """AI service configuration"""
    # LLM Provider selection
    provider: str = os.getenv("LLM_PROVIDER", "kolosal")
    
    # API Keys
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Kolosal.AI (Local LLM)
    kolosal_url: str = os.getenv("KOLOSAL_API_URL", "http://localhost:8080")
    kolosal_model: str = os.getenv("KOLOSAL_MODEL", "Gemma 3 4B:4-bit")
    
    # LLM Settings
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "300"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    # Pose Detection
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    pose_model_complexity: int = 2

@dataclass
class VideoConfig:
    """Video processing configuration"""
    max_size_mb: int = int(os.getenv("MAX_VIDEO_SIZE_MB", "100"))
    supported_formats: List[str] = None
    upload_folder: str = os.getenv("UPLOAD_FOLDER", "uploads")
    pose_data_folder: str = os.getenv("POSE_DATA_FOLDER", "pose_data")
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = os.getenv("SECRET_KEY", "dev-key-change-in-production")
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    allowed_hosts: List[str] = None
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1")
            self.allowed_hosts = hosts.split(",")

@dataclass
class AppConfig:
    """Main application configuration"""
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "5000"))
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Sub-configurations
    database: DatabaseConfig = None
    ai: AIConfig = None
    video: VideoConfig = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        self.database = DatabaseConfig()
        self.ai = AIConfig()
        self.video = VideoConfig()
        self.security = SecurityConfig()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check AI provider configuration
        if self.ai.provider == "openai" and not self.ai.openai_key:
            issues.append("OpenAI API key is required when using OpenAI provider")
        elif self.ai.provider == "anthropic" and not self.ai.anthropic_key:
            issues.append("Anthropic API key is required when using Anthropic provider")
        elif self.ai.provider not in ["openai", "anthropic", "kolosal"]:
            issues.append(f"Invalid LLM provider: {self.ai.provider}")
        
        # Check directories exist
        os.makedirs(self.video.upload_folder, exist_ok=True)
        os.makedirs(self.video.pose_data_folder, exist_ok=True)
        
        # Check security in production
        if self.environment == "production":
            if self.security.secret_key == "dev-key-change-in-production":
                issues.append("SECRET_KEY must be changed in production")
            if self.debug:
                issues.append("DEBUG should be False in production")
        
        return issues

# Exercise-specific configuration
EXERCISE_CONFIG = {
    'lat_pulldown': {
        'aliases': ['lat pull-down', 'lat pulldown', 'pulldown', 'lat pull down'],
        'analyzer_method': 'analyze_lat_pulldown',
        'primary_muscles': ['lats', 'rhomboids', 'middle_traps'],
        'form_points': ['back_arch', 'shoulder_depression', 'elbow_path']
    },
    'pullup': {
        'aliases': ['pull-up', 'pull up', 'pullup', 'chin-up', 'chin up'],
        'analyzer_method': 'analyze_pullup', 
        'primary_muscles': ['lats', 'biceps', 'rear_delts'],
        'form_points': ['full_extension', 'chin_clearance', 'no_kipping']
    },
    'row': {
        'aliases': ['seated row', 'cable row', 'rowing', 'bent over row'],
        'analyzer_method': 'analyze_row',
        'primary_muscles': ['lats', 'rhomboids', 'rear_delts'],
        'form_points': ['posture', 'scapular_retraction', 'elbow_path']
    }
}

FORM_THRESHOLDS = {
    'min_range_of_motion': 60,      # degrees
    'max_back_arch_score': 0.15,    # normalized score
    'min_rep_consistency': 0.8,     # 80% consistency
    'min_visibility_score': 0.5,    # pose landmark visibility
    'max_processing_time': 300,     # 5 minutes max processing
}

# Global config instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def validate_config() -> bool:
    """Validate configuration and print issues"""
    issues = config.validate()
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Configuration validated successfully")
        return True

def print_config_summary():
    """Print configuration summary"""
    print("🤖 APT Configuration Summary")
    print("=" * 40)
    print(f"Environment: {config.environment}")
    print(f"LLM Provider: {config.ai.provider}")
    print(f"Database: {config.database.uri}")
    print(f"Upload Folder: {config.video.upload_folder}")
    print(f"Max Video Size: {config.video.max_size_mb}MB")
    print(f"Debug Mode: {config.debug}")
    print()

if __name__ == "__main__":
    print_config_summary()
    validate_config()