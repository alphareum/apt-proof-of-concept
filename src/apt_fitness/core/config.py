"""
Core configuration for APT Fitness Assistant
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Environment(Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class AppConfig:
    """Application configuration."""
    
    # App Info
    app_title: str = "AI Fitness Assistant"
    app_icon: str = "🏋️‍♀️"
    version: str = "3.0.0"
    description: str = "Your Personal AI Fitness Coach"
    
    # URLs
    repository_url: str = "https://github.com/alphareum/apt-proof-of-concept"
    issues_url: str = "https://github.com/alphareum/apt-proof-of-concept/issues"
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # File Constraints
    max_image_size_mb: int = 10
    max_video_size_mb: int = 50
    supported_image_formats: List[str] = None
    supported_video_formats: List[str] = None
    
    # Physical Constraints
    min_age: int = 18
    max_age: int = 100
    min_weight: float = 30.0
    max_weight: float = 300.0
    min_height: float = 100.0
    max_height: float = 250.0
    
    # Computer Vision
    pose_detection_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5
    segmentation_model: int = 1
    
    # Database
    database_path: str = "data/fitness_app.db"
    
    # Directories
    data_dir: Path = Path("data")
    processed_images_dir: Path = Path("processed_images")
    temp_uploads_dir: Path = Path("temp_uploads")
    logs_dir: Path = Path("logs")
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.supported_image_formats is None:
            self.supported_image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        if self.supported_video_formats is None:
            self.supported_video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.processed_images_dir.mkdir(exist_ok=True)
        self.temp_uploads_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            environment=Environment(os.getenv("ENVIRONMENT", "development")),
            database_path=os.getenv("DATABASE_PATH", "data/fitness_app.db"),
        )


# Global config instance
config = AppConfig()
