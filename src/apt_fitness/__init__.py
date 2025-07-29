"""
APT Fitness Assistant - AI-Powered Fitness Application

A comprehensive fitness application with computer vision capabilities for:
- Body composition analysis
- Exercise form correction
- Personalized workout recommendations
- Progress tracking and analytics

Version: 3.0.0
"""

from .core.config import AppConfig
from .core.models import UserProfile
from .analyzers.body_composition import BodyCompositionAnalyzer
from .engines.recommendation import RecommendationEngine

__version__ = "3.0.0"
__author__ = "APT Research Team"

__all__ = [
    "AppConfig",
    "UserProfile", 
    "BodyCompositionAnalyzer",
    "RecommendationEngine",
]
