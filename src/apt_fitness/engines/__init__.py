"""
Engines package for APT Fitness Assistant
"""

from .recommendation import RecommendationEngine, get_recommendation_engine, WorkoutRecommendation

__all__ = ["RecommendationEngine", "get_recommendation_engine", "WorkoutRecommendation"]
