"""
Test suite for APT Fitness Assistant
"""

import unittest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestConfiguration(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_import(self):
        """Test that config can be imported."""
        from apt_fitness.core.config import config
        self.assertIsNotNone(config)
        self.assertEqual(config.app_title, "AI Fitness Assistant")
    
    def test_config_directories(self):
        """Test that config creates necessary directories."""
        from apt_fitness.core.config import config
        self.assertTrue(config.data_dir.exists())


class TestModels(unittest.TestCase):
    """Test data models."""
    
    def test_user_profile_creation(self):
        """Test user profile creation."""
        from apt_fitness.core.models import UserProfile, Gender, FitnessLevel
        
        profile = UserProfile(
            name="Test User",
            age=25,
            gender=Gender.MALE,
            height_cm=175,
            weight_kg=70
        )
        
        self.assertEqual(profile.name, "Test User")
        self.assertEqual(profile.age, 25)
        self.assertAlmostEqual(profile.bmi, 22.86, places=2)
    
    def test_bmi_calculation(self):
        """Test BMI calculation."""
        from apt_fitness.core.models import UserProfile, Gender
        
        profile = UserProfile(
            height_cm=170,
            weight_kg=70,
            gender=Gender.MALE
        )
        
        expected_bmi = 70 / (1.7 ** 2)
        self.assertAlmostEqual(profile.bmi, expected_bmi, places=2)


class TestDatabase(unittest.TestCase):
    """Test database functionality."""
    
    def test_database_creation(self):
        """Test database initialization."""
        from apt_fitness.data.database import FitnessDatabase
        
        # Use test database
        db = FitnessDatabase("test_fitness.db")
        self.assertIsNotNone(db)
        
        # Cleanup
        Path("test_fitness.db").unlink(missing_ok=True)


class TestRecommendationEngine(unittest.TestCase):
    """Test recommendation engine."""
    
    def test_recommendation_generation(self):
        """Test workout recommendation generation."""
        from apt_fitness.engines.recommendation import RecommendationEngine
        from apt_fitness.core.models import UserProfile, Gender, FitnessLevel, GoalType
        
        engine = RecommendationEngine()
        
        profile = UserProfile(
            name="Test User",
            age=25,
            gender=Gender.MALE,
            fitness_level=FitnessLevel.BEGINNER,
            primary_goal=GoalType.GENERAL_FITNESS
        )
        
        recommendations = engine.generate_workout_recommendations(profile)
        self.assertIsInstance(recommendations, list)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_bmi_calculation(self):
        """Test BMI calculation utility."""
        from apt_fitness.utils.helpers import calculate_bmi
        
        bmi = calculate_bmi(70, 170)
        expected = 70 / (1.7 ** 2)
        self.assertAlmostEqual(bmi, expected, places=2)
    
    def test_unique_id_generation(self):
        """Test unique ID generation."""
        from apt_fitness.utils.helpers import generate_unique_id
        
        id1 = generate_unique_id("test")
        id2 = generate_unique_id("test")
        
        self.assertNotEqual(id1, id2)
        self.assertTrue(isinstance(id1, str))
        self.assertTrue(len(id1) > 0)


if __name__ == "__main__":
    unittest.main()
