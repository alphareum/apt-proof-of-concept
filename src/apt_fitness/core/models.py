"""
Data models for APT Fitness Assistant
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class Gender(Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ActivityLevel(Enum):
    """Activity level enumeration."""
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


class FitnessLevel(Enum):
    """Fitness level enumeration."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class GoalType(Enum):
    """Fitness goal types."""
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    STRENGTH = "strength"
    ENDURANCE = "endurance"
    FLEXIBILITY = "flexibility"
    GENERAL_FITNESS = "general_fitness"


class EquipmentType(Enum):
    """Available equipment types."""
    NONE = "none"
    BASIC = "basic"  # dumbbells, resistance bands
    HOME_GYM = "home_gym"  # home gym setup
    FULL_GYM = "full_gym"  # commercial gym access


@dataclass
class UserProfile:
    """User profile model."""
    
    # Basic Info
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    age: int = 25
    gender: Gender = Gender.MALE
    
    # Physical Measurements
    height_cm: float = 170.0
    weight_kg: float = 70.0
    target_weight_kg: Optional[float] = None
    
    # Fitness Info
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    fitness_level: FitnessLevel = FitnessLevel.BEGINNER
    primary_goal: GoalType = GoalType.GENERAL_FITNESS
    available_equipment: EquipmentType = EquipmentType.BASIC
    
    # Health Info
    injuries: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    
    # Preferences
    preferred_workout_duration: int = 30  # minutes
    workout_frequency_per_week: int = 3
    preferred_workout_time: str = "morning"  # morning, afternoon, evening
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        return self.weight_kg / (self.height_cm / 100) ** 2
    
    @property
    def bmr(self) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation."""
        if self.gender == Gender.MALE:
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161
    
    @property
    def daily_calories(self) -> float:
        """Calculate daily calorie needs based on activity level."""
        multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTREMELY_ACTIVE: 1.9
        }
        return self.bmr * multipliers[self.activity_level]


@dataclass
class Exercise:
    """Exercise model."""
    
    exercise_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""
    muscle_groups: List[str] = field(default_factory=list)
    equipment_needed: List[str] = field(default_factory=list)
    difficulty_level: int = 1  # 1-5 scale
    calories_per_minute: float = 5.0
    instructions: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    video_url: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class WorkoutSession:
    """Workout session model."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    date: datetime = field(default_factory=datetime.now)
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    duration_minutes: int = 0
    calories_burned: float = 0.0
    notes: str = ""
    mood_before: Optional[int] = None  # 1-10 scale
    mood_after: Optional[int] = None   # 1-10 scale
    difficulty_rating: Optional[int] = None  # 1-10 scale
    completed: bool = False


@dataclass
class BodyMeasurement:
    """Body measurement model."""
    
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    date: datetime = field(default_factory=datetime.now)
    weight_kg: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    muscle_mass_kg: Optional[float] = None
    
    # Body measurements in cm
    waist_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    arms_cm: Optional[float] = None
    thighs_cm: Optional[float] = None
    neck_cm: Optional[float] = None
    hips_cm: Optional[float] = None


@dataclass
class BodyCompositionAnalysis:
    """Body composition analysis model."""
    
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    image_path: str = ""
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Composition metrics
    body_fat_percentage: float = 0.0
    muscle_mass_percentage: float = 0.0
    visceral_fat_level: int = 1  # 1-20 scale
    bmr_estimated: int = 0
    body_shape_classification: str = ""
    confidence_score: float = 0.0
    
    # Additional images
    front_image_path: Optional[str] = None
    side_image_path: Optional[str] = None
    processed_image_path: Optional[str] = None
    
    # Analysis data
    body_measurements: Dict[str, Any] = field(default_factory=dict)
    composition_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyPartMeasurement:
    """Body part measurement model."""
    
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str = ""
    body_part: str = ""
    circumference_cm: float = 0.0
    area_percentage: float = 0.0
    muscle_definition_score: float = 0.0
    fat_distribution_score: float = 0.0
    symmetry_score: float = 0.0


@dataclass 
class Goal:
    """User goal model."""
    
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    goal_type: GoalType = GoalType.GENERAL_FITNESS
    target_value: float = 0.0
    current_value: float = 0.0
    target_date: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, completed, paused


def create_user_profile(**kwargs) -> UserProfile:
    """Factory function to create a user profile."""
    return UserProfile(**kwargs)
