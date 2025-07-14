"""
Enhanced data models for AI Fitness Assistant
Improved data structures with validation and serialization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, date
import json
import uuid
from abc import ABC, abstractmethod

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class ActivityLevel(Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"  
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"

class FitnessLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class GoalType(Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    GENERAL_FITNESS = "general_fitness"
    MAINTENANCE = "maintenance"

class ExerciseCategory(Enum):
    CARDIO = "cardio"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    SPORTS = "sports"
    FUNCTIONAL = "functional"

class EquipmentType(Enum):
    NONE = "none"
    DUMBBELLS = "dumbbells"
    BARBELL = "barbell"
    RESISTANCE_BANDS = "resistance_bands"
    KETTLEBELL = "kettlebell"
    PULL_UP_BAR = "pull_up_bar"
    EXERCISE_BIKE = "exercise_bike"
    TREADMILL = "treadmill"
    YOGA_MAT = "yoga_mat"
    BENCH = "bench"
    CABLE_MACHINE = "cable_machine"
    SMITH_MACHINE = "smith_machine"

@dataclass
class BaseModel(ABC):
    """Base model with common functionality."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum handling."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (datetime, date)):
                result[key] = value.isoformat()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if hasattr(item, 'to_dict') else item for item in value]
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class UserProfile(BaseModel):
    """Enhanced user profile with validation."""
    
    # Basic info
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    age: int = 25
    gender: Gender = Gender.OTHER
    weight: float = 70.0  # kg
    height: float = 170.0  # cm
    
    # Activity and fitness
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    fitness_level: FitnessLevel = FitnessLevel.BEGINNER
    
    # Goals and preferences
    primary_goal: GoalType = GoalType.GENERAL_FITNESS
    secondary_goals: List[GoalType] = field(default_factory=list)
    
    # Workout preferences
    available_time: int = 30  # minutes
    workout_days_per_week: int = 3
    preferred_workout_time: str = "morning"  # morning, afternoon, evening
    
    # Equipment and constraints
    available_equipment: List[EquipmentType] = field(default_factory=list)
    injuries: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    
    # Experience
    years_training: int = 0
    favorite_exercises: List[str] = field(default_factory=list)
    disliked_exercises: List[str] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate profile data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate user profile data."""
        if not (18 <= self.age <= 100):
            raise ValueError("Age must be between 18 and 100")
        
        if not (30 <= self.weight <= 300):
            raise ValueError("Weight must be between 30 and 300 kg")
        
        if not (100 <= self.height <= 250):
            raise ValueError("Height must be between 100 and 250 cm")
        
        if not (5 <= self.available_time <= 180):
            raise ValueError("Available time must be between 5 and 180 minutes")
        
        if not (1 <= self.workout_days_per_week <= 7):
            raise ValueError("Workout days must be between 1 and 7")
    
    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 1)
    
    @property
    def bmi_category(self) -> str:
        """Get BMI category."""
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def update_profile(self, **kwargs):
        """Update profile fields and validation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.validate()

@dataclass 
class Exercise(BaseModel):
    """Enhanced exercise model."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: ExerciseCategory = ExerciseCategory.STRENGTH
    muscle_groups: List[str] = field(default_factory=list)
    equipment: List[EquipmentType] = field(default_factory=list)
    difficulty: int = 1  # 1-5 scale
    
    # Metrics
    calories_per_minute: float = 5.0
    met_value: float = 3.0  # Metabolic equivalent
    
    # Instructions and media
    instructions: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    video_url: Optional[str] = None
    image_url: Optional[str] = None
    
    # Targeting and modifications
    target_reps: Optional[str] = None  # e.g., "8-12", "30 seconds"
    target_sets: Optional[int] = None
    rest_time_seconds: Optional[int] = None
    modifications: Dict[str, str] = field(default_factory=dict)  # easier/harder versions
    
    # Safety and contraindications
    contraindications: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    is_custom: bool = False
    popularity_score: float = 0.0
    
    def calculate_calories(self, duration_minutes: int, user_weight_kg: float) -> float:
        """Calculate calories burned for given duration and user weight."""
        return self.met_value * user_weight_kg * (duration_minutes / 60)
    
    def is_suitable_for_user(self, user_profile: UserProfile) -> bool:
        """Check if exercise is suitable for user."""
        # Check equipment availability
        required_equipment = set(self.equipment)
        available_equipment = set(user_profile.available_equipment)
        if required_equipment and not required_equipment.issubset(available_equipment):
            return False
        
        # Check difficulty vs fitness level
        max_difficulty = {
            FitnessLevel.BEGINNER: 2,
            FitnessLevel.INTERMEDIATE: 4,
            FitnessLevel.ADVANCED: 5
        }
        if self.difficulty > max_difficulty[user_profile.fitness_level]:
            return False
        
        # Check contraindications
        user_conditions = user_profile.injuries + user_profile.medical_conditions
        for condition in user_conditions:
            if any(condition.lower() in contra.lower() for contra in self.contraindications):
                return False
        
        return True

@dataclass
class WorkoutSet(BaseModel):
    """Individual set within a workout."""
    
    exercise_id: str = ""
    reps: Optional[int] = None
    weight: Optional[float] = None  # kg
    duration_seconds: Optional[int] = None
    distance: Optional[float] = None  # km
    rest_after_seconds: int = 60
    notes: Optional[str] = None
    perceived_exertion: Optional[int] = None  # 1-10 RPE scale
    completed: bool = False

@dataclass
class Workout(BaseModel):
    """Complete workout session."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    date: datetime = field(default_factory=datetime.now)
    
    # Workout structure
    exercises: List[str] = field(default_factory=list)  # exercise IDs
    sets: List[WorkoutSet] = field(default_factory=list)
    
    # Session data
    planned_duration_minutes: int = 30
    actual_duration_minutes: Optional[int] = None
    calories_burned: Optional[float] = None
    
    # User feedback
    difficulty_rating: Optional[int] = None  # 1-5 scale
    enjoyment_rating: Optional[int] = None  # 1-5 scale
    energy_before: Optional[int] = None  # 1-10 scale
    energy_after: Optional[int] = None  # 1-10 scale
    notes: Optional[str] = None
    
    # Status
    status: str = "planned"  # planned, in_progress, completed, skipped
    completed_at: Optional[datetime] = None
    
    def mark_completed(self):
        """Mark workout as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        if not self.actual_duration_minutes:
            self.actual_duration_minutes = self.planned_duration_minutes

@dataclass
class BodyMeasurements(BaseModel):
    """Body measurements tracking."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    date: datetime = field(default_factory=datetime.now)
    
    # Basic measurements
    weight: Optional[float] = None  # kg
    body_fat_percentage: Optional[float] = None
    muscle_mass: Optional[float] = None  # kg
    
    # Circumference measurements (cm)
    waist: Optional[float] = None
    chest: Optional[float] = None
    arms: Optional[float] = None
    thighs: Optional[float] = None
    neck: Optional[float] = None
    hips: Optional[float] = None
    forearms: Optional[float] = None
    calves: Optional[float] = None
    
    # Additional metrics
    visceral_fat: Optional[float] = None
    bone_mass: Optional[float] = None  # kg
    water_percentage: Optional[float] = None
    
    # Notes and context
    measurement_conditions: Optional[str] = None  # time of day, hydration status, etc.
    notes: Optional[str] = None

@dataclass
class FitnessGoal(BaseModel):
    """Fitness goal tracking."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    goal_type: GoalType = GoalType.GENERAL_FITNESS
    
    # Goal parameters
    title: str = ""
    description: str = ""
    target_value: Optional[float] = None
    current_value: float = 0.0
    unit: str = ""
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    target_date: Optional[date] = None
    achieved_at: Optional[datetime] = None
    
    # Status
    status: str = "active"  # active, achieved, paused, cancelled
    priority: int = 1  # 1-5 scale
    
    # Progress tracking
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    progress_updates: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if not self.target_value or self.target_value == 0:
            return 0.0
        return min(100, (self.current_value / self.target_value) * 100)
    
    @property
    def is_achieved(self) -> bool:
        """Check if goal is achieved."""
        return self.current_value >= (self.target_value or 0)
    
    def update_progress(self, new_value: float, notes: str = ""):
        """Update goal progress."""
        self.current_value = new_value
        self.progress_updates.append({
            'date': datetime.now().isoformat(),
            'value': new_value,
            'notes': notes
        })
        
        if self.is_achieved and self.status == "active":
            self.status = "achieved"
            self.achieved_at = datetime.now()

@dataclass
class WorkoutPlan(BaseModel):
    """Multi-week workout plan."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    description: str = ""
    
    # Plan structure
    duration_weeks: int = 4
    workouts_per_week: int = 3
    weekly_templates: Dict[int, List[str]] = field(default_factory=dict)  # week -> workout IDs
    
    # Progression
    progression_type: str = "linear"  # linear, undulating, block
    progression_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"  # system, user, trainer
    difficulty_level: FitnessLevel = FitnessLevel.BEGINNER
    
    # Status
    status: str = "draft"  # draft, active, completed, paused
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class NutritionEntry(BaseModel):
    """Nutrition tracking entry."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    date: datetime = field(default_factory=datetime.now)
    
    # Macronutrients
    calories: Optional[float] = None
    protein: Optional[float] = None  # grams
    carbs: Optional[float] = None    # grams
    fat: Optional[float] = None      # grams
    fiber: Optional[float] = None    # grams
    sugar: Optional[float] = None    # grams
    sodium: Optional[float] = None   # mg
    
    # Hydration
    water_intake: Optional[float] = None  # liters
    
    # Meal details
    meal_type: Optional[str] = None  # breakfast, lunch, dinner, snack
    foods: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None

# Factory functions for creating models
def create_user_profile(**kwargs) -> UserProfile:
    """Create user profile with defaults."""
    return UserProfile(**kwargs)

def create_exercise(**kwargs) -> Exercise:
    """Create exercise with defaults."""
    return Exercise(**kwargs)

def create_workout(**kwargs) -> Workout:
    """Create workout with defaults.""" 
    return Workout(**kwargs)

def create_body_measurements(**kwargs) -> BodyMeasurements:
    """Create body measurements with defaults."""
    return BodyMeasurements(**kwargs)

def create_fitness_goal(**kwargs) -> FitnessGoal:
    """Create fitness goal with defaults."""
    return FitnessGoal(**kwargs)