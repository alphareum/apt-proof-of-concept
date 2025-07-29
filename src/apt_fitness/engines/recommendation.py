"""
Recommendation engine for APT Fitness Assistant
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random

from ..core.models import UserProfile, Exercise, FitnessLevel, GoalType, EquipmentType
from ..data.database import get_database

logger = logging.getLogger(__name__)


@dataclass
class WorkoutRecommendation:
    """Workout recommendation model."""
    exercise: Exercise
    sets: int
    reps: int
    duration_minutes: Optional[int] = None
    rest_seconds: int = 60
    weight_kg: Optional[float] = None
    notes: str = ""


class ExerciseDatabase:
    """Exercise database with comprehensive exercise library."""
    
    @staticmethod
    def get_exercises() -> Dict[str, Dict[str, Exercise]]:
        """Get comprehensive exercise database."""
        return {
            "cardio": {
                "running": Exercise(
                    name="Running",
                    category="cardio",
                    muscle_groups=["legs", "cardiovascular"],
                    equipment_needed=[],
                    difficulty_level=2,
                    calories_per_minute=10.0,
                    instructions=[
                        "Start with a light warm-up walk",
                        "Gradually increase pace to comfortable running speed",
                        "Maintain steady breathing rhythm",
                        "Land on balls of feet, not heels"
                    ],
                    tips=["Start slow and build endurance", "Focus on breathing"],
                    contraindications=["severe knee problems", "recent surgery"]
                ),
                "cycling": Exercise(
                    name="Cycling",
                    category="cardio",
                    muscle_groups=["legs", "glutes", "cardiovascular"],
                    equipment_needed=["bicycle"],
                    difficulty_level=2,
                    calories_per_minute=8.0,
                    instructions=[
                        "Adjust seat height properly",
                        "Keep core engaged",
                        "Maintain steady cadence"
                    ],
                    tips=["Start with lower resistance", "Gradually increase intensity"],
                    contraindications=["severe knee problems"]
                ),
                "jumping_jacks": Exercise(
                    name="Jumping Jacks",
                    category="cardio",
                    muscle_groups=["full_body", "cardiovascular"],
                    equipment_needed=[],
                    difficulty_level=1,
                    calories_per_minute=8.0,
                    instructions=[
                        "Start with feet together, arms at sides",
                        "Jump feet apart while raising arms overhead",
                        "Jump back to starting position"
                    ],
                    tips=["Land softly on balls of feet", "Keep core engaged"],
                    contraindications=["ankle injury", "knee problems"]
                ),
                "burpees": Exercise(
                    name="Burpees",
                    category="cardio",
                    muscle_groups=["full_body", "cardiovascular"],
                    equipment_needed=[],
                    difficulty_level=4,
                    calories_per_minute=12.0,
                    instructions=[
                        "Start in standing position",
                        "Drop into squat, hands on floor",
                        "Jump feet back to plank",
                        "Do push-up, jump feet to squat, jump up"
                    ],
                    tips=["Modify by stepping instead of jumping", "Focus on form"],
                    contraindications=["back injury", "wrist problems"]
                )
            },
            "strength": {
                "push_ups": Exercise(
                    name="Push-ups",
                    category="strength",
                    muscle_groups=["chest", "shoulders", "triceps", "core"],
                    equipment_needed=[],
                    difficulty_level=2,
                    calories_per_minute=6.0,
                    instructions=[
                        "Start in plank position",
                        "Lower chest to nearly touch ground",
                        "Push back up to starting position",
                        "Keep body in straight line"
                    ],
                    tips=["Modify on knees if needed", "Keep core tight"],
                    contraindications=["wrist injury", "shoulder problems"]
                ),
                "squats": Exercise(
                    name="Squats",
                    category="strength",
                    muscle_groups=["quadriceps", "glutes", "hamstrings"],
                    equipment_needed=[],
                    difficulty_level=2,
                    calories_per_minute=7.0,
                    instructions=[
                        "Stand with feet shoulder-width apart",
                        "Lower by pushing hips back",
                        "Go down until thighs parallel to ground",
                        "Drive through heels to stand"
                    ],
                    tips=["Keep chest up", "Knees track over toes"],
                    contraindications=["severe knee problems"]
                ),
                "planks": Exercise(
                    name="Planks",
                    category="strength",
                    muscle_groups=["core", "shoulders"],
                    equipment_needed=[],
                    difficulty_level=2,
                    calories_per_minute=4.0,
                    instructions=[
                        "Start in forearm plank position",
                        "Keep body in straight line",
                        "Hold position while breathing normally"
                    ],
                    tips=["Don't let hips sag", "Engage core and glutes"],
                    contraindications=["lower back injury"]
                ),
                "lunges": Exercise(
                    name="Lunges",
                    category="strength",
                    muscle_groups=["legs", "glutes", "core"],
                    equipment_needed=[],
                    difficulty_level=2,
                    calories_per_minute=7.0,
                    instructions=[
                        "Step forward with one foot",
                        "Lower back knee toward ground",
                        "Keep front knee over ankle",
                        "Push off front foot to return"
                    ],
                    tips=["Alternate legs", "Keep torso upright"],
                    contraindications=["knee injury"]
                )
            },
            "flexibility": {
                "yoga_flow": Exercise(
                    name="Yoga Flow",
                    category="flexibility",
                    muscle_groups=["full_body", "flexibility"],
                    equipment_needed=["yoga_mat"],
                    difficulty_level=2,
                    calories_per_minute=3.0,
                    instructions=[
                        "Flow through poses with breath",
                        "Hold each pose for 3-5 breaths",
                        "Focus on alignment"
                    ],
                    tips=["Listen to your body", "Don't force stretches"],
                    contraindications=["recent injury"]
                ),
                "stretching": Exercise(
                    name="Static Stretching",
                    category="flexibility",
                    muscle_groups=["full_body", "flexibility"],
                    equipment_needed=[],
                    difficulty_level=1,
                    calories_per_minute=2.0,
                    instructions=[
                        "Hold each stretch for 15-30 seconds",
                        "Breathe deeply during stretches",
                        "Don't bounce or force"
                    ],
                    tips=["Warm up before stretching", "Focus on major muscle groups"],
                    contraindications=["acute muscle injury"]
                )
            }
        }
    
    @classmethod
    def get_exercises_by_category(cls, category: str) -> Dict[str, Exercise]:
        """Get exercises by category."""
        exercises = cls.get_exercises()
        return exercises.get(category, {})
    
    @classmethod
    def filter_exercises(cls, category: str, equipment_available: List[str], 
                        injuries: List[str]) -> Dict[str, Exercise]:
        """Filter exercises based on available equipment and injuries."""
        exercises = cls.get_exercises_by_category(category)
        filtered = {}
        
        for name, exercise in exercises.items():
            # Check if user has required equipment
            if exercise.equipment_needed:
                if not all(eq in equipment_available for eq in exercise.equipment_needed):
                    continue
            
            # Check for contraindications
            if any(injury in exercise.contraindications for injury in injuries):
                continue
            
            filtered[name] = exercise
        
        return filtered


class RecommendationEngine:
    """Advanced exercise recommendation engine."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.exercise_db = ExerciseDatabase()
        self.db = get_database()
    
    def generate_workout_recommendations(self, user_profile: UserProfile, 
                                       target_duration: int = None) -> List[WorkoutRecommendation]:
        """Generate personalized workout recommendations."""
        try:
            # Use target duration or user preference
            duration = target_duration or user_profile.preferred_workout_duration
            
            # Get available equipment
            equipment_map = {
                EquipmentType.NONE: [],
                EquipmentType.BASIC: ["dumbbells", "resistance_bands", "yoga_mat"],
                EquipmentType.HOME_GYM: ["dumbbells", "resistance_bands", "yoga_mat", "bench", "pull_up_bar"],
                EquipmentType.FULL_GYM: ["dumbbells", "barbells", "machines", "cables", "yoga_mat", "bench"]
            }
            available_equipment = equipment_map.get(user_profile.available_equipment, [])
            
            recommendations = []
            remaining_time = duration
            
            # Determine workout focus based on goal
            focus_distribution = self._get_focus_distribution(user_profile.primary_goal)
            
            # Generate exercises for each category
            for category, time_percentage in focus_distribution.items():
                category_time = int(remaining_time * time_percentage)
                if category_time < 5:  # Minimum 5 minutes per category
                    continue
                
                category_exercises = self.exercise_db.filter_exercises(
                    category, available_equipment, user_profile.injuries
                )
                
                if not category_exercises:
                    continue
                
                # Select exercises for this category
                selected_exercises = self._select_exercises_for_category(
                    category_exercises, category_time, user_profile.fitness_level
                )
                
                recommendations.extend(selected_exercises)
                remaining_time -= category_time
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating workout recommendations: {e}")
            return []
    
    def _get_focus_distribution(self, goal: GoalType) -> Dict[str, float]:
        """Get workout focus distribution based on goal."""
        distributions = {
            GoalType.WEIGHT_LOSS: {
                "cardio": 0.6,
                "strength": 0.3,
                "flexibility": 0.1
            },
            GoalType.MUSCLE_GAIN: {
                "strength": 0.7,
                "cardio": 0.2,
                "flexibility": 0.1
            },
            GoalType.STRENGTH: {
                "strength": 0.8,
                "cardio": 0.1,
                "flexibility": 0.1
            },
            GoalType.ENDURANCE: {
                "cardio": 0.7,
                "strength": 0.2,
                "flexibility": 0.1
            },
            GoalType.FLEXIBILITY: {
                "flexibility": 0.6,
                "strength": 0.2,
                "cardio": 0.2
            },
            GoalType.GENERAL_FITNESS: {
                "cardio": 0.4,
                "strength": 0.4,
                "flexibility": 0.2
            }
        }
        
        return distributions.get(goal, distributions[GoalType.GENERAL_FITNESS])
    
    def _select_exercises_for_category(self, exercises: Dict[str, Exercise], 
                                     target_time: int, fitness_level: FitnessLevel) -> List[WorkoutRecommendation]:
        """Select exercises for a specific category."""
        if not exercises:
            return []
        
        recommendations = []
        used_time = 0
        
        # Determine number of exercises based on time and fitness level
        if target_time <= 10:
            num_exercises = 1
        elif target_time <= 20:
            num_exercises = 2
        else:
            num_exercises = min(3, len(exercises))
        
        # Randomly select exercises
        selected_exercises = random.sample(list(exercises.values()), 
                                         min(num_exercises, len(exercises)))
        
        for exercise in selected_exercises:
            if used_time >= target_time:
                break
            
            # Generate sets/reps based on fitness level and exercise type
            recommendation = self._create_exercise_recommendation(exercise, fitness_level)
            
            # Estimate time for this exercise
            if exercise.category == "cardio":
                exercise_time = recommendation.duration_minutes or 10
            else:
                # Strength/flexibility: sets * (reps * 2 seconds + rest)
                exercise_time = recommendation.sets * (recommendation.reps * 2 + recommendation.rest_seconds) / 60
            
            if used_time + exercise_time <= target_time:
                recommendations.append(recommendation)
                used_time += exercise_time
        
        return recommendations
    
    def _create_exercise_recommendation(self, exercise: Exercise, 
                                      fitness_level: FitnessLevel) -> WorkoutRecommendation:
        """Create exercise recommendation with sets/reps."""
        # Base parameters by fitness level
        level_params = {
            FitnessLevel.BEGINNER: {"sets": 2, "reps_multiplier": 0.8, "duration_multiplier": 0.7},
            FitnessLevel.INTERMEDIATE: {"sets": 3, "reps_multiplier": 1.0, "duration_multiplier": 1.0},
            FitnessLevel.ADVANCED: {"sets": 4, "reps_multiplier": 1.2, "duration_multiplier": 1.3}
        }
        
        params = level_params[fitness_level]
        
        if exercise.category == "cardio":
            return WorkoutRecommendation(
                exercise=exercise,
                sets=1,
                reps=1,
                duration_minutes=int(10 * params["duration_multiplier"]),
                rest_seconds=30
            )
        elif exercise.category == "strength":
            base_reps = 12 if "bodyweight" in exercise.name.lower() else 10
            return WorkoutRecommendation(
                exercise=exercise,
                sets=params["sets"],
                reps=int(base_reps * params["reps_multiplier"]),
                rest_seconds=60
            )
        else:  # flexibility
            return WorkoutRecommendation(
                exercise=exercise,
                sets=1,
                reps=1,
                duration_minutes=int(8 * params["duration_multiplier"]),
                rest_seconds=15
            )
    
    def generate_weekly_plan(self, user_profile: UserProfile) -> Dict[str, List[WorkoutRecommendation]]:
        """Generate a weekly workout plan."""
        try:
            weekly_plan = {}
            days_per_week = user_profile.workout_frequency_per_week
            
            # Plan different focus for each day
            focus_rotation = [
                GoalType.STRENGTH,
                GoalType.ENDURANCE,
                GoalType.FLEXIBILITY,
                GoalType.STRENGTH,
                GoalType.ENDURANCE
            ]
            
            for day in range(days_per_week):
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day]
                
                # Rotate focus or use user's primary goal
                if days_per_week > 3:
                    daily_focus = focus_rotation[day % len(focus_rotation)]
                else:
                    daily_focus = user_profile.primary_goal
                
                # Create temporary profile with different focus
                temp_profile = user_profile
                temp_profile.primary_goal = daily_focus
                
                daily_workout = self.generate_workout_recommendations(temp_profile)
                weekly_plan[day_name] = daily_workout
            
            return weekly_plan
            
        except Exception as e:
            logger.error(f"Error generating weekly plan: {e}")
            return {}
    
    def get_exercise_alternatives(self, exercise: Exercise, 
                                user_profile: UserProfile) -> List[Exercise]:
        """Get alternative exercises for muscle groups."""
        try:
            # Get all exercises from the same category
            category_exercises = self.exercise_db.get_exercises_by_category(exercise.category)
            
            alternatives = []
            for alt_exercise in category_exercises.values():
                # Skip the same exercise
                if alt_exercise.name == exercise.name:
                    continue
                
                # Check if it targets similar muscle groups
                common_muscles = set(exercise.muscle_groups) & set(alt_exercise.muscle_groups)
                if len(common_muscles) > 0:
                    # Check constraints
                    if not any(injury in alt_exercise.contraindications 
                             for injury in user_profile.injuries):
                        alternatives.append(alt_exercise)
            
            return alternatives[:3]  # Return top 3 alternatives
            
        except Exception as e:
            logger.error(f"Error getting exercise alternatives: {e}")
            return []


# Singleton instance
_recommendation_engine = None

def get_recommendation_engine() -> RecommendationEngine:
    """Get singleton recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine
