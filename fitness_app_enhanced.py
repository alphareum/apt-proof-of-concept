"""
Enhanced AI Fitness Assistant Pro
Simplified version that integrates all improvements while maintaining functionality

Author: AI Fitness Team
Version: 3.0.0
"""

import streamlit as st
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@st.cache_data
def get_app_config():
    """Get application configuration."""
    return {
        'app_title': 'AI Fitness Assistant Pro',
        'version': '3.0.0',
        'max_image_size_mb': 10,
        'supported_formats': ['jpg', 'jpeg', 'png'],
        'default_workout_duration': 30,
        'bmr_formula': 'mifflin_st_jeor'
    }

# Enhanced styling
def inject_custom_css():
    """Inject custom CSS for modern UI."""
    st.markdown("""
    <style>
        /* Modern styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 4px solid #667eea;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            transition: transform 0.2s ease-in-out;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .recommendation-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .exercise-card {
            background: white;
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #e9ecef;
            margin: 0.5rem 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-beginner { background: #17a2b8; color: white; }
        .status-intermediate { background: #ffc107; color: white; }
        .status-advanced { background: #dc3545; color: white; }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Enhanced Data Models
class EnhancedUserProfile:
    """Enhanced user profile with validation."""
    
    def __init__(self, **kwargs):
        self.user_id = kwargs.get('user_id', str(uuid.uuid4()))
        self.age = kwargs.get('age', 25)
        self.gender = kwargs.get('gender', 'other')
        self.weight = kwargs.get('weight', 70.0)
        self.height = kwargs.get('height', 170.0)
        self.activity_level = kwargs.get('activity_level', 'moderately_active')
        self.fitness_level = kwargs.get('fitness_level', 'beginner')
        self.primary_goal = kwargs.get('primary_goal', 'general_fitness')
        self.available_time = kwargs.get('available_time', 30)
        self.workout_days_per_week = kwargs.get('workout_days_per_week', 3)
        self.available_equipment = kwargs.get('available_equipment', [])
        self.injuries = kwargs.get('injuries', [])
        self.medical_conditions = kwargs.get('medical_conditions', [])
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
    
    @property
    def bmi(self):
        """Calculate BMI."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 1)
    
    @property
    def bmi_category(self):
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
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'weight': self.weight,
            'height': self.height,
            'activity_level': self.activity_level,
            'fitness_level': self.fitness_level,
            'primary_goal': self.primary_goal,
            'available_time': self.available_time,
            'workout_days_per_week': self.workout_days_per_week,
            'available_equipment': self.available_equipment,
            'injuries': self.injuries,
            'medical_conditions': self.medical_conditions,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Enhanced Exercise Database
class EnhancedExerciseDatabase:
    """Comprehensive exercise database."""
    
    @staticmethod
    @st.cache_data
    def get_exercises():
        """Get comprehensive exercise database."""
        return {
            'cardio': {
                'running': {
                    'name': 'Running',
                    'calories_per_min': 10,
                    'difficulty': 2,
                    'equipment': [],
                    'muscle_groups': ['legs', 'cardiovascular'],
                    'instructions': [
                        'Start with 5-minute warm-up walk',
                        'Maintain steady pace you can sustain',
                        'Land on midfoot, not heel',
                        'Keep arms relaxed at 90-degree angle'
                    ],
                    'tips': ['Increase distance gradually', 'Focus on form over speed'],
                    'contraindications': ['knee injury', 'ankle injury']
                },
                'cycling': {
                    'name': 'Cycling',
                    'calories_per_min': 8,
                    'difficulty': 2,
                    'equipment': ['exercise_bike'],
                    'muscle_groups': ['legs', 'glutes', 'cardiovascular'],
                    'instructions': [
                        'Adjust seat height properly',
                        'Keep core engaged',
                        'Maintain steady cadence'
                    ],
                    'tips': ['Start with lower resistance', 'Gradually increase intensity'],
                    'contraindications': ['severe knee problems']
                },
                'jumping_jacks': {
                    'name': 'Jumping Jacks',
                    'calories_per_min': 8,
                    'difficulty': 1,
                    'equipment': [],
                    'muscle_groups': ['full_body', 'cardiovascular'],
                    'instructions': [
                        'Start with feet together, arms at sides',
                        'Jump feet apart while raising arms overhead',
                        'Jump back to starting position'
                    ],
                    'tips': ['Land softly on balls of feet', 'Keep core engaged'],
                    'contraindications': ['ankle injury', 'knee problems']
                },
                'burpees': {
                    'name': 'Burpees',
                    'calories_per_min': 12,
                    'difficulty': 4,
                    'equipment': [],
                    'muscle_groups': ['full_body', 'cardiovascular'],
                    'instructions': [
                        'Start in standing position',
                        'Drop into squat, hands on floor',
                        'Jump feet back to plank',
                        'Do push-up, jump feet to squat, jump up'
                    ],
                    'tips': ['Modify by stepping instead of jumping', 'Focus on form'],
                    'contraindications': ['back injury', 'wrist problems']
                }
            },
            'strength': {
                'push_ups': {
                    'name': 'Push-ups',
                    'calories_per_min': 6,
                    'difficulty': 2,
                    'equipment': [],
                    'muscle_groups': ['chest', 'shoulders', 'triceps', 'core'],
                    'instructions': [
                        'Start in plank position',
                        'Lower chest to nearly touch ground',
                        'Push back up to starting position',
                        'Keep body in straight line'
                    ],
                    'tips': ['Modify on knees if needed', 'Keep core tight'],
                    'contraindications': ['wrist injury', 'shoulder problems']
                },
                'squats': {
                    'name': 'Squats',
                    'calories_per_min': 7,
                    'difficulty': 2,
                    'equipment': [],
                    'muscle_groups': ['quadriceps', 'glutes', 'hamstrings'],
                    'instructions': [
                        'Stand with feet shoulder-width apart',
                        'Lower by pushing hips back',
                        'Go down until thighs parallel to ground',
                        'Drive through heels to stand'
                    ],
                    'tips': ['Keep chest up', 'Knees track over toes'],
                    'contraindications': ['severe knee problems']
                },
                'planks': {
                    'name': 'Planks',
                    'calories_per_min': 4,
                    'difficulty': 2,
                    'equipment': [],
                    'muscle_groups': ['core', 'shoulders'],
                    'instructions': [
                        'Start in forearm plank position',
                        'Keep body in straight line',
                        'Hold position while breathing normally'
                    ],
                    'tips': ['Don\'t let hips sag', 'Engage core and glutes'],
                    'contraindications': ['lower back injury']
                },
                'lunges': {
                    'name': 'Lunges',
                    'calories_per_min': 7,
                    'difficulty': 2,
                    'equipment': [],
                    'muscle_groups': ['legs', 'glutes', 'core'],
                    'instructions': [
                        'Step forward with one foot',
                        'Lower back knee toward ground',
                        'Keep front knee over ankle',
                        'Push off front foot to return'
                    ],
                    'tips': ['Alternate legs', 'Keep torso upright'],
                    'contraindications': ['knee injury']
                }
            },
            'flexibility': {
                'yoga_flow': {
                    'name': 'Yoga Flow',
                    'calories_per_min': 3,
                    'difficulty': 2,
                    'equipment': ['yoga_mat'],
                    'muscle_groups': ['full_body', 'flexibility'],
                    'instructions': [
                        'Flow through poses with breath',
                        'Hold each pose for 3-5 breaths',
                        'Focus on alignment'
                    ],
                    'tips': ['Listen to your body', 'Don\'t force poses'],
                    'contraindications': ['severe joint problems']
                },
                'stretching': {
                    'name': 'Full Body Stretching',
                    'calories_per_min': 2,
                    'difficulty': 1,
                    'equipment': [],
                    'muscle_groups': ['full_body', 'flexibility'],
                    'instructions': [
                        'Hold each stretch for 15-30 seconds',
                        'Breathe deeply and relax',
                        'Never bounce or force'
                    ],
                    'tips': ['Stretch after warming up', 'Focus on tight areas'],
                    'contraindications': ['acute muscle injury']
                }
            }
        }

# Enhanced Recommendation Engine
class SmartRecommendationEngine:
    """Smart exercise recommendation engine."""
    
    def __init__(self):
        self.exercise_db = EnhancedExerciseDatabase()
    
    def generate_recommendations(self, user_profile: EnhancedUserProfile) -> Dict[str, Any]:
        """Generate personalized exercise recommendations."""
        
        exercises = self.exercise_db.get_exercises()
        
        # Filter exercises based on user constraints
        suitable_exercises = self._filter_exercises(exercises, user_profile)
        
        # Generate recommendations by category
        recommendations = {
            'cardio': self._recommend_by_category(suitable_exercises.get('cardio', {}), user_profile, 'cardio'),
            'strength': self._recommend_by_category(suitable_exercises.get('strength', {}), user_profile, 'strength'),
            'flexibility': self._recommend_by_category(suitable_exercises.get('flexibility', {}), user_profile, 'flexibility')
        }
        
        # Create weekly plan
        weekly_plan = self._create_weekly_plan(recommendations, user_profile)
        
        # Calculate metrics
        estimated_calories = self._calculate_weekly_calories(weekly_plan, user_profile)
        
        return {
            'recommendations': recommendations,
            'weekly_plan': weekly_plan,
            'estimated_weekly_calories': estimated_calories,
            'bmr': self._calculate_bmr(user_profile),
            'daily_calorie_needs': self._calculate_daily_calories(user_profile),
            'safety_tips': self._get_safety_tips(user_profile),
            'nutrition_tips': self._get_nutrition_tips(user_profile)
        }
    
    def _filter_exercises(self, exercises: Dict, user_profile: EnhancedUserProfile) -> Dict:
        """Filter exercises based on user profile."""
        
        filtered = {}
        
        for category, category_exercises in exercises.items():
            filtered[category] = {}
            
            for exercise_id, exercise in category_exercises.items():
                # Check equipment requirements
                required_equipment = set(exercise.get('equipment', []))
                available_equipment = set(user_profile.available_equipment)
                
                if not required_equipment or required_equipment.issubset(available_equipment):
                    # Check difficulty vs fitness level
                    difficulty_limits = {'beginner': 2, 'intermediate': 3, 'advanced': 5}
                    max_difficulty = difficulty_limits.get(user_profile.fitness_level, 2)
                    
                    if exercise.get('difficulty', 1) <= max_difficulty:
                        # Check contraindications
                        contraindications = exercise.get('contraindications', [])
                        user_issues = user_profile.injuries + user_profile.medical_conditions
                        
                        if not any(issue.lower() in contra.lower() 
                                 for issue in user_issues 
                                 for contra in contraindications):
                            filtered[category][exercise_id] = exercise
        
        return filtered
    
    def _recommend_by_category(self, category_exercises: Dict, user_profile: EnhancedUserProfile, category: str) -> List[Dict]:
        """Recommend exercises for a specific category."""
        
        # Goal-based weighting
        goal_weights = {
            'weight_loss': {'cardio': 3, 'strength': 2, 'flexibility': 1},
            'muscle_gain': {'cardio': 1, 'strength': 3, 'flexibility': 1},
            'endurance': {'cardio': 3, 'strength': 1, 'flexibility': 2},
            'general_fitness': {'cardio': 2, 'strength': 2, 'flexibility': 2}
        }
        
        category_weight = goal_weights.get(user_profile.primary_goal, {}).get(category, 1)
        
        recommendations = []
        for exercise_id, exercise in category_exercises.items():
            score = self._calculate_exercise_score(exercise, user_profile, category_weight)
            
            recommendations.append({
                'id': exercise_id,
                'exercise': exercise,
                'score': score,
                'recommended_duration': self._get_duration_recommendation(exercise, user_profile),
                'recommended_sets': self._get_sets_recommendation(exercise, user_profile),
                'recommended_reps': self._get_reps_recommendation(exercise, user_profile)
            })
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]
    
    def _calculate_exercise_score(self, exercise: Dict, user_profile: EnhancedUserProfile, category_weight: int) -> float:
        """Calculate suitability score for an exercise."""
        
        score = 0.5  # Base score
        
        # Category weight from goals
        score += category_weight * 0.2
        
        # Difficulty appropriateness
        difficulty = exercise.get('difficulty', 1)
        fitness_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        user_level = fitness_levels.get(user_profile.fitness_level, 1)
        
        if abs(difficulty - user_level) <= 1:
            score += 0.3
        elif abs(difficulty - user_level) <= 2:
            score += 0.1
        
        # Equipment bonus
        if not exercise.get('equipment') or exercise.get('equipment') == []:
            score += 0.2  # Bonus for bodyweight exercises
        
        # Calorie burn appropriateness
        calories_per_min = exercise.get('calories_per_min', 5)
        if user_profile.primary_goal == 'weight_loss' and calories_per_min >= 8:
            score += 0.2
        
        return min(1.0, score)
    
    def _get_duration_recommendation(self, exercise: Dict, user_profile: EnhancedUserProfile) -> str:
        """Get duration recommendation for exercise."""
        
        base_time = user_profile.available_time
        category = exercise.get('muscle_groups', [''])[0]
        
        if 'cardiovascular' in exercise.get('muscle_groups', []):
            return f"{min(base_time - 10, 25)} minutes"
        elif 'flexibility' in exercise.get('muscle_groups', []):
            return f"10-15 minutes"
        else:
            return f"Include in {base_time}-minute workout"
    
    def _get_sets_recommendation(self, exercise: Dict, user_profile: EnhancedUserProfile) -> str:
        """Get sets recommendation."""
        
        if 'strength' not in exercise.get('muscle_groups', []):
            return "N/A"
        
        fitness_levels = {
            'beginner': '2-3 sets',
            'intermediate': '3-4 sets',
            'advanced': '3-5 sets'
        }
        
        return fitness_levels.get(user_profile.fitness_level, '2-3 sets')
    
    def _get_reps_recommendation(self, exercise: Dict, user_profile: EnhancedUserProfile) -> str:
        """Get reps recommendation."""
        
        if 'strength' not in exercise.get('muscle_groups', []):
            return "N/A"
        
        goal_reps = {
            'weight_loss': '12-15 reps',
            'muscle_gain': '8-12 reps',
            'endurance': '15-20 reps',
            'general_fitness': '10-15 reps'
        }
        
        return goal_reps.get(user_profile.primary_goal, '10-15 reps')
    
    def _create_weekly_plan(self, recommendations: Dict, user_profile: EnhancedUserProfile) -> Dict:
        """Create weekly workout plan."""
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plan = {}
        
        workout_days = user_profile.workout_days_per_week
        
        # Create workout schedule based on goals
        if user_profile.primary_goal == 'weight_loss':
            schedule = ['cardio', 'strength', 'cardio', 'strength', 'mixed', 'flexibility', 'rest']
        elif user_profile.primary_goal == 'muscle_gain':
            schedule = ['strength', 'strength', 'cardio', 'strength', 'rest', 'mixed', 'rest']
        else:
            schedule = ['cardio', 'strength', 'rest', 'mixed', 'strength', 'flexibility', 'rest']
        
        for i, day in enumerate(days):
            if i < workout_days:
                workout_type = schedule[i % len(schedule)]
                if workout_type == 'rest':
                    # If we encounter a rest day but still have workout days to fill,
                    # use a mixed workout instead
                    workout_type = 'mixed'
                plan[day] = self._create_day_plan(workout_type, recommendations, user_profile)
            else:
                plan[day] = {
                    'type': 'Rest Day',
                    'focus': 'Recovery and restoration',
                    'activities': ['Light walking', 'Gentle stretching'],
                    'duration': 15,
                    'instructions': 'Allow your body to recover and prepare for the next workout'
                }
        
        return plan
    
    def _create_day_plan(self, workout_type: str, recommendations: Dict, user_profile: EnhancedUserProfile) -> Dict:
        """Create plan for a specific day."""
        
        if workout_type == 'cardio':
            exercises = recommendations['cardio'][:2]
            return {
                'type': 'Cardio Day',
                'focus': 'Cardiovascular fitness',
                'exercises': [ex['exercise']['name'] for ex in exercises],
                'duration': user_profile.available_time,
                'instructions': 'Focus on maintaining steady pace and breathing'
            }
        
        elif workout_type == 'strength':
            exercises = recommendations['strength'][:4]
            return {
                'type': 'Strength Day', 
                'focus': 'Muscle building',
                'exercises': [ex['exercise']['name'] for ex in exercises],
                'duration': user_profile.available_time,
                'instructions': 'Focus on proper form and controlled movements'
            }
        
        elif workout_type == 'flexibility':
            exercises = recommendations['flexibility'][:2]
            return {
                'type': 'Flexibility Day',
                'focus': 'Mobility and recovery',
                'exercises': [ex['exercise']['name'] for ex in exercises],
                'duration': 20,
                'instructions': 'Hold stretches and breathe deeply'
            }
        
        else:  # mixed
            cardio = recommendations['cardio'][:1]
            strength = recommendations['strength'][:2]
            flexibility = recommendations['flexibility'][:1]
            
            mixed_exercises = []
            mixed_exercises.extend([ex['exercise']['name'] for ex in cardio])
            mixed_exercises.extend([ex['exercise']['name'] for ex in strength])
            mixed_exercises.extend([ex['exercise']['name'] for ex in flexibility])
            
            return {
                'type': 'Mixed Training',
                'focus': 'Balanced workout',
                'exercises': mixed_exercises,
                'duration': user_profile.available_time,
                'instructions': 'Combine cardio, strength, and flexibility'
            }
    
    def _calculate_weekly_calories(self, weekly_plan: Dict, user_profile: EnhancedUserProfile) -> int:
        """Calculate estimated weekly calorie burn."""
        
        total_calories = 0
        
        for day_plan in weekly_plan.values():
            duration = day_plan.get('duration', 0)
            workout_type = day_plan.get('type', '').lower()
            
            if 'cardio' in workout_type:
                calories_per_min = 8
            elif 'strength' in workout_type:
                calories_per_min = 6
            elif 'mixed' in workout_type:
                calories_per_min = 7
            else:
                calories_per_min = 3
            
            total_calories += calories_per_min * duration
        
        return int(total_calories)
    
    def _calculate_bmr(self, user_profile: EnhancedUserProfile) -> float:
        """Calculate Basal Metabolic Rate."""
        
        if user_profile.gender == 'male':
            bmr = 10 * user_profile.weight + 6.25 * user_profile.height - 5 * user_profile.age + 5
        else:
            bmr = 10 * user_profile.weight + 6.25 * user_profile.height - 5 * user_profile.age - 161
        
        return round(bmr, 0)
    
    def _calculate_daily_calories(self, user_profile: EnhancedUserProfile) -> float:
        """Calculate daily calorie needs."""
        
        bmr = self._calculate_bmr(user_profile)
        
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        
        multiplier = activity_multipliers.get(user_profile.activity_level, 1.375)
        return round(bmr * multiplier, 0)
    
    def _get_safety_tips(self, user_profile: EnhancedUserProfile) -> List[str]:
        """Get safety tips based on user profile."""
        
        tips = [
            "Always warm up before exercising",
            "Listen to your body and rest when needed",
            "Maintain proper form over intensity",
            "Stay hydrated during workouts"
        ]
        
        if user_profile.fitness_level == 'beginner':
            tips.extend([
                "Start slowly and progress gradually",
                "Consider working with a trainer initially"
            ])
        
        if user_profile.injuries:
            tips.append(f"Be cautious with exercises affecting {', '.join(user_profile.injuries)}")
        
        return tips
    
    def _get_nutrition_tips(self, user_profile: EnhancedUserProfile) -> List[str]:
        """Get nutrition tips based on goals."""
        
        base_tips = [
            "Stay hydrated throughout the day",
            "Eat protein with each meal",
            "Include fruits and vegetables daily"
        ]
        
        goal_tips = {
            'weight_loss': [
                "Create a moderate caloric deficit",
                "Focus on fiber-rich foods",
                "Eat smaller, frequent meals"
            ],
            'muscle_gain': [
                "Ensure adequate protein intake",
                "Don't skip carbohydrates",
                "Consider a slight caloric surplus"
            ],
            'endurance': [
                "Prioritize complex carbohydrates",
                "Time nutrition around workouts",
                "Focus on recovery foods"
            ]
        }
        
        specific_tips = goal_tips.get(user_profile.primary_goal, [])
        return base_tips + specific_tips

# Simple Database (File-based)
class SimpleDatabase:
    """Simple file-based database for development."""
    
    def __init__(self, db_file='fitness_data.json'):
        self.db_file = db_file
        self.data = self._load_data()
    
    def _load_data(self):
        """Load data from file."""
        try:
            with open(self.db_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'users': {}, 'workouts': {}, 'measurements': {}}
    
    def _save_data(self):
        """Save data to file."""
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def save_user_profile(self, user_id: str, profile_data: Dict):
        """Save user profile."""
        self.data['users'][user_id] = profile_data
        self._save_data()
        return True
    
    def get_user_profile(self, user_id: str):
        """Get user profile."""
        return self.data['users'].get(user_id)
    
    def save_workout(self, user_id: str, workout_data: Dict):
        """Save workout session."""
        if user_id not in self.data['workouts']:
            self.data['workouts'][user_id] = []
        
        workout_data['timestamp'] = datetime.now().isoformat()
        self.data['workouts'][user_id].append(workout_data)
        self._save_data()
        return True
    
    def get_workouts(self, user_id: str):
        """Get user workouts."""
        return self.data['workouts'].get(user_id, [])
    
    def save_measurement(self, user_id: str, measurement_data: Dict):
        """Save body measurement."""
        if user_id not in self.data['measurements']:
            self.data['measurements'][user_id] = []
        
        measurement_data['timestamp'] = datetime.now().isoformat()
        self.data['measurements'][user_id].append(measurement_data)
        self._save_data()
        return True
    
    def get_measurements(self, user_id: str):
        """Get user measurements."""
        return self.data['measurements'].get(user_id, [])

# Initialize database
@st.cache_resource
def get_database():
    """Get database instance."""
    return SimpleDatabase()

# Main Application
def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Fitness Assistant Pro",
        page_icon="🏋️‍♀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    inject_custom_css()
    
    # Header
    st.markdown('<div class="main-header">🏋️‍♀️ AI Fitness Assistant Pro</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6c757d;">Your Intelligent Fitness Companion v3.0</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    
    # Check if user has profile
    if st.session_state.user_profile is None:
        render_profile_setup()
    else:
        render_main_app()

def render_profile_setup():
    """Render user profile setup."""
    
    st.markdown("## 👤 Complete Your Profile")
    st.info("Please complete your profile to get personalized recommendations.")
    
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.5)
            activity_level = st.selectbox(
                "Activity Level",
                ["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"],
                index=2
            )
            primary_goal = st.selectbox(
                "Primary Goal",
                ["weight_loss", "muscle_gain", "endurance", "general_fitness"],
                index=3
            )
        
        with col2:
            gender = st.selectbox("Gender", ["male", "female", "other"], index=2)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
            fitness_level = st.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"], index=0)
            available_time = st.slider("Available Time per Workout (minutes)", 15, 120, 30, 5)
        
        workout_days = st.slider("Workout Days per Week", 1, 7, 3)
        
        available_equipment = st.multiselect(
            "Available Equipment",
            ["dumbbells", "barbell", "resistance_bands", "kettlebell", "pull_up_bar", 
             "exercise_bike", "treadmill", "yoga_mat", "bench"],
            help="Select all equipment you have access to"
        )
        
        injuries = st.text_area(
            "Injuries or Limitations (one per line)",
            placeholder="e.g.:\nknee injury\nlower back pain"
        )
        
        submitted = st.form_submit_button("💾 Save Profile", type="primary")
        
        if submitted:
            # Create user profile
            profile = EnhancedUserProfile(
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                activity_level=activity_level,
                fitness_level=fitness_level,
                primary_goal=primary_goal,
                available_time=available_time,
                workout_days_per_week=workout_days,
                available_equipment=available_equipment,
                injuries=[injury.strip() for injury in injuries.split('\n') if injury.strip()]
            )
            
            # Save to database
            db = get_database()
            success = db.save_user_profile(profile.user_id, profile.to_dict())
            
            if success:
                st.session_state.user_profile = profile
                st.success("✅ Profile saved successfully!")
                st.rerun()

def render_main_app():
    """Render main application interface."""
    
    user_profile = st.session_state.user_profile
    
    # Sidebar
    render_sidebar(user_profile)
    
    # Main tabs
    tabs = st.tabs(["📊 Dashboard", "💪 Recommendations", "📈 Progress", "�️ Body Analysis", "�🎯 Goals", "⚙️ Settings"])
    
    with tabs[0]:
        render_dashboard(user_profile)
    
    with tabs[1]:
        render_recommendations(user_profile)
    
    with tabs[2]:
        render_progress(user_profile)
    
    with tabs[3]:
        render_body_composition_tab(user_profile)
    
    with tabs[4]:
        render_goals(user_profile)
    
    with tabs[5]:
        render_settings(user_profile)

def render_sidebar(user_profile: EnhancedUserProfile):
    """Render sidebar with user info."""
    
    st.sidebar.markdown("## 👤 Your Profile")
    st.sidebar.markdown(f"**Age:** {user_profile.age}")
    st.sidebar.markdown(f"**BMI:** {user_profile.bmi} ({user_profile.bmi_category})")
    st.sidebar.markdown(f"**Fitness Level:** {user_profile.fitness_level.title()}")
    st.sidebar.markdown(f"**Primary Goal:** {user_profile.primary_goal.replace('_', ' ').title()}")
    
    st.sidebar.markdown("---")
    
    # Quick stats
    db = get_database()
    workouts = db.get_workouts(user_profile.user_id)
    measurements = db.get_measurements(user_profile.user_id)
    
    st.sidebar.markdown("## 📊 Quick Stats")
    st.sidebar.metric("Total Workouts", len(workouts))
    st.sidebar.metric("Measurements Logged", len(measurements))
    
    # Quick actions
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Update Profile"):
        st.session_state.user_profile = None
        st.rerun()

def render_dashboard(user_profile: EnhancedUserProfile):
    """Render user dashboard."""
    
    st.markdown("## 📊 Your Fitness Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    db = get_database()
    workouts = db.get_workouts(user_profile.user_id)
    measurements = db.get_measurements(user_profile.user_id)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>📈 BMI</h3><h2>{user_profile.bmi}</h2><p>{user_profile.bmi_category}</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>🏃‍♀️ Workouts</h3><h2>{len(workouts)}</h2><p>Total completed</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        engine = SmartRecommendationEngine()
        bmr = engine._calculate_bmr(user_profile)
        st.markdown(
            f'<div class="metric-card"><h3>🔥 BMR</h3><h2>{int(bmr)}</h2><p>Calories/day</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        daily_calories = engine._calculate_daily_calories(user_profile)
        st.markdown(
            f'<div class="metric-card"><h3>🍎 Daily Needs</h3><h2>{int(daily_calories)}</h2><p>Total calories</p></div>',
            unsafe_allow_html=True
        )
    
    # Recent activity
    st.markdown("### 📅 Recent Activity")
    
    if workouts:
        recent_workouts = sorted(workouts, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        for workout in recent_workouts:
            with st.expander(f"🏋️‍♀️ {workout.get('type', 'Workout')} - {workout.get('timestamp', '')[:10]}"):
                st.write(f"**Duration:** {workout.get('duration', 'N/A')} minutes")
                st.write(f"**Exercises:** {', '.join(workout.get('exercises', []))}")
                if workout.get('notes'):
                    st.write(f"**Notes:** {workout['notes']}")
    else:
        st.info("No workouts logged yet. Check out the Recommendations tab to get started!")

def render_recommendations(user_profile: EnhancedUserProfile):
    """Render exercise recommendations."""
    
    st.markdown("## 💪 Personalized Exercise Recommendations")
    
    # Generate recommendations
    with st.spinner("🤖 Generating personalized recommendations..."):
        engine = SmartRecommendationEngine()
        recommendations = engine.generate_recommendations(user_profile)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Weekly Calories", f"{recommendations['estimated_weekly_calories']:,}")
    
    with col2:
        st.metric("BMR", f"{int(recommendations['bmr'])}")
    
    with col3:
        st.metric("Daily Calories", f"{int(recommendations['daily_calorie_needs'])}")
    
    # Recommendations by category
    rec_tabs = st.tabs(["🏃‍♀️ Cardio", "🏋️‍♀️ Strength", "🧘‍♀️ Flexibility", "📅 Weekly Plan"])
    
    with rec_tabs[0]:
        render_category_recommendations("Cardio", recommendations['recommendations']['cardio'])
    
    with rec_tabs[1]:
        render_category_recommendations("Strength", recommendations['recommendations']['strength'])
    
    with rec_tabs[2]:
        render_category_recommendations("Flexibility", recommendations['recommendations']['flexibility'])
    
    with rec_tabs[3]:
        render_weekly_plan(recommendations['weekly_plan'])
    
    # Tips
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛡️ Safety Tips")
        for tip in recommendations['safety_tips']:
            st.markdown(f"• {tip}")
    
    with col2:
        st.markdown("### 🍎 Nutrition Tips")
        for tip in recommendations['nutrition_tips']:
            st.markdown(f"• {tip}")

def render_category_recommendations(category: str, recommendations: List[Dict]):
    """Render recommendations for a specific category."""
    
    st.markdown(f"### {category} Exercises")
    
    if not recommendations:
        st.warning(f"No suitable {category.lower()} exercises found.")
        return
    
    for rec in recommendations:
        exercise = rec['exercise']
        
        with st.expander(f"{exercise['name']} (Score: {rec['score']:.1f}/1.0)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Difficulty:** {exercise['difficulty']}/5")
                st.markdown(f"**Calories/min:** {exercise['calories_per_min']}")
                if rec.get('recommended_duration') != "N/A":
                    st.markdown(f"**Duration:** {rec['recommended_duration']}")
                if rec.get('recommended_sets') != "N/A":
                    st.markdown(f"**Sets:** {rec['recommended_sets']}")
                if rec.get('recommended_reps') != "N/A":
                    st.markdown(f"**Reps:** {rec['recommended_reps']}")
            
            with col2:
                st.markdown("**Target Muscles:**")
                for muscle in exercise['muscle_groups']:
                    st.markdown(f"• {muscle.title()}")
            
            if exercise.get('instructions'):
                st.markdown("**Instructions:**")
                for i, instruction in enumerate(exercise['instructions'], 1):
                    st.markdown(f"{i}. {instruction}")
            
            if exercise.get('tips'):
                st.markdown("**Tips:**")
                for tip in exercise['tips']:
                    st.markdown(f"💡 {tip}")

def render_weekly_plan(weekly_plan: Dict):
    """Render weekly workout plan."""
    
    st.markdown("### 📅 Your Weekly Plan")
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in days:
        if day in weekly_plan:
            plan = weekly_plan[day]
            
            with st.expander(f"📅 {day} - {plan['type']}"):
                st.markdown(f"**Focus:** {plan.get('focus', 'General fitness')}")
                st.markdown(f"**Duration:** {plan.get('duration', 30)} minutes")
                
                if plan.get('exercises'):
                    st.markdown("**Exercises:**")
                    for exercise in plan['exercises']:
                        st.markdown(f"• {exercise}")
                
                if plan.get('activities'):
                    st.markdown("**Activities:**")
                    for activity in plan['activities']:
                        st.markdown(f"• {activity}")
                
                if plan.get('instructions'):
                    st.info(f"💡 {plan['instructions']}")

def render_progress(user_profile: EnhancedUserProfile):
    """Render progress tracking."""
    
    st.markdown("## 📈 Your Fitness Progress")
    
    # Progress summary
    render_progress_summary(user_profile)
    
    # Detailed progress charts
    render_progress_charts(user_profile)

def render_body_composition_tab(user_profile):
    """Render body composition analysis tab."""
    try:
        from body_composition_ui import render_body_composition_analysis
        # Override user_id for the body composition analysis
        st.session_state['body_comp_user_id'] = user_profile.user_id
        render_body_composition_analysis()
    except ImportError:
        st.error("❌ Body composition analysis is not available.")
        st.info("Please install the required dependencies:")
        st.code("pip install opencv-python mediapipe scikit-learn tensorflow keras scipy")
        
        st.markdown("### What is Body Composition Analysis?")
        st.markdown("""
        Body composition analysis uses computer vision and AI to analyze your body from photos:
        
        - **Body Fat Percentage**: Estimated using pose landmarks and body ratios
        - **Muscle Mass**: Calculated based on body shape and proportions  
        - **BMR Estimation**: Basal Metabolic Rate for calorie planning
        - **Body Shape Classification**: Athletic, pear, apple, etc.
        - **Progress Tracking**: Compare changes over time
        - **Detailed Measurements**: Shoulder width, waist-to-hip ratio, etc.
        
        **How it works:**
        1. Upload a full-body photo
        2. AI extracts pose landmarks
        3. Machine learning models estimate composition
        4. Results are saved for progress tracking
        
        **Tips for best results:**
        - Use good lighting
        - Wear fitted clothing
        - Stand straight facing camera
        - Use consistent photo conditions
        """)

def render_progress_summary(user_profile: EnhancedUserProfile):
    """Render progress summary section."""
    
    st.markdown("### Summary")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    db = get_database()
    workouts = db.get_workouts(user_profile.user_id)
    measurements = db.get_measurements(user_profile.user_id)
    
    with col1:
        st.metric("Total Workouts", len(workouts))
    
    with col2:
        st.metric("Measurements Logged", len(measurements))
    
    with col3:
        # Calculate progress towards goals
        goals = db.data.get('goals', {}).get(user_profile.user_id, [])
        if goals:
            total_target = sum(goal.get('target_value', 0) for goal in goals)
            total_current = sum(goal.get('current_value', 0) for goal in goals)
            progress = min(total_current / total_target, 1) if total_target > 0 else 0
            st.progress(progress)
            st.markdown(f"**Overall Goal Progress:** {total_current:.1f} / {total_target:.1f}")
        else:
            st.markdown("**Overall Goal Progress:** No goals set")

def render_progress_charts(user_profile: EnhancedUserProfile):
    """Render detailed progress charts."""
    
    st.markdown("### Detailed Progress")
    
    # Workout history chart
    db = get_database()
    workouts = db.get_workouts(user_profile.user_id)
    
    if workouts:
        # Prepare data for chart
        workout_dates = [datetime.fromisoformat(w['timestamp']).date() for w in workouts]
        workout_counts = [len(w.get('exercises', [])) for w in workouts]
        
        # Plot
        st.line_chart(data={'Date': workout_dates, 'Exercises': workout_counts}, x='Date', y='Exercises')
    else:
        st.info("No workout data available for charting.")
    
    # Measurement history chart
    measurements = db.get_measurements(user_profile.user_id)
    
    if measurements:
        # Prepare data for chart
        measurement_dates = [datetime.fromisoformat(m['timestamp']).date() for m in measurements]
        weights = [m.get('weight', 0) for m in measurements]
        body_fats = [m.get('body_fat_percentage', 0) for m in measurements]
        
        # Plot
        st.line_chart(data={'Date': measurement_dates, 'Weight': weights, 'Body Fat %': body_fats}, x='Date', y=['Weight', 'Body Fat %'])
    else:
        st.info("No measurement data available for charting.")

def goal_form():
    """Render the goal creation form."""
    
    st.subheader("Set a New Goal")
    
    with st.form("goal_form"):
        goal_type = st.selectbox("Goal Type", ["Weight Loss", "Muscle Gain", "Strength", "Endurance"])
        target_value = st.number_input("Target Value", min_value=0.0, value=10.0, step=0.5)
        current_value = st.number_input("Current Value", min_value=0.0, value=0.0, step=0.5)
        target_date = st.date_input("Target Date", value=datetime.now().date() + timedelta(days=90))
        description = st.text_area("Goal Description", placeholder="Describe your goal...")
        
        submitted = st.form_submit_button("🎯 Create Goal")
        
        if submitted:
            # Save goal to database
            db = get_database()
            user_id = st.session_state.user_profile.user_id
            
            goal_data = {
                'type': goal_type,
                'target_value': target_value,
                'current_value': current_value,
                'target_date': target_date.isoformat(),
                'description': description,
                'created_at': datetime.now().isoformat()
            }
            
            # Simple append to file-based DB
            db.data.setdefault('goals', {}).setdefault(user_id, []).append(goal_data)
            db._save_data()
            
            st.success("✅ Goal created successfully!")
            st.session_state.user_profile.updated_at = datetime.now()  # Update profile timestamp
            st.experimental_rerun()

def render_goal_card(goal):
    """Render an individual goal card."""
    
    goal_type = goal.get('type', 'N/A')
    target_value = goal.get('target_value', 'N/A')
    current_value = goal.get('current_value', 'N/A')
    target_date = goal.get('target_date', 'N/A')
    description = goal.get('description', 'N/A')
    created_at = goal.get('created_at', 'N/A')
    
    with st.expander(f"🎯 {goal_type} Goal", expanded=True):
        st.markdown(f"**Target Value:** {target_value}")
        st.markdown(f"**Current Value:** {current_value}")
        st.markdown(f"**Target Date:** {target_date}")
        st.markdown(f"**Description:** {description}")
        st.markdown(f"**Created At:** {created_at}")
        
        # Progress bar
        if current_value and target_value:
            progress = min(max(current_value / target_value, 0), 1)
            st.progress(progress)
        
        # Edit and delete buttons
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("✏️ Edit", key=f"edit_{goal_type}"):
                st.session_state.editing_goal = goal
                st.experimental_rerun()
        
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{goal_type}", type="danger"):
                db = get_database()
                user_id = st.session_state.user_profile.user_id
                
                # Remove from file-based DB
                db.data['goals'][user_id] = [g for g in db.data['goals'].get(user_id, []) if g != goal]
                db._save_data()
