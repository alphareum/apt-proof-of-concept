"""
Enhanced Exercise Recommendation Engine
Improved algorithms, personalization, and progression planning

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import random
from models import (
    UserProfile, Exercise, Workout, WorkoutSet, FitnessGoal,
    GoalType, ExerciseCategory, FitnessLevel, ActivityLevel
)
from database import get_database

logger = logging.getLogger(__name__)

@dataclass
class RecommendationSettings:
    """Settings for exercise recommendations."""
    
    # Recommendation parameters
    max_exercises_per_workout: int = 8
    min_exercises_per_workout: int = 3
    variety_factor: float = 0.3  # 0-1, higher = more variety
    progression_factor: float = 0.1  # weekly progression rate
    
    # Safety parameters
    max_difficulty_increase: int = 1  # per progression cycle
    rest_day_frequency: int = 1  # rest days between intense workouts
    injury_buffer_weeks: int = 2  # weeks to avoid exercises after injury

class AdvancedExerciseRecommendationEngine:
    """Advanced recommendation engine with ML-inspired features."""
    
    def __init__(self):
        self.db = get_database()
        self.exercise_library = self._load_exercise_library()
        self.settings = RecommendationSettings()
    
    def _load_exercise_library(self) -> Dict[str, Exercise]:
        """Load comprehensive exercise library."""
        
        # Enhanced exercise database with detailed information
        exercises_data = {
            # Cardio exercises
            'running': {
                'name': 'Running',
                'category': ExerciseCategory.CARDIO,
                'muscle_groups': ['legs', 'core', 'cardiovascular'],
                'equipment': [],
                'difficulty': 2,
                'calories_per_minute': 10,
                'met_value': 8.0,
                'instructions': [
                    'Start with a 5-minute warm-up walk',
                    'Maintain a steady pace you can sustain',
                    'Land on midfoot, not heel',
                    'Keep arms relaxed at 90-degree angle',
                    'Breathe rhythmically - in through nose, out through mouth'
                ],
                'tips': [
                    'Increase distance gradually (10% rule)',
                    'Focus on form over speed',
                    'Stay hydrated before, during, and after'
                ],
                'contraindications': ['knee injury', 'ankle injury', 'shin splints'],
                'target_sets': 1,
                'modifications': {
                    'easier': 'Walk-run intervals',
                    'harder': 'Hill running or interval sprints'
                }
            },
            
            'cycling': {
                'name': 'Cycling',
                'category': ExerciseCategory.CARDIO,
                'muscle_groups': ['legs', 'glutes', 'cardiovascular'],
                'equipment': ['exercise_bike'],
                'difficulty': 2,
                'calories_per_minute': 8,
                'met_value': 6.8,
                'instructions': [
                    'Adjust seat height so leg is almost fully extended',
                    'Keep core engaged throughout',
                    'Maintain steady cadence (80-100 RPM)',
                    'Use both push and pull motions'
                ],
                'contraindications': ['severe knee problems'],
                'modifications': {
                    'easier': 'Lower resistance, steady pace',
                    'harder': 'Hill intervals or high-intensity intervals'
                }
            },
            
            # Strength exercises
            'push_ups': {
                'name': 'Push-ups',
                'category': ExerciseCategory.STRENGTH,
                'muscle_groups': ['chest', 'shoulders', 'triceps', 'core'],
                'equipment': [],
                'difficulty': 2,
                'calories_per_minute': 6,
                'met_value': 3.8,
                'instructions': [
                    'Start in plank position, hands slightly wider than shoulders',
                    'Keep body in straight line from head to heels',
                    'Lower chest to nearly touch ground',
                    'Push back up to starting position',
                    'Keep core tight throughout movement'
                ],
                'tips': [
                    'Focus on quality over quantity',
                    'Keep elbows at 45-degree angle',
                    'Breathe in on the way down, out on the way up'
                ],
                'contraindications': ['wrist injury', 'shoulder impingement'],
                'target_reps': '8-15',
                'target_sets': 3,
                'rest_time_seconds': 60,
                'modifications': {
                    'easier': 'Knee push-ups or incline push-ups',
                    'harder': 'Decline push-ups or diamond push-ups'
                }
            },
            
            'squats': {
                'name': 'Squats',
                'category': ExerciseCategory.STRENGTH,
                'muscle_groups': ['quadriceps', 'glutes', 'hamstrings', 'core'],
                'equipment': [],
                'difficulty': 2,
                'calories_per_minute': 7,
                'met_value': 5.0,
                'instructions': [
                    'Stand with feet shoulder-width apart',
                    'Keep chest up and core engaged',
                    'Lower by pushing hips back and bending knees',
                    'Go down until thighs are parallel to ground',
                    'Drive through heels to return to standing'
                ],
                'tips': [
                    'Keep knees aligned over toes',
                    'Weight should be on heels and mid-foot',
                    'Imagine sitting back into a chair'
                ],
                'contraindications': ['severe knee problems', 'hip impingement'],
                'target_reps': '10-20',
                'target_sets': 3,
                'rest_time_seconds': 60,
                'modifications': {
                    'easier': 'Chair-assisted squats or partial range',
                    'harder': 'Jump squats or goblet squats with weight'
                }
            },
            
            'planks': {
                'name': 'Planks',
                'category': ExerciseCategory.STRENGTH,
                'muscle_groups': ['core', 'shoulders', 'back'],
                'equipment': [],
                'difficulty': 2,
                'calories_per_minute': 4,
                'met_value': 3.0,
                'instructions': [
                    'Start in push-up position on forearms',
                    'Keep body in straight line from head to heels',
                    'Engage core and glutes',
                    'Hold position while breathing normally',
                    'Avoid sagging hips or raising butt'
                ],
                'tips': [
                    'Quality over duration - maintain perfect form',
                    'Look at floor about 12 inches in front of hands',
                    'Breathe steadily throughout'
                ],
                'contraindications': ['lower back injury', 'wrist problems'],
                'target_reps': '30-60 seconds',
                'target_sets': 3,
                'rest_time_seconds': 30,
                'modifications': {
                    'easier': 'Knee plank or incline plank',
                    'harder': 'Single-arm plank or plank with leg lifts'
                }
            },
            
            # Add more exercises...
            'deadlifts': {
                'name': 'Deadlifts',
                'category': ExerciseCategory.STRENGTH,
                'muscle_groups': ['hamstrings', 'glutes', 'back', 'core'],
                'equipment': ['dumbbells', 'barbell'],
                'difficulty': 4,
                'calories_per_minute': 8,
                'met_value': 6.0,
                'instructions': [
                    'Stand with feet hip-width apart, weights in front of thighs',
                    'Hinge at hips, keeping chest up and back straight',
                    'Lower weights while pushing hips back',
                    'Keep weights close to body throughout movement',
                    'Drive hips forward to return to standing'
                ],
                'contraindications': ['lower back injury', 'herniated disc'],
                'target_reps': '6-12',
                'target_sets': 3,
                'modifications': {
                    'easier': 'Romanian deadlift with light weights',
                    'harder': 'Single-leg deadlifts or sumo deadlifts'
                }
            },
            
            # Flexibility exercises
            'yoga_flow': {
                'name': 'Yoga Flow',
                'category': ExerciseCategory.FLEXIBILITY,
                'muscle_groups': ['full_body', 'core', 'flexibility'],
                'equipment': ['yoga_mat'],
                'difficulty': 2,
                'calories_per_minute': 3,
                'met_value': 2.5,
                'instructions': [
                    'Begin in mountain pose with feet hip-width apart',
                    'Flow through poses with controlled breathing',
                    'Hold each pose for 3-5 breaths',
                    'Focus on alignment and breath awareness',
                    'End in relaxation pose'
                ],
                'contraindications': ['severe joint problems'],
                'target_reps': '20-60 minutes',
                'modifications': {
                    'easier': 'Chair yoga or restorative poses',
                    'harder': 'Advanced poses and longer holds'
                }
            }
        }
        
        # Convert to Exercise objects
        exercises = {}
        for key, data in exercises_data.items():
            exercise = Exercise(
                id=key,
                name=data['name'],
                category=data['category'],
                muscle_groups=data['muscle_groups'],
                equipment=data.get('equipment', []),
                difficulty=data['difficulty'],
                calories_per_minute=data['calories_per_minute'],
                met_value=data['met_value'],
                instructions=data['instructions'],
                tips=data.get('tips', []),
                contraindications=data.get('contraindications', []),
                target_reps=data.get('target_reps'),
                target_sets=data.get('target_sets'),
                rest_time_seconds=data.get('rest_time_seconds'),
                modifications=data.get('modifications', {})
            )
            exercises[key] = exercise
        
        return exercises
    
    def generate_personalized_recommendations(self, user_profile: UserProfile, 
                                            workout_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive personalized exercise recommendations."""
        
        try:
            logger.info(f"Generating recommendations for user {user_profile.user_id}")
            
            # Analyze user preferences and history
            preferences = self._analyze_user_preferences(user_profile, workout_history or [])
            
            # Generate recommendations by category
            recommendations = {
                'cardio': self._recommend_cardio_exercises(user_profile, preferences),
                'strength': self._recommend_strength_exercises(user_profile, preferences),
                'flexibility': self._recommend_flexibility_exercises(user_profile, preferences),
                'weekly_plan': {},
                'progression_plan': {},
                'nutrition_tips': [],
                'safety_guidelines': []
            }
            
            # Create weekly workout plan
            recommendations['weekly_plan'] = self._create_weekly_plan(
                user_profile, recommendations, preferences
            )
            
            # Generate progression plan
            recommendations['progression_plan'] = self._create_progression_plan(
                user_profile, recommendations
            )
            
            # Add nutrition and safety recommendations
            recommendations['nutrition_tips'] = self._get_nutrition_recommendations(user_profile)
            recommendations['safety_guidelines'] = self._get_safety_guidelines(user_profile)
            
            # Calculate metrics
            recommendations['estimated_weekly_calories'] = self._calculate_weekly_calories(
                recommendations['weekly_plan'], user_profile
            )
            
            recommendations['recommendation_confidence'] = self._calculate_confidence_score(
                user_profile, len(workout_history or [])
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': f'Failed to generate recommendations: {str(e)}'}
    
    def _analyze_user_preferences(self, user_profile: UserProfile, 
                                workout_history: List[Dict]) -> Dict[str, Any]:
        """Analyze user preferences from profile and history."""
        
        preferences = {
            'preferred_exercises': set(user_profile.favorite_exercises),
            'avoided_exercises': set(user_profile.disliked_exercises),
            'exercise_frequency': {},
            'preferred_duration': user_profile.available_time,
            'intensity_preference': 'moderate',
            'variety_score': 0.5
        }
        
        # Analyze workout history for patterns
        if workout_history:
            exercise_counts = {}
            total_workouts = len(workout_history)
            
            for workout in workout_history:
                if 'exercises' in workout:
                    exercises = json.loads(workout.get('exercises', '[]'))
                    for exercise in exercises:
                        exercise_name = exercise if isinstance(exercise, str) else exercise.get('name', '')
                        exercise_counts[exercise_name] = exercise_counts.get(exercise_name, 0) + 1
            
            # Calculate frequency preferences
            preferences['exercise_frequency'] = {
                exercise: count / total_workouts 
                for exercise, count in exercise_counts.items()
            }
            
            # Calculate variety preference based on exercise diversity
            unique_exercises = len(exercise_counts)
            if total_workouts > 0:
                preferences['variety_score'] = min(unique_exercises / (total_workouts * 3), 1.0)
        
        # Adjust based on fitness level
        intensity_mapping = {
            FitnessLevel.BEGINNER: 'low',
            FitnessLevel.INTERMEDIATE: 'moderate',
            FitnessLevel.ADVANCED: 'high'
        }
        preferences['intensity_preference'] = intensity_mapping[user_profile.fitness_level]
        
        return preferences
    
    def _recommend_cardio_exercises(self, user_profile: UserProfile, 
                                  preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend cardio exercises based on user profile."""
        
        cardio_exercises = [
            ex for ex in self.exercise_library.values() 
            if ex.category == ExerciseCategory.CARDIO and ex.is_suitable_for_user(user_profile)
        ]
        
        recommendations = []
        for exercise in cardio_exercises:
            score = self._calculate_exercise_score(exercise, user_profile, preferences)
            if score > 0.3:  # Threshold for inclusion
                recommendation = {
                    'exercise': exercise,
                    'score': score,
                    'recommended_duration': self._get_recommended_duration(exercise, user_profile),
                    'intensity_level': self._get_intensity_level(exercise, user_profile),
                    'weekly_frequency': self._get_weekly_frequency(exercise, user_profile),
                    'progression_notes': self._get_progression_notes(exercise, user_profile)
                }
                recommendations.append(recommendation)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]
    
    def _recommend_strength_exercises(self, user_profile: UserProfile,
                                    preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend strength exercises based on user profile."""
        
        strength_exercises = [
            ex for ex in self.exercise_library.values()
            if ex.category == ExerciseCategory.STRENGTH and ex.is_suitable_for_user(user_profile)
        ]
        
        # Ensure balanced muscle group coverage
        muscle_groups_covered = set()
        recommendations = []
        
        for exercise in sorted(strength_exercises, 
                             key=lambda x: self._calculate_exercise_score(x, user_profile, preferences),
                             reverse=True):
            
            score = self._calculate_exercise_score(exercise, user_profile, preferences)
            
            # Prioritize exercises that cover new muscle groups
            new_muscle_groups = set(exercise.muscle_groups) - muscle_groups_covered
            if new_muscle_groups or len(recommendations) < 3:
                muscle_groups_covered.update(exercise.muscle_groups)
                
                recommendation = {
                    'exercise': exercise,
                    'score': score,
                    'recommended_sets': self._get_recommended_sets(exercise, user_profile),
                    'recommended_reps': self._get_recommended_reps(exercise, user_profile),
                    'rest_time': exercise.rest_time_seconds or 60,
                    'weekly_frequency': self._get_weekly_frequency(exercise, user_profile),
                    'progression_plan': self._get_strength_progression(exercise, user_profile)
                }
                recommendations.append(recommendation)
        
        return recommendations[:6]
    
    def _recommend_flexibility_exercises(self, user_profile: UserProfile,
                                       preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend flexibility exercises."""
        
        flexibility_exercises = [
            ex for ex in self.exercise_library.values()
            if ex.category == ExerciseCategory.FLEXIBILITY and ex.is_suitable_for_user(user_profile)
        ]
        
        recommendations = []
        for exercise in flexibility_exercises:
            score = self._calculate_exercise_score(exercise, user_profile, preferences)
            
            recommendation = {
                'exercise': exercise,
                'score': score,
                'recommended_duration': '10-15 minutes',
                'frequency': 'Daily or after workouts',
                'focus_areas': exercise.muscle_groups,
                'benefits': self._get_flexibility_benefits(exercise)
            }
            recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def _calculate_exercise_score(self, exercise: Exercise, user_profile: UserProfile,
                                preferences: Dict[str, Any]) -> float:
        """Calculate suitability score for an exercise."""
        
        score = 0.5  # Base score
        
        # Goal alignment
        goal_weights = {
            GoalType.WEIGHT_LOSS: {'cardio': 0.4, 'strength': 0.3, 'flexibility': 0.1},
            GoalType.MUSCLE_GAIN: {'cardio': 0.1, 'strength': 0.5, 'flexibility': 0.1},
            GoalType.ENDURANCE: {'cardio': 0.5, 'strength': 0.2, 'flexibility': 0.2},
            GoalType.STRENGTH: {'cardio': 0.2, 'strength': 0.5, 'flexibility': 0.1},
            GoalType.FLEXIBILITY: {'cardio': 0.1, 'strength': 0.2, 'flexibility': 0.5},
            GoalType.GENERAL_FITNESS: {'cardio': 0.3, 'strength': 0.3, 'flexibility': 0.3}
        }
        
        category_str = exercise.category.value
        primary_goal_weight = goal_weights.get(user_profile.primary_goal, {}).get(category_str, 0.2)
        score += primary_goal_weight
        
        # Secondary goals
        for goal in user_profile.secondary_goals:
            secondary_weight = goal_weights.get(goal, {}).get(category_str, 0.1)
            score += secondary_weight * 0.3  # Reduced weight for secondary goals
        
        # Fitness level appropriateness
        level_diff = abs(exercise.difficulty - user_profile.fitness_level.value.__hash__() % 3 + 1)
        if level_diff == 0:
            score += 0.2
        elif level_diff == 1:
            score += 0.1
        else:
            score -= 0.1
        
        # User preferences
        if exercise.name.lower() in [ex.lower() for ex in preferences['preferred_exercises']]:
            score += 0.3
        elif exercise.name.lower() in [ex.lower() for ex in preferences['avoided_exercises']]:
            score -= 0.5
        
        # Frequency in history (variety consideration)
        historical_frequency = preferences['exercise_frequency'].get(exercise.name, 0)
        if historical_frequency > 0.7:  # Too frequent
            score -= 0.2 * preferences['variety_score']
        elif 0.2 <= historical_frequency <= 0.5:  # Good frequency
            score += 0.1
        
        # Equipment availability
        if not exercise.equipment or all(eq in user_profile.available_equipment for eq in exercise.equipment):
            score += 0.1
        else:
            score -= 0.3
        
        # Safety considerations
        if any(injury.lower() in contra.lower() 
               for injury in user_profile.injuries 
               for contra in exercise.contraindications):
            score -= 0.8
        
        return max(0, min(1, score))
    
    def _create_weekly_plan(self, user_profile: UserProfile, 
                          recommendations: Dict[str, List], 
                          preferences: Dict[str, Any]) -> Dict[str, Dict]:
        """Create a balanced weekly workout plan."""
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_plan = {}
        
        workout_days = user_profile.workout_days_per_week
        rest_days = 7 - workout_days
        
        # Create workout schedule based on user preferences
        workout_schedule = self._create_workout_schedule(workout_days, user_profile.primary_goal)
        
        # Add rest days
        workout_schedule.extend(['rest'] * rest_days)
        
        for i, day in enumerate(days):
            if i < len(workout_schedule):
                workout_type = workout_schedule[i]
                
                if workout_type == 'cardio':
                    weekly_plan[day] = self._create_cardio_day(
                        recommendations['cardio'], user_profile
                    )
                elif workout_type == 'strength':
                    weekly_plan[day] = self._create_strength_day(
                        recommendations['strength'], user_profile
                    )
                elif workout_type == 'mixed':
                    weekly_plan[day] = self._create_mixed_day(
                        recommendations, user_profile
                    )
                elif workout_type == 'flexibility':
                    weekly_plan[day] = self._create_flexibility_day(
                        recommendations['flexibility'], user_profile
                    )
                else:  # rest
                    weekly_plan[day] = {
                        'type': 'Rest Day',
                        'focus': 'Recovery and light activity',
                        'activities': ['Light walking', 'Gentle stretching', 'Hydration focus'],
                        'duration': 15,
                        'notes': 'Listen to your body and rest if needed'
                    }
        
        return weekly_plan
    
    def _create_workout_schedule(self, workout_days: int, primary_goal: GoalType) -> List[str]:
        """Create optimal workout schedule based on goals."""
        
        schedule_templates = {
            3: {
                GoalType.WEIGHT_LOSS: ['cardio', 'strength', 'mixed'],
                GoalType.MUSCLE_GAIN: ['strength', 'strength', 'cardio'],
                GoalType.ENDURANCE: ['cardio', 'cardio', 'strength'],
                GoalType.GENERAL_FITNESS: ['cardio', 'strength', 'mixed']
            },
            4: {
                GoalType.WEIGHT_LOSS: ['cardio', 'strength', 'cardio', 'mixed'],
                GoalType.MUSCLE_GAIN: ['strength', 'strength', 'cardio', 'strength'],
                GoalType.ENDURANCE: ['cardio', 'strength', 'cardio', 'cardio'],
                GoalType.GENERAL_FITNESS: ['cardio', 'strength', 'mixed', 'flexibility']
            },
            5: {
                GoalType.WEIGHT_LOSS: ['cardio', 'strength', 'cardio', 'mixed', 'cardio'],
                GoalType.MUSCLE_GAIN: ['strength', 'strength', 'cardio', 'strength', 'mixed'],
                GoalType.ENDURANCE: ['cardio', 'strength', 'cardio', 'mixed', 'cardio'],
                GoalType.GENERAL_FITNESS: ['cardio', 'strength', 'mixed', 'strength', 'flexibility']
            }
        }
        
        # Default to 3-day schedule if not specified
        days_template = schedule_templates.get(workout_days, schedule_templates[3])
        return days_template.get(primary_goal, days_template[GoalType.GENERAL_FITNESS])
    
    def _create_cardio_day(self, cardio_recommendations: List[Dict], 
                          user_profile: UserProfile) -> Dict[str, Any]:
        """Create a cardio-focused workout day."""
        
        if not cardio_recommendations:
            return {'type': 'Cardio', 'exercises': [], 'duration': 30}
        
        # Select 1-2 primary cardio exercises
        primary_cardio = cardio_recommendations[0]
        
        return {
            'type': 'Cardio Day',
            'focus': 'Cardiovascular endurance and fat burning',
            'exercises': [
                {
                    'name': primary_cardio['exercise'].name,
                    'duration': primary_cardio['recommended_duration'],
                    'intensity': primary_cardio['intensity_level'],
                    'notes': f"Focus on {primary_cardio['exercise'].tips[0] if primary_cardio['exercise'].tips else 'maintaining steady pace'}"
                }
            ],
            'warm_up': '5-10 minutes light movement',
            'cool_down': '5-10 minutes stretching',
            'total_duration': user_profile.available_time,
            'estimated_calories': self._estimate_calories(primary_cardio['exercise'], 
                                                        user_profile.available_time - 10, 
                                                        user_profile.weight)
        }
    
    def _create_strength_day(self, strength_recommendations: List[Dict],
                           user_profile: UserProfile) -> Dict[str, Any]:
        """Create a strength-focused workout day."""
        
        if not strength_recommendations:
            return {'type': 'Strength', 'exercises': [], 'duration': 30}
        
        # Select 3-4 strength exercises for balanced workout
        selected_exercises = strength_recommendations[:4]
        
        exercises = []
        for rec in selected_exercises:
            exercises.append({
                'name': rec['exercise'].name,
                'sets': rec['recommended_sets'],
                'reps': rec['recommended_reps'],
                'rest': f"{rec['rest_time']} seconds",
                'muscle_groups': rec['exercise'].muscle_groups,
                'tips': rec['exercise'].tips[:2] if rec['exercise'].tips else []
            })
        
        return {
            'type': 'Strength Day',
            'focus': 'Muscle building and strength development',
            'exercises': exercises,
            'warm_up': '5-10 minutes dynamic stretching',
            'cool_down': '5-10 minutes static stretching',
            'total_duration': user_profile.available_time,
            'progression_notes': 'Increase weight/reps when you can complete all sets with 2 reps in reserve'
        }
    
    def _create_mixed_day(self, recommendations: Dict[str, List],
                         user_profile: UserProfile) -> Dict[str, Any]:
        """Create a mixed cardio/strength workout day."""
        
        cardio_exercises = recommendations.get('cardio', [])[:1]
        strength_exercises = recommendations.get('strength', [])[:2]
        flexibility_exercises = recommendations.get('flexibility', [])[:1]
        
        exercises = []
        
        # Add cardio
        if cardio_exercises:
            exercises.append({
                'name': cardio_exercises[0]['exercise'].name,
                'type': 'Cardio',
                'duration': '10-15 minutes',
                'intensity': 'moderate'
            })
        
        # Add strength
        for rec in strength_exercises:
            exercises.append({
                'name': rec['exercise'].name,
                'type': 'Strength',
                'sets': rec['recommended_sets'],
                'reps': rec['recommended_reps']
            })
        
        # Add flexibility
        if flexibility_exercises:
            exercises.append({
                'name': flexibility_exercises[0]['exercise'].name,
                'type': 'Flexibility',
                'duration': '5-10 minutes'
            })
        
        return {
            'type': 'Mixed Training',
            'focus': 'Balanced cardiovascular and strength training',
            'exercises': exercises,
            'total_duration': user_profile.available_time,
            'structure': 'Cardio -> Strength -> Cool-down with flexibility'
        }
    
    def _create_flexibility_day(self, flexibility_recommendations: List[Dict],
                              user_profile: UserProfile) -> Dict[str, Any]:
        """Create a flexibility-focused day."""
        
        return {
            'type': 'Flexibility & Recovery',
            'focus': 'Mobility, flexibility, and recovery',
            'activities': [
                'Full body stretching routine',
                'Foam rolling (if available)', 
                'Deep breathing exercises',
                'Light yoga or tai chi'
            ],
            'duration': 20,
            'benefits': [
                'Improved range of motion',
                'Reduced muscle tension',
                'Better recovery',
                'Stress reduction'
            ]
        }
    
    def _get_nutrition_recommendations(self, user_profile: UserProfile) -> List[str]:
        """Get nutrition recommendations based on goals."""
        
        base_tips = [
            "Stay hydrated - aim for 8-10 glasses of water daily",
            "Eat protein with each meal to support muscle recovery",
            "Include plenty of fruits and vegetables for micronutrients",
            "Time your meals around workouts for optimal energy"
        ]
        
        goal_specific_tips = {
            GoalType.WEIGHT_LOSS: [
                "Create a moderate caloric deficit of 300-500 calories per day",
                "Focus on fiber-rich foods to help with satiety",
                "Consider smaller, more frequent meals"
            ],
            GoalType.MUSCLE_GAIN: [
                "Ensure adequate protein intake (1.6-2.2g per kg body weight)",
                "Don't neglect carbohydrates for workout fuel",
                "Consider a slight caloric surplus"
            ],
            GoalType.ENDURANCE: [
                "Prioritize complex carbohydrates for sustained energy",
                "Consider electrolyte replacement during longer sessions",
                "Focus on recovery nutrition post-workout"
            ]
        }
        
        specific_tips = goal_specific_tips.get(user_profile.primary_goal, [])
        return base_tips + specific_tips
    
    def _get_safety_guidelines(self, user_profile: UserProfile) -> List[str]:
        """Get safety guidelines based on user profile."""
        
        guidelines = [
            "Always warm up before exercising and cool down afterward",
            "Listen to your body and rest when needed",
            "Maintain proper form - quality over quantity",
            "Progress gradually to avoid overuse injuries"
        ]
        
        if user_profile.fitness_level == FitnessLevel.BEGINNER:
            guidelines.extend([
                "Start with bodyweight exercises before adding weights",
                "Consider working with a qualified trainer initially",
                "Don't compare your progress to others"
            ])
        
        if user_profile.injuries:
            guidelines.extend([
                f"Be cautious with exercises affecting {', '.join(user_profile.injuries)}",
                "Consult healthcare providers about exercise modifications",
                "Stop immediately if you feel pain (not to be confused with muscle fatigue)"
            ])
        
        if user_profile.age > 50:
            guidelines.extend([
                "Pay extra attention to joint mobility and flexibility",
                "Consider lower-impact alternatives when possible",
                "Allow for longer recovery periods between intense sessions"
            ])
        
        return guidelines
    
    def _calculate_weekly_calories(self, weekly_plan: Dict[str, Dict], 
                                 user_profile: UserProfile) -> int:
        """Calculate estimated weekly calorie burn."""
        
        total_calories = 0
        
        for day_plan in weekly_plan.values():
            if 'estimated_calories' in day_plan:
                total_calories += day_plan['estimated_calories']
            else:
                # Estimate based on duration and intensity
                duration = day_plan.get('total_duration', day_plan.get('duration', 30))
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
    
    def _estimate_calories(self, exercise: Exercise, duration_minutes: int, 
                          user_weight: float) -> int:
        """Estimate calories burned for specific exercise."""
        return int(exercise.calculate_calories(duration_minutes, user_weight))
    
    def _get_recommended_duration(self, exercise: Exercise, 
                                user_profile: UserProfile) -> str:
        """Get recommended duration for exercise."""
        
        base_duration = user_profile.available_time - 10  # Account for warm-up/cool-down
        
        if exercise.category == ExerciseCategory.CARDIO:
            if user_profile.fitness_level == FitnessLevel.BEGINNER:
                return f"{min(15, base_duration)} minutes"
            elif user_profile.fitness_level == FitnessLevel.INTERMEDIATE:
                return f"{min(25, base_duration)} minutes"
            else:
                return f"{min(35, base_duration)} minutes"
        
        return f"{base_duration} minutes"
    
    def _get_intensity_level(self, exercise: Exercise, user_profile: UserProfile) -> str:
        """Get appropriate intensity level."""
        
        intensity_map = {
            FitnessLevel.BEGINNER: "Low to moderate",
            FitnessLevel.INTERMEDIATE: "Moderate to high", 
            FitnessLevel.ADVANCED: "High intensity"
        }
        
        return intensity_map[user_profile.fitness_level]
    
    def _get_weekly_frequency(self, exercise: Exercise, user_profile: UserProfile) -> str:
        """Get recommended weekly frequency."""
        
        if exercise.category == ExerciseCategory.CARDIO:
            return "3-5 times per week"
        elif exercise.category == ExerciseCategory.STRENGTH:
            return "2-3 times per week"
        else:  # Flexibility
            return "Daily or after each workout"
    
    def _get_recommended_sets(self, exercise: Exercise, user_profile: UserProfile) -> str:
        """Get recommended number of sets."""
        
        if exercise.target_sets:
            return str(exercise.target_sets)
        
        level_sets = {
            FitnessLevel.BEGINNER: "2-3",
            FitnessLevel.INTERMEDIATE: "3-4",
            FitnessLevel.ADVANCED: "3-5"
        }
        
        return level_sets[user_profile.fitness_level]
    
    def _get_recommended_reps(self, exercise: Exercise, user_profile: UserProfile) -> str:
        """Get recommended repetitions."""
        
        if exercise.target_reps:
            return exercise.target_reps
        
        if user_profile.primary_goal == GoalType.STRENGTH:
            return "6-8 reps (heavy weight)"
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            return "8-12 reps"
        elif user_profile.primary_goal == GoalType.ENDURANCE:
            return "12-20 reps"
        else:
            return "10-15 reps"
    
    def _calculate_confidence_score(self, user_profile: UserProfile, 
                                  history_length: int) -> float:
        """Calculate confidence score for recommendations."""
        
        score = 0.5  # Base confidence
        
        # Profile completeness
        profile_fields = [
            user_profile.age, user_profile.weight, user_profile.height,
            user_profile.activity_level, user_profile.fitness_level,
            user_profile.primary_goal
        ]
        
        completeness = sum(1 for field in profile_fields if field) / len(profile_fields)
        score += completeness * 0.3
        
        # History availability
        if history_length > 0:
            history_factor = min(history_length / 10, 1.0)  # Max benefit at 10 workouts
            score += history_factor * 0.2
        
        return min(1.0, score)
    
    def _create_progression_plan(self, user_profile: UserProfile, 
                               recommendations: Dict[str, List]) -> Dict[str, str]:
        """Create a progression plan for the user."""
        
        progression_plans = {
            FitnessLevel.BEGINNER: {
                'week_1_2': 'Focus on learning proper form and establishing routine. Complete workouts at comfortable intensity.',
                'week_3_4': 'Gradually increase workout duration by 5-10 minutes. Add light resistance if exercises feel too easy.',
                'week_5_8': 'Increase intensity moderately. Add more challenging exercise variations when ready.',
                'month_2_3': 'Progress to intermediate exercises. Consider adding an extra workout day.',
                'month_4_6': 'Continue building strength and endurance. Reassess goals and adjust program.'
            },
            FitnessLevel.INTERMEDIATE: {
                'week_1_2': 'Establish consistent routine with current recommendations. Focus on perfect form.',
                'week_3_4': 'Increase weights by 5-10% or add 2-3 reps when exercises feel manageable.',
                'week_5_8': 'Introduce advanced variations and compound movements. Increase workout frequency if desired.',
                'month_2_3': 'Consider periodization - alternate between strength and endurance phases.',
                'month_4_6': 'Explore specialized training methods. Set new challenging goals.'
            },
            FitnessLevel.ADVANCED: {
                'week_1_2': 'Fine-tune current routine. Focus on weak points and technique refinement.',
                'week_3_4': 'Implement advanced training techniques (supersets, drop sets, etc.).',
                'week_5_8': 'Add sport-specific or goal-specific training elements.',
                'month_2_3': 'Consider competition preparation or advanced specialization.',
                'month_4_6': 'Mentor others or explore new fitness disciplines.'
            }
        }
        
        return progression_plans[user_profile.fitness_level]
    
    def _get_progression_notes(self, exercise: Exercise, user_profile: UserProfile) -> str:
        """Get exercise-specific progression notes."""
        
        if exercise.category == ExerciseCategory.CARDIO:
            return "Gradually increase duration by 2-3 minutes per week or add intervals"
        elif exercise.category == ExerciseCategory.STRENGTH:
            return "Increase weight by 2.5-5% when you can complete all sets with 2 reps in reserve"
        else:
            return "Increase hold time or try more advanced variations"
    
    def _get_strength_progression(self, exercise: Exercise, user_profile: UserProfile) -> Dict[str, str]:
        """Get detailed strength progression plan."""
        
        return {
            'beginner': f"Start with {exercise.modifications.get('easier', 'bodyweight version')}",
            'progression': "Add resistance or increase reps gradually",
            'advanced': f"Progress to {exercise.modifications.get('harder', 'advanced variations')}",
            'deload': "Reduce intensity by 20% every 4-6 weeks for recovery"
        }
    
    def _get_flexibility_benefits(self, exercise: Exercise) -> List[str]:
        """Get benefits of flexibility exercises."""
        
        return [
            "Improved range of motion",
            "Reduced muscle tension and stiffness",
            "Better posture and alignment",
            "Enhanced recovery between workouts",
            "Stress reduction and relaxation"
        ]
