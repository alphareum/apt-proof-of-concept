"""
Enhanced AI-Powered Workout Recommendation System with Advanced Planning
Features comprehensive workout planning, periodization, and adaptive recommendations

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import logging
from dataclasses import dataclass, field
import json
import random
from models import (
    UserProfile, Exercise, Workout, WorkoutSet, FitnessGoal,
    GoalType, ExerciseCategory, FitnessLevel, ActivityLevel
)
from database import get_database
from enhanced_exercise_database import ComprehensiveExerciseDatabase, EnhancedExercise

logger = logging.getLogger(__name__)

@dataclass
class WorkoutPhase:
    """Represents a training phase in periodization."""
    name: str
    duration_weeks: int
    focus: str
    intensity_percentage: float
    volume_percentage: float
    recovery_emphasis: float
    description: str

@dataclass
class WorkoutDay:
    """Detailed workout day structure."""
    day_name: str
    workout_type: str
    focus_areas: List[str]
    exercises: List[Dict[str, Any]]
    total_duration: int
    warm_up_duration: int
    cool_down_duration: int
    intensity_level: str
    estimated_calories: int
    notes: List[str]

@dataclass
class WeeklyPlan:
    """Complete weekly workout plan."""
    week_number: int
    phase: str
    workout_days: List[WorkoutDay]
    rest_days: List[str]
    total_weekly_volume: int
    progressive_overload_notes: List[str]
    nutrition_focus: List[str]

@dataclass
class WorkoutProgram:
    """Complete multi-week workout program."""
    program_name: str
    total_weeks: int
    phases: List[WorkoutPhase]
    weekly_plans: List[WeeklyPlan]
    goal_alignment: str
    progression_strategy: str
    assessment_schedule: List[str]

class EnhancedRecommendationEngine:
    """Advanced AI-powered workout recommendation system."""
    
    def __init__(self):
        self.db = get_database()
        self.exercise_database = ComprehensiveExerciseDatabase()
        self.exercise_library = self.exercise_database.get_all_exercises()
        self.workout_templates = self._load_workout_templates()
        self.periodization_models = self._load_periodization_models()
    
    def generate_complete_program(self, user_profile: UserProfile, 
                                program_weeks: int = 12) -> WorkoutProgram:
        """Generate a complete periodized workout program."""
        
        try:
            # Analyze user needs and create phases
            phases = self._create_periodization_phases(user_profile, program_weeks)
            
            # Generate weekly plans for each phase
            weekly_plans = []
            week_counter = 1
            
            for phase in phases:
                for week in range(phase.duration_weeks):
                    weekly_plan = self._create_weekly_plan(
                        user_profile, phase, week_counter, week + 1
                    )
                    weekly_plans.append(weekly_plan)
                    week_counter += 1
            
            # Create complete program
            program = WorkoutProgram(
                program_name=f"{user_profile.primary_goal.value.title()} Program",
                total_weeks=program_weeks,
                phases=phases,
                weekly_plans=weekly_plans,
                goal_alignment=self._analyze_goal_alignment(user_profile),
                progression_strategy=self._create_progression_strategy(user_profile),
                assessment_schedule=self._create_assessment_schedule(program_weeks)
            )
            
            return program
            
        except Exception as e:
            logger.error(f"Error generating complete program: {e}")
            raise
    
    def generate_adaptive_recommendations(self, user_profile: UserProfile,
                                        workout_history: List[Dict] = None,
                                        preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate adaptive recommendations based on user history and preferences."""
        
        try:
            # Analyze user data
            user_analysis = self._analyze_user_comprehensively(user_profile, workout_history)
            
            # Get base recommendations
            base_recommendations = self._get_base_recommendations(user_profile)
            
            # Apply adaptive algorithms
            adaptive_recommendations = self._apply_adaptive_algorithms(
                base_recommendations, user_analysis, preferences or {}
            )
            
            # Create workout varieties
            workout_varieties = self._create_workout_varieties(
                adaptive_recommendations, user_profile
            )
            
            # Generate progression plan
            progression_plan = self._create_detailed_progression_plan(
                user_profile, adaptive_recommendations
            )
            
            return {
                'daily_workouts': adaptive_recommendations,
                'workout_varieties': workout_varieties,
                'progression_plan': progression_plan,
                'user_analysis': user_analysis,
                'adaptation_notes': self._generate_adaptation_notes(user_analysis),
                'recovery_recommendations': self._get_recovery_recommendations(user_profile),
                'nutrition_timing': self._get_nutrition_timing_recommendations(user_profile)
            }
            
        except Exception as e:
            logger.error(f"Error generating adaptive recommendations: {e}")
            return {'error': f'Failed to generate recommendations: {str(e)}'}
    
    def _create_periodization_phases(self, user_profile: UserProfile, 
                                   total_weeks: int) -> List[WorkoutPhase]:
        """Create periodization phases based on user goals."""
        
        phases = []
        
        if user_profile.primary_goal == GoalType.STRENGTH:
            # Strength-focused periodization
            phases = [
                WorkoutPhase(
                    name="Anatomical Adaptation",
                    duration_weeks=3,
                    focus="Movement patterns and base conditioning",
                    intensity_percentage=60,
                    volume_percentage=70,
                    recovery_emphasis=0.8,
                    description="Build movement quality and prepare body for higher intensities"
                ),
                WorkoutPhase(
                    name="Strength Building",
                    duration_weeks=4,
                    focus="Progressive overload and strength gains",
                    intensity_percentage=80,
                    volume_percentage=85,
                    recovery_emphasis=0.7,
                    description="Focus on heavy compound movements and strength development"
                ),
                WorkoutPhase(
                    name="Peak Strength",
                    duration_weeks=3,
                    focus="Maximum strength expression",
                    intensity_percentage=90,
                    volume_percentage=60,
                    recovery_emphasis=0.9,
                    description="Peak strength work with reduced volume"
                ),
                WorkoutPhase(
                    name="Deload & Recovery",
                    duration_weeks=2,
                    focus="Recovery and adaptation",
                    intensity_percentage=50,
                    volume_percentage=40,
                    recovery_emphasis=1.0,
                    description="Allow body to recover and supercompensate"
                )
            ]
        
        elif user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            # Weight loss periodization
            phases = [
                WorkoutPhase(
                    name="Base Building",
                    duration_weeks=4,
                    focus="Aerobic base and movement quality",
                    intensity_percentage=65,
                    volume_percentage=75,
                    recovery_emphasis=0.8,
                    description="Build cardiovascular base and establish consistent habits"
                ),
                WorkoutPhase(
                    name="Fat Burning Focus",
                    duration_weeks=6,
                    focus="High-intensity fat burning protocols",
                    intensity_percentage=75,
                    volume_percentage=90,
                    recovery_emphasis=0.6,
                    description="Maximize caloric expenditure through varied training"
                ),
                WorkoutPhase(
                    name="Metabolic Boost",
                    duration_weeks=2,
                    focus="Metabolic conditioning and variety",
                    intensity_percentage=85,
                    volume_percentage=80,
                    recovery_emphasis=0.7,
                    description="Challenge metabolism with diverse high-intensity work"
                )
            ]
        
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            # Muscle building periodization
            phases = [
                WorkoutPhase(
                    name="Volume Accumulation",
                    duration_weeks=4,
                    focus="High volume hypertrophy work",
                    intensity_percentage=70,
                    volume_percentage=100,
                    recovery_emphasis=0.7,
                    description="Accumulate training volume for muscle growth stimulus"
                ),
                WorkoutPhase(
                    name="Intensification",
                    duration_weeks=4,
                    focus="Progressive overload emphasis",
                    intensity_percentage=80,
                    volume_percentage=85,
                    recovery_emphasis=0.75,
                    description="Increase intensity while maintaining adequate volume"
                ),
                WorkoutPhase(
                    name="Specialization",
                    duration_weeks=3,
                    focus="Target weak areas and refinement",
                    intensity_percentage=75,
                    volume_percentage=90,
                    recovery_emphasis=0.8,
                    description="Address specific muscle groups and movement patterns"
                ),
                WorkoutPhase(
                    name="Recovery Integration",
                    duration_weeks=1,
                    focus="Active recovery and assessment",
                    intensity_percentage=50,
                    volume_percentage=30,
                    recovery_emphasis=1.0,
                    description="Recovery week to consolidate gains"
                )
            ]
        
        else:  # General fitness or endurance
            phases = [
                WorkoutPhase(
                    name="Foundation",
                    duration_weeks=4,
                    focus="General fitness and movement quality",
                    intensity_percentage=65,
                    volume_percentage=70,
                    recovery_emphasis=0.8,
                    description="Build fundamental fitness and movement patterns"
                ),
                WorkoutPhase(
                    name="Development",
                    duration_weeks=6,
                    focus="Balanced fitness development",
                    intensity_percentage=75,
                    volume_percentage=85,
                    recovery_emphasis=0.7,
                    description="Develop all aspects of fitness in balanced manner"
                ),
                WorkoutPhase(
                    name="Refinement",
                    duration_weeks=2,
                    focus="Skill refinement and performance",
                    intensity_percentage=80,
                    volume_percentage=75,
                    recovery_emphasis=0.8,
                    description="Refine skills and optimize performance"
                )
            ]
        
        # Adjust phases to fit total weeks
        total_phase_weeks = sum(phase.duration_weeks for phase in phases)
        if total_phase_weeks != total_weeks:
            adjustment_factor = total_weeks / total_phase_weeks
            for phase in phases:
                phase.duration_weeks = max(1, round(phase.duration_weeks * adjustment_factor))
        
        return phases
    
    def _create_weekly_plan(self, user_profile: UserProfile, 
                          phase: WorkoutPhase, week_number: int, 
                          phase_week: int) -> WeeklyPlan:
        """Create detailed weekly plan for specific phase and week."""
        
        workout_days = []
        rest_days = []
        
        # Determine workout frequency based on user profile and phase
        workout_frequency = min(user_profile.workout_days_per_week, 6)
        
        # Create workout schedule
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        workout_schedule = self._create_workout_schedule(
            user_profile, phase, workout_frequency
        )
        
        for i, day_name in enumerate(days_of_week):
            if i < len(workout_schedule):
                workout_type = workout_schedule[i]
                if workout_type != 'rest':
                    workout_day = self._create_detailed_workout_day(
                        day_name, workout_type, user_profile, phase, phase_week
                    )
                    workout_days.append(workout_day)
                else:
                    rest_days.append(day_name)
            else:
                rest_days.append(day_name)
        
        # Calculate weekly volume
        total_weekly_volume = sum(day.total_duration for day in workout_days)
        
        # Generate progressive overload notes
        overload_notes = self._generate_progressive_overload_notes(
            phase, phase_week, user_profile
        )
        
        # Generate nutrition focus
        nutrition_focus = self._generate_weekly_nutrition_focus(
            user_profile, phase, workout_days
        )
        
        return WeeklyPlan(
            week_number=week_number,
            phase=phase.name,
            workout_days=workout_days,
            rest_days=rest_days,
            total_weekly_volume=total_weekly_volume,
            progressive_overload_notes=overload_notes,
            nutrition_focus=nutrition_focus
        )
    
    def _create_detailed_workout_day(self, day_name: str, workout_type: str,
                                   user_profile: UserProfile, phase: WorkoutPhase,
                                   phase_week: int) -> WorkoutDay:
        """Create detailed workout day with exercises and structure."""
        
        # Get exercises for this workout type
        exercises = self._select_exercises_for_workout(
            workout_type, user_profile, phase, phase_week
        )
        
        # Calculate durations
        base_duration = user_profile.available_time
        warm_up_duration = max(5, base_duration // 8)
        cool_down_duration = max(5, base_duration // 10)
        main_workout_duration = base_duration - warm_up_duration - cool_down_duration
        
        # Determine intensity level
        intensity_level = self._determine_workout_intensity(phase, phase_week, workout_type)
        
        # Estimate calories
        estimated_calories = self._estimate_workout_calories(
            exercises, main_workout_duration, user_profile, intensity_level
        )
        
        # Generate workout notes
        notes = self._generate_workout_notes(workout_type, phase, exercises)
        
        # Determine focus areas
        focus_areas = self._determine_focus_areas(workout_type, exercises)
        
        return WorkoutDay(
            day_name=day_name,
            workout_type=workout_type,
            focus_areas=focus_areas,
            exercises=exercises,
            total_duration=base_duration,
            warm_up_duration=warm_up_duration,
            cool_down_duration=cool_down_duration,
            intensity_level=intensity_level,
            estimated_calories=estimated_calories,
            notes=notes
        )
    
    def _select_exercises_for_workout(self, workout_type: str, user_profile: UserProfile,
                                    phase: WorkoutPhase, phase_week: int) -> List[Dict[str, Any]]:
        """Select appropriate exercises for specific workout type and phase."""
        
        exercises = []
        
        if workout_type == 'upper_strength':
            exercises = self._get_upper_body_strength_exercises(user_profile, phase)
        elif workout_type == 'lower_strength':
            exercises = self._get_lower_body_strength_exercises(user_profile, phase)
        elif workout_type == 'full_body_strength':
            exercises = self._get_full_body_strength_exercises(user_profile, phase)
        elif workout_type == 'cardio_hiit':
            exercises = self._get_hiit_exercises(user_profile, phase)
        elif workout_type == 'cardio_steady':
            exercises = self._get_steady_state_cardio_exercises(user_profile, phase)
        elif workout_type == 'functional_training':
            exercises = self._get_functional_exercises(user_profile, phase)
        elif workout_type == 'flexibility_mobility':
            exercises = self._get_flexibility_exercises(user_profile, phase)
        else:  # mixed or general
            exercises = self._get_mixed_workout_exercises(user_profile, phase)
        
        # Apply phase-specific modifications
        exercises = self._apply_phase_modifications(exercises, phase, phase_week)
        
        return exercises
    
    def _get_upper_body_strength_exercises(self, user_profile: UserProfile, 
                                         phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get upper body strength exercises."""
        
        # Get exercises from database
        strength_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.STRENGTH)
        
        # Filter for upper body
        upper_body_muscles = ['chest', 'shoulders', 'triceps', 'biceps', 'back']
        upper_body_exercises = {
            name: ex for name, ex in strength_exercises.items()
            if any(muscle in upper_body_muscles for muscle in ex.muscle_groups)
        }
        
        # Filter by equipment availability
        available_equipment = getattr(user_profile, 'available_equipment', [])
        suitable_exercises = {
            name: ex for name, ex in upper_body_exercises.items()
            if not ex.equipment_needed or all(eq in available_equipment for eq in ex.equipment_needed)
        }
        
        # Convert to workout format
        exercises = []
        for name, ex in list(suitable_exercises.items())[:4]:  # Top 4 exercises
            exercise_dict = {
                'name': ex.name,
                'category': 'strength',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': ex.technique_cues,
                'progressions': ex.progressions
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_lower_body_strength_exercises(self, user_profile: UserProfile, 
                                         phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get lower body strength exercises."""
        
        base_exercises = [
            {
                'name': 'Squats',
                'category': 'strength',
                'muscle_groups': ['quadriceps', 'glutes', 'hamstrings'],
                'equipment_needed': [],
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Feet shoulder-width apart',
                    'Keep chest up and core tight',
                    'Drive through heels'
                ],
                'progressions': self._get_exercise_progressions('squats', user_profile.fitness_level)
            },
            {
                'name': 'Lunges',
                'category': 'strength',
                'muscle_groups': ['quadriceps', 'glutes', 'hamstrings'],
                'equipment_needed': [],
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Step forward into lunge',
                    'Keep front knee over ankle',
                    'Lower back knee toward ground'
                ],
                'progressions': self._get_exercise_progressions('lunges', user_profile.fitness_level)
            },
            {
                'name': 'Deadlifts',
                'category': 'strength',
                'muscle_groups': ['hamstrings', 'glutes', 'back'],
                'equipment_needed': ['dumbbells'],
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Keep back straight',
                    'Hinge at hips',
                    'Drive through heels'
                ],
                'progressions': self._get_exercise_progressions('deadlifts', user_profile.fitness_level)
            }
        ]
        
        # Filter based on available equipment
        available_exercises = [
            ex for ex in base_exercises 
            if not ex['equipment_needed'] or 
            all(eq in user_profile.available_equipment for eq in ex['equipment_needed'])
        ]
        
        return available_exercises[:4]
    
    def _get_hiit_exercises(self, user_profile: UserProfile, 
                          phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get HIIT exercises."""
        
        hiit_exercises = [
            {
                'name': 'Burpees',
                'category': 'cardio',
                'muscle_groups': ['full_body'],
                'equipment_needed': [],
                'work_time': 30,
                'rest_time': 30,
                'rounds': self._calculate_hiit_rounds(user_profile, phase),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Start standing, drop to squat',
                    'Jump back to plank',
                    'Jump feet to hands, jump up'
                ],
                'modifications': {
                    'easier': 'Step back to plank instead of jumping',
                    'harder': 'Add push-up in plank position'
                }
            },
            {
                'name': 'Mountain Climbers',
                'category': 'cardio',
                'muscle_groups': ['core', 'shoulders', 'legs'],
                'equipment_needed': [],
                'work_time': 45,
                'rest_time': 15,
                'rounds': self._calculate_hiit_rounds(user_profile, phase),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Start in plank position',
                    'Alternate bringing knees to chest',
                    'Keep hips level'
                ],
                'modifications': {
                    'easier': 'Slower pace with full foot placement',
                    'harder': 'Faster pace or add cross-body movement'
                }
            },
            {
                'name': 'High Knees',
                'category': 'cardio',
                'muscle_groups': ['legs', 'core'],
                'equipment_needed': [],
                'work_time': 30,
                'rest_time': 30,
                'rounds': self._calculate_hiit_rounds(user_profile, phase),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': [
                    'Lift knees to hip height',
                    'Pump arms naturally',
                    'Stay on balls of feet'
                ],
                'modifications': {
                    'easier': 'Marching in place with knee lifts',
                    'harder': 'Add arm movements or increase pace'
                }
            }
        ]
        
        return hiit_exercises[:4]
    
    def _calculate_sets(self, user_profile: UserProfile, phase: WorkoutPhase) -> str:
        """Calculate appropriate number of sets."""
        
        base_sets = {
            FitnessLevel.BEGINNER: 2,
            FitnessLevel.INTERMEDIATE: 3,
            FitnessLevel.ADVANCED: 4
        }
        
        phase_modifier = phase.volume_percentage / 100
        calculated_sets = int(base_sets[user_profile.fitness_level] * phase_modifier)
        calculated_sets = max(1, min(calculated_sets, 6))  # Ensure reasonable range
        
        return f"{calculated_sets}"
    
    def _calculate_reps(self, user_profile: UserProfile, phase: WorkoutPhase, 
                       exercise_type: str) -> str:
        """Calculate appropriate number of reps."""
        
        rep_ranges = {
            'strength': {
                FitnessLevel.BEGINNER: (8, 12),
                FitnessLevel.INTERMEDIATE: (6, 10),
                FitnessLevel.ADVANCED: (4, 8)
            },
            'hypertrophy': {
                FitnessLevel.BEGINNER: (10, 15),
                FitnessLevel.INTERMEDIATE: (8, 12),
                FitnessLevel.ADVANCED: (6, 12)
            },
            'endurance': {
                FitnessLevel.BEGINNER: (12, 20),
                FitnessLevel.INTERMEDIATE: (15, 25),
                FitnessLevel.ADVANCED: (20, 30)
            }
        }
        
        # Determine exercise type based on phase focus
        if phase.focus in ['Maximum strength expression', 'Progressive overload']:
            rep_type = 'strength'
        elif phase.focus in ['High volume hypertrophy', 'muscle growth']:
            rep_type = 'hypertrophy'
        else:
            rep_type = 'endurance'
        
        min_reps, max_reps = rep_ranges[rep_type][user_profile.fitness_level]
        
        # Apply intensity modifier
        if phase.intensity_percentage > 85:
            max_reps = min(max_reps, min_reps + 3)
        elif phase.intensity_percentage < 65:
            min_reps = max(min_reps, max_reps - 5)
        
        return f"{min_reps}-{max_reps}"
    
    def _calculate_rest_time(self, phase: WorkoutPhase, exercise_type: str) -> int:
        """Calculate appropriate rest time between sets."""
        
        base_rest_times = {
            'strength': 180,  # 3 minutes
            'hypertrophy': 90,  # 1.5 minutes
            'endurance': 60,   # 1 minute
            'cardio': 30      # 30 seconds
        }
        
        base_rest = base_rest_times.get(exercise_type, 60)
        
        # Adjust based on phase intensity
        intensity_modifier = phase.intensity_percentage / 100
        rest_time = int(base_rest * (0.7 + 0.6 * intensity_modifier))
        
        # Apply recovery emphasis
        rest_time = int(rest_time * phase.recovery_emphasis)
        
        return max(30, min(rest_time, 300))  # Keep between 30 seconds and 5 minutes
    
    def _get_lower_body_strength_exercises(self, user_profile: UserProfile, 
                                         phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get lower body strength exercises."""
        
        # Get exercises from database
        strength_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.STRENGTH)
        
        # Filter for lower body
        lower_body_muscles = ['quadriceps', 'glutes', 'hamstrings', 'calves']
        lower_body_exercises = {
            name: ex for name, ex in strength_exercises.items()
            if any(muscle in lower_body_muscles for muscle in ex.muscle_groups)
        }
        
        # Convert to workout format
        exercises = []
        for name, ex in list(lower_body_exercises.items())[:4]:
            exercise_dict = {
                'name': ex.name,
                'category': 'strength',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': ex.technique_cues,
                'progressions': ex.progressions
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_full_body_strength_exercises(self, user_profile: UserProfile, 
                                        phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get full body strength exercises."""
        
        # Get functional exercises which typically target full body
        functional_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.FUNCTIONAL)
        
        exercises = []
        for name, ex in list(functional_exercises.items())[:3]:
            exercise_dict = {
                'name': ex.name,
                'category': 'functional',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'strength'),
                'rest_seconds': self._calculate_rest_time(phase, 'strength'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': ex.technique_cues,
                'progressions': ex.progressions
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_hiit_exercises(self, user_profile: UserProfile, 
                          phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get HIIT exercises."""
        
        # Get high-intensity exercises
        functional_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.FUNCTIONAL)
        cardio_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.CARDIO)
        
        # Combine and filter for high-intensity
        hiit_candidates = {**functional_exercises, **cardio_exercises}
        hiit_exercises = {
            name: ex for name, ex in hiit_candidates.items()
            if ex.calories_per_minute >= 8  # High intensity threshold
        }
        
        exercises = []
        for name, ex in list(hiit_exercises.items())[:4]:
            exercise_dict = {
                'name': ex.name,
                'category': 'cardio',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'work_time': 30,
                'rest_time': 30,
                'rounds': self._calculate_hiit_rounds(user_profile, phase),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': ex.technique_cues,
                'modifications': ex.modifications
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_steady_state_cardio_exercises(self, user_profile: UserProfile, 
                                         phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get steady state cardio exercises."""
        
        cardio_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.CARDIO)
        
        # Filter for moderate intensity
        steady_state = {
            name: ex for name, ex in cardio_exercises.items()
            if 6 <= ex.calories_per_minute <= 10  # Moderate intensity
        }
        
        exercises = []
        for name, ex in list(steady_state.items())[:3]:
            exercise_dict = {
                'name': ex.name,
                'category': 'cardio',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'duration': user_profile.available_time - 10,  # Leave time for warm-up/cool-down
                'intensity': 'moderate',
                'technique_cues': ex.technique_cues,
                'progressions': ex.progressions
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_functional_exercises(self, user_profile: UserProfile, 
                                phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get functional training exercises."""
        
        functional_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.FUNCTIONAL)
        
        exercises = []
        for name, ex in list(functional_exercises.items())[:4]:
            exercise_dict = {
                'name': ex.name,
                'category': 'functional',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'sets': self._calculate_sets(user_profile, phase),
                'reps': self._calculate_reps(user_profile, phase, 'endurance'),
                'rest_seconds': self._calculate_rest_time(phase, 'endurance'),
                'intensity_modifier': phase.intensity_percentage / 100,
                'technique_cues': ex.technique_cues,
                'progressions': ex.progressions
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_flexibility_exercises(self, user_profile: UserProfile, 
                                 phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get flexibility and mobility exercises."""
        
        flexibility_exercises = self.exercise_database.get_exercises_by_category(ExerciseCategory.FLEXIBILITY)
        
        exercises = []
        for name, ex in list(flexibility_exercises.items())[:3]:
            exercise_dict = {
                'name': ex.name,
                'category': 'flexibility',
                'muscle_groups': ex.muscle_groups,
                'equipment_needed': ex.equipment_needed,
                'duration': ex.target_duration or "10-15 minutes",
                'intensity': 'low',
                'technique_cues': ex.technique_cues,
                'breathing_pattern': ex.breathing_pattern
            }
            exercises.append(exercise_dict)
        
        return exercises
    
    def _get_mixed_workout_exercises(self, user_profile: UserProfile, 
                                   phase: WorkoutPhase) -> List[Dict[str, Any]]:
        """Get mixed workout combining different exercise types."""
        
        # Get a balanced mix
        exercises = []
        
        # Add 1-2 cardio exercises
        cardio = self._get_steady_state_cardio_exercises(user_profile, phase)[:1]
        exercises.extend(cardio)
        
        # Add 2-3 strength exercises
        strength = self._get_full_body_strength_exercises(user_profile, phase)[:2]
        exercises.extend(strength)
        
        # Add 1 flexibility exercise
        flexibility = self._get_flexibility_exercises(user_profile, phase)[:1]
        exercises.extend(flexibility)
        
        return exercises
    
    def _calculate_hiit_rounds(self, user_profile: UserProfile, phase: WorkoutPhase) -> int:
        """Calculate appropriate number of HIIT rounds."""
        
        base_rounds = {
            FitnessLevel.BEGINNER: 4,
            FitnessLevel.INTERMEDIATE: 6,
            FitnessLevel.ADVANCED: 8
        }
        
        rounds = base_rounds[user_profile.fitness_level]
        
        # Adjust for phase intensity
        if phase.intensity_percentage > 85:
            rounds = min(rounds + 2, 12)
        elif phase.intensity_percentage < 65:
            rounds = max(rounds - 2, 3)
        
        return rounds
    
    def _apply_phase_modifications(self, exercises: List[Dict[str, Any]], 
                                 phase: WorkoutPhase, phase_week: int) -> List[Dict[str, Any]]:
        """Apply phase-specific modifications to exercises."""
        
        for exercise in exercises:
            # Apply intensity modifier
            if 'intensity_modifier' in exercise:
                exercise['intensity_modifier'] *= phase.intensity_percentage / 100
            
            # Adjust volume for phase
            if 'sets' in exercise:
                current_sets = int(exercise['sets'])
                adjusted_sets = max(1, int(current_sets * phase.volume_percentage / 100))
                exercise['sets'] = str(adjusted_sets)
            
            # Add phase-specific notes
            if not 'notes' in exercise:
                exercise['notes'] = []
            
            exercise['notes'].append(f"Phase: {phase.name} (Week {phase_week})")
        
        return exercises
    
    def _create_workout_schedule(self, user_profile: UserProfile, 
                               phase: WorkoutPhase, workout_frequency: int) -> List[str]:
        """Create workout schedule based on frequency and goals."""
        
        if user_profile.primary_goal == GoalType.STRENGTH:
            if workout_frequency <= 3:
                return ['upper_strength', 'lower_strength', 'full_body_strength']
            else:
                return ['upper_strength', 'lower_strength', 'upper_strength', 'lower_strength', 'functional_training']
        
        elif user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            if workout_frequency <= 3:
                return ['cardio_hiit', 'full_body_strength', 'cardio_steady']
            else:
                return ['cardio_hiit', 'upper_strength', 'cardio_steady', 'lower_strength', 'functional_training']
        
        elif user_profile.primary_goal == GoalType.ENDURANCE:
            if workout_frequency <= 3:
                return ['cardio_steady', 'functional_training', 'cardio_hiit']
            else:
                return ['cardio_steady', 'functional_training', 'cardio_hiit', 'cardio_steady', 'flexibility_mobility']
        
        else:  # General fitness
            if workout_frequency <= 3:
                return ['full_body_strength', 'cardio_steady', 'functional_training']
            else:
                return ['upper_strength', 'cardio_hiit', 'lower_strength', 'cardio_steady', 'flexibility_mobility']
    
    def _determine_workout_intensity(self, phase: WorkoutPhase, phase_week: int, workout_type: str) -> str:
        """Determine workout intensity level."""
        
        base_intensity = phase.intensity_percentage
        
        # Adjust for workout type
        if 'hiit' in workout_type or 'functional' in workout_type:
            base_intensity *= 1.1
        elif 'flexibility' in workout_type:
            base_intensity *= 0.7
        
        if base_intensity >= 85:
            return "High"
        elif base_intensity >= 70:
            return "Moderate"
        else:
            return "Low"
    
    def _estimate_workout_calories(self, exercises: List[Dict[str, Any]], 
                                 duration: int, user_profile: UserProfile, 
                                 intensity: str) -> int:
        """Estimate calories burned during workout."""
        
        # Base metabolic equivalent values
        intensity_mets = {
            "Low": 3.5,
            "Moderate": 5.5,
            "High": 8.0
        }
        
        met_value = intensity_mets.get(intensity, 5.5)
        
        # Calories = METs × weight (kg) × time (hours)
        weight_kg = getattr(user_profile, 'weight', 70)  # Default 70kg if not available
        time_hours = duration / 60
        
        calories = met_value * weight_kg * time_hours
        
        return int(calories)
    
    def _generate_workout_notes(self, workout_type: str, phase: WorkoutPhase, 
                              exercises: List[Dict[str, Any]]) -> List[str]:
        """Generate helpful workout notes."""
        
        notes = []
        
        # Add phase-specific notes
        notes.append(f"Focus: {phase.focus}")
        
        # Add workout type specific notes
        if 'strength' in workout_type:
            notes.append("Focus on proper form and progressive overload")
        elif 'cardio' in workout_type:
            notes.append("Maintain target heart rate zone")
        elif 'hiit' in workout_type:
            notes.append("Push hard during work intervals, recover during rest")
        elif 'flexibility' in workout_type:
            notes.append("Move slowly and breathe deeply")
        
        # Add intensity notes
        if phase.intensity_percentage > 85:
            notes.append("High intensity session - listen to your body")
        elif phase.intensity_percentage < 65:
            notes.append("Recovery-focused session - quality over quantity")
        
        return notes
    
    def _determine_focus_areas(self, workout_type: str, exercises: List[Dict[str, Any]]) -> List[str]:
        """Determine the focus areas for the workout."""
        
        focus_areas = []
        
        # Get unique muscle groups from exercises
        muscle_groups = set()
        for exercise in exercises:
            # Safety check: ensure exercise is a dictionary
            if not isinstance(exercise, dict):
                # Skip invalid exercises and continue
                continue
            muscle_groups.update(exercise.get('muscle_groups', []))
        
        # Map to focus areas
        if any(muscle in muscle_groups for muscle in ['chest', 'shoulders', 'triceps', 'biceps', 'back']):
            focus_areas.append("Upper Body")
        
        if any(muscle in muscle_groups for muscle in ['quadriceps', 'glutes', 'hamstrings', 'calves']):
            focus_areas.append("Lower Body")
        
        if 'core' in muscle_groups:
            focus_areas.append("Core")
        
        if 'cardiovascular' in muscle_groups:
            focus_areas.append("Cardiovascular")
        
        if not focus_areas:
            focus_areas = [workout_type.replace('_', ' ').title()]
        
        return focus_areas
    
    def _generate_progressive_overload_notes(self, phase: WorkoutPhase, 
                                           phase_week: int, user_profile: UserProfile) -> List[str]:
        """Generate progressive overload guidance."""
        
        notes = []
        
        if phase.focus in ['Progressive overload', 'strength']:
            if phase_week == 1:
                notes.append("Week 1: Focus on establishing proper form")
            elif phase_week == 2:
                notes.append("Week 2: Increase weight by 5-10% if form allows")
            elif phase_week == 3:
                notes.append("Week 3: Push for additional reps or weight")
            else:
                notes.append("Week 4+: Continue progressive increases")
        
        elif phase.focus in ['High volume', 'hypertrophy']:
            notes.append(f"Volume emphasis: Focus on completing all sets with good form")
        
        return notes
    
    def _generate_weekly_nutrition_focus(self, user_profile: UserProfile, 
                                       phase: WorkoutPhase, workout_days: List) -> List[str]:
        """Generate weekly nutrition focus points."""
        
        nutrition_focus = []
        
        # Base on primary goal
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            nutrition_focus.extend([
                "Maintain moderate caloric deficit",
                "Prioritize protein intake (0.8-1g per lb bodyweight)",
                "Time carbohydrates around workouts"
            ])
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            nutrition_focus.extend([
                "Ensure adequate caloric surplus",
                "High protein intake (1-1.2g per lb bodyweight)",
                "Focus on post-workout nutrition"
            ])
        else:
            nutrition_focus.extend([
                "Balanced macronutrient intake",
                "Adequate protein for recovery",
                "Stay well hydrated"
            ])
        
        # Add phase-specific nutrition
        if phase.intensity_percentage > 85:
            nutrition_focus.append("Extra attention to recovery nutrition")
        
        return nutrition_focus
    
    def _load_workout_templates(self) -> Dict[str, Any]:
        """Load pre-built workout templates."""
        return {
            'beginner_templates': {},
            'intermediate_templates': {},
            'advanced_templates': {},
            'goal_specific_templates': {}
        }
    
    def _load_periodization_models(self) -> Dict[str, Any]:
        """Load periodization models for different goals."""
        return {
            'linear_periodization': {},
            'undulating_periodization': {},
            'block_periodization': {},
            'conjugate_method': {}
        }
    
    # Additional analysis methods for adaptive recommendations
    def _analyze_user_comprehensively(self, user_profile: UserProfile, 
                                    workout_history: List[Dict] = None) -> Dict[str, Any]:
        """Comprehensive user analysis for adaptive recommendations."""
        
        analysis = {
            'fitness_assessment': self._assess_current_fitness(user_profile),
            'goal_analysis': self._analyze_goals(user_profile),
            'equipment_availability': self._analyze_equipment(user_profile),
            'time_constraints': self._analyze_time_availability(user_profile),
            'injury_considerations': self._analyze_injuries(user_profile),
            'experience_level': self._assess_experience_level(user_profile, workout_history),
            'motivation_factors': self._identify_motivation_factors(user_profile),
            'potential_barriers': self._identify_potential_barriers(user_profile)
        }
        
        return analysis
    
    def _assess_current_fitness(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Assess current fitness level."""
        return {
            'level': user_profile.fitness_level.value,
            'activity_level': user_profile.activity_level.value,
            'estimated_capacity': 'moderate'  # Would be more sophisticated in real implementation
        }
    
    def _analyze_goals(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze user goals."""
        return {
            'primary_goal': user_profile.primary_goal.value,
            'secondary_goals': [goal.value for goal in user_profile.secondary_goals],
            'goal_specificity': 'moderate'
        }
    
    def _analyze_equipment(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze equipment availability."""
        equipment = getattr(user_profile, 'available_equipment', [])
        return {
            'available_equipment': equipment,
            'equipment_versatility': len(equipment),
            'home_gym_rating': 'basic' if len(equipment) < 3 else 'well_equipped'
        }
    
    def _analyze_time_availability(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze time constraints."""
        return {
            'available_time': user_profile.available_time,
            'workout_frequency': user_profile.workout_days_per_week,
            'time_efficiency_needed': user_profile.available_time < 45
        }
    
    def _analyze_injuries(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze injury considerations."""
        injuries = getattr(user_profile, 'injuries', [])
        return {
            'current_injuries': injuries,
            'injury_risk_level': 'low' if not injuries else 'moderate',
            'modifications_needed': len(injuries) > 0
        }
    
    def _assess_experience_level(self, user_profile: UserProfile, 
                               workout_history: List[Dict] = None) -> Dict[str, Any]:
        """Assess exercise experience level."""
        return {
            'fitness_level': user_profile.fitness_level.value,
            'workout_history_length': len(workout_history) if workout_history else 0,
            'experience_rating': 'novice' if user_profile.fitness_level == FitnessLevel.BEGINNER else 'experienced'
        }
    
    def _identify_motivation_factors(self, user_profile: UserProfile) -> List[str]:
        """Identify user motivation factors."""
        factors = []
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            factors.extend(['Health improvement', 'Aesthetic goals', 'Energy levels'])
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            factors.extend(['Strength gains', 'Physical appearance', 'Athletic performance'])
        elif user_profile.primary_goal == GoalType.ENDURANCE:
            factors.extend(['Athletic performance', 'Health benefits', 'Personal challenge'])
        else:
            factors.extend(['General health', 'Stress management', 'Quality of life'])
        
        return factors
    
    def _identify_potential_barriers(self, user_profile: UserProfile) -> List[str]:
        """Identify potential barriers to success."""
        barriers = []
        
        if user_profile.available_time < 30:
            barriers.append('Limited time availability')
        
        equipment = getattr(user_profile, 'available_equipment', [])
        if len(equipment) < 2:
            barriers.append('Limited equipment access')
        
        if user_profile.fitness_level == FitnessLevel.BEGINNER:
            barriers.append('Learning curve for proper form')
        
        return barriers
    
    def _get_base_recommendations(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Get base exercise recommendations."""
        
        recommendations = self.exercise_database.get_exercise_recommendations(user_profile, max_exercises=15)
        
        # Organize by category
        organized = {
            'strength': [],
            'cardio': [],
            'flexibility': [],
            'functional': []
        }
        
        for exercise in recommendations:
            category = exercise.category.value
            if category in organized:
                organized[category].append({
                    'exercise': exercise,
                    'recommended_sets': exercise.target_sets or "2-3",
                    'recommended_reps': exercise.target_reps or "8-12",
                    'rest_time': exercise.rest_time_seconds
                })
        
        return organized
    
    def _apply_adaptive_algorithms(self, base_recommendations: Dict[str, Any], 
                                 user_analysis: Dict[str, Any], 
                                 preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive algorithms to base recommendations."""
        
        # Start with base recommendations
        adaptive_recs = base_recommendations.copy()
        
        # Apply time constraints
        if user_analysis['time_constraints']['time_efficiency_needed']:
            # Prioritize compound movements and circuits
            for category in adaptive_recs:
                # Filter to more time-efficient exercises
                adaptive_recs[category] = adaptive_recs[category][:3]
        
        # Apply equipment constraints
        equipment_available = user_analysis['equipment_availability']['available_equipment']
        for category in adaptive_recs:
            adaptive_recs[category] = [
                rec for rec in adaptive_recs[category]
                if not rec['exercise'].equipment_needed or 
                all(eq in equipment_available for eq in rec['exercise'].equipment_needed)
            ]
        
        # Apply injury modifications
        if user_analysis['injury_considerations']['modifications_needed']:
            # Apply exercise modifications or substitutions
            for category in adaptive_recs:
                for rec in adaptive_recs[category]:
                    if rec['exercise'].contraindications:
                        # Add modification notes
                        rec['modification_applied'] = True
        
        return adaptive_recs
    
    def _create_workout_varieties(self, recommendations: Dict[str, Any], 
                                user_profile: UserProfile) -> Dict[str, Any]:
        """Create workout varieties for different scenarios."""
        
        varieties = {}
        
        # Quick workout (15-20 minutes)
        varieties['quick_workout'] = {
            'description': 'High-intensity 15-20 minute workout',
            'exercises': [],
            'focus': 'Time-efficient full body training'
        }
        
        # Equipment-free workout
        varieties['no_equipment'] = {
            'description': 'Complete bodyweight workout',
            'exercises': [],
            'focus': 'Bodyweight exercises requiring no equipment'
        }
        
        # Low-intensity recovery workout
        varieties['recovery_workout'] = {
            'description': 'Gentle recovery and mobility session',
            'exercises': [],
            'focus': 'Active recovery and flexibility'
        }
        
        return varieties
    
    def _create_detailed_progression_plan(self, user_profile: UserProfile, 
                                        recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed progression plan."""
        
        progression_plan = {}
        
        # Create 4-week progression blocks
        for week in range(1, 5):
            week_focus = ""
            if week == 1:
                week_focus = "Form and technique establishment"
            elif week == 2:
                week_focus = "Volume increase and adaptation"
            elif week == 3:
                week_focus = "Intensity progression"
            else:
                week_focus = "Peak performance and assessment"
            
            progression_plan[f"Week {week}"] = week_focus
        
        return progression_plan
    
    def _generate_adaptation_notes(self, user_analysis: Dict[str, Any]) -> List[str]:
        """Generate adaptation notes based on user analysis."""
        
        notes = []
        
        if user_analysis['time_constraints']['time_efficiency_needed']:
            notes.append("Workouts optimized for time efficiency")
        
        if user_analysis['injury_considerations']['modifications_needed']:
            notes.append("Exercise modifications applied for injury considerations")
        
        if user_analysis['equipment_availability']['equipment_versatility'] < 3:
            notes.append("Emphasis on bodyweight and minimal equipment exercises")
        
        return notes
    
    def _get_recovery_recommendations(self, user_profile: UserProfile) -> List[str]:
        """Get recovery recommendations."""
        
        recovery_recs = [
            "Aim for 7-9 hours of quality sleep",
            "Stay hydrated throughout the day",
            "Include active recovery on rest days"
        ]
        
        if user_profile.fitness_level == FitnessLevel.BEGINNER:
            recovery_recs.append("Allow 48-72 hours between intense workouts")
        
        if user_profile.activity_level == ActivityLevel.VERY_ACTIVE:
            recovery_recs.append("Consider massage or foam rolling for recovery")
        
        return recovery_recs
    
    def _get_nutrition_timing_recommendations(self, user_profile: UserProfile) -> Dict[str, str]:
        """Get nutrition timing recommendations."""
        
        return {
            'pre_workout': 'Light snack 30-60 minutes before exercise',
            'during_workout': 'Stay hydrated, sports drink for sessions >60 minutes',
            'post_workout': 'Protein + carbs within 30-60 minutes after exercise',
            'daily_focus': 'Balanced meals every 3-4 hours'
        }
    
    def get_workout_for_today(self, user_profile: UserProfile, 
                            current_date: date = None) -> Dict[str, Any]:
        """Get today's specific workout recommendation."""
        
        if current_date is None:
            current_date = date.today()
        
        # Get the current program
        program = self.generate_complete_program(user_profile)
        
        # Calculate which week we're in (this would be stored in user progress)
        week_number = 1  # This would come from user's progress tracking
        
        if week_number <= len(program.weekly_plans):
            weekly_plan = program.weekly_plans[week_number - 1]
            day_of_week = current_date.weekday()  # 0 = Monday, 6 = Sunday
            
            if day_of_week < len(weekly_plan.workout_days):
                today_workout = weekly_plan.workout_days[day_of_week]
                
                return {
                    'workout': today_workout,
                    'week_info': {
                        'week_number': week_number,
                        'phase': weekly_plan.phase,
                        'weekly_focus': weekly_plan.nutrition_focus
                    },
                    'preparation_tips': self._get_workout_preparation_tips(today_workout),
                    'post_workout_guidance': self._get_post_workout_guidance(today_workout)
                }
        
        return {'error': 'No workout scheduled for today'}
    
    def _analyze_goal_alignment(self, user_profile: UserProfile) -> str:
        """Analyze how the program aligns with user goals."""
        
        primary_goal = user_profile.primary_goal.value
        secondary_goals = [goal.value for goal in user_profile.secondary_goals] if user_profile.secondary_goals else []
        
        alignment_description = f"Program specifically designed for {primary_goal.replace('_', ' ')}"
        
        if secondary_goals:
            alignment_description += f" with additional focus on {', '.join([goal.replace('_', ' ') for goal in secondary_goals])}"
        
        # Add specific alignment details based on goal
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            alignment_description += ". Emphasizes calorie burning through combination of strength and cardio training"
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            alignment_description += ". Focuses on progressive overload and muscle hypertrophy through structured resistance training"
        elif user_profile.primary_goal == GoalType.ENDURANCE:
            alignment_description += ". Builds cardiovascular capacity and muscular endurance through varied training intensities"
        elif user_profile.primary_goal == GoalType.STRENGTH:
            alignment_description += ". Develops maximal strength through progressive resistance training and compound movements"
        else:
            alignment_description += ". Provides balanced approach to overall fitness and health improvement"
        
        return alignment_description
    
    def _create_progression_strategy(self, user_profile: UserProfile) -> str:
        """Create progression strategy description."""
        
        strategy_components = []
        
        # Base progression on fitness level
        if user_profile.fitness_level == FitnessLevel.BEGINNER:
            strategy_components.append("Start with form mastery and gradual volume increases")
            strategy_components.append("Focus on establishing consistent workout habits")
        elif user_profile.fitness_level == FitnessLevel.INTERMEDIATE:
            strategy_components.append("Implement progressive overload through weight and volume increases")
            strategy_components.append("Introduce more complex movement patterns")
        else:
            strategy_components.append("Advanced periodization with varied training stimuli")
            strategy_components.append("Focus on performance optimization and specialization")
        
        # Add goal-specific strategies
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            strategy_components.append("Gradually increase workout intensity and frequency")
            strategy_components.append("Progressive calorie burn optimization")
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            strategy_components.append("Systematic load progression (5-10% increases)")
            strategy_components.append("Volume increases every 2-3 weeks")
        elif user_profile.primary_goal == GoalType.ENDURANCE:
            strategy_components.append("Progressive duration and intensity increases")
            strategy_components.append("Periodized training with base building and intensity phases")
        
        return ". ".join(strategy_components)
    
    def _create_assessment_schedule(self, program_weeks: int) -> List[str]:
        """Create assessment schedule for the program."""
        
        assessments = []
        
        # Initial assessment
        assessments.append("Week 1: Baseline fitness assessment and form evaluation")
        
        # Mid-program assessments
        if program_weeks >= 4:
            assessments.append("Week 4: Progress evaluation and program adjustments")
        
        if program_weeks >= 8:
            assessments.append("Week 8: Mid-program comprehensive assessment")
        
        if program_weeks >= 12:
            assessments.append("Week 12: Final assessment and goal achievement review")
        
        # Add quarterly assessments for longer programs
        for week in range(12, program_weeks, 12):
            assessments.append(f"Week {week}: Quarterly progress assessment")
        
        # Final assessment
        if program_weeks > 12:
            assessments.append(f"Week {program_weeks}: Final comprehensive assessment")
        
        return assessments
    
    def _get_workout_preparation_tips(self, workout_day: WorkoutDay) -> List[str]:
        """Get preparation tips for a specific workout."""
        
        tips = [
            "Ensure you're well-hydrated before starting",
            f"Allocate {workout_day.warm_up_duration} minutes for proper warm-up",
            "Have all necessary equipment ready and accessible"
        ]
        
        # Add workout-type specific tips
        if 'strength' in workout_day.workout_type.lower():
            tips.append("Review proper form for each exercise")
            tips.append("Have water bottle nearby for hydration between sets")
        elif 'cardio' in workout_day.workout_type.lower() or 'hiit' in workout_day.workout_type.lower():
            tips.append("Consider pre-workout light snack if training fasted")
            tips.append("Ensure adequate ventilation for intense cardio")
        elif 'flexibility' in workout_day.workout_type.lower():
            tips.append("Ensure comfortable clothing for full range of motion")
            tips.append("Consider playing relaxing music")
        
        # Add intensity-specific tips
        if workout_day.intensity_level == 'High':
            tips.append("Extra attention to warm-up due to high intensity")
            tips.append("Have towel ready for perspiration")
        elif workout_day.intensity_level == 'Low':
            tips.append("Focus on mindful movement and breathing")
        
        return tips
    
    def _get_post_workout_guidance(self, workout_day: WorkoutDay) -> List[str]:
        """Get post-workout guidance for a specific workout."""
        
        guidance = [
            f"Complete {workout_day.cool_down_duration} minute cool-down",
            "Rehydrate with water or electrolyte drink",
            "Log your workout performance for progress tracking"
        ]
        
        # Add workout-type specific guidance
        if 'strength' in workout_day.workout_type.lower():
            guidance.append("Consider protein intake within 30-60 minutes")
            guidance.append("Note any exercises that felt particularly challenging")
        elif 'cardio' in workout_day.workout_type.lower() or 'hiit' in workout_day.workout_type.lower():
            guidance.append("Monitor heart rate recovery")
            guidance.append("Focus on rehydration if high sweat rate")
        elif 'flexibility' in workout_day.workout_type.lower():
            guidance.append("Take time to relax and breathe deeply")
            guidance.append("Note areas of improved flexibility")
        
        # Add intensity-specific guidance
        if workout_day.intensity_level == 'High':
            guidance.append("Extra attention to recovery nutrition")
            guidance.append("Consider light stretching or foam rolling")
        
        # Add estimated recovery time
        if workout_day.estimated_calories > 400:
            guidance.append("Allow 24-48 hours recovery before similar intensity workout")
        else:
            guidance.append("Light activity or full rest tomorrow based on how you feel")
        
        return guidance
    
    def _get_exercise_progressions(self, exercise_name: str, fitness_level: FitnessLevel) -> List[str]:
        """Get exercise progressions based on fitness level."""
        
        progressions = {
            'squats': {
                FitnessLevel.BEGINNER: ['Chair squats', 'Bodyweight squats', 'Goblet squats'],
                FitnessLevel.INTERMEDIATE: ['Goblet squats', 'Barbell squats', 'Front squats'],
                FitnessLevel.ADVANCED: ['Barbell squats', 'Single-leg squats', 'Jump squats']
            },
            'lunges': {
                FitnessLevel.BEGINNER: ['Stationary lunges', 'Reverse lunges', 'Side lunges'],
                FitnessLevel.INTERMEDIATE: ['Walking lunges', 'Elevated lunges', 'Weighted lunges'],
                FitnessLevel.ADVANCED: ['Jump lunges', 'Bulgarian split squats', 'Weighted jump lunges']
            },
            'deadlifts': {
                FitnessLevel.BEGINNER: ['Romanian deadlifts', 'Kettlebell deadlifts', 'Sumo deadlifts'],
                FitnessLevel.INTERMEDIATE: ['Conventional deadlifts', 'Single-leg deadlifts', 'Deficit deadlifts'],
                FitnessLevel.ADVANCED: ['Heavy deadlifts', 'Deficit deadlifts', 'Speed deadlifts']
            },
            'push_ups': {
                FitnessLevel.BEGINNER: ['Wall push-ups', 'Incline push-ups', 'Knee push-ups'],
                FitnessLevel.INTERMEDIATE: ['Standard push-ups', 'Wide-grip push-ups', 'Diamond push-ups'],
                FitnessLevel.ADVANCED: ['One-arm push-ups', 'Clap push-ups', 'Archer push-ups']
            },
            'pull_ups': {
                FitnessLevel.BEGINNER: ['Assisted pull-ups', 'Negative pull-ups', 'Lat pulldowns'],
                FitnessLevel.INTERMEDIATE: ['Standard pull-ups', 'Chin-ups', 'Wide-grip pull-ups'],
                FitnessLevel.ADVANCED: ['Weighted pull-ups', 'One-arm pull-ups', 'Muscle-ups']
            }
        }
        
        # Get progressions for the specific exercise, default to basic if not found
        exercise_progressions = progressions.get(exercise_name.lower(), {
            FitnessLevel.BEGINNER: [f'Basic {exercise_name}'],
            FitnessLevel.INTERMEDIATE: [f'Standard {exercise_name}'],
            FitnessLevel.ADVANCED: [f'Advanced {exercise_name}']
        })
        
        return exercise_progressions.get(fitness_level, [f'Standard {exercise_name}'])
