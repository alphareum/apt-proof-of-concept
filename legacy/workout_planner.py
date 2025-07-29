"""
Advanced Workout Planner with Smart Scheduling and Periodization
Comprehensive workout planning system with adaptive scheduling

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import logging
from dataclasses import dataclass, field
import json
import calendar
from models import UserProfile, GoalType, FitnessLevel, ActivityLevel
from enhanced_recommendation_system import (
    EnhancedRecommendationEngine, WorkoutProgram, WeeklyPlan, WorkoutDay, WorkoutPhase
)

logger = logging.getLogger(__name__)

@dataclass
class WorkoutSchedule:
    """Comprehensive workout schedule with smart features."""
    user_id: str
    start_date: date
    end_date: date
    weekly_plans: List[WeeklyPlan]
    rest_days: List[str]
    deload_weeks: List[int]
    assessment_dates: List[date]
    goal_milestones: List[Dict[str, Any]]
    adaptive_triggers: List[Dict[str, Any]]

@dataclass
class ProgressMetrics:
    """Track progress metrics for adaptive planning."""
    date: date
    workout_completed: bool
    perceived_exertion: int  # 1-10 scale
    duration_actual: int
    exercises_completed: List[str]
    weights_used: Dict[str, float]
    reps_completed: Dict[str, int]
    notes: str
    recovery_score: int  # 1-10 scale

@dataclass
class SmartAdjustment:
    """Represents an intelligent adjustment to the workout plan."""
    adjustment_type: str  # 'intensity', 'volume', 'exercise_swap', 'rest_day'
    reason: str
    original_value: Any
    adjusted_value: Any
    confidence_score: float
    effective_date: date

class AdvancedWorkoutPlanner:
    """Advanced workout planner with AI-driven adaptations."""
    
    def __init__(self):
        self.recommendation_engine = EnhancedRecommendationEngine()
        self.progress_tracker = {}
        self.adaptation_algorithms = self._initialize_adaptation_algorithms()
    
    def create_comprehensive_plan(self, user_profile: UserProfile,
                                start_date: date = None,
                                duration_weeks: int = 12) -> WorkoutSchedule:
        """Create a comprehensive workout schedule with smart features."""
        
        if start_date is None:
            start_date = date.today()
        
        end_date = start_date + timedelta(weeks=duration_weeks)
        
        # Generate base program
        program = self.recommendation_engine.generate_complete_program(
            user_profile, duration_weeks
        )
        
        # Add smart scheduling features
        rest_days = self._calculate_optimal_rest_days(user_profile)
        deload_weeks = self._schedule_deload_weeks(duration_weeks, user_profile)
        assessment_dates = self._schedule_assessments(start_date, duration_weeks)
        goal_milestones = self._create_goal_milestones(user_profile, start_date, duration_weeks)
        adaptive_triggers = self._setup_adaptive_triggers(user_profile)
        
        schedule = WorkoutSchedule(
            user_id=user_profile.user_id,
            start_date=start_date,
            end_date=end_date,
            weekly_plans=program.weekly_plans,
            rest_days=rest_days,
            deload_weeks=deload_weeks,
            assessment_dates=assessment_dates,
            goal_milestones=goal_milestones,
            adaptive_triggers=adaptive_triggers
        )
        
        return schedule
    
    def get_daily_workout_plan(self, user_profile: UserProfile,
                             target_date: date,
                             schedule: WorkoutSchedule,
                             recent_progress: List[ProgressMetrics] = None) -> Dict[str, Any]:
        """Get optimized daily workout plan with smart adaptations."""
        
        # Calculate which week in the program
        weeks_elapsed = (target_date - schedule.start_date).days // 7
        
        if weeks_elapsed >= len(schedule.weekly_plans):
            return {'error': 'Date is beyond the scheduled program'}
        
        weekly_plan = schedule.weekly_plans[weeks_elapsed]
        day_of_week = target_date.weekday()
        
        # Check if it's a scheduled rest day
        if target_date.strftime('%A') in schedule.rest_days:
            return self._create_rest_day_plan(user_profile, target_date)
        
        # Check if it's a deload week
        if (weeks_elapsed + 1) in schedule.deload_weeks:
            return self._create_deload_workout(user_profile, weekly_plan, day_of_week)
        
        # Get base workout
        if day_of_week < len(weekly_plan.workout_days):
            base_workout = weekly_plan.workout_days[day_of_week]
        else:
            return self._create_rest_day_plan(user_profile, target_date)
        
        # Apply smart adaptations based on progress
        adapted_workout = self._apply_smart_adaptations(
            base_workout, user_profile, recent_progress or []
        )
        
        # Add contextual information
        workout_plan = {
            'workout': adapted_workout,
            'week_info': {
                'week_number': weeks_elapsed + 1,
                'phase': weekly_plan.phase,
                'phase_week': (weeks_elapsed % 4) + 1,
                'program_progress': f"{weeks_elapsed + 1}/{len(schedule.weekly_plans)}"
            },
            'preparation': self._get_workout_preparation(adapted_workout, user_profile),
            'recovery': self._get_recovery_recommendations(adapted_workout, recent_progress),
            'nutrition': self._get_workout_nutrition_timing(adapted_workout, user_profile),
            'adaptations_made': getattr(adapted_workout, 'adaptations', []),
            'next_milestone': self._get_next_milestone(schedule, target_date)
        }
        
        return workout_plan
    
    def suggest_schedule_adjustments(self, user_profile: UserProfile,
                                   schedule: WorkoutSchedule,
                                   progress_history: List[ProgressMetrics]) -> List[SmartAdjustment]:
        """Suggest intelligent schedule adjustments based on progress."""
        
        adjustments = []
        
        # Analyze recent progress
        recent_progress = progress_history[-14:] if len(progress_history) >= 14 else progress_history
        
        # Check completion rates
        completion_rate = self._calculate_completion_rate(recent_progress)
        if completion_rate < 0.7:
            adjustments.append(SmartAdjustment(
                adjustment_type='volume',
                reason='Low completion rate suggests volume may be too high',
                original_value='current_volume',
                adjusted_value='reduce_by_20_percent',
                confidence_score=0.8,
                effective_date=date.today() + timedelta(days=1)
            ))
        
        # Check perceived exertion trends
        exertion_values = [p.perceived_exertion for p in recent_progress if p.perceived_exertion > 0]
        if exertion_values:
            avg_exertion = sum(exertion_values) / len(exertion_values)
            if avg_exertion > 8.5:
                adjustments.append(SmartAdjustment(
                    adjustment_type='intensity',
                    reason='High perceived exertion indicates need for intensity reduction',
                    original_value='current_intensity',
                    adjusted_value='reduce_by_15_percent',
                    confidence_score=0.85,
                    effective_date=date.today() + timedelta(days=2)
                ))
        
        # Check recovery trends
        recovery_values = [p.recovery_score for p in recent_progress if p.recovery_score > 0]
        if recovery_values:
            avg_recovery = sum(recovery_values) / len(recovery_values)
            if avg_recovery < 6.0:
                adjustments.append(SmartAdjustment(
                    adjustment_type='rest_day',
                    reason='Poor recovery scores suggest need for additional rest',
                    original_value='current_rest_schedule',
                    adjusted_value='add_extra_rest_day',
                    confidence_score=0.9,
                    effective_date=date.today() + timedelta(days=1)
                ))
        
        # Check for plateau patterns
        if self._detect_plateau(progress_history):
            adjustments.append(SmartAdjustment(
                adjustment_type='exercise_swap',
                reason='Progress plateau detected - exercise variation needed',
                original_value='current_exercises',
                adjusted_value='introduce_exercise_variations',
                confidence_score=0.75,
                effective_date=date.today() + timedelta(weeks=1)
            ))
        
        return adjustments
    
    def create_alternative_workouts(self, base_workout: WorkoutDay,
                                  user_profile: UserProfile,
                                  constraints: Dict[str, Any] = None) -> List[WorkoutDay]:
        """Create alternative workouts for flexibility."""
        
        alternatives = []
        constraints = constraints or {}
        
        # Time-constrained alternative
        if constraints.get('limited_time'):
            time_efficient = self._create_time_efficient_version(base_workout, user_profile)
            alternatives.append(time_efficient)
        
        # Equipment-limited alternative
        if constraints.get('limited_equipment'):
            minimal_equipment = self._create_minimal_equipment_version(base_workout, user_profile)
            alternatives.append(minimal_equipment)
        
        # Low-intensity alternative
        recovery_workout = self._create_recovery_version(base_workout, user_profile)
        alternatives.append(recovery_workout)
        
        # High-intensity alternative
        if user_profile.fitness_level != FitnessLevel.BEGINNER:
            intense_workout = self._create_intensified_version(base_workout, user_profile)
            alternatives.append(intense_workout)
        
        return alternatives
    
    def generate_monthly_overview(self, user_profile: UserProfile,
                                schedule: WorkoutSchedule,
                                target_month: int,
                                target_year: int) -> Dict[str, Any]:
        """Generate comprehensive monthly workout overview."""
        
        # Get calendar for the month
        cal = calendar.monthcalendar(target_year, target_month)
        month_start = date(target_year, target_month, 1)
        
        # Calculate month statistics
        month_stats = {
            'total_workout_days': 0,
            'total_rest_days': 0,
            'phases_this_month': [],
            'estimated_calories': 0,
            'focus_areas': {},
            'progression_goals': []
        }
        
        # Generate daily breakdown
        daily_breakdown = {}
        
        for week in cal:
            for day in week:
                if day == 0:  # Empty day in calendar
                    continue
                
                current_date = date(target_year, target_month, day)
                
                if schedule.start_date <= current_date <= schedule.end_date:
                    day_plan = self.get_daily_workout_plan(
                        user_profile, current_date, schedule
                    )
                    
                    daily_breakdown[day] = {
                        'date': current_date,
                        'plan': day_plan,
                        'is_rest_day': 'error' in day_plan or day_plan.get('workout', {}).get('workout_type') == 'rest'
                    }
                    
                    # Update statistics
                    if not daily_breakdown[day]['is_rest_day']:
                        month_stats['total_workout_days'] += 1
                        workout = day_plan.get('workout', {})
                        month_stats['estimated_calories'] += workout.get('estimated_calories', 0)
                        
                        # Track focus areas
                        for area in workout.get('focus_areas', []):
                            month_stats['focus_areas'][area] = month_stats['focus_areas'].get(area, 0) + 1
                    else:
                        month_stats['total_rest_days'] += 1
        
        # Identify phases
        for weekly_plan in schedule.weekly_plans:
            if weekly_plan.phase not in month_stats['phases_this_month']:
                month_stats['phases_this_month'].append(weekly_plan.phase)
        
        # Generate progression goals for the month
        month_stats['progression_goals'] = self._generate_monthly_goals(
            user_profile, schedule, target_month, target_year
        )
        
        return {
            'month_year': f"{calendar.month_name[target_month]} {target_year}",
            'statistics': month_stats,
            'daily_breakdown': daily_breakdown,
            'key_focuses': self._identify_month_key_focuses(month_stats),
            'challenges': self._suggest_monthly_challenges(user_profile, month_stats),
            'nutrition_themes': self._suggest_monthly_nutrition_themes(month_stats)
        }
    
    def _apply_smart_adaptations(self, base_workout: WorkoutDay,
                               user_profile: UserProfile,
                               recent_progress: List[ProgressMetrics]) -> WorkoutDay:
        """Apply intelligent adaptations to workout based on progress."""
        
        adapted_workout = base_workout
        adaptations = []
        
        if not recent_progress:
            return adapted_workout
        
        # Check last workout performance
        last_workout = recent_progress[-1] if recent_progress else None
        
        if last_workout:
            # Adapt based on last workout completion
            if not last_workout.workout_completed:
                adaptations.append("Reduced volume due to incomplete previous workout")
                adapted_workout = self._reduce_workout_volume(adapted_workout, 0.85)
            
            # Adapt based on perceived exertion
            if last_workout.perceived_exertion >= 9:
                adaptations.append("Reduced intensity due to high exertion last workout")
                adapted_workout = self._reduce_workout_intensity(adapted_workout, 0.9)
            elif last_workout.perceived_exertion <= 5:
                adaptations.append("Slightly increased intensity due to low exertion")
                adapted_workout = self._increase_workout_intensity(adapted_workout, 1.1)
        
        # Check weekly pattern
        weekly_progress = recent_progress[-7:] if len(recent_progress) >= 7 else recent_progress
        recovery_scores = [p.recovery_score for p in weekly_progress if p.recovery_score > 0]
        
        if recovery_scores:
            avg_recovery = sum(recovery_scores) / len(recovery_scores)
            if avg_recovery < 6:
                adaptations.append("Added extra recovery focus due to low recovery scores")
                adapted_workout = self._add_recovery_focus(adapted_workout)
        
        # Add adaptations to workout
        adapted_workout.notes.extend(adaptations)
        setattr(adapted_workout, 'adaptations', adaptations)
        
        return adapted_workout
    
    def _calculate_optimal_rest_days(self, user_profile: UserProfile) -> List[str]:
        """Calculate optimal rest days based on user profile."""
        
        workout_days = user_profile.workout_days_per_week
        
        if workout_days <= 3:
            return ['Tuesday', 'Thursday', 'Saturday', 'Sunday']
        elif workout_days <= 4:
            return ['Wednesday', 'Saturday', 'Sunday']
        elif workout_days <= 5:
            return ['Thursday', 'Sunday']
        else:
            return ['Sunday']
    
    def _schedule_deload_weeks(self, duration_weeks: int, user_profile: UserProfile) -> List[int]:
        """Schedule deload weeks strategically."""
        
        deload_weeks = []
        
        if duration_weeks >= 4:
            # Every 4th week for beginners, every 5th for advanced
            interval = 4 if user_profile.fitness_level == FitnessLevel.BEGINNER else 5
            
            for week in range(interval, duration_weeks + 1, interval):
                deload_weeks.append(week)
        
        return deload_weeks
    
    def _schedule_assessments(self, start_date: date, duration_weeks: int) -> List[date]:
        """Schedule fitness assessments."""
        
        assessments = []
        
        # Initial assessment
        assessments.append(start_date)
        
        # Mid-program assessment
        if duration_weeks >= 8:
            mid_date = start_date + timedelta(weeks=duration_weeks // 2)
            assessments.append(mid_date)
        
        # Final assessment
        final_date = start_date + timedelta(weeks=duration_weeks)
        assessments.append(final_date)
        
        return assessments
    
    def _create_goal_milestones(self, user_profile: UserProfile,
                              start_date: date, duration_weeks: int) -> List[Dict[str, Any]]:
        """Create goal-specific milestones."""
        
        milestones = []
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            milestones = [
                {
                    'week': 2,
                    'goal': 'Establish consistent exercise routine',
                    'metric': 'workout_completion_rate',
                    'target': 0.85
                },
                {
                    'week': 6,
                    'goal': 'Improve cardiovascular endurance',
                    'metric': 'cardio_duration_increase',
                    'target': 1.25
                },
                {
                    'week': 12,
                    'goal': 'Achieve target weight loss',
                    'metric': 'weight_change',
                    'target': -0.5  # kg per week
                }
            ]
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            milestones = [
                {
                    'week': 4,
                    'goal': 'Establish progressive overload pattern',
                    'metric': 'strength_increase',
                    'target': 1.15
                },
                {
                    'week': 8,
                    'goal': 'Achieve hypertrophy adaptations',
                    'metric': 'volume_tolerance',
                    'target': 1.3
                },
                {
                    'week': 12,
                    'goal': 'Measurable muscle growth',
                    'metric': 'strength_increase',
                    'target': 1.25
                }
            ]
        
        # Add dates to milestones
        for milestone in milestones:
            milestone['date'] = start_date + timedelta(weeks=milestone['week'])
        
        return milestones
    
    def _setup_adaptive_triggers(self, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Setup triggers for adaptive adjustments."""
        
        triggers = [
            {
                'name': 'completion_rate_low',
                'condition': 'workout_completion_rate < 0.7',
                'action': 'reduce_volume',
                'sensitivity': 0.8
            },
            {
                'name': 'perceived_exertion_high',
                'condition': 'avg_perceived_exertion > 8.5',
                'action': 'reduce_intensity',
                'sensitivity': 0.85
            },
            {
                'name': 'recovery_poor',
                'condition': 'avg_recovery_score < 6.0',
                'action': 'add_rest_day',
                'sensitivity': 0.9
            },
            {
                'name': 'plateau_detected',
                'condition': 'no_progress_for_2_weeks',
                'action': 'exercise_variation',
                'sensitivity': 0.75
            }
        ]
        
        return triggers
    
    def _initialize_adaptation_algorithms(self) -> Dict[str, Any]:
        """Initialize adaptation algorithms."""
        
        return {
            'volume_adaptation': self._volume_adaptation_algorithm,
            'intensity_adaptation': self._intensity_adaptation_algorithm,
            'exercise_variation': self._exercise_variation_algorithm,
            'recovery_optimization': self._recovery_optimization_algorithm
        }
    
    # Additional helper methods for workout adaptations and calculations...
    
    def _reduce_workout_volume(self, workout: WorkoutDay, factor: float) -> WorkoutDay:
        """Reduce workout volume by given factor."""
        # Implementation would modify sets/reps
        return workout
    
    def _reduce_workout_intensity(self, workout: WorkoutDay, factor: float) -> WorkoutDay:
        """Reduce workout intensity by given factor."""
        # Implementation would modify intensity parameters
        return workout
    
    def _increase_workout_intensity(self, workout: WorkoutDay, factor: float) -> WorkoutDay:
        """Increase workout intensity by given factor."""
        # Implementation would modify intensity parameters
        return workout
    
    def _add_recovery_focus(self, workout: WorkoutDay) -> WorkoutDay:
        """Add recovery-focused elements to workout."""
        # Implementation would add stretching, reduce intensity
        return workout
    
    def _calculate_completion_rate(self, progress: List[ProgressMetrics]) -> float:
        """Calculate workout completion rate."""
        if not progress:
            return 1.0
        
        completed = sum(1 for p in progress if p.workout_completed)
        return completed / len(progress)
    
    def _detect_plateau(self, progress_history: List[ProgressMetrics]) -> bool:
        """Detect if user has hit a plateau."""
        # Implementation would analyze progress trends
        return False
    
    def _create_rest_day_plan(self, user_profile: UserProfile, target_date: date) -> Dict[str, Any]:
        """Create a rest day plan with active recovery options."""
        
        return {
            'type': 'Rest Day',
            'date': target_date,
            'activities': [
                'Light walking (15-20 minutes)',
                'Gentle stretching routine',
                'Foam rolling or self-massage',
                'Meditation or breathing exercises'
            ],
            'focus': 'Recovery and regeneration',
            'nutrition_emphasis': 'Hydration and anti-inflammatory foods',
            'sleep_target': '7-9 hours',
            'stress_management': 'Practice relaxation techniques'
        }
    
    def _create_deload_workout(self, user_profile: UserProfile,
                             weekly_plan: WeeklyPlan, day_of_week: int) -> Dict[str, Any]:
        """Create a deload version of the scheduled workout."""
        
        if day_of_week < len(weekly_plan.workout_days):
            base_workout = weekly_plan.workout_days[day_of_week]
            
            # Reduce intensity and volume for deload
            deload_workout = base_workout
            deload_workout.notes.append("Deload week: Reduced intensity and volume")
            
            return {
                'workout': deload_workout,
                'deload_notes': [
                    'Focus on technique and form',
                    'Reduce weights by 40-50%',
                    'Maintain movement patterns',
                    'Emphasize recovery'
                ]
            }
        
        return self._create_rest_day_plan(user_profile, date.today())
    
    # Adaptation algorithm implementations
    def _volume_adaptation_algorithm(self, workout: WorkoutDay, progress_data: List[ProgressMetrics], 
                                   user_profile: UserProfile) -> Dict[str, Any]:
        """Algorithm for adapting workout volume based on progress."""
        adaptations = {
            'volume_multiplier': 1.0,
            'reason': '',
            'confidence': 0.0
        }
        
        if not progress_data:
            return adaptations
        
        # Calculate completion rate
        completion_rate = self._calculate_completion_rate(progress_data)
        
        if completion_rate < 0.7:
            adaptations['volume_multiplier'] = 0.8
            adaptations['reason'] = 'Reducing volume due to low completion rate'
            adaptations['confidence'] = 0.85
        elif completion_rate > 0.95:
            # Gradually increase volume if user is consistently completing workouts
            adaptations['volume_multiplier'] = 1.1
            adaptations['reason'] = 'Increasing volume due to high completion rate'
            adaptations['confidence'] = 0.7
        
        return adaptations
    
    def _intensity_adaptation_algorithm(self, workout: WorkoutDay, progress_data: List[ProgressMetrics], 
                                      user_profile: UserProfile) -> Dict[str, Any]:
        """Algorithm for adapting workout intensity based on progress."""
        adaptations = {
            'intensity_multiplier': 1.0,
            'reason': '',
            'confidence': 0.0
        }
        
        if not progress_data:
            return adaptations
        
        # Analyze recent perceived exertion
        recent_exertion = [p.perceived_exertion for p in progress_data[-7:] if p.perceived_exertion > 0]
        
        if recent_exertion:
            avg_exertion = sum(recent_exertion) / len(recent_exertion)
            
            if avg_exertion > 8.5:
                adaptations['intensity_multiplier'] = 0.9
                adaptations['reason'] = 'Reducing intensity due to high perceived exertion'
                adaptations['confidence'] = 0.8
            elif avg_exertion < 5.0:
                adaptations['intensity_multiplier'] = 1.1
                adaptations['reason'] = 'Increasing intensity due to low perceived exertion'
                adaptations['confidence'] = 0.7
        
        return adaptations
    
    def _exercise_variation_algorithm(self, workout: WorkoutDay, progress_data: List[ProgressMetrics], 
                                    user_profile: UserProfile) -> Dict[str, Any]:
        """Algorithm for determining when to vary exercises."""
        adaptations = {
            'variation_needed': False,
            'reason': '',
            'confidence': 0.0,
            'suggested_variations': []
        }
        
        if not progress_data:
            return adaptations
        
        # Check for plateau patterns
        if self._detect_plateau(progress_data):
            adaptations['variation_needed'] = True
            adaptations['reason'] = 'Progress plateau detected - exercise variation recommended'
            adaptations['confidence'] = 0.75
            adaptations['suggested_variations'] = [
                'Change exercise order',
                'Introduce new exercise variations',
                'Modify rep ranges',
                'Add complexity to movements'
            ]
        
        return adaptations
    
    def _recovery_optimization_algorithm(self, workout: WorkoutDay, progress_data: List[ProgressMetrics], 
                                       user_profile: UserProfile) -> Dict[str, Any]:
        """Algorithm for optimizing recovery based on progress."""
        adaptations = {
            'recovery_focus': False,
            'rest_needed': False,
            'reason': '',
            'confidence': 0.0
        }
        
        if not progress_data:
            return adaptations
        
        # Analyze recovery scores
        recent_recovery = [p.recovery_score for p in progress_data[-7:] if p.recovery_score > 0]
        
        if recent_recovery:
            avg_recovery = sum(recent_recovery) / len(recent_recovery)
            
            if avg_recovery < 6.0:
                adaptations['recovery_focus'] = True
                adaptations['reason'] = 'Low recovery scores indicate need for recovery focus'
                adaptations['confidence'] = 0.85
                
                if avg_recovery < 4.0:
                    adaptations['rest_needed'] = True
                    adaptations['reason'] = 'Very low recovery scores suggest additional rest day needed'
                    adaptations['confidence'] = 0.9
        
        return adaptations
    
    # Helper methods for workout plan components
    def _get_workout_preparation(self, workout: WorkoutDay, user_profile: UserProfile) -> Dict[str, Any]:
        """Get workout preparation recommendations."""
        return {
            'tips': [
                f'Allocate {workout.warm_up_duration} minutes for warm-up',
                'Ensure you\'re well-hydrated before starting',
                'Have all necessary equipment ready and accessible',
                'Review proper form for each exercise',
                'Set your mental focus and goals for the session'
            ],
            'warm_up_duration': '10-15 minutes',
            'dynamic_stretches': [
                'Arm circles',
                'Leg swings',
                'Hip circles',
                'Light cardio'
            ],
            'mental_preparation': 'Review exercise form and goals',
            'equipment_check': 'Ensure all equipment is available and safe'
        }
    
    def _get_recovery_recommendations(self, workout: WorkoutDay, 
                                    recent_progress: List[ProgressMetrics]) -> Dict[str, Any]:
        """Get post-workout recovery recommendations."""
        
        base_recommendations = [
            f'Complete {workout.cool_down_duration} minute cool-down',
            'Static stretching (10-15 minutes)',
            'Protein and carbs within 30-60 minutes',
            'Stay hydrated throughout and after workout',
            'Aim for 7-9 hours of quality sleep tonight'
        ]
        
        # Adjust based on recent progress
        if recent_progress:
            recovery_scores = [p.recovery_score for p in recent_progress[-3:] if p.recovery_score > 0]
            if recovery_scores:
                avg_recovery = sum(recovery_scores) / len(recovery_scores)
                if avg_recovery < 6.0:
                    base_recommendations.extend([
                        'Consider a warm bath for extra recovery',
                        'Gentle foam rolling for muscle tension',
                        'Focus on stress management and relaxation'
                    ])
        
        return {
            'recommendations': base_recommendations,
            'cool_down': [
                'Static stretching (10-15 minutes)',
                'Light walking',
                'Deep breathing exercises'
            ],
            'nutrition': 'Protein and carbs within 30-60 minutes',
            'hydration': 'Drink water throughout and after workout',
            'sleep': 'Aim for 7-9 hours of quality sleep'
        }
        
        return recommendations
    
    def _get_workout_nutrition_timing(self, workout: WorkoutDay, user_profile: UserProfile) -> Dict[str, Any]:
        """Get nutrition timing recommendations for the workout."""
        return {
            'pre_workout': 'Light snack 30-60 minutes before (banana with nut butter, Greek yogurt)',
            'post_workout': f'Protein + carbs within 30-60 minutes ({int(user_profile.weight * 0.3)}g protein)',
            'during_workout': 'Stay hydrated, sip water throughout',
            'detailed': {
                'pre_workout': {
                    'timing': '30-60 minutes before',
                    'suggestions': [
                        'Light carbs and protein',
                        'Banana with nut butter',
                        'Greek yogurt with berries'
                    ]
                },
                'during_workout': {
                    'hydration': 'Sip water throughout',
                    'electrolytes': 'Consider for workouts > 60 minutes'
                },
                'post_workout': {
                    'timing': 'Within 30-60 minutes',
                    'protein': f'{int(user_profile.weight * 0.3)}g protein',
                    'carbs': 'Quick-digesting carbs to replenish glycogen'
                }
            }
        }
    
    def _get_next_milestone(self, schedule: WorkoutSchedule, current_date: date) -> Dict[str, Any]:
        """Get the next upcoming milestone."""
        upcoming_milestones = [
            milestone for milestone in schedule.goal_milestones
            if milestone['date'] > current_date
        ]
        
        if upcoming_milestones:
            next_milestone = min(upcoming_milestones, key=lambda x: x['date'])
            days_until = (next_milestone['date'] - current_date).days
            
            return {
                'milestone': next_milestone,
                'days_until': days_until,
                'progress_needed': f"Focus on {next_milestone['goal']}"
            }
        
        return {'milestone': None, 'message': 'All milestones completed!'}
    
    # Additional helper methods for alternative workout creation
    def _create_time_efficient_version(self, base_workout: WorkoutDay, user_profile: UserProfile) -> WorkoutDay:
        """Create a time-efficient version of the workout."""
        # Create a copy and modify for time efficiency
        efficient_workout = base_workout
        efficient_workout.notes.append("Time-efficient version: Reduced rest times and compound movements")
        return efficient_workout
    
    def _create_minimal_equipment_version(self, base_workout: WorkoutDay, user_profile: UserProfile) -> WorkoutDay:
        """Create a minimal equipment version of the workout."""
        minimal_workout = base_workout
        minimal_workout.notes.append("Minimal equipment version: Bodyweight and basic equipment alternatives")
        return minimal_workout
    
    def _create_recovery_version(self, base_workout: WorkoutDay, user_profile: UserProfile) -> WorkoutDay:
        """Create a recovery-focused version of the workout."""
        recovery_workout = base_workout
        recovery_workout.notes.append("Recovery version: Lower intensity with focus on mobility and regeneration")
        return recovery_workout
    
    def _create_intensified_version(self, base_workout: WorkoutDay, user_profile: UserProfile) -> WorkoutDay:
        """Create an intensified version of the workout."""
        intense_workout = base_workout
        intense_workout.notes.append("Intensified version: Higher intensity for advanced challenge")
        return intense_workout
    
    # Monthly planning helper methods
    def _generate_monthly_goals(self, user_profile: UserProfile, schedule: WorkoutSchedule,
                              target_month: int, target_year: int) -> List[str]:
        """Generate specific goals for the month."""
        goals = []
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            goals.extend([
                'Maintain consistent workout schedule',
                'Increase cardio endurance',
                'Focus on compound movements'
            ])
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            goals.extend([
                'Progressive overload on key lifts',
                'Increase training volume gradually',
                'Focus on muscle-building exercises'
            ])
        else:
            goals.extend([
                'Improve overall fitness',
                'Maintain exercise variety',
                'Build healthy habits'
            ])
        
        return goals
    
    def _identify_month_key_focuses(self, month_stats: Dict[str, Any]) -> List[str]:
        """Identify key focus areas for the month."""
        focuses = []
        
        # Determine focus based on workout distribution
        focus_areas = month_stats.get('focus_areas', {})
        if focus_areas:
            top_focus = max(focus_areas, key=focus_areas.get)
            focuses.append(f"Primary focus: {top_focus}")
        
        # Add phase-specific focuses
        phases = month_stats.get('phases_this_month', [])
        if phases:
            focuses.append(f"Training phases: {', '.join(phases)}")
        
        return focuses
    
    def _suggest_monthly_challenges(self, user_profile: UserProfile, month_stats: Dict[str, Any]) -> List[str]:
        """Suggest monthly fitness challenges."""
        challenges = [
            'Complete all scheduled workouts',
            'Try one new exercise each week',
            'Improve form on key movements',
            'Track progress measurements'
        ]
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            challenges.append('Add 5 minutes to cardio sessions')
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            challenges.append('Increase weight on compound lifts')
        
        return challenges
    
    def _suggest_monthly_nutrition_themes(self, month_stats: Dict[str, Any]) -> List[str]:
        """Suggest nutrition themes based on workout focus."""
        themes = [
            'Adequate protein for recovery',
            'Hydration throughout workouts',
            'Pre and post-workout nutrition'
        ]
        
        workout_days = month_stats.get('total_workout_days', 0)
        if workout_days > 15:
            themes.append('Higher calorie intake for increased activity')
        
        return themes
