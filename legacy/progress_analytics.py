"""
Advanced Progress Analytics and Tracking System
Comprehensive fitness tracking with detailed analytics

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import logging
from dataclasses import dataclass, field
import json
from models import UserProfile, GoalType
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    WEIGHT = "weight"
    BODY_FAT = "body_fat"
    MUSCLE_MASS = "muscle_mass"
    MEASUREMENTS = "measurements"
    PERFORMANCE = "performance"
    STRENGTH = "strength"
    CARDIO = "cardio"

@dataclass
class BodyMeasurement:
    """Body measurement record."""
    date: date
    weight: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    muscle_mass_kg: Optional[float] = None
    waist_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    bicep_cm: Optional[float] = None
    thigh_cm: Optional[float] = None
    notes: str = ""

@dataclass
class PerformanceMetric:
    """Performance measurement record."""
    date: date
    exercise_name: str
    metric_type: str  # weight, reps, time, distance
    value: float
    unit: str
    notes: str = ""

@dataclass
class WorkoutLog:
    """Detailed workout log entry."""
    date: date
    workout_name: str
    duration_minutes: int
    exercises_completed: List[Dict[str, Any]]
    total_volume: int  # total weight lifted
    calories_burned: int
    perceived_exertion: int  # 1-10 scale
    mood_before: int  # 1-10 scale
    mood_after: int  # 1-10 scale
    notes: str = ""

@dataclass
class ProgressPhoto:
    """Progress photo record."""
    date: date
    photo_path: str
    body_part: str  # front, side, back, specific muscle group
    notes: str = ""

class ProgressAnalytics:
    """Advanced progress tracking and analytics system."""
    
    def __init__(self):
        self.body_measurements = []
        self.performance_metrics = []
        self.workout_logs = []
        self.progress_photos = []
    
    def log_body_measurement(self, measurement: BodyMeasurement):
        """Log a body measurement."""
        self.body_measurements.append(measurement)
        self.body_measurements.sort(key=lambda x: x.date)
    
    def log_performance_metric(self, metric: PerformanceMetric):
        """Log a performance metric."""
        self.performance_metrics.append(metric)
        self.performance_metrics.sort(key=lambda x: x.date)
    
    def log_workout(self, workout: WorkoutLog):
        """Log a workout session."""
        self.workout_logs.append(workout)
        self.workout_logs.sort(key=lambda x: x.date)
    
    def add_progress_photo(self, photo: ProgressPhoto):
        """Add a progress photo."""
        self.progress_photos.append(photo)
        self.progress_photos.sort(key=lambda x: x.date)
    
    def get_weight_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get weight trend analysis."""
        
        cutoff_date = date.today() - timedelta(days=days)
        recent_measurements = [
            m for m in self.body_measurements 
            if m.date >= cutoff_date and m.weight is not None
        ]
        
        if len(recent_measurements) < 2:
            return {
                'trend': 'insufficient_data',
                'change': 0,
                'rate_per_week': 0,
                'measurements': []
            }
        
        weights = [m.weight for m in recent_measurements]
        dates = [m.date for m in recent_measurements]
        
        # Calculate trend
        initial_weight = weights[0]
        final_weight = weights[-1]
        total_change = final_weight - initial_weight
        
        # Calculate rate per week
        days_elapsed = (dates[-1] - dates[0]).days
        rate_per_week = (total_change / max(days_elapsed, 1)) * 7 if days_elapsed > 0 else 0
        
        # Determine trend direction
        if abs(total_change) < 0.5:
            trend = 'stable'
        elif total_change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'change': round(total_change, 1),
            'rate_per_week': round(rate_per_week, 2),
            'measurements': [(m.date, m.weight) for m in recent_measurements]
        }
    
    def get_strength_progress(self, exercise_name: str) -> Dict[str, Any]:
        """Get strength progress for a specific exercise."""
        
        exercise_metrics = [
            m for m in self.performance_metrics 
            if m.exercise_name.lower() == exercise_name.lower() and m.metric_type == 'weight'
        ]
        
        if len(exercise_metrics) < 2:
            return {
                'progress': 'insufficient_data',
                'improvement': 0,
                'best_performance': None,
                'recent_performance': None
            }
        
        # Sort by date
        exercise_metrics.sort(key=lambda x: x.date)
        
        initial_performance = exercise_metrics[0]
        best_performance = max(exercise_metrics, key=lambda x: x.value)
        recent_performance = exercise_metrics[-1]
        
        improvement = recent_performance.value - initial_performance.value
        improvement_percentage = (improvement / initial_performance.value) * 100 if initial_performance.value > 0 else 0
        
        return {
            'progress': 'improving' if improvement > 0 else 'declining' if improvement < 0 else 'stable',
            'improvement': round(improvement, 1),
            'improvement_percentage': round(improvement_percentage, 1),
            'best_performance': {
                'value': best_performance.value,
                'date': best_performance.date,
                'unit': best_performance.unit
            },
            'recent_performance': {
                'value': recent_performance.value,
                'date': recent_performance.date,
                'unit': recent_performance.unit
            },
            'data_points': [(m.date, m.value) for m in exercise_metrics]
        }
    
    def get_workout_consistency(self, days: int = 30) -> Dict[str, Any]:
        """Analyze workout consistency."""
        
        cutoff_date = date.today() - timedelta(days=days)
        recent_workouts = [
            w for w in self.workout_logs 
            if w.date >= cutoff_date
        ]
        
        total_days = days
        workout_days = len(set(w.date for w in recent_workouts))
        consistency_rate = (workout_days / total_days) * 100
        
        # Calculate current streak
        current_streak = self._calculate_current_streak()
        
        # Calculate average workout duration
        avg_duration = np.mean([w.duration_minutes for w in recent_workouts]) if recent_workouts else 0
        
        # Calculate frequency pattern
        weekday_workouts = {}
        for workout in recent_workouts:
            weekday = workout.date.strftime('%A')
            weekday_workouts[weekday] = weekday_workouts.get(weekday, 0) + 1
        
        return {
            'consistency_rate': round(consistency_rate, 1),
            'workout_days': workout_days,
            'total_days': total_days,
            'current_streak': current_streak,
            'avg_duration': round(avg_duration, 1),
            'weekly_pattern': weekday_workouts,
            'total_workouts': len(recent_workouts)
        }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        
        if not self.workout_logs:
            return {'status': 'no_data'}
        
        # Total statistics
        total_workouts = len(self.workout_logs)
        total_minutes = sum(w.duration_minutes for w in self.workout_logs)
        total_calories = sum(w.calories_burned for w in self.workout_logs)
        
        # Average metrics
        avg_duration = total_minutes / total_workouts if total_workouts > 0 else 0
        avg_calories = total_calories / total_workouts if total_workouts > 0 else 0
        
        # Mood analysis
        mood_improvements = []
        for workout in self.workout_logs:
            if workout.mood_before > 0 and workout.mood_after > 0:
                improvement = workout.mood_after - workout.mood_before
                mood_improvements.append(improvement)
        
        avg_mood_improvement = np.mean(mood_improvements) if mood_improvements else 0
        
        # Volume progression (for strength training)
        monthly_volumes = self._calculate_monthly_volumes()
        
        # Most frequent exercises
        exercise_frequency = {}
        for workout in self.workout_logs:
            for exercise in workout.exercises_completed:
                name = exercise.get('name', 'Unknown')
                exercise_frequency[name] = exercise_frequency.get(name, 0) + 1
        
        top_exercises = sorted(exercise_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_workouts': total_workouts,
            'total_minutes': total_minutes,
            'total_calories': total_calories,
            'avg_duration': round(avg_duration, 1),
            'avg_calories': round(avg_calories, 0),
            'avg_mood_improvement': round(avg_mood_improvement, 1),
            'monthly_volumes': monthly_volumes,
            'top_exercises': top_exercises,
            'first_workout': self.workout_logs[0].date if self.workout_logs else None,
            'last_workout': self.workout_logs[-1].date if self.workout_logs else None
        }
    
    def generate_progress_report(self, user_profile: UserProfile, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        
        # Weight analysis
        weight_analysis = self.get_weight_trend(period_days)
        
        # Workout consistency
        consistency = self.get_workout_consistency(period_days)
        
        # Performance analytics
        performance = self.get_performance_analytics()
        
        # Goal-specific insights
        goal_insights = self._generate_goal_insights(user_profile, period_days)
        
        # Recommendations
        recommendations = self._generate_recommendations(user_profile, consistency, weight_analysis)
        
        # Achievement highlights
        achievements = self._identify_achievements(performance, consistency, weight_analysis)
        
        return {
            'period_days': period_days,
            'weight_analysis': weight_analysis,
            'consistency': consistency,
            'performance': performance,
            'goal_insights': goal_insights,
            'recommendations': recommendations,
            'achievements': achievements,
            'generated_date': datetime.now()
        }
    
    def predict_goal_timeline(self, user_profile: UserProfile, target_metric: str, target_value: float) -> Dict[str, Any]:
        """Predict timeline to reach a specific goal."""
        
        if target_metric == 'weight':
            weight_trend = self.get_weight_trend(60)  # 2 months of data
            
            if weight_trend['trend'] == 'insufficient_data':
                return {'prediction': 'insufficient_data'}
            
            current_weight = self.body_measurements[-1].weight if self.body_measurements else user_profile.weight
            weight_diff = target_value - current_weight
            rate_per_week = weight_trend['rate_per_week']
            
            if abs(rate_per_week) < 0.1:
                return {'prediction': 'no_trend', 'message': 'Current trend is too slow to predict'}
            
            weeks_to_goal = weight_diff / rate_per_week
            days_to_goal = weeks_to_goal * 7
            
            if days_to_goal < 0:
                return {'prediction': 'goal_passed', 'message': 'You have already passed this goal!'}
            
            estimated_date = date.today() + timedelta(days=int(days_to_goal))
            
            return {
                'prediction': 'success',
                'estimated_days': int(days_to_goal),
                'estimated_weeks': round(weeks_to_goal, 1),
                'estimated_date': estimated_date,
                'current_rate': rate_per_week,
                'confidence': 'medium' if len(self.body_measurements) > 10 else 'low'
            }
        
        return {'prediction': 'not_supported', 'message': 'Prediction not available for this metric'}
    
    def get_exercise_volume_trends(self, exercise_name: str = None) -> Dict[str, Any]:
        """Get volume trends for exercises."""
        
        if exercise_name:
            # Specific exercise analysis
            exercise_logs = []
            for workout in self.workout_logs:
                for exercise in workout.exercises_completed:
                    if exercise.get('name', '').lower() == exercise_name.lower():
                        volume = exercise.get('sets', 0) * exercise.get('reps', 0) * exercise.get('weight', 0)
                        exercise_logs.append({
                            'date': workout.date,
                            'volume': volume,
                            'sets': exercise.get('sets', 0),
                            'reps': exercise.get('reps', 0),
                            'weight': exercise.get('weight', 0)
                        })
            
            return {'exercise': exercise_name, 'logs': exercise_logs}
        
        else:
            # Overall volume trends
            monthly_volumes = self._calculate_monthly_volumes()
            return {'monthly_volumes': monthly_volumes}
    
    def _calculate_current_streak(self) -> int:
        """Calculate current workout streak."""
        
        if not self.workout_logs:
            return 0
        
        # Get unique workout dates
        workout_dates = sorted(set(w.date for w in self.workout_logs), reverse=True)
        
        if not workout_dates:
            return 0
        
        # Check if user worked out today or yesterday
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        if workout_dates[0] not in [today, yesterday]:
            return 0
        
        # Count consecutive days
        streak = 0
        current_date = workout_dates[0]
        
        for workout_date in workout_dates:
            if workout_date == current_date:
                streak += 1
                current_date -= timedelta(days=1)
            else:
                break
        
        return streak
    
    def _calculate_monthly_volumes(self) -> Dict[str, int]:
        """Calculate total training volume by month."""
        
        monthly_volumes = {}
        
        for workout in self.workout_logs:
            month_key = workout.date.strftime('%Y-%m')
            
            if month_key not in monthly_volumes:
                monthly_volumes[month_key] = 0
            
            monthly_volumes[month_key] += workout.total_volume
        
        return monthly_volumes
    
    def _generate_goal_insights(self, user_profile: UserProfile, period_days: int) -> Dict[str, Any]:
        """Generate insights specific to user's goals."""
        
        insights = {}
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            weight_trend = self.get_weight_trend(period_days)
            if weight_trend['trend'] == 'decreasing':
                insights['weight_loss'] = {
                    'status': 'on_track',
                    'message': f"Great progress! You've lost {abs(weight_trend['change'])} kg in {period_days} days.",
                    'rate': f"{abs(weight_trend['rate_per_week'])} kg/week"
                }
            elif weight_trend['trend'] == 'stable':
                insights['weight_loss'] = {
                    'status': 'plateau',
                    'message': "Weight has been stable. Consider adjusting diet or increasing exercise intensity.",
                    'suggestion': "Try increasing workout frequency or reducing calories by 200/day"
                }
            else:
                insights['weight_loss'] = {
                    'status': 'needs_attention',
                    'message': "Weight is increasing. Review diet and exercise plan.",
                    'suggestion': "Focus on creating a caloric deficit through diet and cardio"
                }
        
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            # Analyze strength progression and workout volume
            consistency = self.get_workout_consistency(period_days)
            if consistency['consistency_rate'] > 70:
                insights['muscle_gain'] = {
                    'status': 'consistent',
                    'message': f"Excellent consistency at {consistency['consistency_rate']}%!",
                    'suggestion': "Continue current routine and ensure adequate protein intake"
                }
            else:
                insights['muscle_gain'] = {
                    'status': 'inconsistent',
                    'message': f"Consistency is at {consistency['consistency_rate']}%. Aim for 70%+",
                    'suggestion': "Schedule workouts in advance and track protein intake"
                }
        
        elif user_profile.primary_goal == GoalType.STRENGTH:
            # Analyze key strength metrics
            strength_exercises = ['bench press', 'squat', 'deadlift', 'overhead press']
            strength_progress = {}
            
            for exercise in strength_exercises:
                progress = self.get_strength_progress(exercise)
                if progress['progress'] != 'insufficient_data':
                    strength_progress[exercise] = progress
            
            if strength_progress:
                improving_exercises = [ex for ex, prog in strength_progress.items() if prog['progress'] == 'improving']
                insights['strength'] = {
                    'status': 'progressing' if improving_exercises else 'plateaued',
                    'improving_exercises': improving_exercises,
                    'message': f"You're improving in {len(improving_exercises)} key exercises"
                }
        
        return insights
    
    def _generate_recommendations(self, user_profile: UserProfile, consistency: Dict, weight_analysis: Dict) -> List[str]:
        """Generate personalized recommendations."""
        
        recommendations = []
        
        # Consistency recommendations
        if consistency['consistency_rate'] < 50:
            recommendations.append("🎯 Focus on consistency - aim for at least 3-4 workouts per week")
        elif consistency['consistency_rate'] < 70:
            recommendations.append("📈 You're doing well! Try to increase workout frequency slightly")
        else:
            recommendations.append("🌟 Excellent consistency! Keep up the great work")
        
        # Duration recommendations
        if consistency['avg_duration'] < 30:
            recommendations.append("⏰ Consider longer workout sessions (45-60 minutes) for better results")
        elif consistency['avg_duration'] > 90:
            recommendations.append("🔄 Your workouts are quite long - ensure you're not overtraining")
        
        # Goal-specific recommendations
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            if weight_analysis['trend'] == 'stable':
                recommendations.append("🔥 Add 2-3 cardio sessions to boost calorie burn")
                recommendations.append("🥗 Review your nutrition plan - you may need a larger caloric deficit")
        
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            recommendations.append("💪 Focus on progressive overload - gradually increase weights")
            recommendations.append("🥩 Ensure you're eating enough protein (2g per kg body weight)")
        
        # Streak recommendations
        if consistency['current_streak'] == 0:
            recommendations.append("🚀 Start a new workout streak - even 3 days can build momentum!")
        elif consistency['current_streak'] < 7:
            recommendations.append(f"🔥 You're on a {consistency['current_streak']}-day streak! Keep it going!")
        
        return recommendations
    
    def _identify_achievements(self, performance: Dict, consistency: Dict, weight_analysis: Dict) -> List[str]:
        """Identify recent achievements and milestones."""
        
        achievements = []
        
        # Workout milestones
        if performance.get('total_workouts', 0) >= 100:
            achievements.append("🏆 Century Club - 100+ workouts completed!")
        elif performance.get('total_workouts', 0) >= 50:
            achievements.append("⭐ Half Century - 50+ workouts completed!")
        elif performance.get('total_workouts', 0) >= 10:
            achievements.append("🎯 Double Digits - 10+ workouts completed!")
        
        # Consistency achievements
        if consistency['current_streak'] >= 30:
            achievements.append("👑 Consistency King/Queen - 30+ day streak!")
        elif consistency['current_streak'] >= 14:
            achievements.append("🔥 Two Week Warrior - 14+ day streak!")
        elif consistency['current_streak'] >= 7:
            achievements.append("💪 Week Champion - 7+ day streak!")
        
        # Time achievements
        total_hours = performance.get('total_minutes', 0) / 60
        if total_hours >= 100:
            achievements.append("⏱️ Time Master - 100+ hours of exercise!")
        elif total_hours >= 50:
            achievements.append("🕐 Half Century Hours - 50+ hours!")
        elif total_hours >= 20:
            achievements.append("⏰ Time Commitment - 20+ hours!")
        
        # Weight loss achievements
        if weight_analysis.get('change', 0) <= -5:
            achievements.append("📉 Major Weight Loss - 5+ kg lost!")
        elif weight_analysis.get('change', 0) <= -2:
            achievements.append("📊 Weight Loss Progress - 2+ kg lost!")
        
        return achievements

    def export_data(self) -> Dict[str, Any]:
        """Export all tracking data for backup or analysis."""
        
        return {
            'body_measurements': [
                {
                    'date': m.date.isoformat(),
                    'weight': m.weight,
                    'body_fat_percentage': m.body_fat_percentage,
                    'muscle_mass_kg': m.muscle_mass_kg,
                    'waist_cm': m.waist_cm,
                    'chest_cm': m.chest_cm,
                    'bicep_cm': m.bicep_cm,
                    'thigh_cm': m.thigh_cm,
                    'notes': m.notes
                } for m in self.body_measurements
            ],
            'performance_metrics': [
                {
                    'date': m.date.isoformat(),
                    'exercise_name': m.exercise_name,
                    'metric_type': m.metric_type,
                    'value': m.value,
                    'unit': m.unit,
                    'notes': m.notes
                } for m in self.performance_metrics
            ],
            'workout_logs': [
                {
                    'date': w.date.isoformat(),
                    'workout_name': w.workout_name,
                    'duration_minutes': w.duration_minutes,
                    'exercises_completed': w.exercises_completed,
                    'total_volume': w.total_volume,
                    'calories_burned': w.calories_burned,
                    'perceived_exertion': w.perceived_exertion,
                    'mood_before': w.mood_before,
                    'mood_after': w.mood_after,
                    'notes': w.notes
                } for w in self.workout_logs
            ],
            'progress_photos': [
                {
                    'date': p.date.isoformat(),
                    'photo_path': p.photo_path,
                    'body_part': p.body_part,
                    'notes': p.notes
                } for p in self.progress_photos
            ]
        }
