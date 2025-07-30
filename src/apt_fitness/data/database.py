"""
Database module for APT Fitness Assistant
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.models import (
    UserProfile, WorkoutSession, BodyMeasurement, 
    BodyCompositionAnalysis, BodyPartMeasurement, Goal
)
from ..core.config import config

logger = logging.getLogger(__name__)


class FitnessDatabase:
    """Centralized database management for fitness app."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        self.db_path = Path(db_path or config.database_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        profile_data TEXT,
                        preferences TEXT
                    )
                """)
                
                # Create workout sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workout_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        date TIMESTAMP,
                        exercises TEXT,
                        duration_minutes INTEGER,
                        calories_burned REAL,
                        notes TEXT,
                        mood_before INTEGER,
                        mood_after INTEGER,
                        difficulty_rating INTEGER,
                        completed BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Create body measurements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS body_measurements (
                        measurement_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        date TIMESTAMP,
                        weight_kg REAL,
                        body_fat_percentage REAL,
                        muscle_mass_kg REAL,
                        waist_cm REAL,
                        chest_cm REAL,
                        arms_cm REAL,
                        thighs_cm REAL,
                        neck_cm REAL,
                        hips_cm REAL,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Create exercise library table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS exercise_library (
                        exercise_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT,
                        muscle_groups TEXT,
                        equipment TEXT,
                        difficulty INTEGER,
                        calories_per_minute REAL,
                        instructions TEXT,
                        tips TEXT,
                        contraindications TEXT,
                        video_url TEXT,
                        image_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create user goals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_goals (
                        goal_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        goal_type TEXT,
                        target_value REAL,
                        current_value REAL,
                        target_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active',
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Create body composition analysis table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS body_composition_analysis (
                        analysis_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        image_path TEXT,
                        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        body_fat_percentage REAL,
                        muscle_mass_percentage REAL,
                        visceral_fat_level INTEGER,
                        bmr_estimated INTEGER,
                        body_shape_classification TEXT,
                        confidence_score REAL,
                        front_image_path TEXT,
                        side_image_path TEXT,
                        processed_image_path TEXT,
                        body_measurements_json TEXT,
                        composition_breakdown_json TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Create body part measurements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS body_part_measurements (
                        measurement_id TEXT PRIMARY KEY,
                        analysis_id TEXT,
                        body_part TEXT,
                        circumference_cm REAL,
                        area_percentage REAL,
                        muscle_definition_score REAL,
                        fat_distribution_score REAL,
                        symmetry_score REAL,
                        FOREIGN KEY (analysis_id) REFERENCES body_composition_analysis (analysis_id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any], 
                         preferences: Dict[str, Any] = None) -> bool:
        """Save or update user profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, profile_data, preferences, last_active)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(profile_data),
                    json.dumps(preferences or {}),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT profile_data, preferences, created_at, last_active
                    FROM users WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "user_id": user_id,
                        "profile_data": json.loads(row[0]) if row[0] else {},
                        "preferences": json.loads(row[1]) if row[1] else {},
                        "created_at": row[2],
                        "last_active": row[3]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def save_workout_session(self, workout: WorkoutSession) -> bool:
        """Save workout session to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO workout_sessions
                    (session_id, user_id, date, exercises, duration_minutes,
                     calories_burned, notes, mood_before, mood_after,
                     difficulty_rating, completed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workout.session_id,
                    workout.user_id,
                    workout.date.isoformat(),
                    json.dumps(workout.exercises),
                    workout.duration_minutes,
                    workout.calories_burned,
                    workout.notes,
                    workout.mood_before,
                    workout.mood_after,
                    workout.difficulty_rating,
                    workout.completed
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving workout session: {e}")
            return False
    
    def get_workout_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get workout history for user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                since_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor.execute("""
                    SELECT * FROM workout_sessions
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date DESC
                """, (user_id, since_date))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting workout history: {e}")
            return []
    
    def save_body_measurement(self, measurement: BodyMeasurement) -> bool:
        """Save body measurement to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO body_measurements
                    (measurement_id, user_id, date, weight_kg, body_fat_percentage,
                     muscle_mass_kg, waist_cm, chest_cm, arms_cm, thighs_cm,
                     neck_cm, hips_cm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.measurement_id,
                    measurement.user_id,
                    measurement.date.isoformat(),
                    measurement.weight_kg,
                    measurement.body_fat_percentage,
                    measurement.muscle_mass_kg,
                    measurement.waist_cm,
                    measurement.chest_cm,
                    measurement.arms_cm,
                    measurement.thighs_cm,
                    measurement.neck_cm,
                    measurement.hips_cm
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving body measurement: {e}")
            return False
    
    def get_body_measurements(self, user_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get body measurements history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                since_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor.execute("""
                    SELECT * FROM body_measurements
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date DESC
                """, (user_id, since_date))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting body measurements: {e}")
            return []
    
    def save_body_composition_analysis(self, analysis: BodyCompositionAnalysis) -> bool:
        """Save body composition analysis to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO body_composition_analysis
                    (analysis_id, user_id, image_path, analysis_date,
                     body_fat_percentage, muscle_mass_percentage, visceral_fat_level,
                     bmr_estimated, body_shape_classification, confidence_score,
                     front_image_path, side_image_path, processed_image_path,
                     body_measurements_json, composition_breakdown_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.analysis_id,
                    analysis.user_id,
                    analysis.image_path,
                    analysis.analysis_date.isoformat(),
                    analysis.body_fat_percentage,
                    analysis.muscle_mass_percentage,
                    analysis.visceral_fat_level,
                    analysis.bmr_estimated,
                    analysis.body_shape_classification,
                    analysis.confidence_score,
                    analysis.front_image_path,
                    analysis.side_image_path,
                    analysis.processed_image_path,
                    json.dumps(analysis.body_measurements),
                    json.dumps(analysis.composition_breakdown)
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving body composition analysis: {e}")
            return False
    
    def save_body_part_measurement(self, measurement: BodyPartMeasurement) -> bool:
        """Save body part measurement to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO body_part_measurements
                    (measurement_id, analysis_id, body_part, circumference_cm,
                     area_percentage, muscle_definition_score, fat_distribution_score,
                     symmetry_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.measurement_id,
                    measurement.analysis_id,
                    measurement.body_part,
                    measurement.circumference_cm,
                    measurement.area_percentage,
                    measurement.muscle_definition_score,
                    measurement.fat_distribution_score,
                    measurement.symmetry_score
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving body part measurement: {e}")
            return False
    
    def get_body_composition_history(self, user_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get body composition analysis history for user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                since_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor.execute("""
                    SELECT * FROM body_composition_analysis
                    WHERE user_id = ? AND analysis_date >= ?
                    ORDER BY analysis_date DESC
                """, (user_id, since_date))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting body composition history: {e}")
            return []
    
    def save_goal(self, goal: Goal) -> bool:
        """Save user goal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_goals
                    (goal_id, user_id, goal_type, target_value, current_value,
                     target_date, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal.goal_id,
                    goal.user_id,
                    goal.goal_type.value,
                    goal.target_value,
                    goal.current_value,
                    goal.target_date.isoformat() if goal.target_date else None,
                    goal.created_at.isoformat(),
                    goal.status
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving goal: {e}")
            return False
    
    def get_user_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user goals."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_goals
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting user goals: {e}")
            return []
    
    def get_analytics_summary(self, user_id: str) -> Dict[str, Any]:
        """Get analytics summary for dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get workout stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_workouts,
                        SUM(duration_minutes) as total_minutes,
                        SUM(calories_burned) as total_calories,
                        AVG(mood_after - mood_before) as avg_mood_improvement
                    FROM workout_sessions 
                    WHERE user_id = ? AND completed = 1
                """, (user_id,))
                
                workout_stats = cursor.fetchone()
                
                # Get recent workout dates for streak calculation
                cursor.execute("""
                    SELECT date FROM workout_sessions
                    WHERE user_id = ? AND completed = 1
                    ORDER BY date DESC
                    LIMIT 30
                """, (user_id,))
                
                workout_dates = [row[0] for row in cursor.fetchall()]
                current_streak = self._calculate_streak(workout_dates)
                
                return {
                    "total_workouts": workout_stats[0] or 0,
                    "total_minutes": workout_stats[1] or 0,
                    "total_calories": workout_stats[2] or 0.0,
                    "avg_mood_improvement": workout_stats[3] or 0.0,
                    "current_streak": current_streak
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def _calculate_streak(self, dates: List[str]) -> int:
        """Calculate current workout streak."""
        if not dates:
            return 0
            
        streak = 0
        today = datetime.now().date()
        
        for date_str in dates:
            workout_date = datetime.fromisoformat(date_str).date()
            days_diff = (today - workout_date).days
            
            if days_diff == streak:
                streak += 1
            elif days_diff == streak + 1:
                # Allow one day gap
                streak += 1
            else:
                break
                
        return streak


# Global database instance
_db_instance = None

def get_database() -> FitnessDatabase:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FitnessDatabase()
    return _db_instance
