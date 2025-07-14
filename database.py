"""
Database layer for AI Fitness Assistant
Handles data persistence, user profiles, and exercise history
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session data structure."""
    user_id: str
    created_at: datetime
    last_active: datetime
    profile_data: Dict[str, Any]
    preferences: Dict[str, Any]

@dataclass
class WorkoutSession:
    """Workout session data structure."""
    session_id: str
    user_id: str
    date: datetime
    exercises: List[Dict[str, Any]]
    duration_minutes: int
    calories_burned: float
    notes: Optional[str] = None
    mood_before: Optional[int] = None  # 1-10 scale
    mood_after: Optional[int] = None   # 1-10 scale
    difficulty_rating: Optional[int] = None  # 1-5 scale

@dataclass
class BodyMeasurement:
    """Body measurement data structure."""
    measurement_id: str
    user_id: str
    date: datetime
    weight: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    muscle_mass: Optional[float] = None
    waist: Optional[float] = None
    chest: Optional[float] = None
    arms: Optional[float] = None
    thighs: Optional[float] = None
    neck: Optional[float] = None
    hips: Optional[float] = None

@dataclass
class BodyCompositionAnalysis:
    """Body composition analysis from images."""
    analysis_id: str
    user_id: str
    image_path: str
    analysis_date: datetime
    body_fat_percentage: float
    muscle_mass_percentage: float
    visceral_fat_level: int
    bmr_estimated: int
    body_shape_classification: str
    confidence_score: float
    analysis_method: str = "computer_vision"
    front_image_path: Optional[str] = None
    side_image_path: Optional[str] = None
    processed_image_path: Optional[str] = None
    body_measurements: Optional[Dict[str, Any]] = None
    composition_breakdown: Optional[Dict[str, Any]] = None

@dataclass
class BodyPartMeasurement:
    """Individual body part measurements."""
    measurement_id: str
    analysis_id: str
    body_part: str
    circumference_cm: float
    area_percentage: float
    muscle_definition_score: float
    fat_distribution_score: float
    symmetry_score: float

class FitnessDatabase:
    """Centralized database management for fitness app."""
    
    def __init__(self, db_path: str = "fitness_app.db"):
        self.db_path = Path(db_path)
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
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Create body measurements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS body_measurements (
                        measurement_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        date TIMESTAMP,
                        weight REAL,
                        body_fat_percentage REAL,
                        muscle_mass REAL,
                        waist REAL,
                        chest REAL,
                        arms REAL,
                        thighs REAL,
                        neck REAL,
                        hips REAL,
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
                        difficulty TEXT,
                        calories_per_min REAL,
                        instructions TEXT,
                        video_url TEXT,
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
                        analysis_method TEXT,
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
                
                # Create progress tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS composition_progress (
                        progress_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        start_analysis_id TEXT,
                        end_analysis_id TEXT,
                        progress_type TEXT,
                        change_percentage REAL,
                        time_period_days INTEGER,
                        trend_direction TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        FOREIGN KEY (start_analysis_id) REFERENCES body_composition_analysis (analysis_id),
                        FOREIGN KEY (end_analysis_id) REFERENCES body_composition_analysis (analysis_id)
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
            if preferences is None:
                preferences = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, profile_data, preferences, last_active)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, json.dumps(profile_data), json.dumps(preferences)))
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
                    SELECT profile_data, preferences FROM users WHERE user_id = ?
                """, (user_id,))
                result = cursor.fetchone()
                
                if result:
                    profile_data = json.loads(result[0])
                    preferences = json.loads(result[1]) if result[1] else {}
                    return {
                        'profile': profile_data,
                        'preferences': preferences
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
                    INSERT INTO workout_sessions 
                    (session_id, user_id, date, exercises, duration_minutes, 
                     calories_burned, notes, mood_before, mood_after, difficulty_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workout.session_id, workout.user_id, workout.date,
                    json.dumps(workout.exercises), workout.duration_minutes,
                    workout.calories_burned, workout.notes,
                    workout.mood_before, workout.mood_after, workout.difficulty_rating
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving workout session: {e}")
            return False
    
    def get_workout_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get workout history for user."""
        try:
            start_date = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM workout_sessions 
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date DESC
                """, (user_id, start_date))
                
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
                    INSERT INTO body_measurements 
                    (measurement_id, user_id, date, weight, body_fat_percentage,
                     muscle_mass, waist, chest, arms, thighs, neck, hips)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.measurement_id, measurement.user_id, measurement.date,
                    measurement.weight, measurement.body_fat_percentage,
                    measurement.muscle_mass, measurement.waist, measurement.chest,
                    measurement.arms, measurement.thighs, measurement.neck, measurement.hips
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving body measurement: {e}")
            return False
    
    def get_measurement_history(self, user_id: str, days: int = 90) -> pd.DataFrame:
        """Get measurement history as DataFrame for analysis."""
        try:
            start_date = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM body_measurements 
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date
                """
                return pd.read_sql_query(query, conn, params=(user_id, start_date))
        except Exception as e:
            logger.error(f"Error getting measurement history: {e}")
            return pd.DataFrame()
    
    def add_exercise_to_library(self, exercise_data: Dict[str, Any]) -> bool:
        """Add custom exercise to library."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO exercise_library 
                    (exercise_id, name, category, muscle_groups, equipment,
                     difficulty, calories_per_min, instructions, video_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exercise_data.get('id', hashlib.md5(exercise_data['name'].encode()).hexdigest()),
                    exercise_data['name'], exercise_data.get('category'),
                    json.dumps(exercise_data.get('muscle_groups', [])),
                    exercise_data.get('equipment'), exercise_data.get('difficulty'),
                    exercise_data.get('calories_per_min'), exercise_data.get('instructions'),
                    exercise_data.get('video_url')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding exercise to library: {e}")
            return False
    
    def get_custom_exercises(self) -> List[Dict[str, Any]]:
        """Get all custom exercises from library."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM exercise_library")
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting custom exercises: {e}")
            return []
    
    def save_user_goal(self, user_id: str, goal_data: Dict[str, Any]) -> bool:
        """Save user fitness goal."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_goals 
                    (goal_id, user_id, goal_type, target_value, current_value, target_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    hashlib.md5(f"{user_id}_{goal_data['type']}_{datetime.now()}".encode()).hexdigest(),
                    user_id, goal_data['type'], goal_data.get('target_value'),
                    goal_data.get('current_value', 0), goal_data.get('target_date')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving user goal: {e}")
            return False
    
    def get_user_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active user goals."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_goals 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY created_at DESC
                """, (user_id,))
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting user goals: {e}")
            return []
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for backup/portability."""
        try:
            data = {
                'profile': self.get_user_profile(user_id),
                'workouts': self.get_workout_history(user_id, days=365),
                'measurements': self.get_measurement_history(user_id, days=365).to_dict('records'),
                'goals': self.get_user_goals(user_id),
                'export_date': datetime.now().isoformat()
            }
            return data
        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return {}
    
    def get_analytics_summary(self, user_id: str) -> Dict[str, Any]:
        """Get analytics summary for dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total workouts
                cursor.execute("""
                    SELECT COUNT(*) FROM workout_sessions WHERE user_id = ?
                """, (user_id,))
                total_workouts = cursor.fetchone()[0]
                
                # Total calories burned
                cursor.execute("""
                    SELECT SUM(calories_burned) FROM workout_sessions WHERE user_id = ?
                """, (user_id,))
                total_calories = cursor.fetchone()[0] or 0
                
                # Total workout time
                cursor.execute("""
                    SELECT SUM(duration_minutes) FROM workout_sessions WHERE user_id = ?
                """, (user_id,))
                total_minutes = cursor.fetchone()[0] or 0
                
                # Recent streak
                cursor.execute("""
                    SELECT date FROM workout_sessions 
                    WHERE user_id = ? 
                    ORDER BY date DESC 
                    LIMIT 30
                """, (user_id,))
                recent_dates = [row[0] for row in cursor.fetchall()]
                
                return {
                    'total_workouts': total_workouts,
                    'total_calories': total_calories,
                    'total_hours': round(total_minutes / 60, 1),
                    'recent_streak': self._calculate_streak(recent_dates),
                    'avg_workout_duration': round(total_minutes / total_workouts, 1) if total_workouts > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def _calculate_streak(self, dates: List[str]) -> int:
        """Calculate current workout streak."""
        if not dates:
            return 0
        
        streak = 0
        current_date = datetime.now().date()
        
        for date_str in dates:
            workout_date = datetime.fromisoformat(date_str).date()
            if (current_date - workout_date).days <= streak + 1:
                streak += 1
                current_date = workout_date
            else:
                break
        
        return streak
    
    def save_body_composition_analysis(self, analysis: BodyCompositionAnalysis) -> bool:
        """Save body composition analysis to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO body_composition_analysis 
                    (analysis_id, user_id, image_path, analysis_date, body_fat_percentage,
                     muscle_mass_percentage, visceral_fat_level, bmr_estimated, 
                     body_shape_classification, confidence_score, analysis_method,
                     front_image_path, side_image_path, processed_image_path,
                     body_measurements_json, composition_breakdown_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.analysis_id, analysis.user_id, analysis.image_path,
                    analysis.analysis_date, analysis.body_fat_percentage,
                    analysis.muscle_mass_percentage, analysis.visceral_fat_level,
                    analysis.bmr_estimated, analysis.body_shape_classification,
                    analysis.confidence_score, analysis.analysis_method,
                    analysis.front_image_path, analysis.side_image_path,
                    analysis.processed_image_path,
                    json.dumps(analysis.body_measurements) if analysis.body_measurements else None,
                    json.dumps(analysis.composition_breakdown) if analysis.composition_breakdown else None
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
                    INSERT INTO body_part_measurements 
                    (measurement_id, analysis_id, body_part, circumference_cm,
                     area_percentage, muscle_definition_score, fat_distribution_score, symmetry_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.measurement_id, measurement.analysis_id, measurement.body_part,
                    measurement.circumference_cm, measurement.area_percentage,
                    measurement.muscle_definition_score, measurement.fat_distribution_score,
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
            start_date = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM body_composition_analysis 
                    WHERE user_id = ? AND analysis_date >= ?
                    ORDER BY analysis_date DESC
                """, (user_id, start_date))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting body composition history: {e}")
            return []
    
    def get_latest_body_composition(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get latest body composition analysis for user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM body_composition_analysis 
                    WHERE user_id = ? 
                    ORDER BY analysis_date DESC 
                    LIMIT 1
                """, (user_id,))
                
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
        except Exception as e:
            logger.error(f"Error getting latest body composition: {e}")
            return None
    
    def calculate_composition_progress(self, user_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Calculate body composition progress over time."""
        try:
            analyses = self.get_body_composition_history(user_id, period_days)
            if len(analyses) < 2:
                return {"error": "Insufficient data for progress calculation"}
            
            latest = analyses[0]
            earliest = analyses[-1]
            
            progress = {
                "period_days": period_days,
                "body_fat_change": latest["body_fat_percentage"] - earliest["body_fat_percentage"],
                "muscle_mass_change": latest["muscle_mass_percentage"] - earliest["muscle_mass_percentage"],
                "visceral_fat_change": latest["visceral_fat_level"] - earliest["visceral_fat_level"],
                "bmr_change": latest["bmr_estimated"] - earliest["bmr_estimated"],
                "trend_analysis": self._analyze_composition_trend(analyses)
            }
            
            return progress
        except Exception as e:
            logger.error(f"Error calculating composition progress: {e}")
            return {}
    
    def _analyze_composition_trend(self, analyses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trends in body composition data."""
        if len(analyses) < 3:
            return {"overall": "insufficient_data"}
        
        body_fat_values = [a["body_fat_percentage"] for a in reversed(analyses)]
        muscle_mass_values = [a["muscle_mass_percentage"] for a in reversed(analyses)]
        
        # Simple trend analysis
        body_fat_trend = "stable"
        muscle_trend = "stable"
        
        if len(body_fat_values) >= 3:
            recent_avg = sum(body_fat_values[-3:]) / 3
            older_avg = sum(body_fat_values[:3]) / 3
            
            if recent_avg < older_avg - 0.5:
                body_fat_trend = "decreasing"
            elif recent_avg > older_avg + 0.5:
                body_fat_trend = "increasing"
        
        if len(muscle_mass_values) >= 3:
            recent_avg = sum(muscle_mass_values[-3:]) / 3
            older_avg = sum(muscle_mass_values[:3]) / 3
            
            if recent_avg > older_avg + 0.5:
                muscle_trend = "increasing"
            elif recent_avg < older_avg - 0.5:
                muscle_trend = "decreasing"
        
        return {
            "body_fat": body_fat_trend,
            "muscle_mass": muscle_trend,
            "overall": "improving" if body_fat_trend == "decreasing" or muscle_trend == "increasing" else "stable"
        }
        
# Singleton instance
_db_instance = None

def get_database() -> FitnessDatabase:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FitnessDatabase()
    return _db_instance
