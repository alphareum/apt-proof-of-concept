from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = db.relationship('Session', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'total_sessions': len(self.sessions)
        }

class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    primary_muscle = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = db.relationship('Session', backref='exercise', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'primary_muscle': self.primary_muscle,
            'description': self.description,
            'total_sessions': len(self.sessions)
        }

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey("exercise.id"), nullable=False)
    performed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Video metadata
    video_path = db.Column(db.String(200))
    video_filename = db.Column(db.String(200))
    video_size_bytes = db.Column(db.Integer)
    fps = db.Column(db.Float)
    total_frames = db.Column(db.Integer)
    duration_seconds = db.Column(db.Float)
    
    # Processing metadata
    pose_data_path = db.Column(db.String(200))
    processing_error = db.Column(db.Text)
    status = db.Column(db.String(20), default="pending")  # pending, processing, processed, error
    rep_count = db.Column(db.Integer)
    
    # Relationships
    feedback = db.relationship('Feedback', backref='session', lazy=True, cascade='all, delete-orphan')
    pose_landmarks = db.relationship('PoseLandmarks', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.name,
            'exercise_id': self.exercise_id,
            'exercise_name': self.exercise.name,
            'performed_at': self.performed_at.isoformat(),
            'status': self.status,
            'rep_count': self.rep_count,
            'duration_seconds': self.duration_seconds,
            'feedback_count': len(self.feedback)
        }

class PoseLandmarks(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.Float, nullable=False)
    landmarks_json = db.Column(db.Text, nullable=False)
    visibility_scores = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_landmarks(self):
        """Parse landmarks JSON"""
        try:
            return json.loads(self.landmarks_json)
        except:
            return []
    
    def get_visibility_scores(self):
        """Parse visibility scores JSON"""
        try:
            return json.loads(self.visibility_scores) if self.visibility_scores else []
        except:
            return []

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=False)
    
    # Core feedback
    summary = db.Column(db.Text, nullable=False)
    form_score = db.Column(db.Float)  # 0.0 - 1.0
    injury_risk_level = db.Column(db.String(20))  # low, medium, high
    recommendations = db.Column(db.Text)
    
    # AI metadata
    pose_analysis_data = db.Column(db.Text)  # JSON of analysis metrics
    llm_model_used = db.Column(db.String(50))
    processing_time_ms = db.Column(db.Integer)
    confidence_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'summary': self.summary,
            'form_score': self.form_score,
            'injury_risk_level': self.injury_risk_level,
            'recommendations': self.recommendations,
            'llm_model_used': self.llm_model_used,
            'created_at': self.created_at.isoformat()
        }
    
    def get_analysis_data(self):
        """Parse pose analysis JSON"""
        try:
            return json.loads(self.pose_analysis_data) if self.pose_analysis_data else {}
        except:
            return {}