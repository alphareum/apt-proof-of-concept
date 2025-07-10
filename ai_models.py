# Add these new columns to your existing models in app.py

# Extended Session model (add these fields to your existing Session class)
class Session(db.Model):
    # ... existing fields ...
    
    # New AI-specific fields
    video_filename = db.Column(db.String(200))           # Original filename
    video_size_bytes = db.Column(db.Integer)             # File size
    fps = db.Column(db.Float)                            # Frames per second
    total_frames = db.Column(db.Integer)                 # Total frame count
    pose_data_path = db.Column(db.String(200))           # Path to extracted pose data
    processing_error = db.Column(db.Text)                # Error messages if processing fails

# New model for storing pose landmarks
class PoseLandmarks(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.Float, nullable=False)      # Time in seconds
    landmarks_json = db.Column(db.Text, nullable=False)  # JSON of pose landmarks
    visibility_scores = db.Column(db.Text)               # Visibility confidence scores
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Enhanced Feedback model (add these fields to existing Feedback class)
class Feedback(db.Model):
    # ... existing fields ...
    
    # New AI-specific fields
    pose_analysis_data = db.Column(db.Text)              # Detailed pose metrics JSON
    llm_model_used = db.Column(db.String(50))            # Which LLM generated feedback
    processing_time_ms = db.Column(db.Integer)           # Total processing time
    confidence_score = db.Column(db.Float)               # AI confidence in analysis