from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
import json
import time
from werkzeug.utils import secure_filename
from ai_processor import PoseAnalyzer, FormAnalyzer, LLMFeedbackGenerator
from config import get_config, validate_config, print_config_summary

# Get configuration
config = get_config()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = config.database.uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.database.track_modifications
app.config['MAX_CONTENT_LENGTH'] = config.video.max_size_mb * 1024 * 1024
app.config['SECRET_KEY'] = config.security.secret_key

# Create upload directories
os.makedirs(config.video.upload_folder, exist_ok=True)
os.makedirs(config.video.pose_data_folder, exist_ok=True)

db = SQLAlchemy(app)
CORS(app, origins=config.security.cors_origins.split(',') if config.security.cors_origins != '*' else '*')

# Validate configuration
if not validate_config():
    print("❌ Configuration validation failed!")
    exit(1)

print_config_summary()

# Initialize AI components
pose_analyzer = PoseAnalyzer()
form_analyzer = FormAnalyzer()

# Initialize LLM generator based on provider
if config.ai.provider == "kolosal":
    llm_generator = LLMFeedbackGenerator(
        provider="kolosal",
        kolosal_url=config.ai.kolosal_url
    )
elif config.ai.provider == "openai":
    llm_generator = LLMFeedbackGenerator(
        api_key=config.ai.openai_key,
        provider="openai"
    )
elif config.ai.provider == "anthropic":
    llm_generator = LLMFeedbackGenerator(
        api_key=config.ai.anthropic_key,
        provider="anthropic"
    )
else:
    raise ValueError(f"Unsupported LLM provider: {config.ai.provider}")

print(f"🤖 Using LLM Provider: {config.ai.provider}")

# Import models (now from separate file)
from models import User, Exercise, Session, PoseLandmarks, Feedback, db

def init_db():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [f.strip('.') for f in config.video.supported_formats]

# ============================================================================
# VIDEO UPLOAD AND AI PROCESSING ROUTES
# ============================================================================

@app.route("/upload-video", methods=["POST"])
def upload_video():
    """Upload video file for AI analysis"""
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Use: {config.video.supported_formats}"}), 400
    
    # Get form data
    user_id = request.form.get('user_id', type=int)
    exercise_id = request.form.get('exercise_id', type=int)
    
    if not user_id or not exercise_id:
        return jsonify({"error": "user_id and exercise_id are required"}), 400
    
    # Verify user and exercise exist
    user = User.query.get(user_id)
    exercise = Exercise.query.get(exercise_id)
    if not user or not exercise:
        return jsonify({"error": "Invalid user_id or exercise_id"}), 404
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(config.video.upload_folder, unique_filename)
    
    try:
        file.save(video_path)
        file_size = os.path.getsize(video_path)
        
        # Create session record
        session = Session(
            user_id=user_id,
            exercise_id=exercise_id,
            video_path=video_path,
            video_filename=filename,
            video_size_bytes=file_size,
            status="uploaded"
        )
        
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            "message": "Video uploaded successfully",
            "session_id": session.id,
            "filename": unique_filename,
            "size_mb": round(file_size / (1024*1024), 2)
        }), 201
        
    except Exception as e:
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

@app.route("/process-ai/<int:session_id>", methods=["POST"])
def process_video_with_ai(session_id):
    """Process uploaded video with real AI analysis"""
    
    session = Session.query.get_or_404(session_id)
    
    if session.status == "processed":
        return jsonify({"error": "Session already processed"}), 400
    
    if not session.video_path or not os.path.exists(session.video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    start_time = time.time()
    
    try:
        # Update status
        session.status = "processing"
        db.session.commit()
        
        print(f"Starting AI processing for session {session_id}...")
        
        # Step 1: Extract poses from video
        print("Step 1: Extracting poses from video...")
        pose_data = pose_analyzer.extract_poses_from_video(session.video_path)
        
        # Update session with video info
        video_info = pose_data['video_info']
        session.fps = video_info['fps']
        session.total_frames = video_info['total_frames']
        session.duration_seconds = video_info['duration_seconds']
        
        # Step 2: Save pose landmarks to database
        print("Step 2: Saving pose landmarks...")
        pose_landmarks_saved = 0
        for pose in pose_data['poses']:
            landmark_record = PoseLandmarks(
                session_id=session.id,
                frame_number=pose['frame_number'],
                timestamp=pose['timestamp'],
                landmarks_json=json.dumps(pose['landmarks']),
                visibility_scores=json.dumps(pose['visibility_scores'])
            )
            db.session.add(landmark_record)
            pose_landmarks_saved += 1
        
        # Step 3: Analyze form based on exercise type
        print("Step 3: Analyzing exercise form...")
        exercise_name = session.exercise.name.lower()
        
        if "lat pull" in exercise_name or "pulldown" in exercise_name:
            analysis = form_analyzer.analyze_lat_pulldown(pose_data['poses'])
        elif "pull-up" in exercise_name or "pullup" in exercise_name:
            analysis = form_analyzer.analyze_pullup(pose_data['poses'])
        else:
            # Generic analysis for other exercises
            analysis = {
                'exercise_type': 'generic',
                'rep_count': len(pose_data['poses']) // 30 if pose_data['poses'] else 0,
                'form_issues': [],
                'metrics': {}
            }
        
        # Update rep count
        session.rep_count = analysis.get('rep_count', 0)
        
        # Step 4: Generate LLM feedback
        print("Step 4: Generating AI feedback...")
        llm_feedback = llm_generator.generate_feedback(
            exercise_name=session.exercise.name,
            analysis_data=analysis,
            user_name=session.user.name
        )
        
        # Step 5: Save feedback to database
        print("Step 5: Saving feedback...")
        feedback = Feedback(
            session_id=session.id,
            summary=llm_feedback['summary'],
            form_score=llm_feedback['form_score'],
            injury_risk_level=llm_feedback['injury_risk_level'],
            recommendations=llm_feedback['recommendations'],
            pose_analysis_data=json.dumps(analysis),
            llm_model_used=config.ai.provider,
            processing_time_ms=int((time.time() - start_time) * 1000),
            confidence_score=llm_feedback.get('confidence_score', 0.8)
        )
        
        # Mark session as processed
        session.status = "processed"
        
        # Save everything
        db.session.add(feedback)
        db.session.commit()
        
        processing_time = time.time() - start_time
        
        print(f"AI processing completed in {processing_time:.2f} seconds")
        
        return jsonify({
            "message": "AI processing completed successfully",
            "session_id": session.id,
            "feedback_id": feedback.id,
            "processing_time_seconds": round(processing_time, 2),
            "pose_landmarks_saved": pose_landmarks_saved,
            "rep_count": session.rep_count,
            "form_score": feedback.form_score,
            "injury_risk_level": feedback.injury_risk_level,
            "llm_generated": llm_feedback.get('llm_generated', True)
        }), 200
        
    except Exception as e:
        # Handle processing errors
        session.status = "error"
        session.processing_error = str(e)
        db.session.commit()
        
        print(f"AI processing error: {e}")
        
        return jsonify({
            "error": "AI processing failed",
            "details": str(e),
            "session_id": session.id
        }), 500

@app.route("/sessions/<int:session_id>/pose-data", methods=["GET"])
def get_pose_data(session_id):
    """Get pose landmarks for a session"""
    session = Session.query.get_or_404(session_id)
    
    landmarks = PoseLandmarks.query.filter_by(session_id=session_id).all()
    
    pose_data = []
    for landmark in landmarks:
        pose_data.append({
            'frame_number': landmark.frame_number,
            'timestamp': landmark.timestamp,
            'landmarks': landmark.get_landmarks(),
            'visibility_scores': landmark.get_visibility_scores()
        })
    
    return jsonify({
        'session_id': session_id,
        'total_landmarks': len(pose_data),
        'fps': session.fps,
        'duration_seconds': session.duration_seconds,
        'pose_data': pose_data
    })

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/')
def hello():
    return jsonify({
        "message": "APT API with Real AI Integration!", 
        "version": "2.0",
        "environment": config.environment,
        "ai_features": ["pose_estimation", "form_analysis", "llm_feedback"]
    })

# ============================================================================
# CRUD ROUTES
# ============================================================================

@app.route("/users", methods=["POST"])
def create_user():
    """Create a new user"""
    data = request.get_json()
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({"error": "Name and email are required"}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 409
    
    user = User(name=data["name"], email=data["email"])
    db.session.add(user)
    db.session.commit()
    return jsonify({"id": user.id, "message": "User created successfully"}), 201

@app.route("/users", methods=["GET"])
def list_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@app.route("/users/<int:user_id>", methods=["GET"])
def read_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@app.route("/exercises", methods=["POST"])
def create_exercise():
    """Create a new exercise"""
    data = request.get_json()
    if not data or not data.get('name') or not data.get('primary_muscle'):
        return jsonify({"error": "Name and primary_muscle are required"}), 400
    
    exercise = Exercise(
        name=data["name"], 
        primary_muscle=data["primary_muscle"],
        description=data.get("description", "")
    )
    db.session.add(exercise)
    db.session.commit()
    return jsonify({"id": exercise.id, "message": "Exercise created successfully"}), 201

@app.route("/exercises", methods=["GET"])
def list_exercises():
    """Get all exercises"""
    exercises = Exercise.query.all()
    return jsonify([exercise.to_dict() for exercise in exercises])

@app.route("/sessions", methods=["POST"])
def create_session():
    """Create a new workout session"""
    data = request.get_json()
    if not data or not data.get('user_id') or not data.get('exercise_id'):
        return jsonify({"error": "user_id and exercise_id are required"}), 400
    
    user = User.query.get(data['user_id'])
    exercise = Exercise.query.get(data['exercise_id'])
    if not user or not exercise:
        return jsonify({"error": "User or exercise not found"}), 404
    
    session = Session(
        user_id=data["user_id"], 
        exercise_id=data["exercise_id"],
        rep_count=data.get("rep_count"), 
        duration_seconds=data.get("duration_seconds")
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({"id": session.id, "message": "Session created successfully"}), 201

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """Get all sessions"""
    sessions = Session.query.all()
    return jsonify([session.to_dict() for session in sessions])

@app.route("/sessions/<int:session_id>", methods=["GET"])
def read_session(session_id):
    """Get a specific session"""
    session = Session.query.get_or_404(session_id)
    return jsonify(session.to_dict())

@app.route("/sessions/<int:session_id>/feedback", methods=["GET"])
def list_session_feedback(session_id):
    """Get all feedback for a specific session"""
    session = Session.query.get_or_404(session_id)
    return jsonify([feedback.to_dict() for feedback in session.feedback])

# Backward compatibility
@app.route("/process/<int:session_id>", methods=["POST"])
def process_video_simulation(session_id):
    """Backward compatibility - redirects to AI processing"""
    return process_video_with_ai(session_id)

# ============================================================================
# AI STATUS AND MONITORING
# ============================================================================

@app.route("/ai-status", methods=["GET"])
def ai_status():
    """Check AI system status"""
    
    status = {
        "pose_analyzer": "ready",
        "form_analyzer": "ready",
        "llm_generator": "unknown",
        "llm_provider": config.ai.provider,
        "environment": config.environment,
        "supported_exercises": ["lat_pulldown", "pullup", "generic"],
        "max_video_size_mb": config.video.max_size_mb,
        "supported_formats": config.video.supported_formats
    }
    
    # Add provider-specific status info
    if config.ai.provider == "kolosal":
        status["kolosal_url"] = config.ai.kolosal_url
        status["local_model"] = config.ai.kolosal_model
    
    # Test LLM connection
    try:
        if llm_generator.test_connection():
            status["llm_generator"] = "ready"
        else:
            status["llm_generator"] = "error - connection failed"
    except Exception as e:
        status["llm_generator"] = f"error - {str(e)}"
        if config.ai.provider == "kolosal":
            status["kolosal_error"] = "Make sure Kolosal.AI server is running with your Gemma model loaded"
    
    return jsonify(status)

if __name__ == '__main__':
    init_db()
    print("🤖 APT API with Real AI Integration Starting...")
    print("📹 Video upload endpoint: POST /upload-video")
    print("🧠 AI processing endpoint: POST /process-ai/<session_id>")
    print("📊 AI status check: GET /ai-status")
    app.run(debug=config.debug, host=config.host, port=config.port)