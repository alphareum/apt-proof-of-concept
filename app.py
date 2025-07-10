from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///apt.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
CORS(app)  # Enable CORS for mobile app integration

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to sessions
    sessions = db.relationship('Session', backref='user', lazy=True)

class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)              # "Lat Pull-Down"
    primary_muscle = db.Column(db.String(80), nullable=False)    # "Lats"
    description = db.Column(db.Text)                             # Optional exercise description
    
    # Relationship to sessions
    sessions = db.relationship('Session', backref='exercise', lazy=True)

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey("exercise.id"), nullable=False)
    performed_at = db.Column(db.DateTime, default=datetime.utcnow)
    video_path = db.Column(db.String(200))                       # local or S3 URL
    status = db.Column(db.String(20), default="pending")         # "pending", "processing", "processed"
    rep_count = db.Column(db.Integer)                            # Detected rep count
    duration_seconds = db.Column(db.Float)                       # Session duration
    
    # Relationship to feedback
    feedback = db.relationship('Feedback', backref='session', lazy=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=False)
    summary = db.Column(db.Text, nullable=False)                 # "Back is arching; drop load 10%"
    form_score = db.Column(db.Float)                             # 0.0 - 1.0 form quality score
    injury_risk_level = db.Column(db.String(20))                 # "low", "medium", "high"
    recommendations = db.Column(db.Text)                         # Specific improvement suggestions
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database
def init_db():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

# Basic route to test the server
@app.route('/')
def hello():
    return jsonify({"message": "APT API is running!", "version": "0.1"})

# ============================================================================
# USER CRUD ROUTES
# ============================================================================

@app.route("/users", methods=["POST"])
def create_user():
    """Create a new user"""
    data = request.get_json()
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({"error": "Name and email are required"}), 400
    
    # Check if email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 409
    
    user = User(name=data["name"], email=data["email"])
    db.session.add(user)
    db.session.commit()
    return jsonify({"id": user.id, "message": "User created successfully"}), 201

@app.route("/users/<int:user_id>", methods=["GET"])
def read_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
        "total_sessions": len(user.sessions)
    })

@app.route("/users", methods=["GET"])
def list_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([{
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "created_at": u.created_at.isoformat(),
        "total_sessions": len(u.sessions)
    } for u in users])

@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    """Update a user"""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if data.get("name"):
        user.name = data["name"]
    if data.get("email"):
        # Check if new email already exists for another user
        existing = User.query.filter_by(email=data["email"]).first()
        if existing and existing.id != user_id:
            return jsonify({"error": "Email already exists"}), 409
        user.email = data["email"]
    
    db.session.commit()
    return jsonify({"message": "User updated successfully"})

@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted successfully"})

# ============================================================================
# EXERCISE CRUD ROUTES
# ============================================================================

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

@app.route("/exercises/<int:exercise_id>", methods=["GET"])
def read_exercise(exercise_id):
    """Get a specific exercise"""
    exercise = Exercise.query.get_or_404(exercise_id)
    return jsonify({
        "id": exercise.id,
        "name": exercise.name,
        "primary_muscle": exercise.primary_muscle,
        "description": exercise.description,
        "total_sessions": len(exercise.sessions)
    })

@app.route("/exercises", methods=["GET"])
def list_exercises():
    """Get all exercises"""
    exercises = Exercise.query.all()
    return jsonify([{
        "id": e.id,
        "name": e.name,
        "primary_muscle": e.primary_muscle,
        "description": e.description,
        "total_sessions": len(e.sessions)
    } for e in exercises])

@app.route("/exercises/<int:exercise_id>", methods=["PUT"])
def update_exercise(exercise_id):
    """Update an exercise"""
    exercise = Exercise.query.get_or_404(exercise_id)
    data = request.get_json()
    
    if data.get("name"):
        exercise.name = data["name"]
    if data.get("primary_muscle"):
        exercise.primary_muscle = data["primary_muscle"]
    if "description" in data:
        exercise.description = data["description"]
    
    db.session.commit()
    return jsonify({"message": "Exercise updated successfully"})

@app.route("/exercises/<int:exercise_id>", methods=["DELETE"])
def delete_exercise(exercise_id):
    """Delete an exercise"""
    exercise = Exercise.query.get_or_404(exercise_id)
    db.session.delete(exercise)
    db.session.commit()
    return jsonify({"message": "Exercise deleted successfully"})

# ============================================================================
# SESSION CRUD ROUTES
# ============================================================================

@app.route("/sessions", methods=["POST"])
def create_session():
    """Create a new workout session"""
    data = request.get_json()
    if not data or not data.get('user_id') or not data.get('exercise_id'):
        return jsonify({"error": "user_id and exercise_id are required"}), 400
    
    # Verify user and exercise exist
    user = User.query.get(data['user_id'])
    exercise = Exercise.query.get(data['exercise_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    if not exercise:
        return jsonify({"error": "Exercise not found"}), 404
    
    session = Session(
        user_id=data["user_id"],
        exercise_id=data["exercise_id"],
        video_path=data.get("video_path", ""),
        status=data.get("status", "pending"),
        rep_count=data.get("rep_count"),
        duration_seconds=data.get("duration_seconds")
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({"id": session.id, "message": "Session created successfully"}), 201

@app.route("/sessions/<int:session_id>", methods=["GET"])
def read_session(session_id):
    """Get a specific session"""
    session = Session.query.get_or_404(session_id)
    return jsonify({
        "id": session.id,
        "user_id": session.user_id,
        "user_name": session.user.name,
        "exercise_id": session.exercise_id,
        "exercise_name": session.exercise.name,
        "performed_at": session.performed_at.isoformat(),
        "video_path": session.video_path,
        "status": session.status,
        "rep_count": session.rep_count,
        "duration_seconds": session.duration_seconds,
        "feedback_count": len(session.feedback)
    })

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """Get all sessions"""
    sessions = Session.query.all()
    return jsonify([{
        "id": s.id,
        "user_name": s.user.name,
        "exercise_name": s.exercise.name,
        "performed_at": s.performed_at.isoformat(),
        "status": s.status,
        "rep_count": s.rep_count,
        "duration_seconds": s.duration_seconds
    } for s in sessions])

@app.route("/sessions/<int:session_id>", methods=["PUT"])
def update_session(session_id):
    """Update a session"""
    session = Session.query.get_or_404(session_id)
    data = request.get_json()
    
    if data.get("video_path"):
        session.video_path = data["video_path"]
    if data.get("status"):
        session.status = data["status"]
    if "rep_count" in data:
        session.rep_count = data["rep_count"]
    if "duration_seconds" in data:
        session.duration_seconds = data["duration_seconds"]
    
    db.session.commit()
    return jsonify({"message": "Session updated successfully"})

@app.route("/sessions/<int:session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session"""
    session = Session.query.get_or_404(session_id)
    db.session.delete(session)
    db.session.commit()
    return jsonify({"message": "Session deleted successfully"})

# ============================================================================
# AI PROCESSING PIPELINE (SIMULATION)
# ============================================================================

@app.route("/process/<int:session_id>", methods=["POST"])
def process_video(session_id):
    """
    Simulate AI video processing pipeline
    This is where your pose estimation + LLM would run
    """
    session = Session.query.get_or_404(session_id)
    
    # Prevent reprocessing
    if session.status == "processed":
        return jsonify({"error": "Session already processed"}), 400
    
    # Update session status to processing
    session.status = "processing"
    db.session.commit()
    
    # Simulate different feedback based on exercise type
    exercise_name = session.exercise.name.lower()
    
    if "lat pull" in exercise_name or "pulldown" in exercise_name:
        feedback_data = {
            "summary": "Lat pull-down analysis: Good range of motion detected. Minor form issue: back arching observed in reps 8-12. Recommend reducing weight by 10-15%.",
            "form_score": 0.72,
            "injury_risk_level": "medium", 
            "recommendations": "1. Maintain neutral spine throughout movement. 2. Engage core before initiating pull. 3. Control eccentric phase. 4. Consider lighter weight to master form."
        }
    elif "pull-up" in exercise_name or "pullup" in exercise_name:
        feedback_data = {
            "summary": "Pull-up analysis: Strong concentric phase. Incomplete range of motion detected - not reaching full extension at bottom.",
            "form_score": 0.68,
            "injury_risk_level": "low",
            "recommendations": "1. Complete full range of motion - arms fully extended at bottom. 2. Control descent speed. 3. Avoid swinging momentum."
        }
    elif "row" in exercise_name:
        feedback_data = {
            "summary": "Seated row analysis: Good posture maintenance. Slight forward head posture detected. Excellent rep consistency.",
            "form_score": 0.83,
            "injury_risk_level": "low",
            "recommendations": "1. Keep chin tucked, eyes forward. 2. Squeeze shoulder blades at peak contraction. 3. Maintain current weight."
        }
    else:
        # Generic feedback for other exercises
        feedback_data = {
            "summary": f"{session.exercise.name} analysis: Movement pattern analyzed. Form assessment completed.",
            "form_score": 0.75,
            "injury_risk_level": "low",
            "recommendations": "Continue with current technique. Focus on controlled movement patterns."
        }
    
    # Add rep count simulation based on duration (rough estimate)
    if not session.rep_count and session.duration_seconds:
        estimated_reps = max(1, int(session.duration_seconds / 3.5))  # ~3.5 seconds per rep
        session.rep_count = estimated_reps
    
    # Create AI-generated feedback
    feedback = Feedback(
        session_id=session.id,
        summary=feedback_data["summary"],
        form_score=feedback_data["form_score"],
        injury_risk_level=feedback_data["injury_risk_level"],
        recommendations=feedback_data["recommendations"]
    )
    
    # Mark session as processed
    session.status = "processed"
    
    # Save everything
    db.session.add(feedback)
    db.session.commit()
    
    return jsonify({
        "message": "Video processing completed",
        "feedback_id": feedback.id,
        "session_id": session.id,
        "form_score": feedback.form_score,
        "injury_risk_level": feedback.injury_risk_level,
        "processing_time_ms": 850  # Simulate processing time
    }), 200

@app.route("/sessions/<int:session_id>/process-status", methods=["GET"])
def get_processing_status(session_id):
    """Check processing status of a session"""
    session = Session.query.get_or_404(session_id)
    
    response_data = {
        "session_id": session.id,
        "status": session.status,
        "user_name": session.user.name,
        "exercise_name": session.exercise.name,
        "performed_at": session.performed_at.isoformat()
    }
    
    if session.status == "processed" and session.feedback:
        latest_feedback = session.feedback[-1]  # Get most recent feedback
        response_data.update({
            "feedback_id": latest_feedback.id,
            "form_score": latest_feedback.form_score,
            "injury_risk_level": latest_feedback.injury_risk_level,
            "summary": latest_feedback.summary
        })
    
    return jsonify(response_data)

# ============================================================================
# FEEDBACK CRUD ROUTES
# ============================================================================

@app.route("/feedback", methods=["POST"])
def create_feedback():
    """Create feedback for a session"""
    data = request.get_json()
    if not data or not data.get('session_id') or not data.get('summary'):
        return jsonify({"error": "session_id and summary are required"}), 400
    
    # Verify session exists
    session = Session.query.get(data['session_id'])
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    feedback = Feedback(
        session_id=data["session_id"],
        summary=data["summary"],
        form_score=data.get("form_score"),
        injury_risk_level=data.get("injury_risk_level", "low"),
        recommendations=data.get("recommendations", "")
    )
    db.session.add(feedback)
    db.session.commit()
    return jsonify({"id": feedback.id, "message": "Feedback created successfully"}), 201

@app.route("/feedback/<int:feedback_id>", methods=["GET"])
def read_feedback(feedback_id):
    """Get specific feedback"""
    feedback = Feedback.query.get_or_404(feedback_id)
    return jsonify({
        "id": feedback.id,
        "session_id": feedback.session_id,
        "user_name": feedback.session.user.name,
        "exercise_name": feedback.session.exercise.name,
        "summary": feedback.summary,
        "form_score": feedback.form_score,
        "injury_risk_level": feedback.injury_risk_level,
        "recommendations": feedback.recommendations,
        "created_at": feedback.created_at.isoformat()
    })

@app.route("/sessions/<int:session_id>/feedback", methods=["GET"])
def list_session_feedback(session_id):
    """Get all feedback for a specific session"""
    session = Session.query.get_or_404(session_id)
    return jsonify([{
        "id": f.id,
        "summary": f.summary,
        "form_score": f.form_score,
        "injury_risk_level": f.injury_risk_level,
        "recommendations": f.recommendations,
        "created_at": f.created_at.isoformat()
    } for f in session.feedback])

@app.route("/feedback/<int:feedback_id>", methods=["PUT"])
def update_feedback(feedback_id):
    """Update feedback"""
    feedback = Feedback.query.get_or_404(feedback_id)
    data = request.get_json()
    
    if data.get("summary"):
        feedback.summary = data["summary"]
    if "form_score" in data:
        feedback.form_score = data["form_score"]
    if data.get("injury_risk_level"):
        feedback.injury_risk_level = data["injury_risk_level"]
    if "recommendations" in data:
        feedback.recommendations = data["recommendations"]
    
    db.session.commit()
    return jsonify({"message": "Feedback updated successfully"})

@app.route("/feedback/<int:feedback_id>", methods=["DELETE"])
def delete_feedback(feedback_id):
    """Delete feedback"""
    feedback = Feedback.query.get_or_404(feedback_id)
    db.session.delete(feedback)
    db.session.commit()
    return jsonify({"message": "Feedback deleted successfully"})

if __name__ == '__main__':
    init_db()  # Create tables when running directly
    app.run(debug=True, host='0.0.0.0', port=5000)