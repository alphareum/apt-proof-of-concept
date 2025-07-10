# AI-Driven Personal Trainer (APT) - Proof of Concept

A REST API proof of concept for an AI-powered personal trainer that analyzes exercise form using computer vision and provides real-time feedback.

## 🎯 Overview

This proof of concept demonstrates the backend architecture for a mobile fitness app that:
- Records user exercise videos
- Analyzes form using AI (pose estimation + LLM)  
- Provides personalized feedback and injury risk assessment
- Tracks workout sessions and progress

## 🏗️ Architecture

- **Backend**: Flask + SQLAlchemy REST API
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **AI Pipeline**: Simulated (ready for MovePose-M + GPT-4o integration)
- **Deployment**: Docker-ready

## 📊 Database Schema

- **Users**: Basic user profiles
- **Exercises**: Exercise definitions (lat pull-down, pull-ups, rows)
- **Sessions**: Individual workout records with video paths
- **Feedback**: AI-generated form analysis and recommendations

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install flask flask_sqlalchemy flask-cors
```

### 2. Run the API
```bash
python app.py
```
Server runs on `http://localhost:5000`

### 3. Test the API

**Create a user:**
```bash
curl -X POST http://localhost:5000/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice Chen","email":"alice@example.com"}'
```

**Create an exercise:**
```bash
curl -X POST http://localhost:5000/exercises \
  -H "Content-Type: application/json" \
  -d '{"name":"Lat Pull-Down", "primary_muscle":"Lats"}'
```

**Create a workout session:**
```bash
curl -X POST http://localhost:5000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":1, "exercise_id":1, "rep_count":12, "duration_seconds":45.5}'
```

**Trigger AI analysis:**
```bash
curl -X POST http://localhost:5000/process/1
```

## 📱 API Endpoints

### Users
- `POST /users` - Create user
- `GET /users` - List all users  
- `GET /users/{id}` - Get user details
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

### Exercises  
- `POST /exercises` - Create exercise
- `GET /exercises` - List all exercises
- `GET /exercises/{id}` - Get exercise details
- `PUT /exercises/{id}` - Update exercise
- `DELETE /exercises/{id}` - Delete exercise

### Sessions
- `POST /sessions` - Create workout session
- `GET /sessions` - List all sessions
- `GET /sessions/{id}` - Get session details  
- `PUT /sessions/{id}` - Update session
- `DELETE /sessions/{id}` - Delete session

### AI Processing
- `POST /process/{session_id}` - Trigger AI video analysis
- `GET /sessions/{session_id}/process-status` - Check processing status

### Feedback
- `GET /sessions/{session_id}/feedback` - Get session feedback
- `GET /feedback/{id}` - Get specific feedback
- `PUT /feedback/{id}` - Update feedback
- `DELETE /feedback/{id}` - Delete feedback

## 🤖 AI Simulation

Currently simulates different feedback based on exercise type:
- **Lat Pull-Down**: Analyzes back posture, range of motion
- **Pull-Up**: Checks full extension, momentum
- **Seated Row**: Evaluates posture, head position

Ready to integrate with:
- MovePose-M for pose estimation
- GPT-4o-Micro for feedback generation

## 🔜 Production Roadmap

- [ ] Replace SQLite with PostgreSQL
- [ ] Add JWT authentication
- [ ] Integrate real pose estimation (MovePose-M)
- [ ] Add LLM feedback generation (GPT-4o)
- [ ] Implement video upload to S3
- [ ] Add user dashboard and analytics
- [ ] Mobile app development (React Native)

## 📈 Technical Specifications

**Target Performance:**
- Form analysis: <1 second latency
- Rep counting: 98%+ accuracy
- Form fault detection: F1 ≥ 0.85

**Privacy:**
- Raw video never leaves device
- Only pose keypoints sent to cloud
- GDPR + PDPL compliant

## 🐳 Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install flask flask_sqlalchemy flask-cors
CMD ["python", "app.py"]
```

```bash
docker build -t apt-api .
docker run -p 5000:5000 apt-api
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a proof of concept. For the full production system:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Built for the future of fitness coaching** 🏋️‍♀️💪