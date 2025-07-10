import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="APT - AI Personal Trainer Dashboard",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(method, endpoint, data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code in [200, 201]:
            return response.json(), True
        else:
            return f"Error {response.status_code}: {response.text}", False
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure Flask server is running on localhost:5000", False
    except Exception as e:
        return f"Error: {str(e)}", False

def main():
    # Header
    st.markdown('<div class="main-header">🏋️‍♀️ AI Personal Trainer Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard",
        "👥 User Management", 
        "💪 Exercise Management",
        "🎯 Workout Sessions",
        "🤖 AI Analysis",
        "📝 Feedback Review"
    ])
    
    # Check API connection
    api_status, api_ok = make_api_request("GET", "/")
    if not api_ok:
        st.error(api_status)
        st.stop()
    else:
        st.sidebar.success("✅ API Connected")
    
    # Route to different pages
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "👥 User Management":
        show_user_management()
    elif page == "💪 Exercise Management":
        show_exercise_management()
    elif page == "🎯 Workout Sessions":
        show_session_management()
    elif page == "🤖 AI Analysis":
        show_ai_analysis()
    elif page == "📝 Feedback Review":
        show_feedback_review()

def show_dashboard():
    st.header("📊 Dashboard Overview")
    
    # Get data for dashboard
    users_data, _ = make_api_request("GET", "/users")
    exercises_data, _ = make_api_request("GET", "/exercises")
    sessions_data, _ = make_api_request("GET", "/sessions")
    
    if isinstance(users_data, list) and isinstance(exercises_data, list) and isinstance(sessions_data, list):
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Users", len(users_data))
        with col2:
            st.metric("💪 Total Exercises", len(exercises_data))
        with col3:
            st.metric("🎯 Total Sessions", len(sessions_data))
        with col4:
            processed_sessions = len([s for s in sessions_data if s.get('status') == 'processed'])
            st.metric("✅ Processed Sessions", processed_sessions)
        
        st.divider()
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Users")
            if users_data:
                users_df = pd.DataFrame(users_data)
                st.dataframe(users_df[['name', 'email', 'total_sessions']], use_container_width=True)
            else:
                st.info("No users found")
        
        with col2:
            st.subheader("Recent Sessions")
            if sessions_data:
                sessions_df = pd.DataFrame(sessions_data)
                st.dataframe(sessions_df[['user_name', 'exercise_name', 'status', 'rep_count']], use_container_width=True)
            else:
                st.info("No sessions found")

def show_user_management():
    st.header("👥 User Management")
    
    tab1, tab2 = st.tabs(["Create User", "View Users"])
    
    with tab1:
        st.subheader("Create New User")
        
        with st.form("create_user_form"):
            name = st.text_input("Full Name", placeholder="Enter user's full name")
            email = st.text_input("Email", placeholder="user@example.com")
            
            submitted = st.form_submit_button("Create User", type="primary")
            
            if submitted:
                if name and email:
                    data = {"name": name, "email": email}
                    result, success = make_api_request("POST", "/users", data)
                    
                    if success:
                        st.success(f"✅ User '{name}' created successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    with tab2:
        st.subheader("All Users")
        
        users_data, success = make_api_request("GET", "/users")
        
        if success and isinstance(users_data, list):
            if users_data:
                users_df = pd.DataFrame(users_data)
                users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users found. Create your first user!")
        else:
            st.error("Failed to load users")

def show_exercise_management():
    st.header("💪 Exercise Management")
    
    tab1, tab2 = st.tabs(["Create Exercise", "View Exercises"])
    
    with tab1:
        st.subheader("Add New Exercise")
        
        with st.form("create_exercise_form"):
            name = st.text_input("Exercise Name", placeholder="e.g., Lat Pull-Down")
            primary_muscle = st.selectbox("Primary Muscle Group", [
                "Lats", "Chest", "Shoulders", "Biceps", "Triceps", 
                "Back", "Legs", "Core", "Glutes", "Hamstrings", "Quadriceps"
            ])
            description = st.text_area("Description (optional)", placeholder="Brief description of the exercise")
            
            submitted = st.form_submit_button("Add Exercise", type="primary")
            
            if submitted:
                if name and primary_muscle:
                    data = {
                        "name": name, 
                        "primary_muscle": primary_muscle,
                        "description": description
                    }
                    result, success = make_api_request("POST", "/exercises", data)
                    
                    if success:
                        st.success(f"✅ Exercise '{name}' added successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Please fill in required fields")
    
    with tab2:
        st.subheader("All Exercises")
        
        exercises_data, success = make_api_request("GET", "/exercises")
        
        if success and isinstance(exercises_data, list):
            if exercises_data:
                exercises_df = pd.DataFrame(exercises_data)
                st.dataframe(exercises_df, use_container_width=True)
            else:
                st.info("No exercises found. Add your first exercise!")
        else:
            st.error("Failed to load exercises")

def show_session_management():
    st.header("🎯 Workout Session Management")
    
    tab1, tab2 = st.tabs(["Create Session", "View Sessions"])
    
    with tab1:
        st.subheader("Record New Workout Session")
        
        # Get users and exercises for dropdowns
        users_data, _ = make_api_request("GET", "/users")
        exercises_data, _ = make_api_request("GET", "/exercises")
        
        if isinstance(users_data, list) and isinstance(exercises_data, list) and users_data and exercises_data:
            with st.form("create_session_form"):
                user_options = {f"{u['name']} ({u['email']})": u['id'] for u in users_data}
                exercise_options = {f"{e['name']} - {e['primary_muscle']}": e['id'] for e in exercises_data}
                
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                selected_exercise = st.selectbox("Select Exercise", list(exercise_options.keys()))
                
                col1, col2 = st.columns(2)
                with col1:
                    rep_count = st.number_input("Rep Count", min_value=1, max_value=100, value=10)
                with col2:
                    duration = st.number_input("Duration (seconds)", min_value=1.0, max_value=300.0, value=30.0, step=0.5)
                
                video_path = st.text_input("Video Path (optional)", placeholder="videos/session_001.mp4")
                
                submitted = st.form_submit_button("Record Session", type="primary")
                
                if submitted:
                    data = {
                        "user_id": user_options[selected_user],
                        "exercise_id": exercise_options[selected_exercise],
                        "rep_count": rep_count,
                        "duration_seconds": duration,
                        "video_path": video_path if video_path else f"videos/session_{int(time.time())}.mp4"
                    }
                    
                    result, success = make_api_request("POST", "/sessions", data)
                    
                    if success:
                        st.success(f"✅ Workout session recorded successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
        else:
            st.warning("⚠️ You need to create at least one user and one exercise first!")
    
    with tab2:
        st.subheader("All Workout Sessions")
        
        sessions_data, success = make_api_request("GET", "/sessions")
        
        if success and isinstance(sessions_data, list):
            if sessions_data:
                sessions_df = pd.DataFrame(sessions_data)
                sessions_df['performed_at'] = pd.to_datetime(sessions_df['performed_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Add status indicators
                status_map = {'pending': '⏳', 'processing': '🔄', 'processed': '✅'}
                sessions_df['status_icon'] = sessions_df['status'].map(status_map)
                sessions_df['Status'] = sessions_df['status_icon'] + ' ' + sessions_df['status']
                
                display_df = sessions_df[['user_name', 'exercise_name', 'Status', 'rep_count', 'duration_seconds', 'performed_at']]
                display_df.columns = ['User', 'Exercise', 'Status', 'Reps', 'Duration (s)', 'Performed At']
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No workout sessions found. Record your first session!")
        else:
            st.error("Failed to load sessions")

def show_ai_analysis():
    st.header("🤖 AI Analysis Pipeline")
    
    # Get pending sessions
    sessions_data, success = make_api_request("GET", "/sessions")
    
    if success and isinstance(sessions_data, list):
        pending_sessions = [s for s in sessions_data if s['status'] in ['pending', 'processing']]
        
        if pending_sessions:
            st.subheader("Sessions Ready for AI Analysis")
            
            for session in pending_sessions:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{session['user_name']}** - {session['exercise_name']}")
                        st.write(f"Reps: {session['rep_count']} | Duration: {session['duration_seconds']}s")
                        st.write(f"Status: {session['status']}")
                    
                    with col2:
                        session_id = session['id']
                        if st.button(f"🤖 Analyze", key=f"analyze_{session_id}"):
                            with st.spinner("Running AI analysis..."):
                                result, success = make_api_request("POST", f"/process/{session_id}")
                                
                                if success:
                                    st.success("✅ Analysis completed!")
                                    st.json(result)
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result}")
                    
                    with col3:
                        if st.button("📊 Status", key=f"status_{session_id}"):
                            status_data, status_success = make_api_request("GET", f"/sessions/{session_id}/process-status")
                            if status_success:
                                st.json(status_data)
                    
                    st.divider()
        else:
            st.info("🎉 All sessions have been processed! Create new sessions to see AI analysis in action.")
            
            # Show recently processed sessions
            processed_sessions = [s for s in sessions_data if s['status'] == 'processed']
            if processed_sessions:
                st.subheader("Recently Processed Sessions")
                recent_sessions = sorted(processed_sessions, key=lambda x: x['performed_at'], reverse=True)[:5]
                
                for session in recent_sessions:
                    st.write(f"✅ **{session['user_name']}** - {session['exercise_name']} (Processed)")
    else:
        st.error("Failed to load session data")

def show_feedback_review():
    st.header("📝 Feedback Review")
    
    sessions_data, success = make_api_request("GET", "/sessions")
    
    if success and isinstance(sessions_data, list):
        processed_sessions = [s for s in sessions_data if s['status'] == 'processed']
        
        if processed_sessions:
            session_options = {f"{s['user_name']} - {s['exercise_name']} ({s['performed_at'][:10]})": s['id'] for s in processed_sessions}
            
            selected_session = st.selectbox("Select Session to View Feedback", list(session_options.keys()))
            
            if selected_session:
                session_id = session_options[selected_session]
                feedback_data, feedback_success = make_api_request("GET", f"/sessions/{session_id}/feedback")
                
                if feedback_success and isinstance(feedback_data, list) and feedback_data:
                    for feedback in feedback_data:
                        with st.container():
                            # Form score visualization
                            form_score = feedback.get('form_score', 0)
                            if form_score:
                                score_percentage = int(form_score * 100)
                                st.metric("Form Score", f"{score_percentage}%")
                            
                            # Injury risk level
                            risk_level = feedback.get('injury_risk_level', 'unknown')
                            risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                            risk_icon = risk_colors.get(risk_level, '⚪')
                            st.write(f"**Injury Risk:** {risk_icon} {risk_level.title()}")
                            
                            # Summary
                            st.write("**AI Analysis Summary:**")
                            st.write(feedback['summary'])
                            
                            # Recommendations
                            if feedback.get('recommendations'):
                                st.write("**Recommendations:**")
                                st.write(feedback['recommendations'])
                            
                            st.write(f"*Generated: {feedback['created_at'][:19]}*")
                            st.divider()
                else:
                    st.info("No feedback found for this session")
        else:
            st.info("No processed sessions found. Complete some AI analysis first!")
    else:
        st.error("Failed to load session data")

if __name__ == "__main__":
    main()