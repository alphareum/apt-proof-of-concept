import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import io

# API Configuration
API_BASE_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="APT - AI Personal Trainer Dashboard v2.0",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
    .ai-status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .video-upload-zone {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(method, endpoint, data=None, files=None):
    """Enhanced API request with file upload support"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
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

def get_ai_status():
    """Get detailed AI system status"""
    status_data, success = make_api_request("GET", "/ai-status")
    return status_data if success else {}

def main():
    # Header with version info
    st.markdown('<div class="main-header">🏋️‍♀️ AI Personal Trainer Dashboard v2.0</div>', unsafe_allow_html=True)
    
    # Sidebar with enhanced navigation
    st.sidebar.title("🧭 Navigation")
    
    # AI Status in sidebar
    ai_status = get_ai_status()
    if ai_status:
        llm_status = ai_status.get('llm_generator', 'unknown')
        if llm_status == 'ready':
            st.sidebar.success(f"🤖 AI: {ai_status.get('llm_provider', 'unknown').title()} Ready")
        else:
            st.sidebar.error(f"🤖 AI: {llm_status}")
        
        st.sidebar.info(f"🌍 Environment: {ai_status.get('environment', 'unknown')}")
    
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard",
        "👥 User Management", 
        "💪 Exercise Management",
        "🎯 Workout Sessions",
        "📹 Video Upload",
        "🤖 AI Analysis",
        "📝 Feedback Review",
        "⚙️ System Status"
    ])
    
    # Check API connection
    api_response, api_ok = make_api_request("GET", "/")
    if not api_ok:
        st.error(api_response)
        st.stop()
    else:
        version = api_response.get('version', 'unknown')
        st.sidebar.success(f"✅ API Connected (v{version})")
    
    # Route to pages
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "👥 User Management":
        show_user_management()
    elif page == "💪 Exercise Management":
        show_exercise_management()
    elif page == "🎯 Workout Sessions":
        show_session_management()
    elif page == "📹 Video Upload":
        show_video_upload()
    elif page == "🤖 AI Analysis":
        show_ai_analysis()
    elif page == "📝 Feedback Review":
        show_feedback_review()
    elif page == "⚙️ System Status":
        show_system_status()

def show_dashboard():
    st.header("📊 Dashboard Overview")
    
    # Enhanced AI status card
    ai_status = get_ai_status()
    if ai_status:
        st.markdown(f"""
        <div class="ai-status-card">
            <h3>🤖 AI System Status</h3>
            <p><strong>Provider:</strong> {ai_status.get('llm_provider', 'unknown').title()}</p>
            <p><strong>Status:</strong> {ai_status.get('llm_generator', 'unknown')}</p>
            <p><strong>Environment:</strong> {ai_status.get('environment', 'unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get data for dashboard
    users_data, _ = make_api_request("GET", "/users")
    exercises_data, _ = make_api_request("GET", "/exercises")
    sessions_data, _ = make_api_request("GET", "/sessions")
    
    if isinstance(users_data, list) and isinstance(exercises_data, list) and isinstance(sessions_data, list):
        # Enhanced metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("👥 Total Users", len(users_data))
        with col2:
            st.metric("💪 Total Exercises", len(exercises_data))
        with col3:
            st.metric("🎯 Total Sessions", len(sessions_data))
        with col4:
            processed_sessions = len([s for s in sessions_data if s.get('status') == 'processed'])
            st.metric("✅ Processed", processed_sessions)
        with col5:
            avg_form_score = 0
            if processed_sessions > 0:
                # This would need API enhancement to get average form scores
                avg_form_score = 78  # Placeholder
            st.metric("📈 Avg Form Score", f"{avg_form_score}%")
        
        st.divider()
        
        # Recent activity with enhanced info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Recent Users")
            if users_data:
                users_df = pd.DataFrame(users_data)
                users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%m-%d %H:%M')
                display_df = users_df[['name', 'email', 'total_sessions', 'created_at']]
                display_df.columns = ['Name', 'Email', 'Sessions', 'Created']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No users found")
        
        with col2:
            st.subheader("🎯 Recent Sessions")
            if sessions_data:
                sessions_df = pd.DataFrame(sessions_data)
                sessions_df['performed_at'] = pd.to_datetime(sessions_df['performed_at']).dt.strftime('%m-%d %H:%M')
                
                # Enhanced status display
                status_map = {'pending': '⏳', 'processing': '🔄', 'processed': '✅', 'error': '❌'}
                sessions_df['status_icon'] = sessions_df['status'].map(status_map)
                sessions_df['Status'] = sessions_df['status_icon'] + ' ' + sessions_df['status']
                
                display_df = sessions_df[['user_name', 'exercise_name', 'Status', 'performed_at']].head(10)
                display_df.columns = ['User', 'Exercise', 'Status', 'Time']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No sessions found")

def show_video_upload():
    """New video upload functionality"""
    st.header("📹 Video Upload & AI Analysis")
    
    # Get users and exercises for the form
    users_data, _ = make_api_request("GET", "/users")
    exercises_data, _ = make_api_request("GET", "/exercises")
    
    if not users_data or not exercises_data:
        st.warning("⚠️ You need to create at least one user and one exercise before uploading videos!")
        return
    
    st.markdown("""
    <div class="video-upload-zone">
        <h3>📹 Upload Exercise Video</h3>
        <p>Upload a video of your workout for AI-powered form analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("video_upload_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # User selection
            user_options = {f"{u['name']} ({u['email']})": u['id'] for u in users_data}
            selected_user = st.selectbox("👤 Select User", list(user_options.keys()))
        
        with col2:
            # Exercise selection
            exercise_options = {f"{e['name']} - {e['primary_muscle']}": e['id'] for e in exercises_data}
            selected_exercise = st.selectbox("💪 Select Exercise", list(exercise_options.keys()))
        
        # Video file upload
        uploaded_file = st.file_uploader(
            "Choose video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV (max 100MB)"
        )
        
        submitted = st.form_submit_button("🚀 Upload & Analyze", type="primary")
        
        if submitted and uploaded_file:
            user_id = user_options[selected_user]
            exercise_id = exercise_options[selected_exercise]
            
            # Create form data for upload
            files = {'video': uploaded_file}
            data = {'user_id': user_id, 'exercise_id': exercise_id}
            
            with st.spinner("📤 Uploading video..."):
                result, success = make_api_request("POST", "/upload-video", data=data, files=files)
                
                if success:
                    session_id = result.get('session_id')
                    file_size = result.get('size_mb')
                    
                    st.success(f"✅ Video uploaded successfully! ({file_size}MB)")
                    st.info(f"📋 Session ID: {session_id}")
                    
                    # Automatically trigger AI analysis
                    with st.spinner("🤖 Running AI analysis..."):
                        time.sleep(1)  # Small delay for better UX
                        ai_result, ai_success = make_api_request("POST", f"/process-ai/{session_id}")
                        
                        if ai_success:
                            st.success("🎉 AI analysis completed!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Form Score", f"{int(ai_result.get('form_score', 0) * 100)}%")
                            with col2:
                                risk_level = ai_result.get('injury_risk_level', 'unknown')
                                risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                                risk_icon = risk_colors.get(risk_level, '⚪')
                                st.metric("⚠️ Injury Risk", f"{risk_icon} {risk_level.title()}")
                            with col3:
                                st.metric("🔄 Rep Count", ai_result.get('rep_count', 0))
                            
                            # Show detailed feedback
                            if 'summary' in ai_result:
                                st.subheader("📝 AI Feedback")
                                st.write(ai_result['summary'])
                        else:
                            st.error(f"❌ AI analysis failed: {ai_result}")
                else:
                    st.error(f"❌ Upload failed: {result}")

def show_ai_analysis():
    st.header("🤖 AI Analysis Pipeline")
    
    # Enhanced AI status
    ai_status = get_ai_status()
    if ai_status.get('llm_generator') != 'ready':
        st.error("❌ AI system not ready. Check system status.")
        return
    
    # Get pending sessions
    sessions_data, success = make_api_request("GET", "/sessions")
    
    if success and isinstance(sessions_data, list):
        pending_sessions = [s for s in sessions_data if s['status'] in ['pending', 'uploaded']]
        processing_sessions = [s for s in sessions_data if s['status'] == 'processing']
        
        # Show processing sessions first
        if processing_sessions:
            st.subheader("🔄 Currently Processing")
            for session in processing_sessions:
                st.info(f"🔄 Processing: {session['user_name']} - {session['exercise_name']}")
        
        if pending_sessions:
            st.subheader("⏳ Sessions Ready for AI Analysis")
            
            for session in pending_sessions:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{session['user_name']}** - {session['exercise_name']}")
                        st.write(f"Reps: {session.get('rep_count', 'N/A')} | Duration: {session.get('duration_seconds', 'N/A')}s")
                        st.write(f"Status: {session['status']}")
                    
                    with col2:
                        session_id = session['id']
                        if st.button(f"🤖 Analyze", key=f"analyze_{session_id}"):
                            with st.spinner("🔄 Running AI analysis..."):
                                # Use the new endpoint
                                result, success = make_api_request("POST", f"/process-ai/{session_id}")
                                
                                if success:
                                    st.success("✅ Analysis completed!")
                                    
                                    # Enhanced result display
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        form_score = result.get('form_score', 0)
                                        st.metric("📊 Form Score", f"{int(form_score * 100)}%")
                                    with col_b:
                                        st.metric("🔄 Reps", result.get('rep_count', 0))
                                    with col_c:
                                        processing_time = result.get('processing_time_seconds', 0)
                                        st.metric("⏱️ Time", f"{processing_time:.1f}s")
                                    
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result}")
                    
                    with col3:
                        if st.button("📊 Details", key=f"details_{session_id}"):
                            session_detail, detail_success = make_api_request("GET", f"/sessions/{session_id}")
                            if detail_success:
                                st.json(session_detail)
                    
                    st.divider()
        else:
            st.info("🎉 All sessions have been processed!")
            
            # Show recently processed sessions with enhanced info
            processed_sessions = [s for s in sessions_data if s['status'] == 'processed']
            if processed_sessions:
                st.subheader("✅ Recently Processed Sessions")
                recent_sessions = sorted(processed_sessions, key=lambda x: x['performed_at'], reverse=True)[:5]
                
                for session in recent_sessions:
                    st.success(f"✅ **{session['user_name']}** - {session['exercise_name']} (Processed)")

def show_system_status():
    """New system status page"""
    st.header("⚙️ System Status & Configuration")
    
    # Get comprehensive status
    ai_status = get_ai_status()
    api_info, _ = make_api_request("GET", "/")
    
    # API Status
    st.subheader("🌐 API Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📡 API Version", api_info.get('version', 'unknown'))
    with col2:
        st.metric("🌍 Environment", api_info.get('environment', 'unknown'))
    with col3:
        features = api_info.get('ai_features', [])
        st.metric("🎯 AI Features", len(features))
    
    # AI System Details
    if ai_status:
        st.subheader("🤖 AI System Details")
        
        # Status indicators
        components = {
            'Pose Analyzer': ai_status.get('pose_analyzer', 'unknown'),
            'Form Analyzer': ai_status.get('form_analyzer', 'unknown'),
            'LLM Generator': ai_status.get('llm_generator', 'unknown')
        }
        
        cols = st.columns(len(components))
        for i, (component, status) in enumerate(components.items()):
            with cols[i]:
                if status == 'ready':
                    st.success(f"✅ {component}")
                else:
                    st.error(f"❌ {component}")
        
        # LLM Provider Info
        st.subheader("🧠 LLM Configuration")
        provider = ai_status.get('llm_provider', 'unknown')
        
        if provider == 'kolosal':
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Provider:** {provider.title()}")
                st.info(f"**URL:** {ai_status.get('kolosal_url', 'N/A')}")
            with col2:
                st.info(f"**Model:** {ai_status.get('local_model', 'N/A')}")
                if ai_status.get('kolosal_error'):
                    st.error(ai_status['kolosal_error'])
        
        # Supported Features
        st.subheader("🎯 Supported Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Exercise Types:**")
            exercises = ai_status.get('supported_exercises', [])
            for exercise in exercises:
                st.write(f"• {exercise}")
        
        with col2:
            st.write("**Video Formats:**")
            formats = ai_status.get('supported_formats', [])
            for fmt in formats:
                st.write(f"• {fmt}")
            
            max_size = ai_status.get('max_video_size_mb', 0)
            st.write(f"**Max Video Size:** {max_size}MB")
    
    # System Performance (placeholder for future metrics)
    st.subheader("📈 System Performance")
    st.info("🚧 Performance metrics will be added in future updates")

# Keep all the existing functions with minor enhancements
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
                
                submitted = st.form_submit_button("Record Session", type="primary")
                
                if submitted:
                    data = {
                        "user_id": user_options[selected_user],
                        "exercise_id": exercise_options[selected_exercise],
                        "rep_count": rep_count,
                        "duration_seconds": duration
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
                
                # Enhanced status indicators
                status_map = {'pending': '⏳', 'uploaded': '📤', 'processing': '🔄', 'processed': '✅', 'error': '❌'}
                sessions_df['status_icon'] = sessions_df['status'].map(status_map)
                sessions_df['Status'] = sessions_df['status_icon'] + ' ' + sessions_df['status']
                
                display_df = sessions_df[['user_name', 'exercise_name', 'Status', 'rep_count', 'duration_seconds', 'performed_at']]
                display_df.columns = ['User', 'Exercise', 'Status', 'Reps', 'Duration (s)', 'Performed At']
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No workout sessions found. Record your first session!")
        else:
            st.error("Failed to load sessions")

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
                            # Enhanced visualization
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                form_score = feedback.get('form_score', 0)
                                if form_score:
                                    score_percentage = int(form_score * 100)
                                    st.metric("📊 Form Score", f"{score_percentage}%")
                            
                            with col2:
                                risk_level = feedback.get('injury_risk_level', 'unknown')
                                risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                                risk_icon = risk_colors.get(risk_level, '⚪')
                                st.metric("⚠️ Injury Risk", f"{risk_icon} {risk_level.title()}")
                            
                            with col3:
                                llm_model = feedback.get('llm_model_used', 'unknown')
                                st.metric("🤖 AI Model", llm_model.title())
                            
                            # Detailed feedback
                            st.subheader("📝 AI Analysis Summary")
                            st.write(feedback['summary'])
                            
                            if feedback.get('recommendations'):
                                st.subheader("💡 Recommendations")
                                st.write(feedback['recommendations'])
                            
                            st.caption(f"Generated: {feedback['created_at'][:19]} | Processing time: {feedback.get('processing_time_ms', 0)}ms")
                            st.divider()
                else:
                    st.info("No feedback found for this session")
        else:
            st.info("No processed sessions found. Complete some AI analysis first!")
    else:
        st.error("Failed to load session data")

if __name__ == "__main__":
    main()