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
    """Enhanced API request with better error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        # Set timeout for all requests
        timeout = 30 if files else 10
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            if files:
                # For file uploads, don't set Content-Type header - let requests handle it
                response = requests.post(url, data=data, files=files, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, timeout=timeout)
        else:
            return f"Unsupported method: {method}", False
        
        if response.status_code in [200, 201]:
            try:
                return response.json(), True
            except json.JSONDecodeError:
                return response.text, True
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"HTTP {response.status_code}")
            except json.JSONDecodeError:
                error_msg = f"HTTP {response.status_code}: {response.text}"
            return error_msg, False
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure Flask server is running on localhost:5000", False
    except requests.exceptions.Timeout:
        return "❌ Request timed out. The server might be busy processing.", False
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {str(e)}", False
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", False

def get_ai_status():
    """Get detailed AI system status with error handling"""
    try:
        status_data, success = make_api_request("GET", "/ai-status")
        if success and isinstance(status_data, dict):
            return status_data
        else:
            return {}
    except Exception:
        return {}

def safe_get(data, key, default=None):
    """Safely get value from dict/list"""
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            return data[key] if 0 <= key < len(data) else default
        else:
            return default
    except (KeyError, IndexError, TypeError):
        return default

def main():
    # Header with version info
    st.markdown('<div class="main-header">🏋️‍♀️ AI Personal Trainer Dashboard v2.0</div>', unsafe_allow_html=True)
    
    # Sidebar with enhanced navigation
    st.sidebar.title("🧭 Navigation")
    
    # AI Status in sidebar with error handling
    try:
        ai_status = get_ai_status()
        if ai_status:
            llm_status = safe_get(ai_status, 'llm_generator', 'unknown')
            llm_provider = safe_get(ai_status, 'llm_provider', 'unknown')
            environment = safe_get(ai_status, 'environment', 'unknown')
            
            if llm_status == 'ready':
                st.sidebar.success(f"🤖 AI: {llm_provider.title()} Ready")
            else:
                st.sidebar.error(f"🤖 AI: {llm_status}")
            
            st.sidebar.info(f"🌍 Environment: {environment}")
        else:
            st.sidebar.warning("🤖 AI Status Unknown")
    except Exception as e:
        st.sidebar.error(f"🤖 AI Status Error: {str(e)}")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "📊 Dashboard"
    
    # Page selection
    page_options = [
        "📊 Dashboard",
        "👥 User Management", 
        "💪 Exercise Management",
        "🎯 Workout Sessions",
        "📹 Video Upload",
        "🤖 AI Analysis",
        "📝 Feedback Review",
        "⚙️ System Status"
    ]
    
    page = st.sidebar.selectbox("Choose a page", page_options, 
                               index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0)
    
    # Update session state
    st.session_state.page = page
    
    # Check API connection
    api_response, api_ok = make_api_request("GET", "/")
    if not api_ok:
        st.error("❌ API Connection Failed")
        st.error(api_response)
        st.info("Make sure your Flask server is running: `python app.py`")
        st.stop()
    else:
        try:
            if isinstance(api_response, dict):
                version = safe_get(api_response, 'version', 'unknown')
                st.sidebar.success(f"✅ API Connected (v{version})")
            else:
                st.sidebar.success("✅ API Connected")
        except Exception:
            st.sidebar.success("✅ API Connected")
    
    # Route to pages with error handling
    try:
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
    except Exception as e:
        st.error(f"❌ Page error: {str(e)}")
        st.error("Please try refreshing the page or contact support.")

def show_dashboard():
    st.header("📊 Dashboard Overview")
    
    # Enhanced AI status card with error handling
    try:
        ai_status = get_ai_status()
        if ai_status:
            llm_provider = safe_get(ai_status, 'llm_provider', 'unknown')
            llm_generator = safe_get(ai_status, 'llm_generator', 'unknown')
            environment = safe_get(ai_status, 'environment', 'unknown')
            
            st.markdown(f"""
            <div class="ai-status-card">
                <h3>🤖 AI System Status</h3>
                <p><strong>Provider:</strong> {llm_provider.title()}</p>
                <p><strong>Status:</strong> {llm_generator}</p>
                <p><strong>Environment:</strong> {environment}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load AI status: {str(e)}")
    
    # Get data for dashboard with error handling
    users_data, users_success = make_api_request("GET", "/users")
    exercises_data, exercises_success = make_api_request("GET", "/exercises")
    sessions_data, sessions_success = make_api_request("GET", "/sessions")
    
    # Validate data types
    if not users_success or not isinstance(users_data, list):
        users_data = []
    if not exercises_success or not isinstance(exercises_data, list):
        exercises_data = []
    if not sessions_success or not isinstance(sessions_data, list):
        sessions_data = []
    
    # Enhanced metrics with error handling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    try:
        with col1:
            st.metric("👥 Total Users", len(users_data))
        with col2:
            st.metric("💪 Total Exercises", len(exercises_data))
        with col3:
            st.metric("🎯 Total Sessions", len(sessions_data))
        with col4:
            processed_sessions = len([s for s in sessions_data if safe_get(s, 'status') == 'processed'])
            st.metric("✅ Processed", processed_sessions)
        with col5:
            # Calculate average form score placeholder
            avg_form_score = 75 if processed_sessions > 0 else 0
            st.metric("📈 Avg Form Score", f"{avg_form_score}%")
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")
    
    st.divider()
    
    # Recent activity with enhanced error handling
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Recent Users")
        try:
            if users_data:
                # Safely create DataFrame
                safe_users = []
                for user in users_data:
                    safe_user = {
                        'name': safe_get(user, 'name', 'Unknown'),
                        'email': safe_get(user, 'email', 'No email'),
                        'total_sessions': safe_get(user, 'total_sessions', 0),
                        'created_at': safe_get(user, 'created_at', 'Unknown')
                    }
                    safe_users.append(safe_user)
                
                if safe_users:
                    users_df = pd.DataFrame(safe_users)
                    # Safe datetime conversion
                    try:
                        users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%m-%d %H:%M')
                    except:
                        users_df['created_at'] = users_df['created_at'].astype(str)
                    
                    display_df = users_df[['name', 'email', 'total_sessions', 'created_at']]
                    display_df.columns = ['Name', 'Email', 'Sessions', 'Created']
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No valid user data found")
            else:
                st.info("No users found")
        except Exception as e:
            st.error(f"Error displaying users: {str(e)}")
    
    with col2:
        st.subheader("🎯 Recent Sessions")
        try:
            if sessions_data:
                # Safely create DataFrame
                safe_sessions = []
                for session in sessions_data[:10]:  # Limit to 10 recent sessions
                    safe_session = {
                        'user_name': safe_get(session, 'user_name', 'Unknown User'),
                        'exercise_name': safe_get(session, 'exercise_name', 'Unknown Exercise'),
                        'status': safe_get(session, 'status', 'unknown'),
                        'performed_at': safe_get(session, 'performed_at', 'Unknown')
                    }
                    safe_sessions.append(safe_session)
                
                if safe_sessions:
                    sessions_df = pd.DataFrame(safe_sessions)
                    
                    # Safe datetime conversion
                    try:
                        sessions_df['performed_at'] = pd.to_datetime(sessions_df['performed_at']).dt.strftime('%m-%d %H:%M')
                    except:
                        sessions_df['performed_at'] = sessions_df['performed_at'].astype(str)
                    
                    # Enhanced status display with safe mapping
                    status_map = {'pending': '⏳', 'uploaded': '📤', 'processing': '🔄', 'processed': '✅', 'error': '❌'}
                    sessions_df['status_icon'] = sessions_df['status'].map(lambda x: status_map.get(x, '❓'))
                    sessions_df['Status'] = sessions_df['status_icon'] + ' ' + sessions_df['status']
                    
                    display_df = sessions_df[['user_name', 'exercise_name', 'Status', 'performed_at']]
                    display_df.columns = ['User', 'Exercise', 'Status', 'Time']
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No valid session data found")
            else:
                st.info("No sessions found")
        except Exception as e:
            st.error(f"Error displaying sessions: {str(e)}")

def show_video_upload():
    """Fixed video upload functionality"""
    st.header("📹 Video Upload & AI Analysis")
    
    # Get users and exercises for the form with better error handling
    users_data, users_success = make_api_request("GET", "/users")
    exercises_data, exercises_success = make_api_request("GET", "/exercises")
    
    # Better error handling and validation
    if not users_success:
        st.error("❌ Failed to load users. Please check API connection.")
        st.error(f"Error: {users_data}")
        return
    
    if not exercises_success:
        st.error("❌ Failed to load exercises. Please check API connection.")
        st.error(f"Error: {exercises_data}")
        return
    
    # Ensure we have lists
    if not isinstance(users_data, list):
        st.error("❌ Invalid user data format received from API.")
        return
    
    if not isinstance(exercises_data, list):
        st.error("❌ Invalid exercise data format received from API.")
        return
    
    if not users_data or not exercises_data:
        st.warning("⚠️ You need to create at least one user and one exercise before uploading videos!")
        
        # Quick links to create data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👥 Create User"):
                st.session_state.page = "👥 User Management"
                st.rerun()
        with col2:
            if st.button("💪 Create Exercise"):
                st.session_state.page = "💪 Exercise Management"
                st.rerun()
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
            # User selection with safe data handling
            try:
                user_options = {}
                for user in users_data:
                    name = safe_get(user, 'name', 'Unknown')
                    email = safe_get(user, 'email', 'No email')
                    user_id = safe_get(user, 'id')
                    
                    if user_id is not None:
                        user_options[f"{name} ({email})"] = user_id
                
                if not user_options:
                    st.error("❌ No valid users found. Please create users first.")
                    st.stop()
                
                selected_user = st.selectbox("👤 Select User", list(user_options.keys()))
            except Exception as e:
                st.error(f"❌ Error processing user data: {e}")
                st.json(users_data)  # Show raw data for debugging
                st.stop()
        
        with col2:
            # Exercise selection with safe data handling
            try:
                exercise_options = {}
                for exercise in exercises_data:
                    name = safe_get(exercise, 'name', 'Unknown')
                    primary_muscle = safe_get(exercise, 'primary_muscle', 'Unknown')
                    exercise_id = safe_get(exercise, 'id')
                    
                    if exercise_id is not None:
                        exercise_options[f"{name} - {primary_muscle}"] = exercise_id
                
                if not exercise_options:
                    st.error("❌ No valid exercises found. Please create exercises first.")
                    st.stop()
                
                selected_exercise = st.selectbox("💪 Select Exercise", list(exercise_options.keys()))
            except Exception as e:
                st.error(f"❌ Error processing exercise data: {e}")
                st.json(exercises_data)  # Show raw data for debugging
                st.stop()
        
        # Video file upload
        uploaded_file = st.file_uploader(
            "Choose video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV (max 100MB)"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Upload & Analyze", type="primary")
        
        if submitted:
            if uploaded_file is None:
                st.error("❌ Please select a video file to upload.")
                return
            
            try:
                user_id = user_options[selected_user]
                exercise_id = exercise_options[selected_exercise]
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Create form data for upload
                files = {'video': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {'user_id': str(user_id), 'exercise_id': str(exercise_id)}
                
                with st.spinner("📤 Uploading video..."):
                    result, success = make_api_request("POST", "/upload-video", data=data, files=files)
                    
                    if success and isinstance(result, dict):
                        session_id = safe_get(result, 'session_id')
                        file_size = safe_get(result, 'size_mb', 0)
                        
                        st.success(f"✅ Video uploaded successfully! ({file_size}MB)")
                        st.info(f"📋 Session ID: {session_id}")
                        
                        # Automatically trigger AI analysis
                        if session_id:
                            with st.spinner("🤖 Running AI analysis..."):
                                time.sleep(1)  # Small delay for better UX
                                ai_result, ai_success = make_api_request("POST", f"/process-ai/{session_id}")
                                
                                if ai_success and isinstance(ai_result, dict):
                                    st.success("🎉 AI analysis completed!")
                                    
                                    # Display results safely
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        form_score = safe_get(ai_result, 'form_score', 0)
                                        if form_score:
                                            st.metric("📊 Form Score", f"{int(form_score * 100)}%")
                                    with col2:
                                        risk_level = safe_get(ai_result, 'injury_risk_level', 'unknown')
                                        risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                                        risk_icon = risk_colors.get(risk_level, '⚪')
                                        st.metric("⚠️ Injury Risk", f"{risk_icon} {risk_level.title()}")
                                    with col3:
                                        rep_count = safe_get(ai_result, 'rep_count', 0)
                                        if rep_count:
                                            st.metric("🔄 Rep Count", rep_count)
                                    
                                    # Show detailed feedback
                                    st.subheader("📝 AI Feedback")
                                    
                                    # Get the actual feedback
                                    feedback_data, feedback_success = make_api_request("GET", f"/sessions/{session_id}/feedback")
                                    if feedback_success and isinstance(feedback_data, list) and feedback_data:
                                        latest_feedback = feedback_data[-1]  # Get the most recent feedback
                                        summary = safe_get(latest_feedback, 'summary', 'No summary available')
                                        recommendations = safe_get(latest_feedback, 'recommendations', '')
                                        
                                        st.write("**Summary:**")
                                        st.write(summary)
                                        
                                        if recommendations:
                                            st.write("**Recommendations:**")
                                            st.write(recommendations)
                                    else:
                                        st.warning("No detailed feedback available yet.")
                                else:
                                    st.error(f"❌ AI analysis failed: {ai_result}")
                        else:
                            st.error("❌ No session ID returned from upload")
                    else:
                        st.error(f"❌ Upload failed: {result}")
                        
            except Exception as e:
                st.error(f"❌ Upload error: {str(e)}")
    
    # Add a section showing recent uploads with error handling
    st.subheader("📋 Recent Uploads")
    try:
        recent_sessions, sessions_success = make_api_request("GET", "/sessions")
        
        if sessions_success and isinstance(recent_sessions, list):
            # Filter for recently uploaded sessions
            uploaded_sessions = [s for s in recent_sessions 
                               if safe_get(s, 'status') in ['uploaded', 'processing', 'processed']]
            
            if uploaded_sessions:
                # Show last 5 uploads
                for session in uploaded_sessions[-5:]:
                    status = safe_get(session, 'status', 'unknown')
                    user_name = safe_get(session, 'user_name', 'Unknown User')
                    exercise_name = safe_get(session, 'exercise_name', 'Unknown Exercise')
                    
                    status_icons = {'uploaded': '📤', 'processing': '🔄', 'processed': '✅'}
                    status_icon = status_icons.get(status, '❓')
                    
                    st.write(f"{status_icon} {user_name} - {exercise_name} ({status})")
            else:
                st.info("No recent uploads found.")
        else:
            st.info("Could not load session data.")
    except Exception as e:
        st.error(f"Error loading recent uploads: {str(e)}")

def show_ai_analysis():
    st.header("🤖 AI Analysis Pipeline")
    
    # Enhanced AI status with error handling
    ai_status = get_ai_status()
    llm_generator_status = safe_get(ai_status, 'llm_generator', 'unknown')
    
    if llm_generator_status != 'ready':
        st.error("❌ AI system not ready. Check system status.")
        if ai_status:
            st.json(ai_status)
        return
    
    # Get pending sessions with error handling
    try:
        sessions_data, success = make_api_request("GET", "/sessions")
        
        if not success or not isinstance(sessions_data, list):
            st.error("❌ Could not load sessions data")
            return
        
        pending_sessions = [s for s in sessions_data if safe_get(s, 'status') in ['pending', 'uploaded']]
        processing_sessions = [s for s in sessions_data if safe_get(s, 'status') == 'processing']
        
        # Show processing sessions first
        if processing_sessions:
            st.subheader("🔄 Currently Processing")
            for session in processing_sessions:
                user_name = safe_get(session, 'user_name', 'Unknown User')
                exercise_name = safe_get(session, 'exercise_name', 'Unknown Exercise')
                st.info(f"🔄 Processing: {user_name} - {exercise_name}")
        
        if pending_sessions:
            st.subheader("⏳ Sessions Ready for AI Analysis")
            
            for session in pending_sessions:
                try:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        session_id = safe_get(session, 'id')
                        user_name = safe_get(session, 'user_name', 'Unknown User')
                        exercise_name = safe_get(session, 'exercise_name', 'Unknown Exercise')
                        rep_count = safe_get(session, 'rep_count', 'N/A')
                        duration = safe_get(session, 'duration_seconds', 'N/A')
                        status = safe_get(session, 'status', 'unknown')
                        
                        with col1:
                            st.write(f"**{user_name}** - {exercise_name}")
                            st.write(f"Reps: {rep_count} | Duration: {duration}s")
                            st.write(f"Status: {status}")
                        
                        with col2:
                            if session_id and st.button(f"🤖 Analyze", key=f"analyze_{session_id}"):
                                with st.spinner("🔄 Running AI analysis..."):
                                    result, success = make_api_request("POST", f"/process-ai/{session_id}")
                                    
                                    if success and isinstance(result, dict):
                                        st.success("✅ Analysis completed!")
                                        
                                        # Enhanced result display
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            form_score = safe_get(result, 'form_score', 0)
                                            if form_score:
                                                st.metric("📊 Form Score", f"{int(form_score * 100)}%")
                                        with col_b:
                                            rep_count_result = safe_get(result, 'rep_count', 0)
                                            st.metric("🔄 Reps", rep_count_result)
                                        with col_c:
                                            processing_time = safe_get(result, 'processing_time_seconds', 0)
                                            st.metric("⏱️ Time", f"{processing_time:.1f}s")
                                        
                                        time.sleep(2)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {result}")
                        
                        with col3:
                            if session_id and st.button("📊 Details", key=f"details_{session_id}"):
                                session_detail, detail_success = make_api_request("GET", f"/sessions/{session_id}")
                                if detail_success:
                                    st.json(session_detail)
                        
                        st.divider()
                        
                except Exception as e:
                    st.error(f"Error processing session {safe_get(session, 'id', 'unknown')}: {str(e)}")
        else:
            st.info("🎉 All sessions have been processed!")
            
            # Show recently processed sessions
            processed_sessions = [s for s in sessions_data if safe_get(s, 'status') == 'processed']
            if processed_sessions:
                st.subheader("✅ Recently Processed Sessions")
                try:
                    # Sort by performed_at safely
                    recent_sessions = sorted(processed_sessions, 
                                           key=lambda x: safe_get(x, 'performed_at', ''), 
                                           reverse=True)[:5]
                    
                    for session in recent_sessions:
                        user_name = safe_get(session, 'user_name', 'Unknown User')
                        exercise_name = safe_get(session, 'exercise_name', 'Unknown Exercise')
                        st.success(f"✅ **{user_name}** - {exercise_name} (Processed)")
                except Exception as e:
                    st.error(f"Error displaying processed sessions: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error in AI analysis page: {str(e)}")

def show_system_status():
    """System status page with enhanced error handling"""
    st.header("⚙️ System Status & Configuration")
    
    try:
        # Get comprehensive status
        ai_status = get_ai_status()
        api_info, api_success = make_api_request("GET", "/")
        
        # API Status
        st.subheader("🌐 API Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_success and isinstance(api_info, dict):
                version = safe_get(api_info, 'version', 'unknown')
                st.metric("📡 API Version", version)
            else:
                st.metric("📡 API Version", "Error")
        
        with col2:
            if api_success and isinstance(api_info, dict):
                environment = safe_get(api_info, 'environment', 'unknown')
                st.metric("🌍 Environment", environment)
            else:
                st.metric("🌍 Environment", "Error")
        
        with col3:
            if api_success and isinstance(api_info, dict):
                features = safe_get(api_info, 'ai_features', [])
                features_count = len(features) if isinstance(features, list) else 0
                st.metric("🎯 AI Features", features_count)
            else:
                st.metric("🎯 AI Features", "Error")
        
        # AI System Details
        if ai_status:
            st.subheader("🤖 AI System Details")
            
            # Status indicators
            components = {
                'Pose Analyzer': safe_get(ai_status, 'pose_analyzer', 'unknown'),
                'Form Analyzer': safe_get(ai_status, 'form_analyzer', 'unknown'),
                'LLM Generator': safe_get(ai_status, 'llm_generator', 'unknown')
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
            provider = safe_get(ai_status, 'llm_provider', 'unknown')
            
            if provider == 'kolosal':
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Provider:** {provider.title()}")
                    kolosal_url = safe_get(ai_status, 'kolosal_url', 'N/A')
                    st.info(f"**URL:** {kolosal_url}")
                with col2:
                    local_model = safe_get(ai_status, 'local_model', 'N/A')
                    st.info(f"**Model:** {local_model}")
                    kolosal_error = safe_get(ai_status, 'kolosal_error')
                    if kolosal_error:
                        st.error(kolosal_error)
            else:
                st.info(f"**Provider:** {provider.title()}")
            
            # Supported Features
            st.subheader("🎯 Supported Features")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Exercise Types:**")
                exercises = safe_get(ai_status, 'supported_exercises', [])
                if isinstance(exercises, list):
                    for exercise in exercises:
                        st.write(f"• {exercise}")
                else:
                    st.write("• Error loading exercises")
            
            with col2:
                st.write("**Video Formats:**")
                formats = safe_get(ai_status, 'supported_formats', [])
                if isinstance(formats, list):
                    for fmt in formats:
                        st.write(f"• {fmt}")
                else:
                    st.write("• Error loading formats")
                
                max_size = safe_get(ai_status, 'max_video_size_mb', 0)
                st.write(f"**Max Video Size:** {max_size}MB")
        else:
            st.error("❌ Could not load AI status")
        
        # System Performance (placeholder for future metrics)
        st.subheader("📈 System Performance")
        st.info("🚧 Performance metrics will be added in future updates")
        
    except Exception as e:
        st.error(f"Error loading system status: {str(e)}")

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
        
        try:
            users_data, success = make_api_request("GET", "/users")
            
            if success and isinstance(users_data, list):
                if users_data:
                    # Safely create DataFrame
                    safe_users = []
                    for user in users_data:
                        safe_user = {
                            'id': safe_get(user, 'id', 'N/A'),
                            'name': safe_get(user, 'name', 'Unknown'),
                            'email': safe_get(user, 'email', 'No email'),
                            'total_sessions': safe_get(user, 'total_sessions', 0),
                            'created_at': safe_get(user, 'created_at', 'Unknown')
                        }
                        safe_users.append(safe_user)
                    
                    users_df = pd.DataFrame(safe_users)
                    
                    # Safe datetime conversion
                    try:
                        users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass  # Keep original format if conversion fails
                    
                    st.dataframe(users_df, use_container_width=True)
                else:
                    st.info("No users found. Create your first user!")
            else:
                st.error(f"Failed to load users: {users_data}")
        except Exception as e:
            st.error(f"Error in user management: {str(e)}")

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
        
        try:
            exercises_data, success = make_api_request("GET", "/exercises")
            
            if success and isinstance(exercises_data, list):
                if exercises_data:
                    # Safely create DataFrame
                    safe_exercises = []
                    for exercise in exercises_data:
                        safe_exercise = {
                            'id': safe_get(exercise, 'id', 'N/A'),
                            'name': safe_get(exercise, 'name', 'Unknown'),
                            'primary_muscle': safe_get(exercise, 'primary_muscle', 'Unknown'),
                            'description': safe_get(exercise, 'description', ''),
                            'total_sessions': safe_get(exercise, 'total_sessions', 0)
                        }
                        safe_exercises.append(safe_exercise)
                    
                    exercises_df = pd.DataFrame(safe_exercises)
                    st.dataframe(exercises_df, use_container_width=True)
                else:
                    st.info("No exercises found. Add your first exercise!")
            else:
                st.error(f"Failed to load exercises: {exercises_data}")
        except Exception as e:
            st.error(f"Error in exercise management: {str(e)}")

def show_session_management():
    st.header("🎯 Workout Session Management")
    
    tab1, tab2 = st.tabs(["Create Session", "View Sessions"])
    
    with tab1:
        st.subheader("Record New Workout Session")
        
        # Get users and exercises for dropdowns
        users_data, users_success = make_api_request("GET", "/users")
        exercises_data, exercises_success = make_api_request("GET", "/exercises")
        
        if (users_success and isinstance(users_data, list) and users_data and 
            exercises_success and isinstance(exercises_data, list) and exercises_data):
            
            try:
                with st.form("create_session_form"):
                    # Safe option creation
                    user_options = {}
                    for user in users_data:
                        name = safe_get(user, 'name', 'Unknown')
                        email = safe_get(user, 'email', 'No email')
                        user_id = safe_get(user, 'id')
                        if user_id is not None:
                            user_options[f"{name} ({email})"] = user_id
                    
                    exercise_options = {}
                    for exercise in exercises_data:
                        name = safe_get(exercise, 'name', 'Unknown')
                        primary_muscle = safe_get(exercise, 'primary_muscle', 'Unknown')
                        exercise_id = safe_get(exercise, 'id')
                        if exercise_id is not None:
                            exercise_options[f"{name} - {primary_muscle}"] = exercise_id
                    
                    if not user_options or not exercise_options:
                        st.error("❌ No valid users or exercises found")
                        st.stop()
                    
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
            except Exception as e:
                st.error(f"Error creating session: {str(e)}")
        else:
            st.warning("⚠️ You need to create at least one user and one exercise first!")
    
    with tab2:
        st.subheader("All Workout Sessions")
        
        try:
            sessions_data, success = make_api_request("GET", "/sessions")
            
            if success and isinstance(sessions_data, list):
                if sessions_data:
                    # Safely create DataFrame
                    safe_sessions = []
                    for session in sessions_data:
                        safe_session = {
                            'id': safe_get(session, 'id', 'N/A'),
                            'user_name': safe_get(session, 'user_name', 'Unknown User'),
                            'exercise_name': safe_get(session, 'exercise_name', 'Unknown Exercise'),
                            'status': safe_get(session, 'status', 'unknown'),
                            'rep_count': safe_get(session, 'rep_count', 'N/A'),
                            'duration_seconds': safe_get(session, 'duration_seconds', 'N/A'),
                            'performed_at': safe_get(session, 'performed_at', 'Unknown')
                        }
                        safe_sessions.append(safe_session)
                    
                    sessions_df = pd.DataFrame(safe_sessions)
                    
                    # Safe datetime conversion
                    try:
                        sessions_df['performed_at'] = pd.to_datetime(sessions_df['performed_at']).dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass  # Keep original format if conversion fails
                    
                    # Enhanced status indicators with safe mapping
                    status_map = {'pending': '⏳', 'uploaded': '📤', 'processing': '🔄', 'processed': '✅', 'error': '❌'}
                    sessions_df['status_icon'] = sessions_df['status'].map(lambda x: status_map.get(x, '❓'))
                    sessions_df['Status'] = sessions_df['status_icon'] + ' ' + sessions_df['status']
                    
                    display_df = sessions_df[['user_name', 'exercise_name', 'Status', 'rep_count', 'duration_seconds', 'performed_at']]
                    display_df.columns = ['User', 'Exercise', 'Status', 'Reps', 'Duration (s)', 'Performed At']
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No workout sessions found. Record your first session!")
            else:
                st.error(f"Failed to load sessions: {sessions_data}")
        except Exception as e:
            st.error(f"Error in session management: {str(e)}")

def show_feedback_review():
    st.header("📝 Feedback Review")
    
    try:
        sessions_data, success = make_api_request("GET", "/sessions")
        
        if not success or not isinstance(sessions_data, list):
            st.error("❌ Failed to load sessions data")
            return
        
        processed_sessions = [s for s in sessions_data if safe_get(s, 'status') == 'processed']
        
        if processed_sessions:
            # Safe session options creation
            session_options = {}
            for session in processed_sessions:
                user_name = safe_get(session, 'user_name', 'Unknown User')
                exercise_name = safe_get(session, 'exercise_name', 'Unknown Exercise')
                performed_at = safe_get(session, 'performed_at', 'Unknown')
                session_id = safe_get(session, 'id')
                
                if session_id is not None:
                    # Safe date formatting
                    try:
                        date_str = performed_at[:10] if len(performed_at) >= 10 else performed_at
                    except:
                        date_str = 'Unknown'
                    
                    session_options[f"{user_name} - {exercise_name} ({date_str})"] = session_id
            
            if session_options:
                selected_session = st.selectbox("Select Session to View Feedback", list(session_options.keys()))
                
                if selected_session:
                    session_id = session_options[selected_session]
                    feedback_data, feedback_success = make_api_request("GET", f"/sessions/{session_id}/feedback")
                    
                    if feedback_success and isinstance(feedback_data, list) and feedback_data:
                        for feedback in feedback_data:
                            try:
                                with st.container():
                                    # Enhanced visualization with safe data access
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        form_score = safe_get(feedback, 'form_score', 0)
                                        if form_score:
                                            score_percentage = int(form_score * 100)
                                            st.metric("📊 Form Score", f"{score_percentage}%")
                                        else:
                                            st.metric("📊 Form Score", "N/A")
                                    
                                    with col2:
                                        risk_level = safe_get(feedback, 'injury_risk_level', 'unknown')
                                        risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                                        risk_icon = risk_colors.get(risk_level, '⚪')
                                        st.metric("⚠️ Injury Risk", f"{risk_icon} {risk_level.title()}")
                                    
                                    with col3:
                                        llm_model = safe_get(feedback, 'llm_model_used', 'unknown')
                                        st.metric("🤖 AI Model", llm_model.title())
                                    
                                    # Detailed feedback
                                    st.subheader("📝 AI Analysis Summary")
                                    summary = safe_get(feedback, 'summary', 'No summary available')
                                    st.write(summary)
                                    
                                    recommendations = safe_get(feedback, 'recommendations')
                                    if recommendations:
                                        st.subheader("💡 Recommendations")
                                        st.write(recommendations)
                                    
                                    # Metadata
                                    created_at = safe_get(feedback, 'created_at', 'Unknown')
                                    processing_time = safe_get(feedback, 'processing_time_ms', 0)
                                    
                                    try:
                                        created_display = created_at[:19] if len(created_at) >= 19 else created_at
                                    except:
                                        created_display = created_at
                                    
                                    st.caption(f"Generated: {created_display} | Processing time: {processing_time}ms")
                                    st.divider()
                                    
                            except Exception as e:
                                st.error(f"Error displaying feedback: {str(e)}")
                    else:
                        st.info("No feedback found for this session")
            else:
                st.info("No valid processed sessions found")
        else:
            st.info("No processed sessions found. Complete some AI analysis first!")
            
    except Exception as e:
        st.error(f"Error in feedback review: {str(e)}")

if __name__ == "__main__":
    main()