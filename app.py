"""
AI Fitness Assistant Pro - Main Application Entry Point

Enhanced fitness application with comprehensive features:
- Advanced body composition analysis
- AI-powered exercise recommendations  
- Real-time form correction with pose detection
- Progress tracking and analytics
- Goal setting and achievement monitoring

Version: 3.0.0
Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import streamlit as st
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Safe imports with error handling
try:
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    st.warning("⚠️ Computer vision libraries not available. Some features may be limited.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    # Import custom modules
    from models import UserProfile, create_user_profile, Gender, ActivityLevel, FitnessLevel, GoalType
    from database import get_database
    from recommendation_engine import AdvancedExerciseRecommendationEngine
    
    # Import enhanced recommendation system
    try:
        from enhanced_recommendation_system import EnhancedRecommendationEngine
        from workout_planner import AdvancedWorkoutPlanner
        from workout_planner_ui import WorkoutPlannerUI
        ENHANCED_PLANNER_AVAILABLE = True
    except ImportError:
        ENHANCED_PLANNER_AVAILABLE = False
        st.warning("⚠️ Enhanced workout planner not available. Using basic recommendations.")
    
    # Import body composition analysis
    try:
        from body_composition_analyzer import get_body_analyzer
        BODY_COMP_AVAILABLE = True
    except ImportError:
        BODY_COMP_AVAILABLE = False

    # Import enhanced UI components
    try:
        from enhanced_ui import EnhancedFitnessUI
        ENHANCED_UI_AVAILABLE = True
    except ImportError:
        ENHANCED_UI_AVAILABLE = False

    # Initialize enhanced UI if available
    if ENHANCED_UI_AVAILABLE:
        enhanced_ui = EnhancedFitnessUI()

except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.stop()

def render_nutrition_tab(user_profile: UserProfile):
    """Render nutrition and meal planning tab."""
    
    if ENHANCED_UI_AVAILABLE:
        enhanced_ui.render_nutrition_tab(user_profile)
    else:
        st.error("🍎 Enhanced nutrition features not available. Please install required dependencies.")
        st.markdown("""
        ### 🍎 Nutrition Planning (Coming Soon)
        
        This comprehensive nutrition system will include:
        
        **🎯 Daily Nutrition:**
        - Personalized macro targets (calories, protein, carbs, fat)
        - Automatic meal plan generation
        - Hydration tracking
        - Supplement recommendations
        
        **📅 Meal Planning:**
        - Weekly meal plans
        - Recipe recommendations
        - Dietary restriction support
        - Shopping list generation
        
        **🍳 Recipe Management:**
        - Recipe database with nutritional info
        - Custom recipe creation
        - Meal prep instructions
        - Difficulty and time filters
        
        **📊 Nutrition Analytics:**
        - Intake tracking over time
        - Macro distribution analysis
        - Goal progress monitoring
        - Nutritional insights
        """)

def render_social_tab(user_profile: UserProfile):
    """Render social and community features tab."""
    
    if ENHANCED_UI_AVAILABLE:
        enhanced_ui.render_social_tab(user_profile)
    else:
        st.error("👥 Social features not available. Please install required dependencies.")
        st.markdown("""
        ### 👥 Community & Social Features (Coming Soon)
        
        Connect with the fitness community through:
        
        **🏆 Achievements System:**
        - Unlock badges for milestones
        - Track your fitness journey
        - Celebrate accomplishments
        - Progress recognition
        
        **🎯 Challenges:**
        - Join community challenges
        - Create personal goals
        - Weekly fitness competitions
        - Streak tracking
        
        **📈 Leaderboards:**
        - Community rankings
        - Workout frequency leaders
        - Streak champions
        - Achievement leaders
        
        **📱 Social Sharing:**
        - Share workout results
        - Community feed
        - Motivational posts
        - Progress celebrations
        
        **🌟 Motivation Hub:**
        - Daily inspiration
        - Personal stats
        - Progress tracking
        - Achievement highlights
        """)

def render_analytics_tab(user_profile: UserProfile):
    """Render advanced analytics and progress tracking tab with enhanced features."""
    
    if ENHANCED_UI_AVAILABLE:
        enhanced_ui.render_analytics_tab(user_profile)
    else:
        st.markdown("### 📊 Advanced Analytics & Progress Tracking")
        
        # Import enhanced progress tracking
        try:
            from enhanced_progress_tracking import (
                render_enhanced_progress_photos_tab,
                render_video_form_analysis_tab, 
                render_enhanced_measurements_tab
            )
            
            # Enhanced analytics tabs
            analytics_tabs = st.tabs([
                "📈 Progress Overview",
                "📸 Progress Photos + AI",
                "🎥 Video Form Analysis", 
                "📏 Enhanced Measurements",
                "💪 Performance Tracking",
                "📊 Comprehensive Reports"
            ])
            
            with analytics_tabs[0]:
                render_progress_overview(user_profile)
            
            with analytics_tabs[1]:
                render_enhanced_progress_photos_tab(user_profile)
            
            with analytics_tabs[2]:
                render_video_form_analysis_tab(user_profile)
            
            with analytics_tabs[3]:
                render_enhanced_measurements_tab(user_profile)
            
            with analytics_tabs[4]:
                render_performance_tracking_tab(user_profile)
            
            with analytics_tabs[5]:
                render_comprehensive_reports_tab(user_profile)
                
        except ImportError:
            st.error("📊 Enhanced analytics features not available. Please install required dependencies.")
            render_basic_analytics_tab(user_profile)

def render_progress_overview(user_profile: UserProfile):
    """Render progress overview with key metrics."""
    
    st.markdown("#### 📈 Progress Overview Dashboard")
    
    # Key metrics in columns
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        # Simulated workout consistency
        workout_consistency = 85  # Would come from database
        st.metric("Workout Consistency", f"{workout_consistency}%", "+5%")
    
    with metric_cols[1]:
        # Progress photos taken
        photo_count = len(st.session_state.get('progress_photos_with_analysis', []))
        st.metric("Progress Photos", photo_count, "+1 this week")
    
    with metric_cols[2]:
        # Form analysis sessions
        video_count = len(st.session_state.get('video_form_analysis', []))
        st.metric("Form Analysis", video_count, "Recent session")
    
    with metric_cols[3]:
        # Goals achieved
        goals_achieved = 7  # Would come from database
        st.metric("Goals Achieved", goals_achieved, "+2 this month")
    
    # Progress trends chart
    st.markdown("#### 📊 Progress Trends (Last 12 Weeks)")
    
    # Create sample trend data
    import plotly.graph_objects as go
    weeks = list(range(1, 13))
    weight_trend = [80 - i*0.3 + (i%3)*0.5 for i in weeks]
    strength_trend = [100 + i*2.5 + (i%4)*1.2 for i in weeks]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=weight_trend, name="Weight (kg)", line=dict(color='#FF6B6B')))
    fig.add_trace(go.Scatter(x=weeks, y=strength_trend, name="Strength Index", line=dict(color='#4ECDC4'), yaxis="y2"))
    
    fig.update_layout(
        title="Weight and Strength Progress",
        xaxis_title="Week",
        yaxis=dict(title="Weight (kg)", side="left"),
        yaxis2=dict(title="Strength Index", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activities
    st.markdown("#### 🔄 Recent Activities")
    
    activities = [
        {"date": "2024-01-22", "activity": "Progress photo analyzed", "type": "photo", "details": "AI noted improved posture"},
        {"date": "2024-01-21", "activity": "Squat form analyzed", "type": "video", "details": "Form score: 87/100"},
        {"date": "2024-01-20", "activity": "Body measurements recorded", "type": "measurement", "details": "Waist: -1.2cm from last month"},
        {"date": "2024-01-19", "activity": "Workout completed", "type": "workout", "details": "Upper body strength session"}
    ]
    
    for activity in activities:
        activity_type_icons = {
            "photo": "📸",
            "video": "🎥", 
            "measurement": "📏",
            "workout": "💪"
        }
        
        icon = activity_type_icons.get(activity["type"], "📝")
        st.write(f"{icon} **{activity['date']}** - {activity['activity']}")
        st.caption(activity["details"])

def render_performance_tracking_tab(user_profile: UserProfile):
    """Render performance tracking with video analysis integration."""
    
    st.markdown("#### 💪 Performance Tracking Dashboard")
    
    # Performance metrics
    st.markdown("##### 🎯 Key Performance Indicators")
    
    perf_cols = st.columns(3)
    
    with perf_cols[0]:
        st.metric("Average Form Score", "84.2", "+2.3 pts")
        st.metric("Video Sessions", "12", "+3 this month")
    
    with perf_cols[1]:
        st.metric("Technique Improvements", "8", "+2 exercises")
        st.metric("Best Streak", "14 days", "Current")
    
    with perf_cols[2]:
        st.metric("Personal Records", "5", "+1 this week")
        st.metric("Injury Prevention", "98%", "Excellent")
    
    # Exercise-specific tracking
    st.markdown("##### 🏋️ Exercise-Specific Performance")
    
    exercise_data = {
        "Squat": {"form_score": 87, "last_analysis": "2024-01-21", "improvements": "Depth improved, knee tracking better"},
        "Deadlift": {"form_score": 82, "last_analysis": "2024-01-19", "improvements": "Bar path more consistent"},
        "Push-up": {"form_score": 91, "last_analysis": "2024-01-18", "improvements": "Excellent alignment maintained"}
    }
    
    for exercise, data in exercise_data.items():
        with st.expander(f"🎯 {exercise} Performance", expanded=False):
            ex_col1, ex_col2 = st.columns(2)
            
            with ex_col1:
                st.metric("Current Form Score", f"{data['form_score']}/100")
                st.write(f"**Last Analysis:** {data['last_analysis']}")
            
            with ex_col2:
                st.markdown("**Recent Improvements:**")
                st.write(data['improvements'])
                
                if st.button(f"📹 New Analysis", key=f"analysis_{exercise}"):
                    st.info(f"Upload a new {exercise} video in the Video Analysis tab")
    
    # Form improvement suggestions
    st.markdown("##### 📋 Personalized Improvement Plan")
    
    improvements = [
        {"exercise": "Squat", "focus": "Ankle mobility", "priority": "High", "timeline": "2 weeks"},
        {"exercise": "Deadlift", "focus": "Hip hinge pattern", "priority": "Medium", "timeline": "3 weeks"},
        {"exercise": "General", "focus": "Core stability", "priority": "Medium", "timeline": "Ongoing"}
    ]
    
    for improvement in improvements:
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        icon = priority_color.get(improvement["priority"], "🔵")
        
        st.write(f"{icon} **{improvement['exercise']}**: Focus on {improvement['focus']} ({improvement['timeline']})")

def render_comprehensive_reports_tab(user_profile: UserProfile):
    """Render comprehensive progress reports."""
    
    st.markdown("#### � Comprehensive Progress Reports")
    
    # Report generation options
    report_col1, report_col2 = st.columns(2)
    
    with report_col1:
        report_period = st.selectbox(
            "Report Period",
            ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months", "All Time"]
        )
    
    with report_col2:
        report_type = st.selectbox(
            "Report Type", 
            ["Complete Summary", "Body Composition Focus", "Performance Focus", "Goal Achievement"]
        )
    
    if st.button("📄 Generate Report"):
        with st.spinner("Generating comprehensive report..."):
            # Simulate report generation
            st.success("✅ Report generated successfully!")
            
            # Report sections
            st.markdown("### 📋 Progress Report Summary")
            
            # Executive summary
            st.markdown("#### 🎯 Executive Summary")
            st.info(f"""
            **Report Period:** {report_period}
            **Report Type:** {report_type}
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            Your fitness journey shows consistent improvement across multiple metrics. 
            Key highlights include improved form scores, steady body composition changes, 
            and strong workout consistency.
            """)
            
            # Key achievements
            st.markdown("#### 🏆 Key Achievements")
            achievements = [
                "✅ Maintained 85%+ workout consistency",
                "✅ Improved squat form score by 12 points", 
                "✅ Reduced body fat percentage by 2.1%",
                "✅ Completed 5 new personal records",
                "✅ Perfect injury prevention record"
            ]
            
            for achievement in achievements:
                st.success(achievement)
            
            # Areas for improvement
            st.markdown("#### 🎯 Areas for Continued Focus")
            focus_areas = [
                "� Ankle mobility for deeper squat depth",
                "🔄 Deadlift bar path consistency", 
                "🔄 Core stability during overhead movements",
                "🔄 Recovery and sleep optimization"
            ]
            
            for area in focus_areas:
                st.warning(area)
            
            # Download options
            st.markdown("#### 💾 Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Export PDF"):
                    st.info("PDF export feature coming soon")
            
            with export_col2:
                if st.button("📊 Export Data"):
                    st.info("Data export feature coming soon")
            
            with export_col3:
                if st.button("📧 Email Report"):
                    st.info("Email feature coming soon")

def render_basic_analytics_tab(user_profile: UserProfile):
    """Fallback basic analytics when enhanced features unavailable."""
    
    st.markdown("### 📊 Basic Analytics")
    st.info("Enhanced analytics features are not available. Please install required dependencies for full functionality.")
    
    # Basic metrics
    st.markdown("#### 📈 Basic Progress Tracking")
    
    basic_cols = st.columns(3)
    
    with basic_cols[0]:
        st.metric("Workouts This Month", "12", "+3")
    
    with basic_cols[1]:
        st.metric("Current Streak", "7 days", "+1")
    
    with basic_cols[2]:
        st.metric("Goals Achieved", "5", "+2")
    
    # Simple progress visualization
    st.markdown("#### 📊 Simple Progress Chart")
    
    # Create basic chart
    import plotly.express as px
    import pandas as pd
    
    # Sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-22', freq='D')
    progress_data = pd.DataFrame({
        'Date': dates,
        'Workout_Completed': [1 if i % 3 != 0 else 0 for i in range(len(dates))],
        'Satisfaction': [7 + (i % 4) + np.random.randint(-1, 2) for i in range(len(dates))]
    })
    
    fig = px.bar(progress_data, x='Date', y='Workout_Completed', 
                 title="Daily Workout Completion")
    st.plotly_chart(fig, use_container_width=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    
    # User profile and authentication
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    
    # Application state
    if 'profile_complete' not in st.session_state:
        st.session_state.profile_complete = False
    
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    if 'recommendations_cache' not in st.session_state:
        st.session_state.recommendations_cache = {}
    
    # Navigation state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    # Enhanced features session state
    if 'show_profile_creation' not in st.session_state:
        st.session_state.show_profile_creation = False
    
    if 'progress_photos_with_analysis' not in st.session_state:
        st.session_state.progress_photos_with_analysis = []
    
    if 'video_form_analysis' not in st.session_state:
        st.session_state.video_form_analysis = []
    
    if 'workout_variations_cache' not in st.session_state:
        st.session_state.workout_variations_cache = {}

def create_new_profile_callback():
    """Callback function to trigger new profile creation."""
    st.session_state.show_profile_creation = True
    st.session_state.user_profile = None
    st.session_state.profile_complete = False
    st.rerun()

def check_user_profile():
    """Check if user has a complete profile."""
    
    if st.session_state.user_profile is None:
        return False
    
    # Check for required profile fields
    required_fields = ['age', 'weight', 'height', 'gender', 'activity_level', 'fitness_level']
    profile_dict = st.session_state.user_profile.to_dict()
    
    return all(profile_dict.get(field) is not None for field in required_fields)

def render_profile_setup():
    """Render profile setup interface."""
    
    st.header("👤 Complete Your Profile")
    st.info("Please complete your profile to get personalized recommendations and track your progress.")
    
    # Simple profile form
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
            gender = st.selectbox("Gender", ["male", "female"])
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
            activity_level = st.selectbox("Activity Level", [
                "sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"
            ])
            fitness_level = st.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"])
        
        primary_goal = st.selectbox("Primary Goal", [
            "weight_loss", "muscle_gain", "endurance", "strength", "general_fitness"
        ])
        
        submitted = st.form_submit_button("Create Profile")
        
        if submitted:
            try:
                new_profile = create_user_profile(
                    age=age,
                    gender=Gender(gender),
                    height=height,
                    weight=weight,
                    activity_level=ActivityLevel(activity_level),
                    fitness_level=FitnessLevel(fitness_level),
                    primary_goal=GoalType(primary_goal)
                )
                
                st.session_state.user_profile = new_profile
                st.session_state.user_id = new_profile.user_id
                st.session_state.profile_complete = True
                st.success("Profile created successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating profile: {e}")

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Setup page configuration
    st.set_page_config(
        page_title="AI Fitness Assistant Pro",
        page_icon="🏋️‍♀️"
    )
    
    # Render header
    st.title("🏋️‍♀️ AI Fitness Assistant Pro")
    st.subheader("Your Intelligent Fitness Companion")
    
    # Handle profile creation callback
    if st.session_state.get('show_profile_creation', False):
        render_profile_setup()
        return
    
    # Check if user has completed profile
    if not check_user_profile():
        render_profile_setup()
        return
    
    # Main application interface
    render_main_application()

def render_main_application():
    """Render the main application interface."""
    
    user_profile = st.session_state.user_profile
    
    # Sidebar with user info and navigation
    render_sidebar(user_profile)
    
    # Main content area with enhanced tabs
    tabs = st.tabs([
        "📊 Body Analysis", 
        "💪 Recommendations", 
        "📅 Workout Planner",
        "� Nutrition & Meals",
        "👥 Community & Social",
        "📈 Advanced Analytics",
        "🎯 Form Correction",
        "🏆 Goal Management"
    ])
    
    with tabs[0]:
        render_body_analysis_tab(user_profile)
    
    with tabs[1]:
        render_recommendations_tab(user_profile)
    
    with tabs[2]:
        render_workout_planner_tab(user_profile)
    
    with tabs[3]:
        render_nutrition_tab(user_profile)
    
    with tabs[4]:
        render_social_tab(user_profile)
    
    with tabs[5]:
        render_analytics_tab(user_profile)
    
    with tabs[6]:
        render_form_correction_tab(user_profile)
    
    with tabs[7]:
        render_goal_management_tab(user_profile)

def render_sidebar(user_profile: UserProfile):
    """Render sidebar with user information and controls."""
    
    st.sidebar.header("👤 Your Profile")
    
    # User summary
    st.sidebar.write(f"**Age:** {user_profile.age}")
    st.sidebar.write(f"**BMI:** {user_profile.bmi} ({user_profile.bmi_category})")
    st.sidebar.write(f"**Fitness Level:** {user_profile.fitness_level.value.title()}")
    st.sidebar.write(f"**Primary Goal:** {user_profile.primary_goal.value.replace('_', ' ').title()}")
    
    st.sidebar.divider()
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Update Profile"):
        st.session_state.profile_complete = False
        st.rerun()
    
    if st.sidebar.button("� Create New Profile", on_click=create_new_profile_callback):
        pass  # Callback handles the logic
    
    if st.sidebar.button("�📊 Generate New Recommendations"):
        # Clear recommendations cache
        st.session_state.recommendations_cache = {}
        st.success("Recommendations will be regenerated!")
    
    if st.sidebar.button("📥 Export Data"):
        db = get_database()
        user_data = db.export_user_data(user_profile.user_id)
        if user_data:
            st.sidebar.download_button(
                "💾 Download Data",
                data=json.dumps(user_data, indent=2),
                file_name=f"fitness_data_{user_profile.user_id}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    st.sidebar.markdown("---")
    
    # Application info
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown("**AI Fitness Assistant Pro v3.0**")
    st.sidebar.markdown("Your intelligent fitness companion")
    st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/alphareum/apt-proof-of-concept)")
    st.sidebar.markdown("[🐛 Report Issues](https://github.com/alphareum/apt-proof-of-concept/issues)")
    st.sidebar.markdown(f"Profile ID: `{user_profile.user_id[:8]}...`")

def render_body_analysis_tab(user_profile: UserProfile):
    """Render body analysis tab."""
    
    st.markdown("## 📊 Body Composition Analysis")
    
    analysis_tabs = st.tabs(["📸 Image Analysis", "📏 Manual Measurements", "📈 Progress History"])
    
    with analysis_tabs[0]:
        render_image_analysis(user_profile)
    
    with analysis_tabs[1]:
        render_manual_measurements(user_profile)
    
    with analysis_tabs[2]:
        render_measurement_history(user_profile)

def render_image_analysis(user_profile: UserProfile):
    """Render image-based body analysis."""
    
    st.markdown("### 📸 Upload Body Photo for Analysis")
    st.info("Upload a clear, full-body photo for AI-powered body composition analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="For best results, use good lighting and wear fitted clothing"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Body Composition", type="primary"):
            with st.spinner("🤖 Analyzing image..."):
                try:
                    # Convert to bytes for analysis
                    img_bytes = uploaded_file.getvalue()
                    
                    # Perform analysis (simplified version)
                    analysis_result = analyze_body_image(img_bytes, user_profile)
                    
                    if 'error' in analysis_result:
                        st.error(f"❌ Analysis failed: {analysis_result['error']}")
                    else:
                        display_analysis_results(analysis_result, user_profile)
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

def analyze_body_image(img_bytes: bytes, user_profile: UserProfile) -> Dict[str, Any]:
    """Analyze body composition from image using advanced CV."""
    
    if not BODY_COMP_AVAILABLE:
        return {
            'error': 'Body composition analysis not available',
            'message': 'Please install required dependencies: opencv-python, mediapipe, scikit-learn'
        }
    
    try:
        # Save uploaded image temporarily
        import tempfile
        import hashlib
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_analysis_{timestamp}_{hashlib.md5(img_bytes).hexdigest()[:8]}.jpg"
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / filename
        
        # Save image
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        
        # Perform analysis using the body composition analyzer
        analyzer = get_body_analyzer()
        analysis_result = analyzer.analyze_image(
            image_path=str(temp_path),
            user_id=user_profile.user_id
        )
        
        # Clean up temporary file
        try:
            temp_path.unlink()
        except:
            pass
        
        if analysis_result.get('success', False):
            return {
                'success': True,
                'analysis_id': analysis_result['analysis_id'],
                'estimated_body_fat': analysis_result['body_fat_percentage'],
                'muscle_mass_percentage': analysis_result['muscle_mass_percentage'],
                'visceral_fat_level': analysis_result['visceral_fat_level'],
                'bmr_estimated': analysis_result['bmr_estimated'],
                'body_shape': analysis_result['body_shape'],
                'confidence': analysis_result['confidence'],
                'measurements': analysis_result.get('measurements', {}),
                'breakdown': analysis_result.get('breakdown', {}),
                'processed_image_path': analysis_result.get('processed_image_path', ''),
                'recommendations': generate_recommendations_from_analysis(analysis_result, user_profile)
            }
        else:
            return {
                'error': analysis_result.get('error', 'Analysis failed'),
                'confidence': 0.0
            }
        
    except Exception as e:
        return {'error': f'Analysis error: {str(e)}'}

def generate_recommendations_from_analysis(analysis: Dict[str, Any], user_profile: UserProfile) -> List[str]:
    """Generate personalized recommendations based on body composition analysis."""
    
    recommendations = []
    
    body_fat = analysis.get('body_fat_percentage', 0)
    muscle_mass = analysis.get('muscle_mass_percentage', 0)
    visceral_fat = analysis.get('visceral_fat_level', 0)
    
    # Body fat recommendations
    if body_fat < 10:
        recommendations.append("⚠️ Body fat is very low - consider consulting a nutritionist")
    elif body_fat < 15:
        recommendations.append("💪 Excellent body fat level - focus on maintaining muscle mass")
    elif body_fat < 25:
        recommendations.append("✅ Healthy body fat range - continue current routine")
    elif body_fat < 30:
        recommendations.append("🏃 Consider increasing cardio and reducing calorie intake")
    else:
        recommendations.append("🎯 Focus on weight loss through diet and exercise combination")
    
    # Muscle mass recommendations
    if muscle_mass < 30:
        recommendations.append("🏋️ Add strength training to build muscle mass")
    elif muscle_mass > 45:
        recommendations.append("💪 Excellent muscle mass - focus on maintenance")
    else:
        recommendations.append("⚖️ Good muscle mass - continue strength training")
    
    # Visceral fat recommendations
    if visceral_fat > 15:
        recommendations.append("⚠️ High visceral fat - prioritize cardio and core exercises")
    elif visceral_fat > 10:
        recommendations.append("🏃 Moderate visceral fat - add more aerobic exercise")
    else:
        recommendations.append("✅ Good visceral fat level - maintain current activity")
    
    # Age-specific recommendations
    if user_profile.age > 40:
        recommendations.append("🧘 Consider adding flexibility and balance training")
    
    # Goal-specific recommendations
    if hasattr(user_profile, 'primary_goal'):
        if 'weight_loss' in str(user_profile.primary_goal).lower():
            recommendations.append("🍎 Focus on creating a moderate caloric deficit")
        elif 'muscle' in str(user_profile.primary_goal).lower():
            recommendations.append("🥩 Ensure adequate protein intake (0.8-1g per lb bodyweight)")
    
    return recommendations

def estimate_body_fat_from_profile(user_profile: UserProfile) -> float:
    """Estimate body fat percentage based on user profile."""
    
    # Basic estimation using BMI and demographic factors
    bmi = user_profile.bmi
    age = user_profile.age
    
    if user_profile.gender == Gender.MALE:
        base_bf = (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        base_bf = (1.20 * bmi) + (0.23 * age) - 5.4
    
    # Adjust for activity level
    activity_adjustments = {
        ActivityLevel.SEDENTARY: 1.2,
        ActivityLevel.LIGHTLY_ACTIVE: 1.0,
        ActivityLevel.MODERATELY_ACTIVE: 0.9,
        ActivityLevel.VERY_ACTIVE: 0.8,
        ActivityLevel.EXTREMELY_ACTIVE: 0.7
    }
    
    adjustment = activity_adjustments.get(user_profile.activity_level, 1.0)
    estimated_bf = base_bf * adjustment
    
    return max(3, min(50, estimated_bf))

def display_analysis_results(results: Dict[str, Any], user_profile: UserProfile):
    """Display comprehensive body composition analysis results."""
    
    if results.get('success', False):
        st.success("✅ Analysis completed successfully!")
        
        # Main metrics
        st.markdown("### 📊 Body Composition Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Body Fat %",
                f"{results['estimated_body_fat']:.1f}%",
                help="AI-estimated body fat percentage"
            )
        
        with col2:
            st.metric(
                "Muscle Mass %",
                f"{results['muscle_mass_percentage']:.1f}%",
                help="Estimated muscle mass percentage"
            )
        
        with col3:
            st.metric(
                "BMR",
                f"{results['bmr_estimated']} cal/day",
                help="Estimated Basal Metabolic Rate"
            )
        
        with col4:
            st.metric(
                "Confidence",
                f"{results['confidence']:.2f}",
                help="Analysis confidence score"
            )
        
        # Additional details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Body Assessment")
            st.write(f"**Body Shape:** {results['body_shape']}")
            st.write(f"**Visceral Fat Level:** {results['visceral_fat_level']}/20")
            
            # Health assessment
            body_fat = results['estimated_body_fat']
            if body_fat < 10:
                health_status = "⚠️ Very Low (may be unhealthy)"
            elif body_fat < 15:
                health_status = "✅ Athletic"
            elif body_fat < 25:
                health_status = "✅ Healthy"
            elif body_fat < 30:
                health_status = "⚠️ Above Average"
            else:
                health_status = "🔴 High (health risk)"
            
            st.write(f"**Health Status:** {health_status}")
        
        with col2:
            if "breakdown" in results:
                st.markdown("### 📈 Composition Breakdown")
                breakdown = results["breakdown"]
                
                st.write(f"**Fat Mass:** {breakdown.get('fat_mass_kg', 0):.1f} kg")
                st.write(f"**Muscle Mass:** {breakdown.get('muscle_mass_kg', 0):.1f} kg")
                st.write(f"**Bone Mass:** {breakdown.get('bone_mass_kg', 0):.1f} kg")
                st.write(f"**Water %:** {breakdown.get('water_percentage', 0):.1f}%")
        
        # Body measurements
        if "measurements" in results:
            st.markdown("### 📏 Body Measurements")
            measurements = results["measurements"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Upper Body**")
                st.write(f"Shoulder Width: {measurements.get('shoulder_width', 0):.1f} px")
                st.write(f"Arm Length: {measurements.get('left_arm_length', 0):.1f} px")
            
            with col2:
                st.write("**Core**")
                st.write(f"Waist Width: {measurements.get('waist_width', 0):.1f} px")
                st.write(f"Hip Width: {measurements.get('hip_width', 0):.1f} px")
            
            with col3:
                st.write("**Lower Body**")
                st.write(f"Body Height: {measurements.get('body_height', 0):.1f} px")
                st.write(f"Leg Length: {measurements.get('left_leg_length', 0):.1f} px")
        
        # Processed image
        if results.get("processed_image_path"):
            st.markdown("### 🖼️ Analysis Visualization")
            try:
                processed_image = Image.open(results["processed_image_path"])
                st.image(processed_image, caption="Analysis Visualization", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display processed image: {e}")
        
        # Recommendations
        if results.get('recommendations'):
            st.markdown("### 💡 Personalized Recommendations")
            for rec in results['recommendations']:
                st.markdown(f"• {rec}")
        
        # Save results to database
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("💾 Save Results", type="primary"):
                if save_analysis_to_database(results, user_profile):
                    st.success("✅ Analysis saved!")
                    st.balloons()
                else:
                    st.error("❌ Failed to save analysis")
    
    else:
        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        st.info("💡 Tips for better results:")
        st.markdown("""
        - Use good lighting (natural light preferred)
        - Wear fitted clothing
        - Stand straight facing camera
        - Ensure full body is visible
        - Use consistent conditions for progress tracking
        """)

def save_analysis_to_database(results: Dict[str, Any], user_profile: UserProfile) -> bool:
    """Save body composition analysis results to database."""
    
    try:
        db = get_database()
        
        # If we have a full analysis with analysis_id, it's already saved
        if results.get('analysis_id'):
            st.info("Analysis already saved with comprehensive data")
            return True
        
        # Otherwise, save as a basic body measurement for backward compatibility
        from database import BodyMeasurement
        import uuid
        from datetime import datetime
        
        measurement = BodyMeasurement(
            measurement_id=str(uuid.uuid4()),
            user_id=user_profile.user_id,
            date=datetime.now(),
            body_fat_percentage=results.get('estimated_body_fat'),
            muscle_mass=results.get('muscle_mass_percentage')
        )
        
        return db.save_body_measurement(measurement)
        
    except Exception as e:
        st.error(f"Error saving analysis: {e}")
        return False

def render_manual_measurements(user_profile: UserProfile):
    """Render manual measurements input."""
    
    st.markdown("### 📏 Enter Manual Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=float(user_profile.weight), step=0.1)
        waist = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, value=80.0, step=0.5)
        chest = st.number_input("Chest (cm)", min_value=60.0, max_value=200.0, value=90.0, step=0.5)
        neck = st.number_input("Neck (cm)", min_value=25.0, max_value=60.0, value=35.0, step=0.5)
    
    with col2:
        body_fat = st.number_input("Body Fat % (if known)", min_value=3.0, max_value=50.0, value=15.0, step=0.1)
        arms = st.number_input("Arms (cm)", min_value=15.0, max_value=60.0, value=30.0, step=0.5)
        thighs = st.number_input("Thighs (cm)", min_value=30.0, max_value=100.0, value=55.0, step=0.5)
        hips = st.number_input("Hips (cm)", min_value=60.0, max_value=200.0, value=90.0, step=0.5)
    
    measurement_notes = st.text_area("Notes", placeholder="Any additional notes about measurements or conditions...")
    
    if st.button("💾 Save Measurements", type="primary"):
        save_manual_measurements(user_profile, {
            'weight': weight,
            'waist': waist,
            'chest': chest,
            'neck': neck,
            'body_fat_percentage': body_fat,
            'arms': arms,
            'thighs': thighs,
            'hips': hips,
            'notes': measurement_notes
        })
        st.success("✅ Measurements saved successfully!")

def save_manual_measurements(user_profile: UserProfile, measurements: Dict[str, Any]):
    """Save manual measurements to database."""
    
    db = get_database()
    
    from models import BodyMeasurements
    import uuid
    
    measurement = BodyMeasurements(
        id=str(uuid.uuid4()),
        user_id=user_profile.user_id,
        weight=measurements.get('weight'),
        waist=measurements.get('waist'),
        chest=measurements.get('chest'),
        neck=measurements.get('neck'),
        body_fat_percentage=measurements.get('body_fat_percentage'),
        arms=measurements.get('arms'),
        thighs=measurements.get('thighs'),
        hips=measurements.get('hips'),
        notes=measurements.get('notes')
    )
    
    db.save_body_measurement(measurement)

def render_measurement_history(user_profile: UserProfile):
    """Render measurement history and progress including body composition analysis."""
    
    st.markdown("### 📈 Measurement History & Progress")
    
    # Create tabs for different types of history
    history_tabs = st.tabs(["📊 Body Composition", "📏 Manual Measurements", "📈 Progress Charts"])
    
    with history_tabs[0]:
        render_body_composition_history(user_profile)
    
    with history_tabs[1]:
        render_manual_measurement_history(user_profile)
    
    with history_tabs[2]:
        render_progress_charts(user_profile)

def render_body_composition_history(user_profile: UserProfile):
    """Render body composition analysis history."""
    
    st.markdown("#### 🏋️ Body Composition Analysis History")
    
    if BODY_COMP_AVAILABLE:
        db = get_database()
        
        # Get analysis history
        days = st.selectbox("Show data for last:", [30, 60, 90, 180, 365], index=2, key="comp_history_days")
        history = db.get_body_composition_history(user_profile.user_id, days)
        
        if history:
            st.success(f"Found {len(history)} analyses in the last {days} days")
            
            # Display history table
            display_data = []
            for analysis in history:
                display_data.append({
                    "Date": analysis["analysis_date"][:10],
                    "Body Fat %": f"{analysis['body_fat_percentage']:.1f}%",
                    "Muscle Mass %": f"{analysis['muscle_mass_percentage']:.1f}%",
                    "BMR": f"{analysis['bmr_estimated']} cal",
                    "Body Shape": analysis["body_shape_classification"],
                    "Confidence": f"{analysis['confidence_score']:.2f}"
                })
            
            st.dataframe(display_data, use_container_width=True)
            
            # Show progress if enough data
            if len(history) >= 2:
                st.markdown("#### 📈 Progress Analysis")
                progress = db.calculate_composition_progress(user_profile.user_id, days)
                
                if "error" not in progress:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        body_fat_change = progress.get("body_fat_change", 0)
                        delta_color = "inverse" if body_fat_change < 0 else "normal"
                        st.metric(
                            "Body Fat Change",
                            f"{body_fat_change:+.1f}%",
                            delta=f"{body_fat_change:+.1f}%",
                            delta_color=delta_color
                        )
                    
                    with col2:
                        muscle_change = progress.get("muscle_mass_change", 0)
                        delta_color = "normal" if muscle_change > 0 else "inverse"
                        st.metric(
                            "Muscle Mass Change",
                            f"{muscle_change:+.1f}%",
                            delta=f"{muscle_change:+.1f}%",
                            delta_color=delta_color
                        )
                    
                    with col3:
                        bmr_change = progress.get("bmr_change", 0)
                        st.metric(
                            "BMR Change",
                            f"{bmr_change:+.0f} cal/day",
                            delta=f"{bmr_change:+.0f} cal"
                        )
                    
                    # Trend analysis
                    if "trend_analysis" in progress:
                        trends = progress["trend_analysis"]
                        st.markdown("##### 🔍 Trend Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            body_fat_trend = trends.get("body_fat", "stable")
                            if body_fat_trend == "decreasing":
                                st.success("🔥 Body fat decreasing")
                            elif body_fat_trend == "increasing":
                                st.warning("📈 Body fat increasing")
                            else:
                                st.info("➡️ Body fat stable")
                        
                        with col2:
                            muscle_trend = trends.get("muscle_mass", "stable")
                            if muscle_trend == "increasing":
                                st.success("💪 Muscle mass increasing")
                            elif muscle_trend == "decreasing":
                                st.warning("📉 Muscle mass decreasing")
                            else:
                                st.info("➡️ Muscle mass stable")
                        
                        with col3:
                            overall_trend = trends.get("overall", "stable")
                            if overall_trend == "improving":
                                st.success("🎯 Overall improving")
                            else:
                                st.info("➡️ Overall stable")
        else:
            st.info("No body composition analyses found. Upload a photo for analysis!")
    else:
        st.warning("Body composition analysis not available. Install required dependencies.")

def render_manual_measurement_history(user_profile: UserProfile):
    """Render manual measurement history."""
    
    st.markdown("#### 📏 Manual Measurements History")
    
    db = get_database()
    measurements_df = db.get_measurement_history(user_profile.user_id)
    
    if not measurements_df.empty:
        # Show recent measurements
        recent_measurements = measurements_df.tail(10).sort_values('date', ascending=False)
        st.dataframe(recent_measurements, use_container_width=True)
    else:
        st.info("No manual measurements recorded. Add measurements in the Manual Measurements tab!")

def render_progress_charts(user_profile: UserProfile):
    """Render progress visualization charts."""
    
    st.markdown("#### 📈 Progress Visualization")
    
    try:
        db = get_database()
        
        # Body composition charts
        if BODY_COMP_AVAILABLE:
            history = db.get_body_composition_history(user_profile.user_id, 180)
            
            if len(history) >= 2:
                # Prepare data for plotting
                dates = [analysis["analysis_date"][:10] for analysis in reversed(history)]
                body_fat = [analysis["body_fat_percentage"] for analysis in reversed(history)]
                muscle_mass = [analysis["muscle_mass_percentage"] for analysis in reversed(history)]
                bmr = [analysis["bmr_estimated"] for analysis in reversed(history)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Body Fat % Trend**")
                    chart_data = {"Date": dates, "Body Fat %": body_fat}
                    st.line_chart(chart_data, x="Date", y="Body Fat %")
                
                with col2:
                    st.markdown("**Muscle Mass % Trend**")
                    chart_data = {"Date": dates, "Muscle Mass %": muscle_mass}
                    st.line_chart(chart_data, x="Date", y="Muscle Mass %")
                
                # BMR trend
                st.markdown("**BMR Trend**")
                chart_data = {"Date": dates, "BMR": bmr}
                st.line_chart(chart_data, x="Date", y="BMR")
            else:
                st.info("Need at least 2 body composition analyses to show trends")
        
        # Manual measurements charts
        measurements_df = db.get_measurement_history(user_profile.user_id)
        if not measurements_df.empty and 'weight' in measurements_df.columns:
            st.markdown("**Weight Progress**")
            weight_data = measurements_df[['date', 'weight']].dropna()
            if not weight_data.empty:
                st.line_chart(weight_data.set_index('date'))
        
    except Exception as e:
        st.error(f"Error creating charts: {e}")

def create_measurement_progress_chart(df):
    """Create measurement progress chart."""
    try:
        import plotly.express as px
        
        if 'weight' in df.columns and not df['weight'].isna().all():
            fig = px.line(df, x='date', y='weight', title='Weight Progress Over Time')
            fig.update_xaxis(title='Date')
            fig.update_yaxis(title='Weight (kg)')
            return fig
    except ImportError:
        st.warning("Plotly not available for charts. Install plotly for better visualizations.")
    
    return None

def render_recommendations_tab(user_profile: UserProfile):
    """Render exercise recommendations tab."""
    
    st.header("💪 Exercise Recommendations")
    st.info("Getting personalized exercise recommendations based on your profile...")
    
    # Check if enhanced recommendations are available
    if ENHANCED_PLANNER_AVAILABLE:
        try:
            # Use enhanced recommendation engine
            enhanced_engine = EnhancedRecommendationEngine()
            enhanced_recommendations = enhanced_engine.generate_adaptive_recommendations(user_profile)
            
            if 'error' not in enhanced_recommendations:
                render_enhanced_recommendations(enhanced_recommendations, user_profile)
            else:
                st.error(f"Error generating enhanced recommendations: {enhanced_recommendations['error']}")
                render_basic_recommendations(user_profile)
        except Exception as e:
            st.warning(f"Enhanced recommendations unavailable: {str(e)}")
            render_basic_recommendations(user_profile)
    else:
        render_basic_recommendations(user_profile)

def render_enhanced_recommendations(recommendations: Dict[str, Any], user_profile: UserProfile):
    """Render enhanced AI-powered recommendations with 4 distinct program options."""
    
    st.markdown("### 🤖 AI-Powered Workout Recommendations")
    
    # Check if we have program options (4 distinct programs)
    program_options = recommendations.get('program_options', [])
    
    if program_options and len(program_options) >= 4:
        st.markdown("#### 🎯 Choose Your Program")
        st.info("Based on your profile, here are 4 distinctly different workout programs tailored for you:")
        
        # Program selection
        program_tabs = st.tabs([
            f"Option 1: {program_options[0]['name'][:30]}...",
            f"Option 2: {program_options[1]['name'][:30]}...", 
            f"Option 3: {program_options[2]['name'][:30]}...",
            f"Option 4: {program_options[3]['name'][:30]}..."
        ])
        
        for i, (tab, program) in enumerate(zip(program_tabs, program_options)):
            with tab:
                # Program overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**{program['name']}**")
                    st.write(program['description'])
                    
                    # Key details
                    st.markdown("**Program Details:**")
                    st.write(f"• Duration: {program['duration_per_session']} min per session")
                    st.write(f"• Frequency: {program['sessions_per_week']} sessions/week") 
                    st.write(f"• Style: {program['intensity_style'].replace('_', ' ').title()}")
                    st.write(f"• Equipment: {', '.join(program['equipment_needed'][:3])}")
                
                with col2:
                    # Differentiation score
                    diff_score = program.get('differentiation_score', 0.5)
                    st.metric("Uniqueness", f"{diff_score*100:.0f}%")
                    
                    # Best for
                    st.markdown("**Best For:**")
                    for item in program.get('best_for', [])[:3]:
                        st.write(f"• {item}")
                
                # Pros and Cons
                pros_cons_col1, pros_cons_col2 = st.columns(2)
                
                with pros_cons_col1:
                    st.markdown("**✅ Advantages:**")
                    for pro in program.get('pros', [])[:4]:
                        st.success(f"• {pro}")
                
                with pros_cons_col2:
                    st.markdown("**⚠️ Considerations:**")
                    for con in program.get('cons', [])[:4]:
                        st.warning(f"• {con}")
                
                # Sample workout
                if 'sample_week' in program:
                    st.markdown("**📋 Sample Week Overview:**")
                    sample_week = program['sample_week']
                    
                    if 'monday' in sample_week:
                        monday_workout = sample_week['monday']
                        with st.expander(f"Sample: {monday_workout['workout_name']}", expanded=False):
                            st.write(f"**Duration:** {monday_workout['duration']} minutes")
                            st.write(f"**Structure:** {monday_workout['structure']}")
                            st.write(f"**Focus:** {monday_workout['focus']}")
                            
                            if 'exercises' in monday_workout:
                                st.markdown("**Key Exercises:**")
                                for ex in monday_workout['exercises'][:3]:
                                    st.write(f"• {ex['name']}: {ex.get('sets', '')} sets x {ex.get('reps', ex.get('duration', 'N/A'))}")
                
                # Selection button
                if st.button(f"Select Option {i+1}", key=f"select_program_{i}"):
                    st.session_state.selected_program = program
                    st.success(f"✅ Selected: {program['name']}")
                    st.rerun()
        
        # Program comparison
        if st.checkbox("📊 Compare All Programs"):
            comparison = recommendations.get('program_comparison', {})
            if comparison:
                st.markdown("#### � Program Comparison")
                
                # Create comparison table
                comparison_data = []
                for i, program in enumerate(program_options):
                    comparison_data.append({
                        'Program': f"Option {i+1}",
                        'Name': program['name'][:40],
                        'Duration': f"{program['duration_per_session']} min",
                        'Frequency': f"{program['sessions_per_week']}/week",
                        'Equipment': ', '.join(program['equipment_needed'][:2]),
                        'Style': program['intensity_style'].replace('_', ' ').title()
                    })
                
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Key differences
                st.markdown("**🔍 Key Differences:**")
                for diff in comparison.get('key_differences', []):
                    st.write(f"• {diff}")
    
    # Show selected program details
    if 'selected_program' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Your Selected Program")
        
        selected = st.session_state.selected_program
        st.success(f"Currently using: **{selected['name']}**")
        
        # Show today's workout if available
        daily_workouts = recommendations.get('daily_workouts', {})
        if daily_workouts:
            st.markdown("#### Today's Workout")
            for workout_type, workout_details in daily_workouts.items():
                with st.expander(f"Today: {workout_type.replace('_', ' ').title()}", expanded=True):
                    if isinstance(workout_details, dict) and 'exercises' in workout_details:
                        for exercise in workout_details['exercises']:
                            st.markdown(f"**{exercise.get('name', 'Exercise')}**")
                            st.markdown(f"Sets: {exercise.get('sets', 'N/A')} | Reps: {exercise.get('reps', 'N/A')}")
                            
                            if exercise.get('technique_cues'):
                                st.caption("💡 " + ", ".join(exercise['technique_cues'][:2]))
    
    # Workout varieties section  
    workout_varieties = recommendations.get('workout_varieties', {})
    if workout_varieties:
        st.markdown("### 🔄 Alternative Workout Options")
        st.info("When you want to mix things up, try these alternatives:")
        
        variety_cols = st.columns(2)
        for i, (variety_name, variety_details) in enumerate(workout_varieties.items()):
            col = variety_cols[i % 2]
            
            with col:
                with st.expander(f"🔄 {variety_details.get('name', variety_name)}", expanded=False):
                    st.write(variety_details.get('description', 'Alternative workout option'))
                    st.write(f"**Duration:** {variety_details.get('duration', 'N/A')} minutes")
                    st.write(f"**Calorie Burn:** {variety_details.get('calorie_burn', 'Moderate')}")
                    st.write(f"**Best When:** {variety_details.get('best_when', 'Anytime')}")
                    
                    # Show sample workout
                    if 'sample_workout' in variety_details:
                        sample = variety_details['sample_workout']
                        if 'exercises' in sample:
                            st.markdown("**Sample Exercises:**")
                            for ex in sample['exercises'][:3]:
                                st.write(f"• {ex}")
    
    # User analysis insights
    user_analysis = recommendations.get('user_analysis', {})
    if user_analysis:
        st.markdown("### 🧠 AI Analysis Insights")
        
        analysis_cols = st.columns(2)
        with analysis_cols[0]:
            fitness_assessment = user_analysis.get('fitness_assessment', {})
            if fitness_assessment:
                st.markdown("**Fitness Assessment:**")
                for key, value in fitness_assessment.items():
                    st.markdown(f"• {key.replace('_', ' ').title()}: {value}")
        
        with analysis_cols[1]:
            motivation_factors = user_analysis.get('motivation_factors', [])
            if motivation_factors:
                st.markdown("**Motivation Factors:**")
                for factor in motivation_factors:
                    st.markdown(f"• {factor}")

def render_basic_recommendations(user_profile: UserProfile):
    """Render basic recommendations as fallback."""
    
    st.markdown("### 📋 Basic Recommendations")
    
    # Basic recommendations display
    recommendations = generate_basic_recommendations(user_profile)
    for rec in recommendations:
        st.subheader(rec['name'])
        st.write(f"**Type:** {rec['type']}")
        st.write(f"**Duration:** {rec['duration']}")
        st.write(f"**Description:** {rec['description']}")
        st.divider()

def render_workout_planner_tab(user_profile: UserProfile):
    """Render the enhanced workout planner tab."""
    
    if not ENHANCED_PLANNER_AVAILABLE:
        st.error("❌ Enhanced workout planner is not available. Please ensure all required dependencies are installed.")
        return
    
    try:
        # Initialize the workout planner UI
        planner_ui = WorkoutPlannerUI()
        
        # Render the complete workout planner interface
        planner_ui.render_workout_planner_tab(user_profile)
        
    except Exception as e:
        st.error(f"Error loading workout planner: {str(e)}")
        st.markdown("### 📅 Basic Workout Planning")
        st.info("The enhanced workout planner is temporarily unavailable. Here's a basic weekly schedule:")
        
        # Fallback basic schedule
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days:
            if day in ['Tuesday', 'Thursday', 'Sunday']:
                st.markdown(f"**{day}:** Rest Day")
            else:
                workout_type = "Cardio" if day in ['Monday', 'Friday'] else "Strength"
                st.markdown(f"**{day}:** {workout_type} Training - {user_profile.available_time} minutes")

def render_form_correction_tab(user_profile: UserProfile):
    """Render form correction tab."""
    
    st.markdown("## 🎯 Workout Form Correction")
    
    # Simplified form correction interface
    render_simplified_form_correction()

def render_simplified_form_correction():
    """Render simplified form correction interface."""
    
    st.info("📹 Form correction feature helps you perfect your exercise technique using AI pose detection.")
    
    exercise_type = st.selectbox(
        "Select Exercise",
        ["Squat", "Push-up", "Deadlift", "Plank", "Lunge"],
        help="Choose the exercise you want to analyze"
    )
    
    st.markdown(f"### 🏋️‍♀️ {exercise_type} Form Guide")
    
    # Exercise-specific tips
    form_tips = {
        "Squat": [
            "Keep your chest up and core engaged",
            "Knees should track over your toes",
            "Lower until thighs are parallel to ground",
            "Drive through your heels to stand up"
        ],
        "Push-up": [
            "Maintain straight line from head to heels",
            "Hands slightly wider than shoulders",
            "Lower chest to nearly touch ground",
            "Keep core tight throughout movement"
        ],
        "Deadlift": [
            "Keep the bar close to your body",
            "Hinge at hips, not knees",
            "Maintain neutral spine",
            "Drive hips forward to lift"
        ],
        "Plank": [
            "Body should form straight line",
            "Engage core and glutes",
            "Don't let hips sag or pike up",
            "Breathe normally throughout hold"
        ],
        "Lunge": [
            "Step forward with control",
            "Keep front knee over ankle",
            "Lower back knee toward ground",
            "Push off front foot to return"
        ]
    }
    
    for tip in form_tips.get(exercise_type, []):
        st.markdown(f"✅ {tip}")
    
    st.info("💡 Upload a video or use your camera for real-time form analysis (feature coming soon)!")

def render_progress_tracking_tab(user_profile: UserProfile):
    """Render progress tracking tab with body composition analysis."""
    
    st.markdown("## 📊 Progress Tracking")
    
    # Create tabs for different progress views
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🔄 Body Composition Progress", "📋 Detailed Analysis"])
    
    with tab1:
        st.subheader("Progress Overview")
        st.info("Your progress tracking will be displayed here.")
        
        # Basic progress metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Workouts This Week", "3")
        col2.metric("Total Calories Burned", "450")
        col3.metric("Streak Days", "7")
    
    with tab2:
        render_body_composition_progress(user_profile)
    
    with tab3:
        render_detailed_analysis_comparison(user_profile)

def render_goal_management_tab(user_profile: UserProfile):
    """Render goal management tab."""
    
    # Simplified goal management interface
    render_simplified_goal_management(user_profile)

def render_simplified_goal_management(user_profile: UserProfile):
    """Render simplified goal management interface."""
    
    st.markdown("## 🏆 Goal Management")
    
    st.info("Set and track your fitness goals to stay motivated and measure progress.")
    
    # Goal creation
    st.markdown("### ➕ Create New Goal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        goal_type = st.selectbox(
            "Goal Type",
            ["Weight Loss", "Muscle Gain", "Strength", "Endurance", "Body Fat Reduction"],
            help="Choose your primary goal type"
        )
        
        target_value = st.number_input(
            "Target Value",
            min_value=0.0,
            value=10.0,
            step=0.5,
            help="Your target value (kg, %, etc.)"
        )
    
    with col2:
        current_value = st.number_input(
            "Current Value",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Your current progress value"
        )
        
        target_date = st.date_input(
            "Target Date",
            value=datetime.now().date() + timedelta(days=90),
            help="When you want to achieve this goal"
        )
    
    goal_description = st.text_area(
        "Goal Description",
        placeholder="Describe your goal in detail...",
        help="Add details about your goal for better tracking"
    )
    
    if st.button("🎯 Create Goal", type="primary"):
        # Save goal to database
        save_fitness_goal(user_profile, {
            'type': goal_type,
            'target_value': target_value,
            'current_value': current_value,
            'target_date': target_date,
            'description': goal_description
        })
        st.success("✅ Goal created successfully!")
    
    # Display existing goals
    st.markdown("### 📋 Your Goals")
    display_user_goals(user_profile)

def save_fitness_goal(user_profile: UserProfile, goal_data: Dict[str, Any]):
    """Save fitness goal to database."""
    
    db = get_database()
    db.save_user_goal(user_profile.user_id, goal_data)

def display_user_goals(user_profile: UserProfile):
    """Display user's fitness goals."""
    
    db = get_database()
    goals = db.get_user_goals(user_profile.user_id)
    
    if goals:
        for goal in goals:
            with st.expander(f"🎯 {goal['goal_type']} - {goal.get('target_value', 'No target')} by {goal.get('target_date', 'No date')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Current:** {goal.get('current_value', 0)}")
                    st.markdown(f"**Target:** {goal.get('target_value', 'Not set')}")
                
                with col2:
                    st.markdown(f"**Created:** {goal.get('created_at', 'Unknown')}")
                    st.markdown(f"**Status:** {goal.get('status', 'Active').title()}")
                
                # Progress calculation
                if goal.get('target_value') and goal.get('target_value') > 0:
                    progress = (goal.get('current_value', 0) / goal['target_value']) * 100
                    st.progress(min(progress / 100, 1.0))
                    st.markdown(f"Progress: {progress:.1f}%")
    else:
        st.info("No goals set yet. Create your first goal above!")

def render_body_composition_progress(user_profile: UserProfile):
    """Render body composition progress tracking."""
    
    if not BODY_COMP_AVAILABLE:
        st.warning("Body composition analysis not available. Please install required dependencies.")
        return
    
    st.markdown("### 🔄 Body Composition Progress")
    
    # Get historical data
    try:
        db = get_database()
        
        history = db.get_body_composition_history(user_profile.user_id, days=30)
        
        if not history:
            st.info("No body composition analyses found. Complete an analysis first to track progress.")
            return
        
        # Display latest analysis
        latest = history[0]
        st.markdown("#### Latest Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Body Fat %",
                f"{latest['body_fat_percentage']:.1f}%",
                delta=f"{latest['body_fat_percentage'] - (history[1]['body_fat_percentage'] if len(history) > 1 else latest['body_fat_percentage']):.1f}%" if len(history) > 1 else None
            )
        
        with col2:
            st.metric(
                "Muscle Mass %",
                f"{latest['muscle_mass_percentage']:.1f}%",
                delta=f"{latest['muscle_mass_percentage'] - (history[1]['muscle_mass_percentage'] if len(history) > 1 else latest['muscle_mass_percentage']):.1f}%" if len(history) > 1 else None
            )
        
        with col3:
            st.metric(
                "BMR",
                f"{int(latest['bmr_estimated'])} cal",
                delta=f"{int(latest['bmr_estimated'] - (history[1]['bmr_estimated'] if len(history) > 1 else latest['bmr_estimated']))}" if len(history) > 1 else None
            )
        
        with col4:
            st.metric(
                "Analyses",
                len(history),
                delta=None
            )
        
        # Progress charts
        if len(history) >= 2:
            st.markdown("#### Progress Charts")
            
            # Prepare data for plotting
            dates = [analysis['analysis_date'] for analysis in reversed(history)]
            body_fat = [analysis['body_fat_percentage'] for analysis in reversed(history)]
            muscle_mass = [analysis['muscle_mass_percentage'] for analysis in reversed(history)]
            bmr_values = [analysis['bmr_estimated'] for analysis in reversed(history)]
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Body composition chart
                fig_comp = plt.figure(figsize=(10, 6))
                plt.plot(dates, body_fat, marker='o', label='Body Fat %', color='red', alpha=0.7)
                plt.plot(dates, muscle_mass, marker='s', label='Muscle Mass %', color='green', alpha=0.7)
                plt.title('Body Composition Progress')
                plt.xlabel('Date')
                plt.ylabel('Percentage (%)')
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_comp)
            
            with col2:
                # BMR chart
                fig_bmr = plt.figure(figsize=(10, 6))
                plt.plot(dates, bmr_values, marker='o', label='BMR', color='blue', alpha=0.7)
                plt.title('Basal Metabolic Rate Progress')
                plt.xlabel('Date')
                plt.ylabel('Calories')
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_bmr)
        
        # Progress statistics
        if len(history) >= 2:
            progress_stats = db.calculate_composition_progress(user_profile.user_id)
            if progress_stats:
                st.markdown("#### Progress Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Body Fat Change**")
                    fat_change = progress_stats.get('body_fat_change', 0)
                    if fat_change < 0:
                        st.success(f"📉 Decreased by {abs(fat_change):.1f}%")
                    elif fat_change > 0:
                        st.warning(f"📈 Increased by {fat_change:.1f}%")
                    else:
                        st.info("➡️ No change")
                
                with col2:
                    st.markdown("**Muscle Mass Change**")
                    muscle_change = progress_stats.get('muscle_mass_change', 0)
                    if muscle_change > 0:
                        st.success(f"📈 Increased by {muscle_change:.1f}%")
                    elif muscle_change < 0:
                        st.warning(f"📉 Decreased by {abs(muscle_change):.1f}%")
                    else:
                        st.info("➡️ No change")
                
                with col3:
                    st.markdown("**BMR Change**")
                    bmr_change = progress_stats.get('bmr_change', 0)
                    if bmr_change > 0:
                        st.success(f"📈 Increased by {int(bmr_change)} cal")
                    elif bmr_change < 0:
                        st.warning(f"📉 Decreased by {int(abs(bmr_change))} cal")
                    else:
                        st.info("➡️ No change")
    
    except Exception as e:
        st.error(f"Error loading body composition progress: {str(e)}")

def render_detailed_analysis_comparison(user_profile: UserProfile):
    """Render detailed analysis comparison."""
    
    if not BODY_COMP_AVAILABLE:
        st.warning("Body composition analysis not available. Please install required dependencies.")
        return
    
    st.markdown("### 📋 Detailed Analysis Comparison")
    
    try:
        db = get_database()
        
        history = db.get_body_composition_history(user_profile.user_id, days=180)
        
        if len(history) < 2:
            st.info("At least 2 analyses are needed for comparison. Complete more analyses to see comparisons.")
            return
        
        # Analysis selection
        st.markdown("#### Select Analyses to Compare")
        
        col1, col2 = st.columns(2)
        
        # Prepare options
        analysis_options = {
            f"{analysis['analysis_date'][:19]} - {analysis['body_shape_classification']}": analysis
            for analysis in history
        }
        
        with col1:
            analysis1_key = st.selectbox(
                "First Analysis",
                options=list(analysis_options.keys()),
                index=0
            )
            analysis1 = analysis_options[analysis1_key]
        
        with col2:
            analysis2_key = st.selectbox(
                "Second Analysis",
                options=list(analysis_options.keys()),
                index=1 if len(analysis_options) > 1 else 0
            )
            analysis2 = analysis_options[analysis2_key]
        
        # Comparison display
        if analysis1 != analysis2:
            st.markdown("#### Comparison Results")
            
            # Create comparison table
            comparison_data = {
                "Metric": [
                    "Analysis Date",
                    "Body Fat %",
                    "Muscle Mass %",
                    "BMR (calories)",
                    "Body Shape",
                    "Health Assessment",
                    "Confidence Score"
                ],
                "First Analysis": [
                    analysis1['analysis_date'][:19],
                    f"{analysis1['body_fat_percentage']:.1f}%",
                    f"{analysis1['muscle_mass_percentage']:.1f}%",
                    f"{int(analysis1['bmr_estimated'])}",
                    analysis1['body_shape_classification'],
                    analysis1['health_assessment'],
                    f"{analysis1['confidence_score']:.1f}%"
                ],
                "Second Analysis": [
                    analysis2['analysis_date'][:19],
                    f"{analysis2['body_fat_percentage']:.1f}%",
                    f"{analysis2['muscle_mass_percentage']:.1f}%",
                    f"{int(analysis2['bmr_estimated'])}",
                    analysis2['body_shape_classification'],
                    analysis2['health_assessment'],
                    f"{analysis2['confidence_score']:.1f}%"
                ],
                "Change": [
                    "---",
                    f"{analysis2['body_fat_percentage'] - analysis1['body_fat_percentage']:+.1f}%",
                    f"{analysis2['muscle_mass_percentage'] - analysis1['muscle_mass_percentage']:+.1f}%",
                    f"{int(analysis2['bmr_estimated'] - analysis1['bmr_estimated']):+}",
                    "---" if analysis1['body_shape_classification'] == analysis2['body_shape_classification'] else f"{analysis1['body_shape_classification']} → {analysis2['body_shape_classification']}",
                    "---" if analysis1['health_assessment'] == analysis2['health_assessment'] else f"{analysis1['health_assessment']} → {analysis2['health_assessment']}",
                    f"{analysis2['confidence_score'] - analysis1['confidence_score']:+.1f}%"
                ]
            }
            
            import pandas as pd
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Highlight significant changes
            st.markdown("#### Key Changes")
            
            changes = []
            
            # Body fat change
            fat_diff = analysis2['body_fat_percentage'] - analysis1['body_fat_percentage']
            if abs(fat_diff) >= 1.0:
                if fat_diff > 0:
                    changes.append(f"🔴 Body fat increased by {fat_diff:.1f}%")
                else:
                    changes.append(f"🟢 Body fat decreased by {abs(fat_diff):.1f}%")
            
            # Muscle mass change
            muscle_diff = analysis2['muscle_mass_percentage'] - analysis1['muscle_mass_percentage']
            if abs(muscle_diff) >= 1.0:
                if muscle_diff > 0:
                    changes.append(f"🟢 Muscle mass increased by {muscle_diff:.1f}%")
                else:
                    changes.append(f"🔴 Muscle mass decreased by {abs(muscle_diff):.1f}%")
            
            # BMR change
            bmr_diff = analysis2['bmr_estimated'] - analysis1['bmr_estimated']
            if abs(bmr_diff) >= 50:
                if bmr_diff > 0:
                    changes.append(f"🟢 BMR increased by {int(bmr_diff)} calories")
                else:
                    changes.append(f"🔴 BMR decreased by {int(abs(bmr_diff))} calories")
            
            # Body shape change
            if analysis1['body_shape_classification'] != analysis2['body_shape_classification']:
                changes.append(f"📏 Body shape changed from {analysis1['body_shape_classification']} to {analysis2['body_shape_classification']}")
            
            # Health assessment change
            if analysis1['health_assessment'] != analysis2['health_assessment']:
                changes.append(f"❤️ Health assessment changed from {analysis1['health_assessment']} to {analysis2['health_assessment']}")
            
            if changes:
                for change in changes:
                    st.markdown(f"- {change}")
            else:
                st.info("No significant changes detected between the selected analyses.")
        
        else:
            st.warning("Please select two different analyses to compare.")
    
    except Exception as e:
        st.error(f"Error loading analysis comparison: {str(e)}")

def generate_basic_recommendations(user_profile: UserProfile):
    """Generate basic exercise recommendations."""
    return [
        {
            'name': 'Walking',
            'type': 'Cardio',
            'duration': '30 minutes',
            'description': 'A great low-impact exercise to start with.'
        },
        {
            'name': 'Push-ups',
            'type': 'Strength',
            'duration': '3 sets of 10',
            'description': 'Build upper body strength.'
        },
        {
            'name': 'Squats',
            'type': 'Strength',
            'duration': '3 sets of 15',
            'description': 'Strengthen your lower body.'
        }
    ]

if __name__ == "__main__":
    main()
