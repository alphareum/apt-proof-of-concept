"""
APT Fitness Assistant - Main Streamlit Application

AI-Powered fitness application with comprehensive features including:
- Advanced body composition analysis
- AI-powered exercise recommendations  
- Real-time form correction with pose detection
- Progress tracking and analytics
- Goal setting and achievement monitoring

This is the main entry point for the Streamlit application.
Run with: streamlit run main.py

Version: 3.0.0
Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import sys
import logging
import streamlit as st
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="APT Fitness Assistant",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import core modules from the new structure
try:
    from apt_fitness import AppConfig, UserProfile, BodyCompositionAnalyzer, RecommendationEngine
    from apt_fitness.core.models import Gender, ActivityLevel, FitnessLevel, GoalType, EquipmentType
    from apt_fitness.ui.components import UIComponents
    from apt_fitness.engines.recommendation import get_recommendation_engine
    from apt_fitness.data.database import get_database
    CORE_AVAILABLE = True
    logger.info("✅ Core APT Fitness modules loaded successfully")
except ImportError as e:
    CORE_AVAILABLE = False
    st.error(f"❌ Core modules not available: {e}")
    logger.error(f"Core import error: {e}")

# Optional imports with graceful fallbacks
try:
    from PIL import Image
    import cv2
    import mediapipe as mp
    VISION_AVAILABLE = True
    logger.info("✅ Computer vision libraries available")
except ImportError as e:
    VISION_AVAILABLE = False
    st.warning(f"⚠️ Computer vision libraries not available: {e}")
    logger.warning(f"Vision import error: {e}")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
    logger.info("✅ Plotting libraries available")
except ImportError as e:
    PLOTTING_AVAILABLE = False
    st.warning("⚠️ Plotting libraries not available. Charts may be limited.")
    logger.warning(f"Plotting import error: {e}")

try:
    from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
    BODY_COMP_AVAILABLE = True
    logger.info("✅ Body composition analyzer available")
except ImportError as e:
    BODY_COMP_AVAILABLE = False
    logger.warning(f"Body composition analyzer not available: {e}")


class APTFitnessApp:
    """Main APT Fitness Application class."""
    
    def __init__(self):
        """Initialize the APT Fitness application."""
        self.ui = UIComponents() if CORE_AVAILABLE else None
        self.recommendation_engine = None
        self.body_analyzer = None
        self.database = None
        
        # Initialize components if available
        if CORE_AVAILABLE:
            try:
                self.recommendation_engine = get_recommendation_engine()
                self.database = get_database()
                if BODY_COMP_AVAILABLE:
                    self.body_analyzer = BodyCompositionAnalyzer()
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
        
        # Initialize session state
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        
        # User profile and authentication
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = None
        
        if 'profile_complete' not in st.session_state:
            st.session_state.profile_complete = False
        
        if 'show_profile_edit' not in st.session_state:
            st.session_state.show_profile_edit = False
        
        # Application state
        if 'recommendations_cache' not in st.session_state:
            st.session_state.recommendations_cache = {}
        
        if 'weekly_plan_cache' not in st.session_state:
            st.session_state.weekly_plan_cache = {}
        
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = {
                'total_workouts': 0,
                'total_minutes': 0,
                'total_calories': 0,
                'current_streak': 0
            }
        
        # Body analysis data
        if 'measurements_history' not in st.session_state:
            st.session_state.measurements_history = []
        
        if 'body_analysis_history' not in st.session_state:
            st.session_state.body_analysis_history = []
    
    def check_user_profile(self) -> bool:
        """Check if user has a complete profile."""
        if not st.session_state.user_profile:
            return False
        
        profile = st.session_state.user_profile
        required_fields = ['name', 'age', 'height_cm', 'weight_kg', 'gender', 
                          'activity_level', 'fitness_level', 'primary_goal']
        
        try:
            for field in required_fields:
                if not hasattr(profile, field) or getattr(profile, field) is None:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking profile: {e}")
            return False
    
    def render_header(self):
        """Render application header."""
        st.title("🏋️‍♀️ APT Fitness Assistant")
        st.subheader("Your AI-Powered Fitness Companion")
        
        # Feature availability status
        if CORE_AVAILABLE:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if VISION_AVAILABLE:
                    st.success("✅ Computer Vision")
                else:
                    st.error("❌ Computer Vision")
            
            with col2:
                if BODY_COMP_AVAILABLE:
                    st.success("✅ Body Analysis")
                else:
                    st.error("❌ Body Analysis")
            
            with col3:
                if PLOTTING_AVAILABLE:
                    st.success("✅ Analytics")
                else:
                    st.error("❌ Analytics")
            
            with col4:
                if self.recommendation_engine:
                    st.success("✅ AI Recommendations")
                else:
                    st.error("❌ AI Recommendations")
        
        st.divider()
    
    def render_profile_setup(self):
        """Render profile setup interface."""
        st.header("👤 Complete Your Profile")
        st.info("Please complete your profile to get personalized recommendations and track your progress.")
        
        if CORE_AVAILABLE and self.ui:
            # Use the UI components for profile creation
            new_profile = self.ui.render_profile_form()
            
            if new_profile:
                st.session_state.user_profile = new_profile
                st.session_state.profile_complete = True
                st.success("✅ Profile created successfully!")
                st.rerun()
        else:
            st.error("❌ Profile creation not available. Core modules missing.")
    
    def render_sidebar(self):
        """Render sidebar with user information and controls."""
        if not st.session_state.user_profile:
            return
        
        profile = st.session_state.user_profile
        
        if CORE_AVAILABLE and self.ui:
            self.ui.render_sidebar(profile)
        else:
            # Fallback sidebar
            st.sidebar.header(f"👋 Hello, {getattr(profile, 'name', 'User')}!")
            st.sidebar.write(f"**Age:** {getattr(profile, 'age', 'N/A')}")
            st.sidebar.write(f"**Goal:** {getattr(profile, 'primary_goal', 'N/A')}")
            
            if st.sidebar.button("Edit Profile"):
                st.session_state.show_profile_edit = True
    
    def render_dashboard_tab(self):
        """Render main dashboard tab."""
        st.header("📊 Your Fitness Dashboard")
        
        if not st.session_state.user_profile:
            st.info("Complete your profile to see your personalized dashboard.")
            return
        
        profile = st.session_state.user_profile
        
        if CORE_AVAILABLE and self.ui:
            # Use UI components for dashboard
            self.ui.render_metrics_dashboard(profile, st.session_state.analytics_data)
        else:
            # Basic dashboard fallback
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Workouts", st.session_state.analytics_data['total_workouts'])
            
            with col2:
                st.metric("Total Minutes", st.session_state.analytics_data['total_minutes'])
            
            with col3:
                st.metric("Calories Burned", f"{st.session_state.analytics_data['total_calories']:.0f}")
            
            with col4:
                st.metric("Current Streak", f"{st.session_state.analytics_data['current_streak']} days")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Generate Workout", type="primary"):
                if self.recommendation_engine:
                    st.session_state.recommendations_cache = {}  # Clear cache
                    st.success("New workout will be generated!")
                else:
                    st.error("Recommendation engine not available")
        
        with col2:
            if st.button("📊 Body Analysis"):
                st.session_state.current_tab = 2  # Switch to body analysis tab
                st.rerun()
        
        with col3:
            if st.button("📈 View Progress"):
                st.session_state.current_tab = 4  # Switch to analytics tab
                st.rerun()
    
    def render_recommendations_tab(self):
        """Render exercise recommendations tab."""
        st.header("💪 Exercise Recommendations")
        
        if not st.session_state.user_profile:
            st.info("Complete your profile to get personalized recommendations.")
            return
        
        profile = st.session_state.user_profile
        
        if not self.recommendation_engine:
            st.error("❌ Recommendation engine not available.")
            return
        
        # Generate recommendations if not cached
        cache_key = f"{profile.user_id}_{profile.primary_goal.value}_{profile.fitness_level.value}"
        
        if cache_key not in st.session_state.recommendations_cache:
            with st.spinner("🤖 Generating personalized recommendations..."):
                try:
                    recommendations = self.recommendation_engine.generate_workout_recommendations(profile)
                    st.session_state.recommendations_cache[cache_key] = recommendations
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    return
        
        recommendations = st.session_state.recommendations_cache.get(cache_key, [])
        
        if not recommendations:
            st.warning("No recommendations available. Try adjusting your profile settings.")
            return
        
        # Display workout summary
        if CORE_AVAILABLE and self.ui:
            self.ui.render_workout_summary(recommendations)
        
        st.divider()
        
        # Display individual exercises
        st.subheader("🎯 Your Recommended Exercises")
        
        for i, rec in enumerate(recommendations):
            if CORE_AVAILABLE and self.ui:
                self.ui.render_workout_card(rec, i)
            else:
                # Fallback display
                with st.container():
                    st.write(f"**{rec.exercise.name}**")
                    st.write(f"Category: {rec.exercise.category}")
                    st.write(f"Sets: {rec.sets}, Reps: {rec.reps}")
                    if rec.duration_minutes:
                        st.write(f"Duration: {rec.duration_minutes} minutes")
            
            st.divider()
        
        # Generate new recommendations
        if st.button("🔄 Generate New Recommendations"):
            st.session_state.recommendations_cache = {}
            st.rerun()
    
    def render_workout_planner_tab(self):
        """Render workout planner tab."""
        st.header("📅 Workout Planner")
        
        if not st.session_state.user_profile:
            st.info("Complete your profile to create workout plans.")
            return
        
        profile = st.session_state.user_profile
        
        if not self.recommendation_engine:
            st.error("❌ Workout planner not available.")
            return
        
        # Generate weekly plan if not cached
        cache_key = f"weekly_{profile.user_id}_{profile.workout_frequency_per_week}"
        
        if cache_key not in st.session_state.weekly_plan_cache:
            with st.spinner("📅 Creating your weekly workout plan..."):
                try:
                    weekly_plan = self.recommendation_engine.generate_weekly_plan(profile)
                    st.session_state.weekly_plan_cache[cache_key] = weekly_plan
                except Exception as e:
                    st.error(f"Error generating weekly plan: {e}")
                    return
        
        weekly_plan = st.session_state.weekly_plan_cache.get(cache_key, {})
        
        if CORE_AVAILABLE and self.ui:
            self.ui.render_weekly_plan_view(weekly_plan)
        else:
            # Fallback weekly plan display
            if weekly_plan:
                for day, workouts in weekly_plan.items():
                    st.subheader(day)
                    if workouts:
                        for workout in workouts:
                            st.write(f"• {workout.exercise.name}: {workout.sets}x{workout.reps}")
                    else:
                        st.write("Rest day")
                    st.divider()
        
        # Plan controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Regenerate Plan"):
                st.session_state.weekly_plan_cache = {}
                st.rerun()
        
        with col2:
            if st.button("💾 Save Plan"):
                st.success("Plan saved! (Feature coming soon)")
    
    def render_body_analysis_tab(self):
        """Render body analysis tab."""
        st.header("📊 Body Composition Analysis")
        
        if not VISION_AVAILABLE:
            st.warning("⚠️ Computer vision features not available. Limited analysis only.")
        
        # Analysis options
        analysis_tabs = st.tabs(["📸 Image Analysis", "📏 Manual Entry", "📈 History"])
        
        with analysis_tabs[0]:
            self.render_image_analysis()
        
        with analysis_tabs[1]:
            self.render_manual_measurements()
        
        with analysis_tabs[2]:
            self.render_analysis_history()
    
    def render_image_analysis(self):
        """Render image-based body analysis."""
        st.subheader("📸 Upload Photo for Analysis")
        st.info("💡 **Tip:** Upload a clear, full-body photo in good lighting for best results")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, full-body photo for analysis",
            key="body_analysis_uploader"
        )
        
        if uploaded_file:
            try:
                # Validate file immediately
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("❌ File too large. Please upload an image smaller than 10MB.")
                    return
                
                # Display the uploaded image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    try:
                        # Convert uploaded file to PIL Image to avoid filename issues
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Could not display image: {e}")
                        return
            
            except Exception as e:
                st.error(f"❌ File processing error: {e}")
                return
            
            with col2:
                st.subheader("📝 Analysis Options")
                
                # Analysis type selection
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Basic Body Assessment", "Posture Analysis", "Progress Comparison"],
                    help="Choose the type of analysis to perform"
                )
                
                # Additional measurement inputs
                st.subheader("📏 Optional Measurements")
                with st.form("image_analysis_form"):
                    current_weight = st.number_input("Current Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
                    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
                    
                    # Body part focus
                    focus_areas = st.multiselect(
                        "Focus Areas",
                        ["Overall Body", "Upper Body", "Lower Body", "Core", "Posture"],
                        default=["Overall Body"]
                    )
                    
                    analysis_notes = st.text_area("Notes", placeholder="Any specific areas of concern or goals...")
                    
                    analyze_button = st.form_submit_button("🔍 Analyze Image", type="primary")
                    
                    if analyze_button:
                        self.process_image_analysis(uploaded_file, {
                            'weight': current_weight,
                            'height': height,
                            'analysis_type': analysis_type,
                            'focus_areas': focus_areas,
                            'notes': analysis_notes
                        })
        
        # Previous analyses section
        if st.session_state.body_analysis_history:
            st.divider()
            st.subheader("📅 Recent Image Analyses")
            
            # Show last 3 analyses
            for i, analysis in enumerate(st.session_state.body_analysis_history[-3:]):
                with st.expander(f"Analysis from {analysis['date'][:16]}"):
                    if 'analysis' in analysis:
                        st.write("**Results:**", analysis['analysis'])
                    if 'measurements' in analysis:
                        st.write("**Measurements:**", analysis['measurements'])
    
    def process_image_analysis(self, uploaded_file, measurement_data):
        """Process the uploaded image and measurement data."""
        
        # Validate file size and type
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("❌ File too large. Please upload an image smaller than 10MB.")
            return
        
        if not uploaded_file.type.startswith('image/'):
            st.error("❌ Invalid file type. Please upload an image file.")
            return
        
        with st.spinner("🤖 Analyzing your image..."):
            try:
                # Enhanced analysis with body composition analyzer if available
                if CORE_AVAILABLE and self.body_analyzer and BODY_COMP_AVAILABLE:
                    # Safely read file data
                    try:
                        file_bytes = uploaded_file.getvalue()
                        if len(file_bytes) == 0:
                            st.error("❌ Empty file detected. Please upload a valid image.")
                            return
                    except Exception as e:
                        st.error(f"❌ Error reading file: {e}")
                        return
                    
                    analysis_result = self.body_analyzer.analyze_image(
                        file_bytes,
                        user_id=getattr(st.session_state.user_profile, 'user_id', 'default_user'),
                        physical_measurements=measurement_data,
                        user_profile={
                            'age': getattr(st.session_state.user_profile, 'age', 30),
                            'gender': getattr(st.session_state.user_profile, 'gender', 'male'),
                            'weight_kg': measurement_data.get('weight', 70),
                            'height_cm': measurement_data.get('height', 170)
                        }
                    )
                else:
                    # Fallback analysis
                    analysis_result = self.perform_basic_image_analysis(measurement_data)
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Results display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Analysis Results")
                    if isinstance(analysis_result, dict):
                        for key, value in analysis_result.items():
                            # Only display values that are suitable for st.metric (primitives)
                            if isinstance(value, (int, float, str)) and not isinstance(value, dict):
                                # Convert value to string if it's not already
                                display_value = str(value)
                                st.metric(key.replace('_', ' ').title(), display_value)
                            elif isinstance(value, dict):
                                # For dictionary values, display as expandable section
                                with st.expander(f"📋 {key.replace('_', ' ').title()}"):
                                    for sub_key, sub_value in value.items():
                                        st.write(f"**{sub_key.replace('_', ' ').title()}:** {sub_value}")
                            else:
                                # For other complex types, just display as text
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(analysis_result)
                
                with col2:
                    st.subheader("💡 Recommendations")
                    self.generate_image_based_recommendations(measurement_data, analysis_result)
                
                # Save to history with error handling
                try:
                    st.session_state.body_analysis_history.append({
                        'date': datetime.now().isoformat(),
                        'analysis': analysis_result,
                        'measurements': measurement_data,
                        'image_name': uploaded_file.name
                    })
                except Exception as e:
                    logger.warning(f"Could not save to history: {e}")
                
            except Exception as e:
                error_msg = str(e)
                if "400" in error_msg or "request" in error_msg.lower():
                    st.error("🔄 **Connection error during analysis.** Please try again or refresh the page.")
                else:
                    st.error(f"❌ Analysis error: {e}")
                logger.error(f"Image analysis error: {e}")
    
    def perform_basic_image_analysis(self, measurement_data):
        """Perform basic analysis when advanced CV is not available."""
        weight = measurement_data.get('weight', 70)
        height = measurement_data.get('height', 170)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Basic body composition estimates
        if bmi < 18.5:
            body_category = "Underweight"
            body_fat_estimate = "Low (10-15%)"
        elif bmi < 25:
            body_category = "Normal weight"
            body_fat_estimate = "Normal (15-20%)"
        elif bmi < 30:
            body_category = "Overweight"
            body_fat_estimate = "Elevated (20-25%)"
        else:
            body_category = "Obese"
            body_fat_estimate = "High (25%+)"
        
        return {
            "BMI": f"{bmi:.1f}",
            "Body Category": body_category,
            "Estimated Body Fat": body_fat_estimate,
            "Analysis Type": measurement_data.get('analysis_type', 'Basic'),
            "Focus Areas": ", ".join(measurement_data.get('focus_areas', ['Overall Body']))
        }
    
    def generate_image_based_recommendations(self, measurement_data, analysis_result):
        """Generate recommendations based on image analysis."""
        analysis_type = measurement_data.get('analysis_type', 'Basic Body Assessment')
        focus_areas = measurement_data.get('focus_areas', ['Overall Body'])
        
        recommendations = []
        
        if 'Overall Body' in focus_areas:
            recommendations.append("🏃‍♀️ Focus on full-body compound exercises")
            recommendations.append("💧 Maintain proper hydration for accurate body composition")
        
        if 'Upper Body' in focus_areas:
            recommendations.append("💪 Include upper body strength training")
            recommendations.append("🤸‍♀️ Work on shoulder mobility and posture")
        
        if 'Lower Body' in focus_areas:
            recommendations.append("🏋️‍♀️ Incorporate squats and lunges")
            recommendations.append("🚶‍♀️ Add more walking or running")
        
        if 'Core' in focus_areas:
            recommendations.append("🧘‍♀️ Practice core-strengthening exercises")
            recommendations.append("🏋️‍♂️ Try planks and deadlifts")
        
        if 'Posture' in focus_areas:
            recommendations.append("🪑 Take breaks from sitting regularly")
            recommendations.append("🧘‍♂️ Consider yoga or stretching routines")
        
        # Display recommendations
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Additional tips based on analysis
        if isinstance(analysis_result, dict) and 'BMI' in analysis_result:
            bmi_value = float(analysis_result['BMI'].split()[0]) if isinstance(analysis_result['BMI'], str) else analysis_result['BMI']
            if bmi_value < 18.5:
                st.info("💡 Consider consulting a nutritionist for healthy weight gain strategies")
            elif bmi_value > 25:
                st.info("💡 Focus on creating a sustainable caloric deficit through diet and exercise")
    
    def render_manual_measurements(self):
        """Render manual measurements entry."""
        st.subheader("📏 Manual Body Measurements")
        
        with st.form("measurements_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
                waist = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, value=80.0)
                chest = st.number_input("Chest (cm)", min_value=60.0, max_value=200.0, value=90.0)
            
            with col2:
                body_fat = st.number_input("Body Fat % (if known)", min_value=3.0, max_value=50.0, value=15.0)
                arms = st.number_input("Arms (cm)", min_value=15.0, max_value=60.0, value=30.0)
                thighs = st.number_input("Thighs (cm)", min_value=30.0, max_value=100.0, value=55.0)
            
            notes = st.text_area("Notes", placeholder="Additional notes about measurements...")
            
            submitted = st.form_submit_button("💾 Save Measurements")
            
            if submitted:
                measurement = {
                    'date': datetime.now().isoformat(),
                    'weight': weight,
                    'waist': waist,
                    'chest': chest,
                    'body_fat': body_fat,
                    'arms': arms,
                    'thighs': thighs,
                    'notes': notes
                }
                
                st.session_state.measurements_history.append(measurement)
                st.success("✅ Measurements saved successfully!")
    
    def render_analysis_history(self):
        """Render analysis history."""
        st.subheader("📈 Analysis History")
        
        if st.session_state.measurements_history:
            if PLOTTING_AVAILABLE:
                # Create progress charts
                df = pd.DataFrame(st.session_state.measurements_history)
                df['date'] = pd.to_datetime(df['date'])
                
                # Weight progress
                if len(df) > 1:
                    fig = px.line(df, x='date', y='weight', title='Weight Progress')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Body fat progress
                if len(df) > 1 and 'body_fat' in df.columns:
                    fig = px.line(df, x='date', y='body_fat', title='Body Fat Progress')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display recent measurements
            st.subheader("Recent Measurements")
            for measurement in st.session_state.measurements_history[-5:]:
                date_str = measurement['date'][:10]
                st.write(f"**{date_str}:** Weight: {measurement['weight']} kg, Body Fat: {measurement['body_fat']}%")
        else:
            st.info("No measurements recorded yet. Add some measurements to track your progress!")
    
    def render_analytics_tab(self):
        """Render analytics and progress tracking tab."""
        st.header("📈 Progress Analytics")
        
        if not st.session_state.user_profile:
            st.info("Complete your profile to see analytics.")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workouts", st.session_state.analytics_data['total_workouts'], "+1")
        
        with col2:
            st.metric("Total Minutes", st.session_state.analytics_data['total_minutes'], "+30")
        
        with col3:
            st.metric("Calories Burned", f"{st.session_state.analytics_data['total_calories']:.0f}", "+250")
        
        with col4:
            st.metric("Current Streak", f"{st.session_state.analytics_data['current_streak']} days", "+1")
        
        # Progress charts
        if PLOTTING_AVAILABLE and st.session_state.measurements_history:
            st.subheader("📊 Progress Charts")
            
            df = pd.DataFrame(st.session_state.measurements_history)
            df['date'] = pd.to_datetime(df['date'])
            
            # Multiple metrics chart
            col1, col2 = st.columns(2)
            
            with col1:
                if len(df) > 1:
                    fig = px.line(df, x='date', y='weight', title='Weight Trend')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(df) > 1 and 'body_fat' in df.columns:
                    fig = px.line(df, x='date', y='body_fat', title='Body Fat Trend')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add measurements to see progress analytics!")
    
    def render_goals_tab(self):
        """Render goals management tab."""
        st.header("🎯 Goals Management")
        
        if not st.session_state.user_profile:
            st.info("Complete your profile to set goals.")
            return
        
        profile = st.session_state.user_profile
        
        # Current goal display
        st.subheader("Current Primary Goal")
        st.write(f"**{profile.primary_goal.value.replace('_', ' ').title()}**")
        
        if hasattr(profile, 'target_weight_kg') and profile.target_weight_kg:
            current_weight = getattr(profile, 'weight_kg', 0)
            target_weight = profile.target_weight_kg
            progress = ((current_weight - target_weight) / current_weight) * 100 if current_weight else 0
            
            st.progress(min(1.0, max(0.0, progress / 100)))
            st.write(f"Progress: {progress:.1f}%")
        
        # Goal creation
        with st.expander("➕ Set New Goal"):
            goal_type = st.selectbox("Goal Type", [g.value for g in GoalType])
            target_value = st.number_input("Target Value", min_value=0.0, value=10.0)
            target_date = st.date_input("Target Date", value=datetime.now().date() + timedelta(days=90))
            
            if st.button("Set Goal"):
                st.success("Goal set successfully! (Feature will be fully implemented)")
        
        # Achievement tracking
        st.subheader("🏆 Achievements")
        achievements = [
            "🥇 First Workout Completed",
            "🔥 7-Day Streak",
            "📈 First Measurement Logged",
            "💪 Profile Completed"
        ]
        
        for achievement in achievements:
            st.write(f"✅ {achievement}")
    
    def run(self):
        """Run the main application."""
        
        # Render header
        self.render_header()
        
        # Check for core availability
        if not CORE_AVAILABLE:
            st.error("❌ Core APT Fitness modules are not available. Please check your installation.")
            st.stop()
        
        # Check for profile or show setup
        if not self.check_user_profile() or st.session_state.get('show_profile_edit', False):
            self.render_profile_setup()
            return
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab_labels = [
            "🏠 Dashboard",
            "💪 Recommendations", 
            "📅 Workout Planner",
            "📊 Body Analysis",
            "📈 Analytics",
            "🎯 Goals"
        ]
        
        tabs = st.tabs(tab_labels)
        
        with tabs[0]:
            self.render_dashboard_tab()
        
        with tabs[1]:
            self.render_recommendations_tab()
        
        with tabs[2]:
            self.render_workout_planner_tab()
        
        with tabs[3]:
            self.render_body_analysis_tab()
        
        with tabs[4]:
            self.render_analytics_tab()
        
        with tabs[5]:
            self.render_goals_tab()


def main():
    """Main application entry point."""
    try:
        # Configure Streamlit to handle errors gracefully
        if 'error_count' not in st.session_state:
            st.session_state.error_count = 0
        
        # Reset error count periodically
        if st.session_state.error_count > 10:
            st.session_state.error_count = 0
            st.rerun()
        
        app = APTFitnessApp()
        app.run()
        
    except Exception as e:
        st.session_state.error_count += 1
        error_msg = str(e)
        
        # Handle specific error types
        if "AxiosError" in error_msg or "400" in error_msg:
            st.error("🔄 **Connection Issue Detected**")
            st.info("This appears to be a temporary communication error. Please try refreshing the page or restarting the application.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh Page"):
                    st.rerun()
            with col2:
                if st.button("🧹 Clear Cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
            with col3:
                if st.button("🔄 Reset Session"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        else:
            st.error(f"❌ Application error: {e}")
            logger.error(f"Application error: {traceback.format_exc()}")
        
        # Show debug info in expander
        with st.expander("🔧 Debug Information"):
            st.code(traceback.format_exc())
            st.write("**Session State:**")
            st.json({k: str(v)[:100] for k, v in st.session_state.items()})


if __name__ == "__main__":
    main()
