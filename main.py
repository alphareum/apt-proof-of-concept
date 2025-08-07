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
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO to see debug messages
logger = logging.getLogger(__name__)

# Configure Streamlit page - only once
if "page_configured" not in st.session_state:
    st.set_page_config(
        page_title="APT Fitness Assistant",
        page_icon="🏋️‍♀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_configured = True

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Lazy import pattern for better performance
@st.cache_resource
def load_core_modules():
    """Lazy load core modules with caching."""
    try:
        from apt_fitness import AppConfig, UserProfile, BodyCompositionAnalyzer, RecommendationEngine
        from apt_fitness.core.models import Gender, ActivityLevel, FitnessLevel, GoalType, EquipmentType
        from apt_fitness.ui.components import UIComponents
        from apt_fitness.engines.recommendation import get_recommendation_engine
        from apt_fitness.data.database import get_database
        
        return {
            'AppConfig': AppConfig,
            'UserProfile': UserProfile,
            'BodyCompositionAnalyzer': BodyCompositionAnalyzer,
            'RecommendationEngine': RecommendationEngine,
            'Gender': Gender,
            'ActivityLevel': ActivityLevel, 
            'FitnessLevel': FitnessLevel,
            'GoalType': GoalType,
            'EquipmentType': EquipmentType,
            'UIComponents': UIComponents,
            'get_recommendation_engine': get_recommendation_engine,
            'get_database': get_database,
            'available': True
        }
    except ImportError as e:
        logger.error(f"Core import error: {e}")
        return {'available': False, 'error': str(e)}

@st.cache_resource
def load_vision_modules():
    """Lazy load computer vision modules."""
    try:
        from PIL import Image
        import cv2
        import mediapipe as mp
        return {'Image': Image, 'cv2': cv2, 'mp': mp, 'available': True}
    except ImportError as e:
        logger.warning(f"Vision import error: {e}")
        return {'available': False, 'error': str(e)}

@st.cache_resource  
def load_plotting_modules():
    """Lazy load plotting modules."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        return {'px': px, 'go': go, 'pd': pd, 'np': np, 'available': True}
    except ImportError as e:
        logger.warning(f"Plotting import error: {e}")
        return {'available': False, 'error': str(e)}

@st.cache_resource
def load_body_composition_analyzer():
    """Lazy load body composition analyzer."""
    try:
        from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
        return {'BodyCompositionAnalyzer': BodyCompositionAnalyzer, 'available': True}
    except ImportError as e:
        logger.warning(f"Body composition analyzer not available: {e}")
        return {'available': False, 'error': str(e)}

@st.cache_resource(ttl=300)  # Cache for 5 minutes only to allow updates
def get_cached_body_analyzer():
    """Create and cache a single body analyzer instance."""
    try:
        if BODY_COMP_AVAILABLE:
            BodyCompositionAnalyzer = BODY_COMP_MODULES['BodyCompositionAnalyzer']
            return BodyCompositionAnalyzer()
        return None
    except Exception as e:
        logger.error(f"Error creating cached body analyzer: {e}")
        return None

# Load modules
CORE_MODULES = load_core_modules()
VISION_MODULES = load_vision_modules()
PLOTTING_MODULES = load_plotting_modules()
BODY_COMP_MODULES = load_body_composition_analyzer()

CORE_AVAILABLE = CORE_MODULES['available']
VISION_AVAILABLE = VISION_MODULES['available'] 
PLOTTING_AVAILABLE = PLOTTING_MODULES['available']
BODY_COMP_AVAILABLE = BODY_COMP_MODULES['available']


class APTFitnessApp:
    """Main APT Fitness Application class."""
    
    def __init__(self):
        """Initialize the APT Fitness application."""
        # Get cached modules
        self.core_modules = CORE_MODULES
        self.vision_modules = VISION_MODULES
        self.plotting_modules = PLOTTING_MODULES
        self.body_comp_modules = BODY_COMP_MODULES
        
        # Initialize UI components
        self.ui = None
        if CORE_AVAILABLE:
            UIComponents = self.core_modules['UIComponents']
            self.ui = UIComponents()
        
        # Initialize other components lazily
        self.recommendation_engine = None
        self.body_analyzer = None
        self.database = None
        
        # Initialize session state once
        self.initialize_session_state()
    
    @st.cache_resource
    def get_recommendation_engine(_self):
        """Get recommendation engine with caching."""
        if CORE_AVAILABLE and _self.recommendation_engine is None:
            try:
                get_recommendation_engine = _self.core_modules['get_recommendation_engine']
                _self.recommendation_engine = get_recommendation_engine()
            except Exception as e:
                logger.error(f"Error initializing recommendation engine: {e}")
        return _self.recommendation_engine
    
    @st.cache_resource
    def get_database(_self):
        """Get database with caching."""
        if CORE_AVAILABLE and _self.database is None:
            try:
                get_database = _self.core_modules['get_database']
                _self.database = get_database()
            except Exception as e:
                logger.error(f"Error initializing database: {e}")
        return _self.database
    
    @st.cache_resource
    def get_body_analyzer(_self):
        """Get body analyzer with caching."""
        # Use the global cached instance instead of creating per-app instance
        return get_cached_body_analyzer()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables efficiently."""
        
        # Clean up any corrupted session state that might cause 400 errors
        if hasattr(st.session_state, '_session_corrupted'):
            for key in list(st.session_state.keys()):
                if key.startswith('body_analysis') or key.startswith('recommendations'):
                    try:
                        del st.session_state[key]
                    except:
                        pass
        
        # Default session state values
        defaults = {
            'user_profile': None,
            'profile_complete': False,
            'show_profile_edit': False,
            'recommendations_cache': {},
            'weekly_plan_cache': {},
            'analytics_data': {
                'total_workouts': 0,
                'total_minutes': 0,
                'total_calories': 0,
                'current_streak': 0
            },
            'measurements_history': [],
            'body_analysis_history': []
        }
        
        # Initialize only missing keys
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        # Limit history sizes to prevent memory issues that cause 400 errors
        if len(st.session_state.get('body_analysis_history', [])) > 20:
            st.session_state.body_analysis_history = st.session_state.body_analysis_history[-10:]
            
        if len(st.session_state.get('measurements_history', [])) > 50:
            st.session_state.measurements_history = st.session_state.measurements_history[-25:]
    
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
    
    @st.fragment 
    def render_header(self):
        """Render application header with fragment optimization."""
        # Only show status if not cached
        if "header_rendered" not in st.session_state:
            st.title("🏋️‍♀️ APT Fitness Assistant")
            st.subheader("Your AI-Powered Fitness Companion")
            
            # Feature availability status - cached
            if CORE_AVAILABLE:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.success("✅ Computer Vision" if VISION_AVAILABLE else "❌ Computer Vision")
                
                with col2:
                    st.success("✅ Body Analysis" if BODY_COMP_AVAILABLE else "❌ Body Analysis")
                
                with col3:
                    st.success("✅ Analytics" if PLOTTING_AVAILABLE else "❌ Analytics")
                
                with col4:
                    engine = self.get_recommendation_engine()
                    st.success("✅ AI Recommendations" if engine else "❌ AI Recommendations")
            
            st.divider()
            st.session_state.header_rendered = True
    
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
                # Use experimental_rerun for better performance
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
                recommendation_engine = self.get_recommendation_engine()
                if recommendation_engine:
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
        
        recommendation_engine = self.get_recommendation_engine()
        if not recommendation_engine:
            st.error("❌ Recommendation engine not available.")
            return
        
        # Generate recommendations if not cached
        cache_key = f"{profile.user_id}_{profile.primary_goal.value}_{profile.fitness_level.value}"
        
        if cache_key not in st.session_state.recommendations_cache:
            with st.spinner("🤖 Generating personalized recommendations..."):
                try:
                    recommendations = recommendation_engine.generate_workout_recommendations(profile)
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
        
        recommendation_engine = self.get_recommendation_engine()
        if not recommendation_engine:
            st.error("❌ Workout planner not available.")
            return
        
        # Generate weekly plan if not cached
        cache_key = f"weekly_{profile.user_id}_{profile.workout_frequency_per_week}"
        
        if cache_key not in st.session_state.weekly_plan_cache:
            with st.spinner("📅 Creating your weekly workout plan..."):
                try:
                    weekly_plan = recommendation_engine.generate_weekly_plan(profile)
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
        
        # Debug option to clear analyzer cache
        if st.checkbox("🔧 Force reload body analyzer (for debugging)", key="force_reload_analyzer"):
            get_cached_body_analyzer.clear()
            st.info("Body analyzer cache cleared - will use latest code")
        
        # File uploader with improved error handling
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, full-body photo for analysis (Max 10MB)",
            key="body_analysis_uploader",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            # Immediate validation to prevent 400 errors
            try:
                # Check file properties immediately
                if not hasattr(uploaded_file, 'size') or uploaded_file.size is None:
                    st.error("❌ Invalid file. Please try uploading again.")
                    st.stop()
                    
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("❌ File too large. Please upload an image smaller than 10MB.")
                    st.stop()
                    
                if uploaded_file.size < 1024:  # Too small
                    st.error("❌ File too small. Please upload a valid image.")
                    st.stop()
            except Exception as validation_error:
                st.error(f"❌ File validation error: {validation_error}")
                st.stop()
            
            try:
                # Display the uploaded image with better error handling
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    try:
                        # Convert uploaded file to PIL Image to avoid filename issues
                        if VISION_AVAILABLE:
                            Image = self.vision_modules['Image']
                            # Create a copy of the file data to avoid conflicts
                            file_copy = uploaded_file.read()
                            uploaded_file.seek(0)  # Reset for later use
                            
                            from io import BytesIO
                            image = Image.open(BytesIO(file_copy))
                            
                            # Validate image format
                            if image.format not in ['JPEG', 'PNG', 'JPG']:
                                st.error("❌ Unsupported image format. Please use JPG or PNG.")
                                st.stop()
                                
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                        else:
                            st.error("❌ Image processing not available")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ Could not display image: {e}")
                        st.info("💡 Try uploading a different image or refresh the page.")
                        st.stop()
            
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
        """Process the uploaded image and measurement data with improved error handling."""
        
        # Enhanced validation with better error handling
        try:
            # Validate file size and type more thoroughly
            if not uploaded_file:
                st.error("❌ No file uploaded. Please select an image file.")
                return
                
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error("❌ File too large. Please upload an image smaller than 10MB.")
                return
            
            if uploaded_file.size < 1024:  # Too small file
                st.error("❌ File too small. Please upload a valid image file.")
                return
            
            if not uploaded_file.type or not uploaded_file.type.startswith('image/'):
                st.error("❌ Invalid file type. Please upload a valid image file (JPG, PNG).")
                return
            
            # Validate measurement data
            if not isinstance(measurement_data, dict):
                st.error("❌ Invalid measurement data. Please check your inputs.")
                return
                
        except Exception as e:
            st.error(f"❌ File validation error: {e}")
            return
        
        with st.spinner("🤖 Analyzing your image..."):
            try:
                # Enhanced analysis with body composition analyzer if available
                body_analyzer = self.get_body_analyzer()
                if CORE_AVAILABLE and body_analyzer and BODY_COMP_AVAILABLE:
                    # Safely read file data with better error handling
                    try:
                        # Reset file pointer to beginning
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.read()
                        
                        if not file_bytes or len(file_bytes) == 0:
                            st.error("❌ Empty file detected. Please upload a valid image.")
                            return
                            
                        # Validate it's actually an image by trying to open it
                        if VISION_AVAILABLE:
                            Image = self.vision_modules['Image']
                            try:
                                test_image = Image.open(uploaded_file)
                                test_image.verify()  # Verify it's a valid image
                                uploaded_file.seek(0)  # Reset after verification
                                file_bytes = uploaded_file.read()
                            except Exception as img_error:
                                st.error(f"❌ Invalid image file: {img_error}")
                                return
                        
                    except Exception as e:
                        st.error(f"❌ Error reading file: {e}")
                        return
                    
                    # Prepare user profile data more safely
                    user_profile = st.session_state.user_profile
                    user_profile_data = {}
                    
                    if user_profile:
                        user_profile_data = {
                            'age': getattr(user_profile, 'age', 30),
                            'gender': str(getattr(user_profile, 'gender', 'male')),
                            'weight_kg': float(measurement_data.get('weight', 70)),
                            'height_cm': float(measurement_data.get('height', 170))
                        }
                    else:
                        user_profile_data = {
                            'age': 30,
                            'gender': 'male',
                            'weight_kg': float(measurement_data.get('weight', 70)),
                            'height_cm': float(measurement_data.get('height', 170))
                        }
                    
                    # Process with timeout and better error handling
                    try:
                        analysis_result = body_analyzer.analyze_image(
                            file_bytes,
                            user_id=getattr(user_profile, 'user_id', f'user_{uuid.uuid4().hex[:8]}'),
                            physical_measurements=measurement_data,
                            user_profile=user_profile_data
                        )
                    except Exception as analysis_error:
                        st.error(f"❌ Analysis failed: {analysis_error}")
                        # Fallback to basic analysis
                        analysis_result = self.perform_basic_image_analysis(measurement_data)
                        
                else:
                    # Fallback analysis
                    analysis_result = self.perform_basic_image_analysis(measurement_data)
                
                # Display results with better error handling
                st.success("✅ Analysis completed!")
                
                # Results display with improved validation
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Analysis Results")
                    try:
                        if isinstance(analysis_result, dict) and analysis_result:
                            for key, value in analysis_result.items():
                                if key in ['breakdown', 'quality_metrics', 'calculation_methods']:
                                    continue  # Skip complex nested objects for main display
                                    
                                # Only display values that are suitable for st.metric (primitives)
                                if isinstance(value, (int, float)):
                                    # Format numeric values appropriately
                                    if key.endswith('_percentage') or 'percentage' in key.lower():
                                        display_value = f"{value:.1f}%"
                                    elif key in ['bmr_estimated', 'visceral_fat_level']:
                                        display_value = f"{value:.0f}"
                                    else:
                                        display_value = f"{value:.1f}"
                                    st.metric(key.replace('_', ' ').title(), display_value)
                                elif isinstance(value, str):
                                    st.metric(key.replace('_', ' ').title(), value)
                                elif isinstance(value, dict) and key in ['ratios', 'measurements']:
                                    # For dictionary values, display as expandable section
                                    with st.expander(f"📋 {key.replace('_', ' ').title()}"):
                                        for sub_key, sub_value in value.items():
                                            if isinstance(sub_value, (int, float)):
                                                st.write(f"**{sub_key.replace('_', ' ').title()}:** {sub_value:.2f}")
                                            else:
                                                st.write(f"**{sub_key.replace('_', ' ').title()}:** {sub_value}")
                        else:
                            st.write("Basic analysis completed")
                            if isinstance(analysis_result, dict):
                                for key, value in analysis_result.items():
                                    st.write(f"**{key}:** {value}")
                    except Exception as display_error:
                        st.warning(f"Could not display some results: {display_error}")
                        st.write("Analysis completed but display formatting failed.")
                
                with col2:
                    st.subheader("💡 Recommendations")
                    try:
                        self.generate_image_based_recommendations(measurement_data, analysis_result)
                    except Exception as rec_error:
                        st.warning(f"Could not generate recommendations: {rec_error}")
                        st.write("• Focus on maintaining a balanced diet")
                        st.write("• Include regular cardiovascular exercise")
                        st.write("• Add strength training to your routine")
                
                # Save to history with better error handling
                try:
                    history_entry = {
                        'date': datetime.now().isoformat(),
                        'analysis': str(analysis_result)[:500],  # Limit size to prevent issues
                        'measurements': measurement_data,
                        'image_name': getattr(uploaded_file, 'name', 'unknown.jpg')
                    }
                    st.session_state.body_analysis_history.append(history_entry)
                    
                    # Limit history size to prevent memory issues
                    if len(st.session_state.body_analysis_history) > 10:
                        st.session_state.body_analysis_history = st.session_state.body_analysis_history[-10:]
                        
                except Exception as save_error:
                    logger.warning(f"Could not save to history: {save_error}")
                
            except Exception as e:
                error_msg = str(e)
                if "400" in error_msg or "request" in error_msg.lower() or "axios" in error_msg.lower():
                    st.error("🔄 **Connection error during analysis.** Please try:")
                    st.write("1. Refresh the page")
                    st.write("2. Upload a smaller image (< 5MB)")
                    st.write("3. Try a different image format (JPG/PNG)")
                    if st.button("🔄 Refresh Page", key="refresh_after_error"):
                        st.rerun()
                else:
                    st.error(f"❌ Analysis error: {e}")
                logger.error(f"Image analysis error: {e}")
                
                # Provide fallback basic analysis
                st.info("Falling back to basic analysis...")
                try:
                    basic_result = self.perform_basic_image_analysis(measurement_data)
                    st.subheader("📊 Basic Analysis Results")
                    for key, value in basic_result.items():
                        st.write(f"**{key}:** {value}")
                except Exception as fallback_error:
                    st.error(f"Even basic analysis failed: {fallback_error}")
    
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
                pd = self.plotting_modules['pd']
                px = self.plotting_modules['px']
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
            
            pd = self.plotting_modules['pd']
            px = self.plotting_modules['px']
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
            if CORE_AVAILABLE:
                GoalType = self.core_modules['GoalType']
                goal_type = st.selectbox("Goal Type", [g.value for g in GoalType])
            else:
                goal_type = st.selectbox("Goal Type", ["Weight Loss", "Muscle Gain", "Endurance"])
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
        
        # Clear caches to ensure latest code is used
        if st.button("🔄 Refresh App & Clear Cache", key="refresh_app_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
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
    """Main application entry point with enhanced error handling for 400 errors."""
    try:
        # Enhanced error counting for better stability
        if 'error_count' not in st.session_state:
            st.session_state.error_count = 0
        
        # Reset on excessive errors to prevent cascading failures
        if st.session_state.error_count > 3:
            st.session_state.error_count = 0
            st.cache_data.clear()
            st.cache_resource.clear()
            # Clear potentially corrupted session state
            corrupted_keys = [key for key in st.session_state.keys() 
                            if 'cache' in key or 'history' in key or 'upload' in key]
            for key in corrupted_keys:
                try:
                    del st.session_state[key]
                except:
                    pass
            st.session_state._session_corrupted = True
        
        # Initialize and run app
        app = APTFitnessApp()
        app.run()

        
    except Exception as e:
        st.session_state.error_count += 1
        error_msg = str(e).lower()
        
        # Enhanced error handling for different types
        if any(keyword in error_msg for keyword in ["axios", "400", "request", "connection", "bad request"]):
            st.error("🔄 **Connection Issue Detected**")
            st.write("This usually happens with large file uploads or network issues.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh Page", key="error_refresh"):
                    st.cache_data.clear()
                    st.rerun()
            with col2:
                if st.button("🧹 Clear Cache", key="clear_cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.clear()
                    st.rerun()
            with col3:
                if st.button("🔄 Restart App", key="restart_app"):
                    # Force complete restart
                    for key in list(st.session_state.keys()):
                        try:
                            del st.session_state[key]
                        except:
                            pass
                    st.rerun()
                    
            st.info("💡 **Tips to avoid this error:**")
            st.write("• Use smaller images (< 5MB)")
            st.write("• Try JPG format instead of PNG")
            st.write("• Ensure stable internet connection")
            st.write("• Close other browser tabs if using a lot of memory")
            
        elif "memory" in error_msg or "size" in error_msg:
            st.error("💾 **Memory Issue** - Try using smaller files or refreshing the page.")
            if st.button("🔄 Clear Memory & Refresh", key="memory_refresh"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        else:
            st.error(f"❌ Application error: {e}")
            logger.error(f"Application error: {e}")
            
            # Provide recovery options
            if st.button("🔄 Try Again", key="generic_refresh"):
                st.rerun()
        
        # Show debug info only after multiple errors
        if st.session_state.error_count > 2:
            with st.expander("🔧 Technical Details"):
                st.code(f"Error: {e}")
                st.code(f"Type: {type(e).__name__}")
                st.write(f"Error count: {st.session_state.error_count}")
    
    except KeyboardInterrupt:
        st.info("👋 Application stopped by user.")


if __name__ == "__main__":
    main()
