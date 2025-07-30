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

import streamlit as st
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util


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
legacy_dir = current_dir / "legacy"

for path in [str(current_dir), str(src_dir), str(legacy_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Safe imports with error handling
try:
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    st.warning(f"⚠️ Computer vision libraries not available: {e}")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Charts may be limited.")

# Try to import from both new and legacy structure
def safe_import(module_name, package=None, legacy_fallback=None):
    """Safely import modules with fallback options."""
    try:
        if package:
            return importlib.import_module(f"{package}.{module_name}")
        return importlib.import_module(module_name)
    except ImportError:
        if legacy_fallback:
            try:
                return importlib.import_module(legacy_fallback)
            except ImportError:
                pass
    return None

def safe_get_enum_value(enum_obj):
    """Safely get the value from an enum, or return the object if it's already a string."""
    if hasattr(enum_obj, 'value'):
        return enum_obj.value
    return str(enum_obj)

def safe_create_enum(enum_class, value):
    """Safely create an enum from a string value."""
    if hasattr(enum_class, '__members__'):
        for enum_member in enum_class:
            if enum_member.value == value:
                return enum_member
    # Return the string value if enum creation fails
    return value

# Import core modules
try:
    # Try new structure first
    from src.apt_fitness.core.models import UserProfile, Gender, ActivityLevel, FitnessLevel, GoalType
    from src.apt_fitness.core.config import AppConfig
    from apt_fitness.analyzers.body_composition import get_body_analyzer
    MODELS_AVAILABLE = True
except ImportError:
    try:
        # Fall back to legacy structure
        from models import UserProfile, create_user_profile, Gender, ActivityLevel, FitnessLevel, GoalType
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
        st.error("❌ Core models not available. Please check installation.")

# Try to import database functionality
try:
    from database import get_database, BodyMeasurement
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    st.warning("⚠️ Database functionality not available.")

# Try to import recommendation engine
try:
    from recommendation_engine import AdvancedExerciseRecommendationEngine
    RECOMMENDATION_ENGINE_AVAILABLE = True
except ImportError:
    RECOMMENDATION_ENGINE_AVAILABLE = False

# Try to import enhanced features
try:
    from enhanced_recommendation_system import EnhancedRecommendationEngine
    from workout_planner import AdvancedWorkoutPlanner
    from workout_planner_ui import WorkoutPlannerUI
    ENHANCED_PLANNER_AVAILABLE = True
except ImportError:
    ENHANCED_PLANNER_AVAILABLE = False

# Try to import body composition analysis
# In main.py, change to:
try:
    from apt_fitness.analyzers.body_composition import get_body_analyzer
    BODY_COMP_AVAILABLE = True
except ImportError:
    BODY_COMP_AVAILABLE = False

# Try to import enhanced UI components
try:
    from enhanced_ui import EnhancedFitnessUI
    ENHANCED_UI_AVAILABLE = True
    enhanced_ui = EnhancedFitnessUI()
except ImportError:
    ENHANCED_UI_AVAILABLE = False

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

def create_user_profile_from_form(age, gender, height, weight, activity_level, fitness_level, primary_goal):
    """Create user profile from form data."""
    try:
        if MODELS_AVAILABLE:
            # Try to create profile with available models
            import uuid
            
            # Convert string values to proper enum objects
            try:
                # Convert string values to enums by finding the matching enum value
                gender_enum = None
                for g in Gender:
                    if g.value == gender:
                        gender_enum = g
                        break
                if not gender_enum:
                    gender_enum = Gender.OTHER
                
                activity_enum = None
                for a in ActivityLevel:
                    if a.value == activity_level:
                        activity_enum = a
                        break
                if not activity_enum:
                    activity_enum = ActivityLevel.MODERATELY_ACTIVE
                
                fitness_enum = None
                for f in FitnessLevel:
                    if f.value == fitness_level:
                        fitness_enum = f
                        break
                if not fitness_enum:
                    fitness_enum = FitnessLevel.BEGINNER
                
                goal_enum = None
                for g in GoalType:
                    if g.value == primary_goal:
                        goal_enum = g
                        break
                if not goal_enum:
                    goal_enum = GoalType.GENERAL_FITNESS
                
                # Try to use proper model classes if available
                from models import create_user_profile
                return create_user_profile(
                    age=age,
                    gender=gender_enum,
                    height=height,
                    weight=weight,
                    activity_level=activity_enum,
                    fitness_level=fitness_enum,
                    primary_goal=goal_enum
                )
            except Exception as enum_error:
                logger.warning(f"Error creating enum values: {enum_error}")
                # Fallback to basic profile with string values
                pass
            
            # Create basic profile structure as fallback
            profile_data = {
                'user_id': str(uuid.uuid4()),
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'activity_level': activity_level,
                'fitness_level': fitness_level,
                'primary_goal': primary_goal
            }
            
            # Try creating with string values if enum creation failed
            try:
                from models import create_user_profile
                return create_user_profile(**profile_data)
            except Exception as create_error:
                logger.warning(f"Error using create_user_profile: {create_error}")
                # Fallback to basic profile object
                class BasicProfile:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                        self.bmi = self.weight / ((self.height / 100) ** 2)
                        self.bmi_category = self._calculate_bmi_category()
                    
                    def _calculate_bmi_category(self):
                        if self.bmi < 18.5:
                            return "Underweight"
                        elif self.bmi < 25:
                            return "Normal"
                        elif self.bmi < 30:
                            return "Overweight"
                        else:
                            return "Obese"
                    
                    def to_dict(self):
                        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
                
                return BasicProfile(**profile_data)
        
        return None
    except Exception as e:
        st.error(f"Error creating profile: {e}")
        return None

def check_user_profile():
    """Check if user has a complete profile."""
    
    if st.session_state.user_profile is None:
        return False
    
    # Check for required profile fields
    required_fields = ['age', 'weight', 'height', 'gender', 'activity_level', 'fitness_level']
    
    try:
        if hasattr(st.session_state.user_profile, 'to_dict'):
            profile_dict = st.session_state.user_profile.to_dict()
        else:
            profile_dict = st.session_state.user_profile.__dict__
        
        return all(profile_dict.get(field) is not None for field in required_fields)
    except:
        return False

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
                new_profile = create_user_profile_from_form(
                    age=age,
                    gender=gender,
                    height=height,
                    weight=weight,
                    activity_level=activity_level,
                    fitness_level=fitness_level,
                    primary_goal=primary_goal
                )
                
                if new_profile:
                    st.session_state.user_profile = new_profile
                    st.session_state.user_id = getattr(new_profile, 'user_id', str(age) + str(weight))
                    st.session_state.profile_complete = True
                    st.success("Profile created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create profile. Please try again.")
            except Exception as e:
                st.error(f"Error creating profile: {e}")

def render_sidebar(user_profile):
    """Render sidebar with user information and controls."""
    
    st.sidebar.header("👤 Your Profile")
    
    # User summary
    st.sidebar.write(f"**Age:** {getattr(user_profile, 'age', 'N/A')}")
    
    try:
        bmi = getattr(user_profile, 'bmi', 0)
        bmi_category = getattr(user_profile, 'bmi_category', 'Unknown')
        st.sidebar.write(f"**BMI:** {bmi:.1f} ({bmi_category})")
    except:
        st.sidebar.write("**BMI:** Not calculated")
    
    # Safe access to enum values
    try:
        fitness_level = getattr(user_profile, 'fitness_level', 'Unknown')
        fitness_level_str = safe_get_enum_value(fitness_level)
        st.sidebar.write(f"**Fitness Level:** {fitness_level_str.replace('_', ' ').title()}")
    except:
        st.sidebar.write("**Fitness Level:** Unknown")
    
    try:
        primary_goal = getattr(user_profile, 'primary_goal', 'Unknown')
        primary_goal_str = safe_get_enum_value(primary_goal)
        st.sidebar.write(f"**Primary Goal:** {primary_goal_str.replace('_', ' ').title()}")
    except:
        st.sidebar.write("**Primary Goal:** Unknown")
    
    st.sidebar.divider()
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Update Profile"):
        st.session_state.profile_complete = False
        st.rerun()
    
    if st.sidebar.button("👤 Create New Profile"):
        st.session_state.show_profile_creation = True
        st.session_state.user_profile = None
        st.session_state.profile_complete = False
        st.rerun()
    
    if st.sidebar.button("🔄 Generate New Recommendations"):
        # Clear recommendations cache
        st.session_state.recommendations_cache = {}
        st.success("Recommendations will be regenerated!")
    
    st.sidebar.markdown("---")
    
    # Application info
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown("**APT Fitness Assistant v3.0**")
    st.sidebar.markdown("Your intelligent fitness companion")
    st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/alphareum/apt-proof-of-concept)")
    
    # Show available features
    st.sidebar.markdown("### 🎯 Available Features")
    if VISION_AVAILABLE:
        st.sidebar.success("✅ Computer Vision")
    else:
        st.sidebar.error("❌ Computer Vision")
    
    if BODY_COMP_AVAILABLE:
        st.sidebar.success("✅ Body Analysis")
    else:
        st.sidebar.error("❌ Body Analysis")
    
    if ENHANCED_PLANNER_AVAILABLE:
        st.sidebar.success("✅ Enhanced Planner")
    else:
        st.sidebar.error("❌ Enhanced Planner")
    
    if DATABASE_AVAILABLE:
        st.sidebar.success("✅ Database")
    else:
        st.sidebar.error("❌ Database")

def render_body_analysis_tab(user_profile):
    """Render body analysis tab."""
    
    st.markdown("## 📊 Body Composition Analysis")
    
    if not VISION_AVAILABLE:
        st.warning("⚠️ Computer vision libraries not available. Body analysis features are limited.")
        render_manual_measurements_only(user_profile)
        return
    
    analysis_tabs = st.tabs(["📸 Image Analysis", "📏 Manual Measurements", "📈 Progress History"])
    
    with analysis_tabs[0]:
        render_image_analysis(user_profile)
    
    with analysis_tabs[1]:
        render_manual_measurements(user_profile)
    
    with analysis_tabs[2]:
        render_measurement_history(user_profile)

def render_manual_measurements_only(user_profile):
    """Render manual measurements when computer vision is not available."""
    
    st.markdown("### 📏 Manual Body Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, 
                                value=float(getattr(user_profile, 'weight', 70.0)), step=0.1)
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
        measurements = {
            'weight': weight,
            'waist': waist,
            'chest': chest,
            'neck': neck,
            'body_fat_percentage': body_fat,
            'arms': arms,
            'thighs': thighs,
            'hips': hips,
            'notes': measurement_notes
        }
        
        # Save to session state if database is not available
        if 'manual_measurements' not in st.session_state:
            st.session_state.manual_measurements = []
        
        measurements['date'] = datetime.now().isoformat()
        measurements['user_id'] = getattr(user_profile, 'user_id', 'default')
        st.session_state.manual_measurements.append(measurements)
        
        st.success("✅ Measurements saved successfully!")

def render_image_analysis(user_profile):
    """Render image-based body analysis."""
    
    st.markdown("### 📸 Upload Body Photo for Analysis")
    st.info("Upload a clear, full-body photo for AI-powered body composition analysis.")
    
    if not BODY_COMP_AVAILABLE:
        st.warning("⚠️ Body composition analyzer not available. Feature coming soon!")
        return
    
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
                    
                    # Perform analysis (simplified version for demo)
                    analysis_result = perform_basic_image_analysis(img_bytes, user_profile)
                    
                    if 'error' not in analysis_result:
                        display_analysis_results(analysis_result, user_profile)
                    else:
                        st.error(f"❌ Analysis failed: {analysis_result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

def perform_basic_image_analysis(img_bytes: bytes, user_profile) -> Dict[str, Any]:
    """Perform basic image analysis when full body comp analyzer is not available."""
    
    try:
        # Create a mock analysis result for demonstration
        import random
        
        # Basic estimates based on user profile
        age = getattr(user_profile, 'age', 30)
        weight = getattr(user_profile, 'weight', 70)
        height = getattr(user_profile, 'height', 170)
        gender = getattr(user_profile, 'gender', 'male')
        
        # Simple BMI-based estimates
        bmi = weight / ((height / 100) ** 2)
        
        if str(gender).lower() == 'male':
            base_bf = max(8, min(25, (1.20 * bmi) + (0.23 * age) - 16.2))
        else:
            base_bf = max(12, min(35, (1.20 * bmi) + (0.23 * age) - 5.4))
        
        # Add some random variation for demo
        body_fat = base_bf + random.uniform(-2, 2)
        muscle_mass = max(30, min(50, 100 - body_fat - random.uniform(15, 25)))
        
        # Calculate BMR using Mifflin-St Jeor equation
        if str(gender).lower() == 'male':
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        
        return {
            'success': True,
            'estimated_body_fat': round(body_fat, 1),
            'muscle_mass_percentage': round(muscle_mass, 1),
            'bmr_estimated': int(bmr),
            'body_shape': 'Rectangular' if bmi < 25 else 'Oval',
            'confidence': 0.75,
            'analysis_type': 'Basic Estimation',
            'note': 'This is a basic estimation. For accurate results, please install the full body composition analyzer.'
        }
        
    except Exception as e:
        return {'error': f'Analysis error: {str(e)}'}

def display_analysis_results(results: Dict[str, Any], user_profile):
    """Display body composition analysis results."""
    
    st.success("✅ Analysis completed!")
    
    if results.get('note'):
        st.info(f"ℹ️ {results['note']}")
    
    # Main metrics
    st.markdown("### 📊 Body Composition Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Body Fat %",
            f"{results['estimated_body_fat']:.1f}%",
            help="Estimated body fat percentage"
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
    
    # Recommendations based on results
    st.markdown("### 💡 Recommendations")
    recommendations = generate_recommendations_from_analysis(results, user_profile)
    for rec in recommendations:
        st.markdown(f"• {rec}")

def generate_recommendations_from_analysis(analysis: Dict[str, Any], user_profile) -> List[str]:
    """Generate recommendations based on analysis results."""
    
    recommendations = []
    body_fat = analysis.get('estimated_body_fat', 20)
    
    if body_fat < 10:
        recommendations.append("⚠️ Body fat is very low - consider consulting a nutritionist")
    elif body_fat < 15:
        recommendations.append("💪 Excellent body fat level - focus on maintaining muscle mass")
    elif body_fat < 25:
        recommendations.append("✅ Healthy body fat range - continue current routine")
    else:
        recommendations.append("🎯 Consider incorporating more cardio and strength training")
    
    # Goal-specific recommendations
    goal = str(getattr(user_profile, 'primary_goal', '')).lower()
    if 'weight_loss' in goal:
        recommendations.append("🍎 Focus on creating a moderate caloric deficit")
    elif 'muscle' in goal:
        recommendations.append("🥩 Ensure adequate protein intake (0.8-1g per lb bodyweight)")
    
    return recommendations

def render_manual_measurements(user_profile):
    """Render manual measurements input."""
    render_manual_measurements_only(user_profile)

def render_measurement_history(user_profile):
    """Render measurement history."""
    
    st.markdown("### 📈 Measurement History")
    
    # Check session state for saved measurements
    if 'manual_measurements' in st.session_state and st.session_state.manual_measurements:
        measurements = st.session_state.manual_measurements
        
        # Display as table
        try:
            import pandas as pd
            df = pd.DataFrame(measurements)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(df.drop(columns=['user_id'], errors='ignore'), use_container_width=True)
            
            # Simple weight chart if available
            if 'weight' in df.columns and len(df) > 1:
                st.markdown("#### Weight Progress")
                st.line_chart(df.set_index('date')['weight'])
        except ImportError:
            # Fallback display without pandas
            st.markdown("#### Measurement Records")
            for i, measurement in enumerate(measurements):
                with st.expander(f"Measurement {i+1} - {measurement.get('date', 'Unknown date')[:10]}"):
                    for key, value in measurement.items():
                        if key not in ['user_id', 'date']:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    else:
        st.info("No measurements recorded yet. Add some measurements to see your progress!")

def render_recommendations_tab(user_profile):
    """Render exercise recommendations tab."""
    
    st.header("💪 Exercise Recommendations")
    
    if ENHANCED_PLANNER_AVAILABLE and RECOMMENDATION_ENGINE_AVAILABLE:
        try:
            render_enhanced_recommendations(user_profile)
        except Exception as e:
            st.warning(f"Enhanced recommendations unavailable: {str(e)}")
            render_basic_recommendations(user_profile)
    else:
        render_basic_recommendations(user_profile)

def render_enhanced_recommendations(user_profile):
    """Render enhanced AI-powered recommendations."""
    
    st.markdown("### 🤖 AI-Powered Workout Recommendations")
    st.info("Generating personalized recommendations based on your profile...")
    
    try:
        # Try to use enhanced recommendation engine
        enhanced_engine = EnhancedRecommendationEngine()
        
        # Create a safe wrapper for the user profile that handles both enum and string values
        class SafeUserProfile:
            def __init__(self, original_profile):
                self._original = original_profile
                # Copy all attributes from original profile
                for attr_name in dir(original_profile):
                    if not attr_name.startswith('_'):
                        value = getattr(original_profile, attr_name)
                        setattr(self, attr_name, value)
            
            def __getattr__(self, name):
                # If attribute doesn't exist, try to get from original
                if hasattr(self._original, name):
                    value = getattr(self._original, name)
                    # Convert string values to actual enum objects if possible
                    if isinstance(value, str) and name in ['primary_goal', 'fitness_level', 'activity_level', 'gender']:
                        try:
                            if name == 'fitness_level':
                                from models import FitnessLevel
                                for enum_member in FitnessLevel:
                                    if enum_member.value == value:
                                        return enum_member
                            elif name == 'activity_level':
                                from models import ActivityLevel
                                for enum_member in ActivityLevel:
                                    if enum_member.value == value:
                                        return enum_member
                            elif name == 'primary_goal':
                                from models import GoalType
                                for enum_member in GoalType:
                                    if enum_member.value == value:
                                        return enum_member
                            elif name == 'gender':
                                from models import Gender
                                for enum_member in Gender:
                                    if enum_member.value == value:
                                        return enum_member
                        except ImportError:
                            pass
                        # Create a mock enum-like object as fallback
                        class MockEnum:
                            def __init__(self, val):
                                self.value = val
                            def __str__(self):
                                return self.value
                        return MockEnum(value)
                    return value
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        safe_profile = SafeUserProfile(user_profile)
        recommendations = enhanced_engine.generate_adaptive_recommendations(safe_profile)
        
        if 'error' not in recommendations:
            # Display program options if available
            program_options = recommendations.get('program_options', [])
            
            if program_options:
                st.markdown("#### 🎯 Recommended Programs")
                
                for i, program in enumerate(program_options[:3]):  # Show top 3
                    with st.expander(f"Option {i+1}: {program.get('name', 'Workout Program')}", expanded=i==0):
                        st.write(program.get('description', 'No description available'))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Duration:** {program.get('duration_per_session', 'N/A')} min")
                            st.write(f"**Frequency:** {program.get('sessions_per_week', 'N/A')}/week")
                        
                        with col2:
                            st.write(f"**Style:** {program.get('intensity_style', 'N/A').replace('_', ' ').title()}")
                            equipment = program.get('equipment_needed', [])
                            st.write(f"**Equipment:** {', '.join(equipment[:3]) if equipment else 'None'}")
                        
                        if st.button(f"Select Program {i+1}", key=f"select_{i}"):
                            st.session_state.selected_program = program
                            st.success(f"Selected: {program.get('name', 'Program')}")
            
            # Show daily workout if available
            daily_workouts = recommendations.get('daily_workouts', {})
            if daily_workouts:
                st.markdown("#### Today's Suggested Workout")
                for workout_name, workout in daily_workouts.items():
                    st.subheader(workout_name.replace('_', ' ').title())
                    if isinstance(workout, dict) and 'exercises' in workout:
                        for exercise in workout['exercises'][:5]:  # Show first 5 exercises
                            st.write(f"• {exercise.get('name', 'Exercise')}: {exercise.get('sets', 'N/A')} sets x {exercise.get('reps', 'N/A')}")
        else:
            st.error(f"Enhanced recommendations failed: {recommendations['error']}")
            render_basic_recommendations(user_profile)
    
    except Exception as e:
        st.error(f"Error generating enhanced recommendations: {str(e)}")
        logger.error(f"Enhanced recommendations error: {str(e)}")
        render_basic_recommendations(user_profile)

def render_basic_recommendations(user_profile):
    """Render basic exercise recommendations."""
    
    st.markdown("### 📋 Basic Recommendations")
    st.info("Based on your profile, here are some recommended exercises:")
    
    # Generate basic recommendations
    fitness_level = safe_get_enum_value(getattr(user_profile, 'fitness_level', 'beginner')).lower()
    goal = safe_get_enum_value(getattr(user_profile, 'primary_goal', 'general_fitness')).lower()
    
    recommendations = []
    
    if 'beginner' in fitness_level:
        recommendations.extend([
            {'name': 'Walking', 'type': 'Cardio', 'duration': '20-30 minutes', 'description': 'Start with daily walks to build endurance'},
            {'name': 'Bodyweight Squats', 'type': 'Strength', 'duration': '2 sets of 8-12', 'description': 'Build lower body strength'},
            {'name': 'Wall Push-ups', 'type': 'Strength', 'duration': '2 sets of 5-10', 'description': 'Begin building upper body strength'}
        ])
    elif 'intermediate' in fitness_level:
        recommendations.extend([
            {'name': 'Jogging', 'type': 'Cardio', 'duration': '30-45 minutes', 'description': 'Moderate intensity cardio'},
            {'name': 'Push-ups', 'type': 'Strength', 'duration': '3 sets of 10-15', 'description': 'Upper body strengthening'},
            {'name': 'Lunges', 'type': 'Strength', 'duration': '3 sets of 10 each leg', 'description': 'Lower body and balance'}
        ])
    else:  # advanced
        recommendations.extend([
            {'name': 'HIIT Training', 'type': 'Cardio', 'duration': '20-30 minutes', 'description': 'High intensity interval training'},
            {'name': 'Weighted Squats', 'type': 'Strength', 'duration': '4 sets of 8-12', 'description': 'Advanced lower body strength'},
            {'name': 'Pull-ups', 'type': 'Strength', 'duration': '3 sets of 5-10', 'description': 'Advanced upper body strength'}
        ])
    
    # Goal-specific additions
    if 'weight_loss' in goal:
        recommendations.append({'name': 'Circuit Training', 'type': 'Mixed', 'duration': '30 minutes', 'description': 'Combines cardio and strength for fat burning'})
    elif 'muscle_gain' in goal:
        recommendations.append({'name': 'Progressive Overload', 'type': 'Strength', 'duration': '45-60 minutes', 'description': 'Gradually increase weights for muscle growth'})
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.subheader(rec['name'])
                st.caption(f"Type: {rec['type']}")
            
            with col2:
                st.metric("Duration", rec['duration'])
            
            with col3:
                st.write(rec['description'])
            
            st.divider()

def render_workout_planner_tab(user_profile):
    """Render workout planner tab."""
    
    st.header("📅 Workout Planner")
    
    if ENHANCED_PLANNER_AVAILABLE:
        try:
            # Create a safe wrapper for the user profile that handles both enum and string values
            class SafeUserProfile:
                def __init__(self, original_profile):
                    self._original = original_profile
                    # Copy all attributes from original profile
                    for attr_name in dir(original_profile):
                        if not attr_name.startswith('_'):
                            value = getattr(original_profile, attr_name)
                            setattr(self, attr_name, value)
                
                def __getattr__(self, name):
                    # If attribute doesn't exist, try to get from original
                    if hasattr(self._original, name):
                        value = getattr(self._original, name)
                        # Convert string values to actual enum objects if possible
                        if isinstance(value, str) and name in ['primary_goal', 'fitness_level', 'activity_level', 'gender']:
                            try:
                                if name == 'fitness_level':
                                    from models import FitnessLevel
                                    for enum_member in FitnessLevel:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'activity_level':
                                    from models import ActivityLevel
                                    for enum_member in ActivityLevel:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'primary_goal':
                                    from models import GoalType
                                    for enum_member in GoalType:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'gender':
                                    from models import Gender
                                    for enum_member in Gender:
                                        if enum_member.value == value:
                                            return enum_member
                            except ImportError:
                                pass
                            # Create a mock enum-like object as fallback
                            class MockEnum:
                                def __init__(self, val):
                                    self.value = val
                                def __str__(self):
                                    return self.value
                            return MockEnum(value)
                        return value
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            safe_profile = SafeUserProfile(user_profile)
            # Try to use enhanced workout planner UI
            planner_ui = WorkoutPlannerUI()
            planner_ui.render_workout_planner_tab(safe_profile)
        except Exception as e:
            st.warning(f"Enhanced planner unavailable: {str(e)}")
            logger.error(f"Enhanced planner error: {str(e)}")
            render_basic_workout_planner(user_profile)
    else:
        render_basic_workout_planner(user_profile)

def render_basic_workout_planner(user_profile):
    """Render basic workout planner."""
    
    st.markdown("### 📅 Weekly Workout Schedule")
    
    # Days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get user preferences
    col1, col2 = st.columns(2)
    with col1:
        workouts_per_week = st.selectbox("Workouts per week", [3, 4, 5, 6], index=1)
    with col2:
        rest_preference = st.selectbox("Prefer rest on", ["Weekends", "Mid-week", "Mixed"])
    
    # Generate basic schedule
    st.markdown("#### Your Suggested Schedule")
    
    fitness_level = safe_get_enum_value(getattr(user_profile, 'fitness_level', 'beginner')).lower()
    
    for i, day in enumerate(days):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader(day)
            
            with col2:
                # Simple logic for workout assignment
                if i < workouts_per_week:
                    if i % 2 == 0:
                        workout_type = "Strength Training"
                        duration = "45 min" if 'advanced' in fitness_level else "30 min"
                    else:
                        workout_type = "Cardio"
                        duration = "30 min" if 'beginner' in fitness_level else "45 min"
                    
                    st.write(f"**{workout_type}** - {duration}")
                    st.caption("Click here to see detailed workout")
                else:
                    st.write("**Rest Day** 🛌")
                    st.caption("Recovery and light stretching")
            
            st.divider()

def render_nutrition_tab(user_profile):
    """Render nutrition tab."""
    
    st.header("🍎 Nutrition & Meal Planning")
    
    if ENHANCED_UI_AVAILABLE:
        try:
            # Create a safe wrapper for the user profile that handles both enum and string values
            class SafeUserProfile:
                def __init__(self, original_profile):
                    self._original = original_profile
                    # Copy all attributes from original profile
                    for attr_name in dir(original_profile):
                        if not attr_name.startswith('_'):
                            value = getattr(original_profile, attr_name)
                            setattr(self, attr_name, value)
                
                def __getattr__(self, name):
                    # If attribute doesn't exist, try to get from original
                    if hasattr(self._original, name):
                        value = getattr(self._original, name)
                        # Convert string values to actual enum objects if possible
                        if isinstance(value, str) and name in ['primary_goal', 'fitness_level', 'activity_level', 'gender']:
                            try:
                                if name == 'fitness_level':
                                    from models import FitnessLevel
                                    for enum_member in FitnessLevel:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'activity_level':
                                    from models import ActivityLevel
                                    for enum_member in ActivityLevel:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'primary_goal':
                                    from models import GoalType
                                    for enum_member in GoalType:
                                        if enum_member.value == value:
                                            return enum_member
                                elif name == 'gender':
                                    from models import Gender
                                    for enum_member in Gender:
                                        if enum_member.value == value:
                                            return enum_member
                            except ImportError:
                                pass
                            # Create a mock enum-like object as fallback
                            class MockEnum:
                                def __init__(self, val):
                                    self.value = val
                                def __str__(self):
                                    return self.value
                            return MockEnum(value)
                        return value
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            safe_profile = SafeUserProfile(user_profile)
            enhanced_ui.render_nutrition_tab(safe_profile)
        except Exception as e:
            st.warning(f"Enhanced nutrition features unavailable: {str(e)}")
            # Get more detailed error info
            import traceback
            logger.error(f"Enhanced nutrition error details: {traceback.format_exc()}")
            render_basic_nutrition(user_profile)
    else:
        render_basic_nutrition(user_profile)

def render_basic_nutrition(user_profile):
    """Render basic nutrition information."""
    
    st.markdown("### 🎯 Nutrition Goals")
    
    # Calculate basic nutritional needs
    try:
        weight = getattr(user_profile, 'weight', 70)
        height = getattr(user_profile, 'height', 170)
        age = getattr(user_profile, 'age', 30)
        gender = safe_get_enum_value(getattr(user_profile, 'gender', 'male')).lower()
        activity_level = safe_get_enum_value(getattr(user_profile, 'activity_level', 'moderately_active')).lower()
        goal = safe_get_enum_value(getattr(user_profile, 'primary_goal', 'general_fitness')).lower()
        
        # Calculate BMR using Mifflin-St Jeor equation
        if gender == 'male':
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        
        # Activity multipliers
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        
        multiplier = activity_multipliers.get(activity_level, 1.55)
        tdee = bmr * multiplier
        
        # Adjust for goals
        if 'weight_loss' in goal:
            target_calories = tdee - 500  # 500 calorie deficit
        elif 'muscle_gain' in goal:
            target_calories = tdee + 300  # 300 calorie surplus
        else:
            target_calories = tdee
        
        # Macronutrient breakdown
        protein_g = weight * 2.2  # 1g per lb bodyweight
        fat_g = target_calories * 0.25 / 9  # 25% of calories from fat
        carb_g = (target_calories - (protein_g * 4) - (fat_g * 9)) / 4
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Calories", f"{int(target_calories)}")
        
        with col2:
            st.metric("Protein", f"{int(protein_g)}g")
        
        with col3:
            st.metric("Carbs", f"{int(carb_g)}g")
        
        with col4:
            st.metric("Fat", f"{int(fat_g)}g")
        
        # Basic meal suggestions
        st.markdown("### 🍽️ Meal Suggestions")
        
        meal_ideas = {
            "Breakfast": ["Oatmeal with berries and protein powder", "Greek yogurt with nuts and honey", "Scrambled eggs with whole grain toast"],
            "Lunch": ["Grilled chicken salad", "Quinoa bowl with vegetables", "Lean protein with brown rice"],
            "Dinner": ["Baked salmon with sweet potato", "Lean beef with roasted vegetables", "Tofu stir-fry with brown rice"],
            "Snacks": ["Apple with almond butter", "Greek yogurt", "Mixed nuts and berries"]
        }
        
        for meal, ideas in meal_ideas.items():
            with st.expander(f"{meal} Ideas"):
                for idea in ideas:
                    st.write(f"• {idea}")
    
    except Exception as e:
        st.error(f"Error calculating nutrition goals: {str(e)}")
        st.info("Please ensure your profile is complete for personalized nutrition recommendations.")

def render_analytics_tab(user_profile):
    """Render analytics and progress tracking tab."""
    
    st.header("📈 Progress Analytics")
    
    # Basic analytics dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Simulated data - in real app this would come from database
        st.metric("Workouts This Week", "4", "+1")
    
    with col2:
        st.metric("Current Streak", "7 days", "+1")
    
    with col3:
        st.metric("Goals Achieved", "3", "+1")
    
    # Progress visualization
    st.markdown("### 📊 Progress Visualization")
    
    if 'manual_measurements' in st.session_state and st.session_state.manual_measurements:
        measurements = st.session_state.manual_measurements
        
        if len(measurements) > 1:
            # Create weight progress chart
            try:
                import pandas as pd
                df = pd.DataFrame(measurements)
                df['date'] = pd.to_datetime(df['date'])
                
                if 'weight' in df.columns:
                    st.markdown("#### Weight Progress")
                    st.line_chart(df.set_index('date')['weight'])
                
                if 'body_fat_percentage' in df.columns:
                    st.markdown("#### Body Fat Progress")
                    st.line_chart(df.set_index('date')['body_fat_percentage'])
            except ImportError:
                st.info("Install pandas for advanced progress charts.")
                # Simple text-based progress display
                st.markdown("#### Recent Measurements")
                for measurement in measurements[-5:]:  # Show last 5
                    date_str = measurement.get('date', 'Unknown')[:10]
                    weight = measurement.get('weight', 'N/A')
                    st.write(f"**{date_str}:** Weight: {weight} kg")
        else:
            st.info("Add more measurements to see progress charts!")
    else:
        st.info("No progress data available yet. Start tracking measurements to see analytics!")

def render_form_correction_tab(user_profile):
    """Render form correction tab."""
    
    st.header("🎯 Workout Form Correction")
    
    st.info("📹 Form correction feature helps you perfect your exercise technique using AI pose detection.")
    
    # Exercise selection
    exercise_type = st.selectbox(
        "Select Exercise",
        ["Squat", "Push-up", "Deadlift", "Plank", "Lunge"],
        help="Choose the exercise you want to analyze"
    )
    
    # Display exercise-specific form tips
    st.markdown(f"### 🏋️‍♀️ {exercise_type} Form Guide")
    
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
    
    # Video upload placeholder
    st.markdown("### 📹 Upload Exercise Video")
    st.info("💡 Video analysis feature coming soon! Upload a video of your exercise for real-time form feedback.")
    
    uploaded_video = st.file_uploader(
        "Choose a video file...",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video of your exercise for analysis"
    )
    
    if uploaded_video is not None:
        st.info("Video uploaded successfully! Analysis feature will be available in a future update.")

def render_goal_management_tab(user_profile):
    """Render goal management tab."""
    
    st.header("🏆 Goal Management")
    
    st.info("Set and track your fitness goals to stay motivated and measure progress.")
    
    # Goal creation form
    st.markdown("### ➕ Create New Goal")
    
    with st.form("goal_form"):
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
                help="Your target value (kg, %, reps, etc.)"
            )
        
        with col2:
            current_value = st.number_input(
                "Current Value", 
                min_value=0.0,
                value=0.0,
                step=0.5,
                help="Your current starting value"
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
        
        submitted = st.form_submit_button("🎯 Create Goal", type="primary")
        
        if submitted:
            # Save goal to session state
            if 'user_goals' not in st.session_state:
                st.session_state.user_goals = []
            
            new_goal = {
                'type': goal_type,
                'target_value': target_value,
                'current_value': current_value,
                'target_date': target_date.isoformat(),
                'description': goal_description,
                'created_at': datetime.now().isoformat(),
                'user_id': getattr(user_profile, 'user_id', 'default')
            }
            
            st.session_state.user_goals.append(new_goal)
            st.success("✅ Goal created successfully!")
    
    # Display existing goals
    st.markdown("### 📋 Your Goals")
    
    if 'user_goals' in st.session_state and st.session_state.user_goals:
        for i, goal in enumerate(st.session_state.user_goals):
            with st.expander(f"🎯 {goal['type']} - Target: {goal['target_value']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Current:** {goal['current_value']}")
                    st.write(f"**Target:** {goal['target_value']}")
                    st.write(f"**Target Date:** {goal['target_date']}")
                
                with col2:
                    # Calculate progress
                    if goal['target_value'] > 0:
                        progress = (goal['current_value'] / goal['target_value']) * 100
                        st.progress(min(progress / 100, 1.0))
                        st.write(f"**Progress:** {progress:.1f}%")
                    
                    # Days remaining
                    try:
                        target_date = datetime.fromisoformat(goal['target_date']).date()
                        days_remaining = (target_date - datetime.now().date()).days
                        st.write(f"**Days Remaining:** {days_remaining}")
                    except:
                        pass
                
                if goal['description']:
                    st.write(f"**Description:** {goal['description']}")
                
                # Update progress button
                if st.button(f"Update Progress", key=f"update_{i}"):
                    new_value = st.number_input(f"New value for {goal['type']}", 
                                               value=goal['current_value'], 
                                               key=f"new_val_{i}")
                    goal['current_value'] = new_value
                    st.success("Progress updated!")
                    st.rerun()
    else:
        st.info("No goals set yet. Create your first goal above!")

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title("🏋️‍♀️ APT Fitness Assistant")
    st.subheader("Your AI-Powered Fitness Companion")
    
    # Check if models are available for profile creation
    if not MODELS_AVAILABLE:
        st.error("❌ Core models not available. Some features may be limited.")
        st.info("Please ensure all dependencies are properly installed.")
    
    # Handle profile creation
    if st.session_state.get('show_profile_creation', False):
        render_profile_setup()
        return
    
    # Check if user has completed profile
    if not check_user_profile():
        render_profile_setup()
        return
    
    # Main application interface
    user_profile = st.session_state.user_profile
    
    # Render sidebar
    render_sidebar(user_profile)
    
    # Main content area with tabs
    tab_labels = [
        "📊 Body Analysis", 
        "💪 Recommendations", 
        "📅 Workout Planner",
        "🍎 Nutrition",
        "📈 Analytics",
        "🎯 Form Correction",
        "🏆 Goals"
    ]
    
    tabs = st.tabs(tab_labels)
    
    with tabs[0]:
        render_body_analysis_tab(user_profile)
    
    with tabs[1]:
        render_recommendations_tab(user_profile)
    
    with tabs[2]:
        render_workout_planner_tab(user_profile)
    
    with tabs[3]:
        render_nutrition_tab(user_profile)
    
    with tabs[4]:
        render_analytics_tab(user_profile)
    
    with tabs[5]:
        render_form_correction_tab(user_profile)
    
    with tabs[6]:
        render_goal_management_tab(user_profile)

if __name__ == "__main__":
    main()
