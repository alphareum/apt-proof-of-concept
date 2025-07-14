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
    
    # Import body composition analysis
    try:
        from body_composition_analyzer import get_body_analyzer
        BODY_COMP_AVAILABLE = True
    except ImportError:
        BODY_COMP_AVAILABLE = False
    
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please ensure all required files are present and dependencies are installed.")
    st.stop()

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
    
    # Main content area with tabs
    tabs = st.tabs([
        "📊 Body Analysis", 
        "💪 Recommendations", 
        "🎯 Form Correction",
        "📈 Progress Tracking",
        "🏆 Goal Management"
    ])
    
    with tabs[0]:
        render_body_analysis_tab(user_profile)
    
    with tabs[1]:
        render_recommendations_tab(user_profile)
    
    with tabs[2]:
        render_form_correction_tab(user_profile)
    
    with tabs[3]:
        render_progress_tracking_tab(user_profile)
    
    with tabs[4]:
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
    
    if st.sidebar.button("📊 Generate New Recommendations"):
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
    
    # Basic recommendations display
    recommendations = generate_basic_recommendations(user_profile)
    for rec in recommendations:
        st.subheader(rec['name'])
        st.write(f"**Type:** {rec['type']}")
        st.write(f"**Duration:** {rec['duration']}")
        st.write(f"**Description:** {rec['description']}")
        st.divider()

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
