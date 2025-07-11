"""
AI Fitness Assistant - Streamlit Application

A comprehensive fitness application with:
- Tab 1: Body Fat Analysis from Images
- Tab 2: Exercise Recommendations based on User Data  
- Tab 3: Workout Form Correction using Pose Detection

Author: AI Fitness Team
Version: 1.0.0
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import math

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("⚠️ MediaPipe not installed. Install with: pip install mediapipe")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration."""
    
    APP_TITLE = "AI Fitness Assistant"
    APP_ICON = "🏋️‍♀️"
    VERSION = "1.0.0"
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm']
    
    # Body fat estimation parameters
    BODY_FAT_FEATURES = ['age', 'weight', 'height', 'activity_level', 'gender']
    
    # Exercise database
    EXERCISES_DB = {
        'cardio': {
            'running': {'calories_per_min': 10, 'difficulty': 'medium', 'equipment': 'none'},
            'cycling': {'calories_per_min': 8, 'difficulty': 'low', 'equipment': 'bike'},
            'swimming': {'calories_per_min': 12, 'difficulty': 'medium', 'equipment': 'pool'},
            'jumping_jacks': {'calories_per_min': 8, 'difficulty': 'low', 'equipment': 'none'},
        },
        'strength': {
            'push_ups': {'calories_per_min': 6, 'difficulty': 'medium', 'equipment': 'none'},
            'squats': {'calories_per_min': 7, 'difficulty': 'medium', 'equipment': 'none'},
            'deadlifts': {'calories_per_min': 8, 'difficulty': 'high', 'equipment': 'weights'},
            'bench_press': {'calories_per_min': 6, 'difficulty': 'high', 'equipment': 'weights'},
            'lat_pulldown': {'calories_per_min': 6, 'difficulty': 'medium', 'equipment': 'machine'},
        },
        'flexibility': {
            'yoga': {'calories_per_min': 3, 'difficulty': 'low', 'equipment': 'mat'},
            'stretching': {'calories_per_min': 2, 'difficulty': 'low', 'equipment': 'none'},
            'pilates': {'calories_per_min': 4, 'difficulty': 'medium', 'equipment': 'mat'},
        }
    }

# Page configuration
st.set_page_config(
    page_title=f"{Config.APP_TITLE} v{Config.VERSION}",
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 1rem 0;
    }
    .exercise-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .pose-point {
        color: #28a745;
        font-weight: bold;
    }
    .pose-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .pose-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class BodyFatAnalyzer:
    """Body fat analysis using anthropometric measurements and visual estimation."""
    
    @staticmethod
    def calculate_body_fat_percentage(age: int, weight: float, height: float, 
                                    gender: str, activity_level: str,
                                    waist: float = None, neck: float = None, 
                                    hip: float = None) -> Dict[str, Any]:
        """Calculate body fat percentage using multiple methods."""
        
        # BMI calculation
        height_m = height / 100  # Convert cm to meters
        bmi = weight / (height_m ** 2)
        
        # Navy Method (if measurements provided)
        navy_bf = None
        if waist and neck:
            if gender.lower() == 'male':
                if waist > 0 and neck > 0:
                    navy_bf = 495 / (1.0324 - 0.19077 * math.log10(waist - neck) + 0.15456 * math.log10(height)) - 450
            elif hip:
                if waist > 0 and neck > 0 and hip > 0:
                    navy_bf = 495 / (1.29579 - 0.35004 * math.log10(waist + hip - neck) + 0.22100 * math.log10(height)) - 450
        
        # Simplified estimation based on BMI, age, and activity
        activity_modifier = {
            'sedentary': 1.2,
            'lightly_active': 1.0,
            'moderately_active': 0.9,
            'very_active': 0.8,
            'extremely_active': 0.7
        }
        
        # Base calculation using BMI and demographic factors
        if gender.lower() == 'male':
            base_bf = (1.20 * bmi) + (0.23 * age) - 16.2
        else:
            base_bf = (1.20 * bmi) + (0.23 * age) - 5.4
        
        # Apply activity modifier
        estimated_bf = base_bf * activity_modifier.get(activity_level, 1.0)
        estimated_bf = max(3, min(50, estimated_bf))  # Clamp between reasonable bounds
        
        # Use Navy method if available, otherwise use estimation
        final_bf = navy_bf if navy_bf and 3 <= navy_bf <= 50 else estimated_bf
        
        # Classification
        if gender.lower() == 'male':
            if final_bf < 6:
                category = "Essential Fat"
            elif final_bf < 14:
                category = "Athletic"
            elif final_bf < 18:
                category = "Fitness"
            elif final_bf < 25:
                category = "Acceptable"
            else:
                category = "Obesity"
        else:  # Female
            if final_bf < 14:
                category = "Essential Fat"
            elif final_bf < 21:
                category = "Athletic"
            elif final_bf < 25:
                category = "Fitness"
            elif final_bf < 32:
                category = "Acceptable"
            else:
                category = "Obesity"
        
        return {
            'body_fat_percentage': round(final_bf, 1),
            'bmi': round(bmi, 1),
            'category': category,
            'navy_method': round(navy_bf, 1) if navy_bf else None,
            'estimation_method': 'Navy Formula' if navy_bf else 'BMI-based Estimation'
        }
    
    @staticmethod
    def analyze_body_composition_from_image(image: np.ndarray) -> Dict[str, Any]:
        """Analyze body composition from image using computer vision techniques."""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple body detection using contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No body detected in image'}
        
        # Find largest contour (assumed to be the body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate basic metrics
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Simple heuristics for body composition estimation
        # These are simplified estimates and would need ML models for accuracy
        density_score = area / (w * h) if w * h > 0 else 0
        
        # Estimate muscle vs fat based on image characteristics
        muscle_indicator = density_score * 100
        fat_indicator = (1 - density_score) * 100
        
        return {
            'body_area': area,
            'aspect_ratio': round(aspect_ratio, 2),
            'muscle_indicator': round(muscle_indicator, 1),
            'fat_indicator': round(fat_indicator, 1),
            'body_detected': True,
            'analysis_method': 'Computer Vision Estimation'
        }

class ExerciseRecommendationEngine:
    """Generate personalized exercise recommendations."""
    
    @staticmethod
    def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation."""
        if gender.lower() == 'male':
            return 10 * weight + 6.25 * height - 5 * age + 5
        else:
            return 10 * weight + 6.25 * height - 5 * age - 161
    
    @staticmethod
    def get_daily_calorie_needs(bmr: float, activity_level: str) -> float:
        """Calculate daily calorie needs based on activity level."""
        multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        return bmr * multipliers.get(activity_level, 1.375)
    
    @staticmethod
    def recommend_exercises(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized exercise recommendations."""
        
        # Extract user data
        goals = user_data.get('goals', [])
        fitness_level = user_data.get('fitness_level', 'beginner')
        available_time = user_data.get('available_time', 30)
        equipment = user_data.get('equipment', [])
        injuries = user_data.get('injuries', [])
        
        recommendations = {
            'cardio': [],
            'strength': [],
            'flexibility': [],
            'weekly_plan': {},
            'calories_burned_estimate': 0
        }
        
        # Filter exercises based on equipment and injuries
        def filter_exercises(exercise_dict, equipment_available, injury_list):
            filtered = {}
            for name, details in exercise_dict.items():
                # Check equipment requirements
                required_equipment = details.get('equipment', 'none')
                if required_equipment == 'none' or required_equipment in equipment_available:
                    # Simple injury check (this would be more sophisticated in practice)
                    if not any(injury.lower() in name.lower() for injury in injury_list):
                        filtered[name] = details
            return filtered
        
        # Get filtered exercises
        available_cardio = filter_exercises(Config.EXERCISES_DB['cardio'], equipment, injuries)
        available_strength = filter_exercises(Config.EXERCISES_DB['strength'], equipment, injuries)
        available_flexibility = filter_exercises(Config.EXERCISES_DB['flexibility'], equipment, injuries)
        
        # Recommend based on goals
        if 'weight_loss' in goals:
            # Prioritize cardio
            recommendations['cardio'] = list(available_cardio.keys())[:3]
            recommendations['strength'] = list(available_strength.keys())[:2]
            recommendations['flexibility'] = list(available_flexibility.keys())[:1]
        
        elif 'muscle_gain' in goals:
            # Prioritize strength training
            recommendations['strength'] = list(available_strength.keys())[:4]
            recommendations['cardio'] = list(available_cardio.keys())[:1]
            recommendations['flexibility'] = list(available_flexibility.keys())[:1]
        
        elif 'endurance' in goals:
            # Balance cardio and strength
            recommendations['cardio'] = list(available_cardio.keys())[:3]
            recommendations['strength'] = list(available_strength.keys())[:2]
            recommendations['flexibility'] = list(available_flexibility.keys())[:1]
        
        else:
            # General fitness
            recommendations['cardio'] = list(available_cardio.keys())[:2]
            recommendations['strength'] = list(available_strength.keys())[:2]
            recommendations['flexibility'] = list(available_flexibility.keys())[:2]
        
        # Create weekly plan
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i, day in enumerate(days):
            if i % 2 == 0:  # Every other day
                if i % 4 == 0:  # Cardio days
                    recommendations['weekly_plan'][day] = {
                        'focus': 'Cardio & Flexibility',
                        'exercises': recommendations['cardio'][:2] + recommendations['flexibility'][:1],
                        'duration': available_time
                    }
                else:  # Strength days
                    recommendations['weekly_plan'][day] = {
                        'focus': 'Strength Training',
                        'exercises': recommendations['strength'][:3],
                        'duration': available_time
                    }
            else:  # Rest or light activity days
                recommendations['weekly_plan'][day] = {
                    'focus': 'Rest or Light Activity',
                    'exercises': recommendations['flexibility'][:1],
                    'duration': 15
                }
        
        # Estimate calories burned
        total_calories = 0
        for day_plan in recommendations['weekly_plan'].values():
            day_calories = 0
            for exercise in day_plan['exercises']:
                # Find exercise in database
                for category in Config.EXERCISES_DB.values():
                    if exercise in category:
                        day_calories += category[exercise]['calories_per_min'] * day_plan['duration']
                        break
            total_calories += day_calories
        
        recommendations['calories_burned_estimate'] = total_calories
        
        return recommendations

class PoseAnalyzer:
    """Analyze workout form using MediaPipe pose detection."""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            self.mp_pose = None
            self.pose = None
            self.mp_drawing = None
    
    def analyze_pose(self, image: np.ndarray, exercise_type: str = 'general') -> Dict[str, Any]:
        """Analyze pose and provide form feedback."""
        
        if not MEDIAPIPE_AVAILABLE:
            return {'error': 'MediaPipe not available'}
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return {'error': 'No pose detected in image'}
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Calculate angles and distances for form analysis
        analysis = self._analyze_form(landmarks, exercise_type)
        
        # Draw pose landmarks on image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        analysis['annotated_image'] = annotated_image
        analysis['landmarks_detected'] = True
        
        return analysis
    
    def _analyze_form(self, landmarks, exercise_type: str) -> Dict[str, Any]:
        """Analyze specific exercise form."""
        
        # Get key landmark positions
        def get_landmark(index):
            return [landmarks[index].x, landmarks[index].y, landmarks[index].z]
        
        # Key body points
        nose = get_landmark(0)
        left_shoulder = get_landmark(11)
        right_shoulder = get_landmark(12)
        left_elbow = get_landmark(13)
        right_elbow = get_landmark(14)
        left_wrist = get_landmark(15)
        right_wrist = get_landmark(16)
        left_hip = get_landmark(23)
        right_hip = get_landmark(24)
        left_knee = get_landmark(25)
        right_knee = get_landmark(26)
        left_ankle = get_landmark(27)
        right_ankle = get_landmark(28)
        
        feedback = {
            'overall_score': 0,
            'specific_feedback': [],
            'warnings': [],
            'good_points': []
        }
        
        # General posture analysis
        # Check shoulder alignment
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_diff < 0.05:
            feedback['good_points'].append("✅ Good shoulder alignment")
            feedback['overall_score'] += 20
        else:
            feedback['warnings'].append("⚠️ Uneven shoulders - check posture")
        
        # Check spine alignment (simplified)
        mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        if abs(nose[0] - (left_shoulder[0] + right_shoulder[0]) / 2) < 0.1:
            feedback['good_points'].append("✅ Good head alignment")
            feedback['overall_score'] += 15
        else:
            feedback['warnings'].append("⚠️ Head forward - maintain neutral spine")
        
        # Exercise-specific analysis
        if exercise_type.lower() == 'squat':
            feedback.update(self._analyze_squat(landmarks))
        elif exercise_type.lower() == 'pushup':
            feedback.update(self._analyze_pushup(landmarks))
        elif exercise_type.lower() == 'deadlift':
            feedback.update(self._analyze_deadlift(landmarks))
        else:
            # General standing posture
            knee_bend = abs(left_knee[1] - left_hip[1])
            if knee_bend > 0.3:
                feedback['good_points'].append("✅ Good knee bend for standing position")
                feedback['overall_score'] += 10
        
        # Calculate final score
        max_possible_score = 100
        feedback['overall_score'] = min(feedback['overall_score'], max_possible_score)
        
        # Add overall assessment
        if feedback['overall_score'] >= 80:
            feedback['assessment'] = "Excellent form!"
        elif feedback['overall_score'] >= 60:
            feedback['assessment'] = "Good form with room for improvement"
        elif feedback['overall_score'] >= 40:
            feedback['assessment'] = "Needs improvement"
        else:
            feedback['assessment'] = "Poor form - risk of injury"
        
        return feedback
    
    def _analyze_squat(self, landmarks) -> Dict[str, Any]:
        """Analyze squat form."""
        feedback = {'specific_feedback': [], 'warnings': [], 'good_points': []}
        
        # Get key points for squat analysis
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        
        # Check squat depth
        hip_knee_distance = abs(left_hip[1] - left_knee[1])
        if hip_knee_distance < 0.1:  # Hip below knee level
            feedback['good_points'].append("✅ Good squat depth")
        else:
            feedback['warnings'].append("⚠️ Try to squat deeper - hips below knees")
        
        # Check knee alignment
        knee_ankle_alignment = abs(left_knee[0] - left_ankle[0])
        if knee_ankle_alignment < 0.05:
            feedback['good_points'].append("✅ Knees aligned over ankles")
        else:
            feedback['warnings'].append("⚠️ Keep knees aligned over ankles")
        
        feedback['specific_feedback'].append("Squat-specific analysis completed")
        return feedback
    
    def _analyze_pushup(self, landmarks) -> Dict[str, Any]:
        """Analyze push-up form."""
        feedback = {'specific_feedback': [], 'warnings': [], 'good_points': []}
        
        # Get key points
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        
        # Check body alignment (plank position)
        shoulder_hip_alignment = abs(left_shoulder[1] - left_hip[1])
        if shoulder_hip_alignment < 0.15:
            feedback['good_points'].append("✅ Good plank position")
        else:
            feedback['warnings'].append("⚠️ Maintain straight line from head to heels")
        
        feedback['specific_feedback'].append("Push-up specific analysis completed")
        return feedback
    
    def _analyze_deadlift(self, landmarks) -> Dict[str, Any]:
        """Analyze deadlift form."""
        feedback = {'specific_feedback': [], 'warnings': [], 'good_points': []}
        
        # Get key points
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        
        # Check back angle (simplified)
        shoulder_hip_distance = abs(left_shoulder[0] - left_hip[0])
        if shoulder_hip_distance < 0.1:
            feedback['good_points'].append("✅ Good back position")
        else:
            feedback['warnings'].append("⚠️ Keep chest up and back straight")
        
        feedback['specific_feedback'].append("Deadlift-specific analysis completed")
        return feedback

def main():
    """Main application entry point."""
    
    # Header
    st.markdown(f'''
    <div class="main-header">
        {Config.APP_ICON} AI Fitness Assistant v{Config.VERSION}
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    setup_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Body Fat Analysis", 
        "💪 Exercise Recommendations", 
        "🎯 Form Correction"
    ])
    
    with tab1:
        show_body_fat_analysis()
    
    with tab2:
        show_exercise_recommendations()
    
    with tab3:
        show_form_correction()

def setup_sidebar():
    """Setup sidebar with user information and quick stats."""
    st.sidebar.title("🏋️‍♀️ User Profile")
    
    # Quick user input for all tabs
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    
    with st.sidebar.form("user_profile"):
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["male", "female"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        
        activity_level = st.selectbox("Activity Level", [
            "sedentary", "lightly_active", "moderately_active", 
            "very_active", "extremely_active"
        ])
        
        if st.form_submit_button("Update Profile"):
            st.session_state.user_data.update({
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'activity_level': activity_level
            })
            st.success("Profile updated!")
    
    # Display current BMI if data available
    if st.session_state.user_data:
        data = st.session_state.user_data
        if 'weight' in data and 'height' in data:
            bmi = data['weight'] / ((data['height'] / 100) ** 2)
            st.sidebar.metric("Current BMI", f"{bmi:.1f}")

def show_body_fat_analysis():
    """Tab 1: Body Fat Analysis from Images."""
    st.header("📊 Body Fat Analysis")
    
    st.markdown("""
    Upload a photo for body composition analysis. This tool uses:
    - **Anthropometric calculations** (Navy Method when measurements provided)
    - **Computer vision analysis** of body shape and composition
    - **BMI-based estimations** with activity level adjustments
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Photo")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=Config.SUPPORTED_IMAGE_FORMATS,
            help="Upload a clear, full-body photo for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array for analysis
            image_array = np.array(image)
            
            # Analyze image
            with st.spinner("🔍 Analyzing body composition..."):
                analyzer = BodyFatAnalyzer()
                image_analysis = analyzer.analyze_body_composition_from_image(image_array)
            
            if 'error' not in image_analysis:
                st.success("✅ Image analysis completed!")
                
                # Display image analysis results
                st.subheader("🖼️ Image Analysis Results")
                
                col1_img, col2_img = st.columns(2)
                with col1_img:
                    st.metric("Muscle Indicator", f"{image_analysis['muscle_indicator']:.1f}%")
                    st.metric("Aspect Ratio", image_analysis['aspect_ratio'])
                
                with col2_img:
                    st.metric("Fat Indicator", f"{image_analysis['fat_indicator']:.1f}%")
                    st.metric("Body Area", f"{image_analysis['body_area']:.0f} pixels")
                
                st.info(f"**Method:** {image_analysis['analysis_method']}")
            else:
                st.error(f"❌ {image_analysis['error']}")
    
    with col2:
        st.subheader("📐 Body Measurements")
        
        if st.session_state.user_data:
            user_data = st.session_state.user_data
            
            with st.form("measurements_form"):
                st.write("**Optional measurements for more accurate analysis:**")
                
                waist = st.number_input("Waist circumference (cm)", min_value=50.0, max_value=150.0, value=80.0)
                neck = st.number_input("Neck circumference (cm)", min_value=25.0, max_value=60.0, value=35.0)
                
                if user_data.get('gender') == 'female':
                    hip = st.number_input("Hip circumference (cm)", min_value=60.0, max_value=150.0, value=90.0)
                else:
                    hip = None
                
                if st.form_submit_button("🧮 Calculate Body Fat", type="primary"):
                    # Calculate body fat percentage
                    analyzer = BodyFatAnalyzer()
                    results = analyzer.calculate_body_fat_percentage(
                        age=user_data['age'],
                        weight=user_data['weight'],
                        height=user_data['height'],
                        gender=user_data['gender'],
                        activity_level=user_data['activity_level'],
                        waist=waist,
                        neck=neck,
                        hip=hip
                    )
                    
                    # Display results
                    st.markdown("""
                    <div class="analysis-card">
                        <h3>📊 Body Composition Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1_res, col2_res, col3_res = st.columns(3)
                    
                    with col1_res:
                        st.metric("Body Fat %", f"{results['body_fat_percentage']}%")
                    
                    with col2_res:
                        st.metric("BMI", results['bmi'])
                    
                    with col3_res:
                        st.metric("Category", results['category'])
                    
                    st.info(f"**Method Used:** {results['estimation_method']}")
                    
                    if results['navy_method']:
                        st.success(f"✅ Navy Method Result: {results['navy_method']}%")
                    
                    # Recommendations based on category
                    if results['category'] in ['Obesity', 'Acceptable']:
                        st.warning("💡 **Recommendation:** Consider increasing physical activity and consulting with a fitness professional.")
                    elif results['category'] in ['Athletic', 'Fitness']:
                        st.success("🎉 **Great job!** You're in a healthy range. Keep up the good work!")
                    elif results['category'] == 'Essential Fat':
                        st.error("⚠️ **Caution:** Body fat may be too low. Consider consulting with a healthcare professional.")
        else:
            st.info("👆 Please fill out your profile in the sidebar first.")

def show_exercise_recommendations():
    """Tab 2: Exercise Recommendations based on User Data."""
    st.header("💪 Personalized Exercise Recommendations")
    
    if not st.session_state.user_data:
        st.warning("👆 Please fill out your profile in the sidebar first.")
        return
    
    user_data = st.session_state.user_data.copy()
    
    # Additional form for exercise preferences
    with st.form("exercise_preferences"):
        st.subheader("🎯 Your Fitness Goals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goals = st.multiselect("Select your goals:", [
                "weight_loss", "muscle_gain", "endurance", "flexibility", 
                "general_fitness", "strength", "cardio_health"
            ])
            
            fitness_level = st.selectbox("Current fitness level:", [
                "beginner", "intermediate", "advanced"
            ])
            
            available_time = st.slider("Available workout time (minutes):", 15, 120, 45)
        
        with col2:
            equipment = st.multiselect("Available equipment:", [
                "none", "weights", "machine", "bike", "pool", "mat", "resistance_bands"
            ])
            
            injuries = st.multiselect("Current injuries/limitations:", [
                "knee", "back", "shoulder", "wrist", "ankle", "neck"
            ])
            
            workout_days = st.slider("Preferred workout days per week:", 1, 7, 4)
        
        if st.form_submit_button("🚀 Get Recommendations", type="primary"):
            # Update user data with preferences
            user_data.update({
                'goals': goals,
                'fitness_level': fitness_level,
                'available_time': available_time,
                'equipment': equipment,
                'injuries': injuries,
                'workout_days': workout_days
            })
            
            # Calculate BMR and daily calories
            engine = ExerciseRecommendationEngine()
            bmr = engine.calculate_bmr(
                user_data['weight'], user_data['height'], 
                user_data['age'], user_data['gender']
            )
            daily_calories = engine.get_daily_calorie_needs(bmr, user_data['activity_level'])
            
            # Get exercise recommendations
            recommendations = engine.recommend_exercises(user_data)
            
            # Display results
            st.success("✅ Recommendations generated!")
            
            # Caloric information
            st.subheader("🔥 Caloric Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Daily BMR", f"{bmr:.0f} cal")
            with col2:
                st.metric("Daily Needs", f"{daily_calories:.0f} cal")
            with col3:
                st.metric("Weekly Exercise Burn", f"{recommendations['calories_burned_estimate']:.0f} cal")
            
            # Exercise recommendations by category
            st.subheader("💪 Recommended Exercises")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏃‍♀️ Cardio**")
                for exercise in recommendations['cardio']:
                    exercise_data = None
                    for category in Config.EXERCISES_DB.values():
                        if exercise in category:
                            exercise_data = category[exercise]
                            break
                    
                    if exercise_data:
                        st.markdown(f"""
                        <div class="exercise-card">
                            <strong>{exercise.replace('_', ' ').title()}</strong><br>
                            <small>🔥 {exercise_data['calories_per_min']} cal/min | 
                            ⭐ {exercise_data['difficulty']} | 
                            🛠️ {exercise_data['equipment']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🏋️‍♀️ Strength**")
                for exercise in recommendations['strength']:
                    exercise_data = None
                    for category in Config.EXERCISES_DB.values():
                        if exercise in category:
                            exercise_data = category[exercise]
                            break
                    
                    if exercise_data:
                        st.markdown(f"""
                        <div class="exercise-card">
                            <strong>{exercise.replace('_', ' ').title()}</strong><br>
                            <small>🔥 {exercise_data['calories_per_min']} cal/min | 
                            ⭐ {exercise_data['difficulty']} | 
                            🛠️ {exercise_data['equipment']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("**🧘‍♀️ Flexibility**")
                for exercise in recommendations['flexibility']:
                    exercise_data = None
                    for category in Config.EXERCISES_DB.values():
                        if exercise in category:
                            exercise_data = category[exercise]
                            break
                    
                    if exercise_data:
                        st.markdown(f"""
                        <div class="exercise-card">
                            <strong>{exercise.replace('_', ' ').title()}</strong><br>
                            <small>🔥 {exercise_data['calories_per_min']} cal/min | 
                            ⭐ {exercise_data['difficulty']} | 
                            🛠️ {exercise_data['equipment']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Weekly plan
            st.subheader("📅 Weekly Workout Plan")
            
            for day, plan in recommendations['weekly_plan'].items():
                with st.expander(f"{day} - {plan['focus']} ({plan['duration']} min)"):
                    st.write(f"**Focus:** {plan['focus']}")
                    st.write(f"**Duration:** {plan['duration']} minutes")
                    st.write("**Exercises:**")
                    for exercise in plan['exercises']:
                        st.write(f"• {exercise.replace('_', ' ').title()}")
    
    # Additional tips
    st.subheader("💡 General Tips")
    st.info("""
    **Remember:**
    - Start slowly if you're a beginner
    - Stay hydrated during workouts
    - Allow rest days for recovery
    - Listen to your body and adjust intensity
    - Consider consulting with a fitness professional
    """)

def show_form_correction():
    """Tab 3: Workout Form Correction using Pose Detection."""
    st.header("🎯 Workout Form Analysis")
    
    if not MEDIAPIPE_AVAILABLE:
        st.error("❌ MediaPipe is not installed. Please install it with: `pip install mediapipe`")
        return
    
    st.markdown("""
    Upload a photo or video of your workout to get real-time form analysis and corrections.
    Our AI analyzes your pose and provides specific feedback for different exercises.
    """)
    
    # Exercise type selection
    exercise_type = st.selectbox("Select exercise type:", [
        "general", "squat", "pushup", "deadlift", "plank", "lunge"
    ])
    
    # Upload options
    upload_option = st.radio("Choose input type:", ["Image", "Video", "Live Camera"])
    
    if upload_option == "Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=Config.SUPPORTED_IMAGE_FORMATS,
            help="Upload a clear image showing your exercise form"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Analyze pose
            with st.spinner("🔍 Analyzing your form..."):
                analyzer = PoseAnalyzer()
                analysis = analyzer.analyze_pose(image_cv, exercise_type)
            
            if 'error' not in analysis:
                with col2:
                    st.subheader("🎯 Pose Analysis")
                    # Convert back to RGB for display
                    annotated_image_rgb = cv2.cvtColor(analysis['annotated_image'], cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, caption="Pose Detection", use_column_width=True)
                
                # Display analysis results
                st.subheader("📊 Form Analysis Results")
                
                # Overall score
                score = analysis['overall_score']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if score >= 80:
                        st.success(f"🎉 **Overall Score: {score}/100**")
                    elif score >= 60:
                        st.warning(f"⚠️ **Overall Score: {score}/100**")
                    else:
                        st.error(f"❌ **Overall Score: {score}/100**")
                
                with col2:
                    st.info(f"**Assessment:** {analysis['assessment']}")
                
                with col3:
                    st.metric("Exercise Type", exercise_type.title())
                
                # Detailed feedback
                col1, col2 = st.columns(2)
                
                with col1:
                    if analysis['good_points']:
                        st.subheader("✅ Good Form Points")
                        for point in analysis['good_points']:
                            st.markdown(f"<div class='pose-point'>{point}</div>", unsafe_allow_html=True)
                
                with col2:
                    if analysis['warnings']:
                        st.subheader("⚠️ Areas for Improvement")
                        for warning in analysis['warnings']:
                            st.markdown(f"<div class='pose-warning'>{warning}</div>", unsafe_allow_html=True)
                
                if analysis['specific_feedback']:
                    st.subheader("🎯 Exercise-Specific Feedback")
                    for feedback in analysis['specific_feedback']:
                        st.info(feedback)
                
            else:
                st.error(f"❌ {analysis['error']}")
                st.info("💡 **Tips for better detection:**\n- Ensure good lighting\n- Stand clear of background\n- Show full body in frame\n- Wear contrasting colors")
    
    elif upload_option == "Video":
        st.info("🎬 Video analysis feature coming soon! For now, please upload individual frames as images.")
    
    elif upload_option == "Live Camera":
        st.info("📹 Live camera analysis feature coming soon! For now, please take a photo and upload it.")
    
    # Exercise-specific tips
    st.subheader("💡 Exercise-Specific Tips")
    
    tips = {
        'squat': """
        **Squat Form Tips:**
        - Keep feet shoulder-width apart
        - Lower until hips are below knees
        - Keep knees aligned over ankles
        - Maintain straight back
        - Drive through heels to stand
        """,
        'pushup': """
        **Push-up Form Tips:**
        - Maintain straight line from head to heels
        - Keep hands slightly wider than shoulders
        - Lower chest to ground
        - Engage core throughout movement
        - Keep elbows at 45-degree angle
        """,
        'deadlift': """
        **Deadlift Form Tips:**
        - Keep bar close to body
        - Maintain neutral spine
        - Drive through heels
        - Keep chest up and shoulders back
        - Hinge at hips, not knees
        """,
        'general': """
        **General Form Tips:**
        - Maintain good posture
        - Keep movements controlled
        - Breathe consistently
        - Listen to your body
        - Focus on quality over quantity
        """
    }
    
    st.info(tips.get(exercise_type, tips['general']))

if __name__ == "__main__":
    main()
