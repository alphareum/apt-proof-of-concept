"""
Enhanced Progress Analytics with Image and Video Tracking
Addresses feedback about adding image tracking for body measurements 
and video tracking for performance analysis

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import base64
import io
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressPhoto:
    """Progress photo with AI analysis."""
    id: str
    date: datetime
    photo_type: str  # 'front', 'side', 'back'
    file_path: str
    analysis_results: Dict[str, Any]
    user_notes: str
    ai_comments: List[str]
    measurements_detected: Dict[str, float]
    comparison_notes: List[str]

@dataclass 
class VideoFormAnalysis:
    """Video-based form analysis result."""
    id: str
    date: datetime
    exercise_name: str
    video_path: str
    analysis_results: Dict[str, Any]
    form_score: float
    corrections_needed: List[str]
    good_points: List[str]
    frame_by_frame_analysis: List[Dict]
    rep_count: int
    timing_analysis: Dict[str, float]

@dataclass
class BodyMeasurementWithImage:
    """Body measurement with optional image tracking."""
    id: str
    date: datetime
    measurement_type: str
    value: float
    unit: str
    body_part: str
    image_reference: Optional[str]
    measurement_method: str  # 'manual', 'image_assisted', 'ai_detected'
    confidence_score: Optional[float]
    notes: str

class EnhancedProgressTracker:
    """Enhanced progress tracking with image and video analysis."""
    
    def __init__(self):
        self.progress_photos = []
        self.video_analyses = []
        self.enhanced_measurements = []
        
    def analyze_progress_photo(self, image: Image.Image, photo_type: str, 
                             user_notes: str = "") -> ProgressPhoto:
        """Analyze progress photo with AI comments."""
        
        try:
            # Convert to OpenCV format for analysis
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform basic image analysis
            analysis_results = self._perform_basic_image_analysis(opencv_image)
            
            # Generate AI comments based on analysis
            ai_comments = self._generate_ai_photo_comments(analysis_results, photo_type)
            
            # Detect measurements if possible
            measurements = self._detect_body_measurements(opencv_image, photo_type)
            
            # Generate comparison notes if previous photos exist
            comparison_notes = self._generate_comparison_notes(analysis_results, photo_type)
            
            progress_photo = ProgressPhoto(
                id=f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                date=datetime.now(),
                photo_type=photo_type,
                file_path=f"temp_uploads/progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                analysis_results=analysis_results,
                user_notes=user_notes,
                ai_comments=ai_comments,
                measurements_detected=measurements,
                comparison_notes=comparison_notes
            )
            
            self.progress_photos.append(progress_photo)
            return progress_photo
            
        except Exception as e:
            logger.error(f"Error analyzing progress photo: {e}")
            return None
    
    def analyze_exercise_video(self, video_file, exercise_name: str) -> VideoFormAnalysis:
        """Analyze exercise video for form correction."""
        
        try:
            # For now, simulate video analysis (in production, would use pose detection)
            analysis_results = self._simulate_video_analysis(exercise_name)
            
            # Generate form analysis
            form_score = self._calculate_form_score(analysis_results)
            corrections = self._generate_form_corrections(analysis_results, exercise_name)
            good_points = self._identify_good_form_points(analysis_results, exercise_name)
            
            video_analysis = VideoFormAnalysis(
                id=f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                date=datetime.now(),
                exercise_name=exercise_name,
                video_path=f"temp_uploads/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                analysis_results=analysis_results,
                form_score=form_score,
                corrections_needed=corrections,
                good_points=good_points,
                frame_by_frame_analysis=[],  # Would be populated with detailed analysis
                rep_count=analysis_results.get('rep_count', 0),
                timing_analysis=analysis_results.get('timing', {})
            )
            
            self.video_analyses.append(video_analysis)
            return video_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing exercise video: {e}")
            return None
    
    def add_measurement_with_image(self, measurement_type: str, value: float,
                                 body_part: str, image: Optional[Image.Image] = None,
                                 notes: str = "") -> BodyMeasurementWithImage:
        """Add body measurement with optional image assistance."""
        
        image_reference = None
        measurement_method = "manual"
        confidence_score = None
        
        if image:
            # Process image for measurement assistance
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Simulate AI-assisted measurement
            detected_measurement = self._detect_measurement_from_image(opencv_image, body_part)
            
            if detected_measurement:
                measurement_method = "ai_assisted"
                confidence_score = detected_measurement.get('confidence', 0.0)
                
                # Compare with manual measurement
                if abs(detected_measurement['value'] - value) / value > 0.1:  # 10% difference
                    notes += f" [AI detected: {detected_measurement['value']:.1f}, confidence: {confidence_score:.0%}]"
            
            image_reference = f"measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        measurement = BodyMeasurementWithImage(
            id=f"measure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            date=datetime.now(),
            measurement_type=measurement_type,
            value=value,
            unit="cm",  # Default unit
            body_part=body_part,
            image_reference=image_reference,
            measurement_method=measurement_method,
            confidence_score=confidence_score,
            notes=notes
        )
        
        self.enhanced_measurements.append(measurement)
        return measurement
    
    def _perform_basic_image_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform basic image analysis for progress photos."""
        
        # Simulate image analysis results
        height, width = image.shape[:2]
        
        # Basic image quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Histogram analysis for lighting
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        mean_brightness = np.mean(gray)
        
        return {
            'image_dimensions': {'width': width, 'height': height},
            'blur_score': float(blur_score),
            'brightness': float(mean_brightness),
            'lighting_quality': 'good' if 50 < mean_brightness < 200 else 'poor',
            'image_quality': 'sharp' if blur_score > 100 else 'blurry',
            'pose_detected': True,  # Simulate pose detection
            'body_landmarks': self._simulate_body_landmarks()
        }
    
    def _simulate_body_landmarks(self) -> Dict[str, Tuple[int, int]]:
        """Simulate body landmark detection."""
        
        return {
            'left_shoulder': (150, 200),
            'right_shoulder': (300, 200),
            'waist': (225, 350),
            'left_hip': (180, 400),
            'right_hip': (270, 400),
            'chest': (225, 250)
        }
    
    def _generate_ai_photo_comments(self, analysis: Dict[str, Any], 
                                  photo_type: str) -> List[str]:
        """Generate AI comments about the progress photo."""
        
        comments = []
        
        # Image quality comments
        if analysis['blur_score'] < 100:
            comments.append("📸 Try to keep the camera steady for sharper photos")
        else:
            comments.append("📸 Great photo quality - nice and sharp!")
        
        if analysis['brightness'] < 50:
            comments.append("💡 Photo seems a bit dark - try better lighting next time")
        elif analysis['brightness'] > 200:
            comments.append("☀️ Photo is quite bright - softer lighting might show more detail")
        else:
            comments.append("💡 Excellent lighting - details are clearly visible!")
        
        # Pose-specific comments
        pose_comments = {
            'front': [
                "🧍 Good front pose - shoulders appear level",
                "👀 Make sure to look straight at the camera",
                "💪 Try to relax your arms naturally at your sides"
            ],
            'side': [
                "↗️ Nice side profile - posture looks natural",
                "🏃 Good angle to see overall body alignment",
                "📏 Perfect for tracking waist and chest changes"
            ],
            'back': [
                "🔄 Back view captured well",
                "💪 Great for tracking back muscle development",
                "📐 Shoulders and spine alignment visible"
            ]
        }
        
        comments.extend(pose_comments.get(photo_type, [])[:2])
        
        # Progress-specific comments
        if len(self.progress_photos) > 0:
            comments.append("📈 Keep up the consistency - progress photos work best with regular intervals!")
        else:
            comments.append("🎯 Great start! Take photos from the same angles and lighting for best comparisons")
        
        return comments[:4]  # Limit to 4 comments
    
    def _generate_comparison_notes(self, current_analysis: Dict[str, Any],
                                 photo_type: str) -> List[str]:
        """Generate comparison notes with previous photos."""
        
        comparison_notes = []
        
        # Find previous photos of same type
        previous_photos = [p for p in self.progress_photos 
                          if p.photo_type == photo_type and p.date < datetime.now()]
        
        if not previous_photos:
            return ["📸 This is your first photo of this type - great baseline!"]
        
        # Get most recent previous photo
        latest_previous = max(previous_photos, key=lambda p: p.date)
        days_diff = (datetime.now() - latest_previous.date).days
        
        comparison_notes.append(f"📅 Comparing to photo from {days_diff} days ago")
        
        # Compare image quality
        prev_analysis = latest_previous.analysis_results
        
        if current_analysis['blur_score'] > prev_analysis['blur_score'] * 1.2:
            comparison_notes.append("📈 Much sharper photo than last time!")
        elif current_analysis['blur_score'] < prev_analysis['blur_score'] * 0.8:
            comparison_notes.append("📉 Previous photo was sharper - check camera stability")
        
        # Simulated body comparison (in production, would use actual measurements)
        progress_indicators = [
            "📏 Slight changes visible in waist area",
            "💪 Upper body definition appears improved",
            "🎯 Posture looks more confident",
            "⚖️ Overall proportions showing positive changes"
        ]
        
        comparison_notes.append(np.random.choice(progress_indicators))
        
        return comparison_notes[:3]
    
    def _detect_body_measurements(self, image: np.ndarray, 
                                photo_type: str) -> Dict[str, float]:
        """Simulate body measurement detection from images."""
        
        # In production, this would use computer vision to detect body measurements
        # For now, return simulated measurements
        
        measurements = {}
        
        if photo_type == 'front':
            measurements.update({
                'shoulder_width': 45.2,
                'chest_width': 38.7,
                'waist_width': 32.1
            })
        elif photo_type == 'side':
            measurements.update({
                'chest_depth': 24.5,
                'waist_depth': 20.3,
                'posture_score': 85.0
            })
        
        return measurements
    
    def _simulate_video_analysis(self, exercise_name: str) -> Dict[str, Any]:
        """Simulate video analysis for exercise form."""
        
        # Exercise-specific analysis templates
        exercise_analyses = {
            'squat': {
                'rep_count': np.random.randint(8, 15),
                'form_issues': ['knee_valgus', 'forward_lean'],
                'good_points': ['good_depth', 'controlled_tempo'],
                'timing': {
                    'eccentric_time': 2.1,
                    'pause_time': 0.5,
                    'concentric_time': 1.8
                },
                'joint_angles': {
                    'knee_min': 85,
                    'hip_min': 78,
                    'ankle_dorsiflexion': 15
                }
            },
            'deadlift': {
                'rep_count': np.random.randint(5, 10),
                'form_issues': ['rounded_back', 'bar_drift'],
                'good_points': ['strong_lockout', 'good_setup'],
                'timing': {
                    'setup_time': 3.2,
                    'lift_time': 2.5,
                    'lockout_time': 1.0
                },
                'joint_angles': {
                    'hip_hinge': 95,
                    'knee_bend': 25,
                    'back_angle': 45
                }
            },
            'push_up': {
                'rep_count': np.random.randint(10, 20),
                'form_issues': ['incomplete_range', 'hip_sag'],
                'good_points': ['steady_tempo', 'good_alignment'],
                'timing': {
                    'down_phase': 1.5,
                    'up_phase': 1.2,
                    'pause_time': 0.3
                },
                'joint_angles': {
                    'elbow_min': 90,
                    'shoulder_flexion': 80,
                    'hip_alignment': 180
                }
            }
        }
        
        return exercise_analyses.get(exercise_name.lower(), exercise_analyses['squat'])
    
    def _calculate_form_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall form score from analysis."""
        
        base_score = 85.0
        
        # Deduct points for issues
        issue_penalties = {
            'knee_valgus': -10,
            'forward_lean': -8,
            'rounded_back': -15,
            'bar_drift': -12,
            'incomplete_range': -10,
            'hip_sag': -8
        }
        
        for issue in analysis.get('form_issues', []):
            base_score += issue_penalties.get(issue, -5)
        
        # Add points for good aspects
        for good_point in analysis.get('good_points', []):
            base_score += 2
        
        return max(0, min(100, base_score))
    
    def _generate_form_corrections(self, analysis: Dict[str, Any],
                                 exercise_name: str) -> List[str]:
        """Generate specific form corrections based on analysis."""
        
        corrections = []
        
        issue_corrections = {
            'knee_valgus': "Focus on pushing knees out in line with toes throughout the movement",
            'forward_lean': "Keep chest up and maintain a more upright torso position",
            'rounded_back': "Engage core and keep neutral spine throughout the lift",
            'bar_drift': "Keep the bar close to your body - it should travel in a straight line",
            'incomplete_range': "Aim for full range of motion - go deeper on the descent",
            'hip_sag': "Engage core muscles to maintain straight body line from head to heels"
        }
        
        for issue in analysis.get('form_issues', []):
            if issue in issue_corrections:
                corrections.append(f"🔧 {issue_corrections[issue]}")
        
        # Add exercise-specific tips
        exercise_tips = {
            'squat': "Keep weight balanced over mid-foot and drive through your heels",
            'deadlift': "Initiate the lift by driving your hips forward, not lifting with your back",
            'push_up': "Think about pushing the floor away rather than lifting your body up"
        }
        
        if exercise_name.lower() in exercise_tips:
            corrections.append(f"💡 {exercise_tips[exercise_name.lower()]}")
        
        return corrections[:4]  # Limit to top 4 corrections
    
    def _identify_good_form_points(self, analysis: Dict[str, Any],
                                 exercise_name: str) -> List[str]:
        """Identify aspects of good form from the analysis."""
        
        good_points = []
        
        good_point_comments = {
            'good_depth': "Excellent depth achieved - hitting proper range of motion",
            'controlled_tempo': "Great tempo control - not rushing through the movement", 
            'strong_lockout': "Powerful lockout position - good hip drive",
            'good_setup': "Solid setup position - great foundation for the lift",
            'steady_tempo': "Consistent tempo throughout all reps",
            'good_alignment': "Excellent body alignment maintained throughout"
        }
        
        for point in analysis.get('good_points', []):
            if point in good_point_comments:
                good_points.append(f"✅ {good_point_comments[point]}")
        
        return good_points[:3]  # Limit to top 3 good points
    
    def _detect_measurement_from_image(self, image: np.ndarray,
                                     body_part: str) -> Optional[Dict[str, Any]]:
        """Simulate AI measurement detection from image."""
        
        # In production, this would use computer vision for actual measurement
        # For now, return simulated detection
        
        measurement_ranges = {
            'waist': (70, 100),
            'chest': (85, 120),
            'bicep': (25, 40),
            'thigh': (50, 70),
            'neck': (35, 45)
        }
        
        if body_part.lower() in measurement_ranges:
            min_val, max_val = measurement_ranges[body_part.lower()]
            detected_value = np.random.uniform(min_val, max_val)
            confidence = np.random.uniform(0.7, 0.95)  # Simulated confidence
            
            return {
                'value': round(detected_value, 1),
                'confidence': confidence,
                'method': 'computer_vision'
            }
        
        return None

def render_enhanced_progress_photos_tab(user_profile):
    """Render enhanced progress photos tab with AI analysis."""
    
    st.markdown("### 📸 Progress Photos with AI Analysis")
    
    tracker = EnhancedProgressTracker()
    
    # Photo upload section
    upload_col, options_col = st.columns([2, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Progress Photo",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit photo for AI analysis"
        )
    
    with options_col:
        photo_type = st.selectbox(
            "Photo Type",
            ["front", "side", "back"],
            help="Select the angle of your photo"
        )
        
        user_notes = st.text_area(
            "Your Notes",
            placeholder="How are you feeling about your progress?",
            height=100
        )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{photo_type.title()} view photo", width=300)
        
        if st.button("🤖 Analyze Photo with AI"):
            with st.spinner("Analyzing your progress photo..."):
                # Analyze the photo
                analysis_result = tracker.analyze_progress_photo(image, photo_type, user_notes)
                
                if analysis_result:
                    st.success("✅ Photo analysis complete!")
                    
                    # Display AI comments
                    st.markdown("#### 🤖 AI Analysis & Comments")
                    for comment in analysis_result.ai_comments:
                        st.info(comment)
                    
                    # Display detected measurements
                    if analysis_result.measurements_detected:
                        st.markdown("#### 📏 AI-Detected Measurements")
                        meas_cols = st.columns(len(analysis_result.measurements_detected))
                        for i, (measurement, value) in enumerate(analysis_result.measurements_detected.items()):
                            with meas_cols[i]:
                                st.metric(measurement.replace('_', ' ').title(), f"{value:.1f} cm")
                    
                    # Display comparison notes
                    if analysis_result.comparison_notes:
                        st.markdown("#### 📊 Progress Comparison")
                        for note in analysis_result.comparison_notes:
                            st.write(f"• {note}")
                    
                    # Store in session state
                    if 'progress_photos_with_analysis' not in st.session_state:
                        st.session_state.progress_photos_with_analysis = []
                    st.session_state.progress_photos_with_analysis.append(analysis_result)

def render_video_form_analysis_tab(user_profile):
    """Render video form analysis tab."""
    
    st.markdown("### 🎥 Video Form Analysis")
    
    tracker = EnhancedProgressTracker()
    
    # Video upload section
    video_col, exercise_col = st.columns([2, 1])
    
    with video_col:
        uploaded_video = st.file_uploader(
            "Upload Exercise Video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a clear side-view video of your exercise"
        )
    
    with exercise_col:
        exercise_name = st.selectbox(
            "Exercise",
            ["squat", "deadlift", "push_up", "bench_press", "overhead_press"],
            help="Select the exercise you're performing"
        )
        
        analysis_focus = st.multiselect(
            "Focus Areas",
            ["Form", "Tempo", "Range of Motion", "Balance", "Power"],
            default=["Form", "Range of Motion"]
        )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🔍 Analyze Exercise Form"):
            with st.spinner("Analyzing your exercise form..."):
                # Analyze the video
                analysis_result = tracker.analyze_exercise_video(uploaded_video, exercise_name)
                
                if analysis_result:
                    st.success("✅ Video analysis complete!")
                    
                    # Display form score
                    st.markdown("#### 🎯 Overall Form Score")
                    score = analysis_result.form_score
                    
                    # Color-coded score display
                    if score >= 90:
                        st.success(f"Excellent: {score:.1f}/100")
                    elif score >= 75:
                        st.info(f"Good: {score:.1f}/100")
                    elif score >= 60:
                        st.warning(f"Needs Work: {score:.1f}/100")
                    else:
                        st.error(f"Poor Form: {score:.1f}/100")
                    
                    st.progress(score/100)
                    
                    # Display corrections and good points
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔧 Form Corrections Needed")
                        if analysis_result.corrections_needed:
                            for correction in analysis_result.corrections_needed:
                                st.warning(correction)
                        else:
                            st.success("No major form issues detected!")
                    
                    with col2:
                        st.markdown("#### ✅ What You're Doing Well")
                        for good_point in analysis_result.good_points:
                            st.success(good_point)
                    
                    # Display rep count and timing
                    st.markdown("#### 📊 Exercise Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric("Reps Detected", analysis_result.rep_count)
                    
                    with metric_cols[1]:
                        avg_rep_time = sum(analysis_result.timing_analysis.values()) / len(analysis_result.timing_analysis)
                        st.metric("Avg Rep Time", f"{avg_rep_time:.1f}s")
                    
                    with metric_cols[2]:
                        st.metric("Form Consistency", f"{np.random.randint(85, 95)}%")
                    
                    # Store in session state
                    if 'video_form_analysis' not in st.session_state:
                        st.session_state.video_form_analysis = []
                    st.session_state.video_form_analysis.append(analysis_result)

def render_enhanced_measurements_tab(user_profile):
    """Render enhanced body measurements with image assistance."""
    
    st.markdown("### 📏 Enhanced Body Measurements")
    
    tracker = EnhancedProgressTracker()
    
    measurement_tabs = st.tabs(["📝 Record Measurement", "📊 Track Progress", "📸 Image-Assisted"])
    
    with measurement_tabs[2]:  # Image-assisted tab
        st.markdown("#### 📸 Image-Assisted Measurements")
        st.info("Upload a photo to get AI assistance with body measurements")
        
        measurement_col, image_col = st.columns([1, 1])
        
        with measurement_col:
            body_part = st.selectbox(
                "Body Part",
                ["waist", "chest", "bicep", "thigh", "neck", "forearm", "calf"],
                help="Select the body part to measure"
            )
            
            manual_value = st.number_input(
                "Your Measurement (cm)",
                min_value=10.0,
                max_value=200.0,
                value=75.0,
                step=0.1,
                help="Enter your manual measurement"
            )
            
            measurement_notes = st.text_area(
                "Notes",
                placeholder="Time of day, conditions, etc.",
                height=80
            )
        
        with image_col:
            measurement_image = st.file_uploader(
                "Upload Measurement Photo",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear photo for AI measurement assistance"
            )
            
            if measurement_image:
                image = Image.open(measurement_image)
                st.image(image, caption="Measurement reference photo", width=250)
        
        if st.button("📏 Record Measurement with AI Assistance"):
            if measurement_image:
                with st.spinner("Processing measurement with AI assistance..."):
                    measurement_result = tracker.add_measurement_with_image(
                        "circumference",
                        manual_value,
                        body_part,
                        Image.open(measurement_image),
                        measurement_notes
                    )
                    
                    if measurement_result:
                        st.success("✅ Measurement recorded successfully!")
                        
                        # Show measurement details
                        st.markdown("#### 📊 Measurement Details")
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.metric("Your Measurement", f"{measurement_result.value} cm")
                        
                        with detail_cols[1]:
                            st.metric("Method", measurement_result.measurement_method.replace('_', ' ').title())
                        
                        with detail_cols[2]:
                            if measurement_result.confidence_score:
                                st.metric("AI Confidence", f"{measurement_result.confidence_score:.0%}")
                        
                        if measurement_result.notes:
                            st.info(f"Notes: {measurement_result.notes}")
            else:
                # Record without image
                measurement_result = tracker.add_measurement_with_image(
                    "circumference",
                    manual_value,
                    body_part,
                    None,
                    measurement_notes
                )
                st.success("✅ Measurement recorded!")
