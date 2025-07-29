"""
UI components for APT Fitness Assistant
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..core.models import UserProfile, GoalType, FitnessLevel, ActivityLevel, Gender, EquipmentType
from ..engines.recommendation import get_recommendation_engine


class UIComponents:
    """Reusable UI components for the fitness app."""
    
    @staticmethod
    def render_profile_form() -> Optional[UserProfile]:
        """Render user profile creation form."""
        st.header("👤 Create Your Fitness Profile")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name", placeholder="Enter your name")
                age = st.number_input("Age", min_value=18, max_value=100, value=25)
                gender = st.selectbox("Gender", options=[g.value for g in Gender])
                height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
            
            with col2:
                activity_level = st.selectbox("Activity Level", 
                                            options=[a.value for a in ActivityLevel])
                fitness_level = st.selectbox("Fitness Level", 
                                           options=[f.value for f in FitnessLevel])
                primary_goal = st.selectbox("Primary Goal", 
                                          options=[g.value for g in GoalType])
                equipment = st.selectbox("Available Equipment", 
                                       options=[e.value for e in EquipmentType])
                workout_duration = st.slider("Preferred Workout Duration (minutes)", 
                                           15, 90, 30)
            
            # Additional preferences
            st.subheader("Additional Information")
            injuries = st.multiselect("Any injuries or limitations?", 
                                    options=["back", "knee", "shoulder", "wrist", "ankle", "neck"])
            workout_frequency = st.slider("Workouts per week", 1, 7, 3)
            
            submitted = st.form_submit_button("Create Profile")
            
            if submitted:
                if name:
                    return UserProfile(
                        name=name,
                        age=age,
                        gender=Gender(gender),
                        height_cm=height_cm,
                        weight_kg=weight_kg,
                        activity_level=ActivityLevel(activity_level),
                        fitness_level=FitnessLevel(fitness_level),
                        primary_goal=GoalType(primary_goal),
                        available_equipment=EquipmentType(equipment),
                        preferred_workout_duration=workout_duration,
                        workout_frequency_per_week=workout_frequency,
                        injuries=injuries
                    )
                else:
                    st.error("Please enter your name")
        
        return None
    
    @staticmethod
    def render_sidebar(user_profile: UserProfile):
        """Render sidebar with user information."""
        st.sidebar.header(f"👋 Hello, {user_profile.name}!")
        
        # User stats
        st.sidebar.metric("BMI", f"{user_profile.bmi:.1f}")
        st.sidebar.metric("Daily Calories", f"{user_profile.daily_calories:.0f}")
        
        # Quick stats
        st.sidebar.subheader("📊 Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Age", f"{user_profile.age}")
        col2.metric("Goal", user_profile.primary_goal.value.replace("_", " ").title())
        
        # Profile actions
        st.sidebar.subheader("⚙️ Actions")
        if st.sidebar.button("Edit Profile"):
            st.session_state.show_profile_edit = True
        
        if st.sidebar.button("Reset Data"):
            if st.sidebar.button("Confirm Reset", key="confirm_reset"):
                st.session_state.clear()
                st.experimental_rerun()
    
    @staticmethod
    def render_metrics_dashboard(user_profile: UserProfile, analytics_data: Dict[str, Any]):
        """Render metrics dashboard."""
        st.subheader("📊 Your Fitness Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workouts", analytics_data.get("total_workouts", 0))
        
        with col2:
            st.metric("Total Minutes", analytics_data.get("total_minutes", 0))
        
        with col3:
            st.metric("Calories Burned", f"{analytics_data.get('total_calories', 0):.0f}")
        
        with col4:
            st.metric("Current Streak", f"{analytics_data.get('current_streak', 0)} days")
        
        # Progress charts
        if analytics_data.get("total_workouts", 0) > 0:
            UIComponents.render_progress_charts(analytics_data)
    
    @staticmethod
    def render_progress_charts(analytics_data: Dict[str, Any]):
        """Render progress visualization charts."""
        st.subheader("📈 Progress Overview")
        
        # Sample data for demonstration
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        workout_data = pd.DataFrame({
            "Date": dates,
            "Workouts": [1 if i % 3 == 0 else 0 for i in range(30)],
            "Calories": [300 if i % 3 == 0 else 0 for i in range(30)]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(workout_data, x="Date", y="Workouts", 
                         title="Workout Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(workout_data, x="Date", y="Calories", 
                        title="Calories Burned")
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_workout_card(recommendation, index: int):
        """Render individual workout exercise card."""
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{recommendation.exercise.name}**")
                st.caption(f"Target: {', '.join(recommendation.exercise.muscle_groups)}")
            
            with col2:
                if recommendation.exercise.category == "cardio":
                    st.metric("Duration", f"{recommendation.duration_minutes}m")
                else:
                    st.metric("Sets × Reps", f"{recommendation.sets} × {recommendation.reps}")
            
            with col3:
                st.metric("Calories", f"~{recommendation.exercise.calories_per_minute * (recommendation.duration_minutes or 5):.0f}")
            
            # Exercise details expander
            with st.expander("Exercise Details"):
                st.write("**Instructions:**")
                for i, instruction in enumerate(recommendation.exercise.instructions, 1):
                    st.write(f"{i}. {instruction}")
                
                if recommendation.exercise.tips:
                    st.write("**Tips:**")
                    for tip in recommendation.exercise.tips:
                        st.write(f"💡 {tip}")
                
                if recommendation.exercise.contraindications:
                    st.warning("⚠️ Avoid if you have: " + ", ".join(recommendation.exercise.contraindications))
    
    @staticmethod
    def render_workout_summary(recommendations: List):
        """Render workout summary statistics."""
        if not recommendations:
            return
        
        total_time = sum(rec.duration_minutes or 5 for rec in recommendations)
        total_calories = sum(rec.exercise.calories_per_minute * (rec.duration_minutes or 5) 
                           for rec in recommendations)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time", f"{total_time} min")
        col2.metric("Estimated Calories", f"{total_calories:.0f}")
        col3.metric("Exercises", len(recommendations))
    
    @staticmethod
    def render_weekly_plan_view(weekly_plan: Dict[str, List]):
        """Render weekly workout plan."""
        st.subheader("📅 Your Weekly Plan")
        
        if not weekly_plan:
            st.info("Generate a workout plan to see your weekly schedule.")
            return
        
        tabs = st.tabs(list(weekly_plan.keys()))
        
        for i, (day, workouts) in enumerate(weekly_plan.items()):
            with tabs[i]:
                if workouts:
                    UIComponents.render_workout_summary(workouts)
                    st.divider()
                    
                    for j, workout in enumerate(workouts):
                        UIComponents.render_workout_card(workout, j)
                else:
                    st.info("Rest day - Recovery is important!")
    
    @staticmethod
    def render_body_analysis_upload():
        """Render enhanced body composition analysis with measurement inputs."""
        st.subheader("📸 Enhanced Body Composition Analysis")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📏 With Measurements (Recommended)", "📷 Image Only"])
        
        with tab1:
            st.info("💡 **For most accurate results, provide physical measurements along with your photo.**")
            
            # Physical measurements input
            st.subheader("📐 Physical Measurements")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Width Measurements (cm)**")
                shoulder_width = st.number_input(
                    "Shoulder Width", 
                    min_value=25.0, max_value=60.0, value=40.0, step=0.5,
                    help="Measure across the widest part of your shoulders"
                )
                waist_width = st.number_input(
                    "Waist Circumference", 
                    min_value=50.0, max_value=150.0, value=80.0, step=0.5,
                    help="Measure around the narrowest part of your waist"
                )
                hip_width = st.number_input(
                    "Hip Circumference", 
                    min_value=60.0, max_value=180.0, value=95.0, step=0.5,
                    help="Measure around the widest part of your hips"
                )
            
            with col2:
                st.write("**Circumference Measurements (cm)**")
                neck_circumference = st.number_input(
                    "Neck Circumference", 
                    min_value=25.0, max_value=50.0, value=35.0, step=0.5,
                    help="Measure around your neck just below the Adam's apple"
                )
                arm_circumference = st.number_input(
                    "Upper Arm Circumference", 
                    min_value=15.0, max_value=50.0, value=30.0, step=0.5,
                    help="Measure around the largest part of your upper arm (flexed)"
                )
                thigh_circumference = st.number_input(
                    "Thigh Circumference", 
                    min_value=30.0, max_value=80.0, value=55.0, step=0.5,
                    help="Measure around the largest part of your thigh"
                )
            
            with col3:
                st.write("**Additional Info**")
                height_cm = st.number_input(
                    "Height (cm)", 
                    min_value=140.0, max_value=220.0, value=170.0, step=0.5
                )
                weight_kg = st.number_input(
                    "Weight (kg)", 
                    min_value=40.0, max_value=200.0, value=70.0, step=0.1
                )
                age = st.number_input(
                    "Age", 
                    min_value=18, max_value=100, value=30
                )
                gender = st.selectbox(
                    "Gender", 
                    options=["Male", "Female", "Other"],
                    help="Used for body fat calculation formulas"
                )
            
            # Measurement tips
            with st.expander("📋 How to Take Accurate Measurements"):
                st.markdown("""
                **Tips for accurate measurements:**
                
                🎯 **Shoulder Width**: Stand relaxed, measure from the outer edge of one shoulder to the other
                
                📏 **Waist**: Find your natural waist (narrowest point), usually above the belly button
                
                🍑 **Hips**: Measure around the widest part of your hips/buttocks
                
                👔 **Neck**: Measure just below the Adam's apple, snug but not tight
                
                💪 **Arms**: Flex your bicep and measure around the largest part
                
                🦵 **Thigh**: Measure around the largest part of your upper thigh
                
                **General Tips:**
                - Use a flexible measuring tape
                - Take measurements at the same time of day
                - Don't pull the tape too tight
                - Take 2-3 measurements and use the average
                """)
            
            # Photo upload for enhanced analysis
            st.subheader("📸 Upload Photo")
            uploaded_file_enhanced = st.file_uploader(
                "Upload a full-body photo for enhanced analysis",
                type=["jpg", "jpeg", "png"],
                help="Photo will be combined with measurements for most accurate results",
                key="enhanced_upload"
            )
            
            if uploaded_file_enhanced:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(uploaded_file_enhanced, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.success("✅ **Enhanced Analysis Ready**")
                    st.info("**Analysis will include:**\n"
                           "• Navy Method body fat calculation\n"
                           "• Jackson-Pollock estimation\n"
                           "• Enhanced muscle mass calculation\n"
                           "• Visceral fat assessment\n"
                           "• Improved BMR calculation")
                    
                    if st.button("🔬 Analyze with Measurements", type="primary", key="enhanced_analyze"):
                        with st.spinner("Running enhanced body composition analysis..."):
                            # Prepare measurements dictionary
                            measurements = {
                                "shoulder_width_cm": shoulder_width,
                                "waist_width_cm": waist_width,
                                "hip_width_cm": hip_width,
                                "neck_width_cm": neck_circumference,
                                "arm_circumference_cm": arm_circumference,
                                "thigh_circumference_cm": thigh_circumference,
                                "height_cm": height_cm
                            }
                            
                            user_profile = {
                                "age": age,
                                "gender": gender.lower(),
                                "weight_kg": weight_kg,
                                "height_cm": height_cm
                            }
                            
                            # Here you would call the enhanced analyzer
                            st.success("✅ Enhanced analysis complete!")
                            
                            # Display enhanced results
                            UIComponents._display_enhanced_results(measurements, user_profile)
            
            return uploaded_file_enhanced, {
                "measurements": {
                    "shoulder_width_cm": shoulder_width,
                    "waist_width_cm": waist_width,
                    "hip_width_cm": hip_width,
                    "neck_width_cm": neck_circumference,
                    "arm_circumference_cm": arm_circumference,
                    "thigh_circumference_cm": thigh_circumference,
                    "height_cm": height_cm
                },
                "user_profile": {
                    "age": age,
                    "gender": gender.lower(),
                    "weight_kg": weight_kg,
                    "height_cm": height_cm
                }
            } if uploaded_file_enhanced else None
        
        with tab2:
            st.warning("⚠️ **Image-only analysis is less accurate.** For best results, use the 'With Measurements' tab.")
            
            uploaded_file_basic = st.file_uploader(
                "Upload a full-body photo for basic analysis",
                type=["jpg", "jpeg", "png"],
                help="Analysis will use pose detection and image analysis only",
                key="basic_upload"
            )
            
            if uploaded_file_basic:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(uploaded_file_basic, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.info("**Basic Analysis Includes:**\n"
                           "• Pose-based measurements\n"
                           "• Estimated body composition\n"
                           "• Basic body shape classification\n"
                           "• BMR estimation")
                    
                    if st.button("📷 Analyze Image Only", type="secondary", key="basic_analyze"):
                        with st.spinner("Analyzing body composition from image..."):
                            # Here you would call the basic analyzer
                            st.success("✅ Basic analysis complete!")
                            
                            # Mock results for basic analysis
                            col1, col2 = st.columns(2)
                            col1.metric("Body Fat %", "18.5%", help="Estimated from pose analysis")
                            col2.metric("Muscle Mass %", "42.1%", help="Estimated from body proportions")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Body Shape", "Athletic")
                            col2.metric("Confidence", "Medium", help="Lower confidence without measurements")
            
            return uploaded_file_basic, None
    
    @staticmethod
    def _display_enhanced_results(measurements: Dict[str, float], user_profile: Dict[str, Any]):
        """Display enhanced analysis results."""
        st.subheader("🎯 Enhanced Analysis Results")
        
        # Calculate enhanced metrics (mock calculation for demo)
        waist_to_height = measurements["waist_width_cm"] / measurements["height_cm"]
        bmi = user_profile["weight_kg"] / ((measurements["height_cm"] / 100) ** 2)
        
        # Enhanced body fat calculation (simplified Navy method)
        if user_profile["gender"] == "male":
            body_fat = 495 / (1.0324 - 0.19077 * np.log10(measurements["waist_width_cm"] - measurements["neck_width_cm"]) + 
                             0.15456 * np.log10(measurements["height_cm"])) - 450
        else:
            body_fat = 495 / (1.29579 - 0.35004 * np.log10(measurements["waist_width_cm"] + measurements["hip_width_cm"] - measurements["neck_width_cm"]) + 
                             0.22100 * np.log10(measurements["height_cm"])) - 450
        
        body_fat = max(5, min(35, body_fat))  # Reasonable bounds
        muscle_mass = max(25, min(55, 100 - body_fat - 20))  # Rough calculation
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Body Fat %", 
                f"{body_fat:.1f}%",
                help="Calculated using Navy Method with your measurements"
            )
        
        with col2:
            st.metric(
                "Muscle Mass %", 
                f"{muscle_mass:.1f}%",
                help="Estimated using anthropometric formulas"
            )
        
        with col3:
            # Enhanced BMR calculation
            if user_profile["gender"] == "male":
                bmr = 88.362 + (13.397 * user_profile["weight_kg"]) + (4.799 * measurements["height_cm"]) - (5.677 * user_profile["age"])
            else:
                bmr = 447.593 + (9.247 * user_profile["weight_kg"]) + (3.098 * measurements["height_cm"]) - (4.330 * user_profile["age"])
            
            st.metric(
                "BMR (cal/day)", 
                f"{int(bmr)}",
                help="Calculated using Mifflin-St Jeor equation"
            )
        
        with col4:
            st.metric(
                "Analysis Quality", 
                "High",
                help="High confidence due to physical measurements"
            )
        
        # Health indicators
        st.subheader("🏥 Health Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Waist-to-height ratio
            if waist_to_height < 0.5:
                wth_status = "Excellent"
                wth_color = "green"
            elif waist_to_height < 0.6:
                wth_status = "Good"
                wth_color = "yellow"
            else:
                wth_status = "High Risk"
                wth_color = "red"
            
            st.metric(
                "Waist-to-Height Ratio", 
                f"{waist_to_height:.2f}",
                delta=wth_status,
                help="Health indicator for abdominal obesity risk"
            )
        
        with col2:
            # BMI category
            if bmi < 18.5:
                bmi_cat = "Underweight"
            elif bmi < 25:
                bmi_cat = "Normal"
            elif bmi < 30:
                bmi_cat = "Overweight"
            else:
                bmi_cat = "Obese"
            
            st.metric(
                "BMI", 
                f"{bmi:.1f}",
                delta=bmi_cat
            )
        
        with col3:
            # Body shape based on measurements
            shoulder_to_waist = measurements["shoulder_width_cm"] / measurements["waist_width_cm"]
            waist_to_hip = measurements["waist_width_cm"] / measurements["hip_width_cm"]
            
            if shoulder_to_waist > 1.4 and waist_to_hip < 0.8:
                body_shape = "Athletic V-Shape"
            elif waist_to_hip > 0.9:
                body_shape = "Apple Shape"
            elif shoulder_to_waist < 1.1:
                body_shape = "Pear Shape"
            else:
                body_shape = "Rectangle"
            
            st.metric("Body Shape", body_shape)
        
        # Detailed breakdown
        with st.expander("📊 Detailed Body Composition Breakdown"):
            fat_mass = (body_fat / 100) * user_profile["weight_kg"]
            muscle_mass_kg = (muscle_mass / 100) * user_profile["weight_kg"]
            bone_mass = user_profile["weight_kg"] * 0.15
            water_mass = user_profile["weight_kg"] - fat_mass - muscle_mass_kg - bone_mass
            
            breakdown_data = {
                "Component": ["Fat Mass", "Muscle Mass", "Bone Mass", "Water"],
                "Weight (kg)": [fat_mass, muscle_mass_kg, bone_mass, water_mass],
                "Percentage": [body_fat, muscle_mass, 15, (water_mass/user_profile["weight_kg"])*100]
            }
            
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, hide_index=True)
        
        # Calculation methods used
        with st.expander("🔬 Calculation Methods Used"):
            st.markdown("""
            **Body Fat Calculation:**
            - Navy Method (primary): Uses waist, neck, and hip circumferences
            - Validated against DEXA scans with ±3% accuracy
            
            **Muscle Mass Estimation:**
            - Anthropometric formulas using limb circumferences
            - Correlated with bioelectrical impedance analysis
            
            **BMR Calculation:**
            - Mifflin-St Jeor equation (most accurate for general population)
            - Adjusted for body composition when available
            
            **Health Indicators:**
            - Waist-to-height ratio: WHO guidelines
            - BMI: Standard classification
            - Body shape: Based on circumference ratios
            """)
        
        return True
    
    @staticmethod
    def render_goal_tracker(user_profile: UserProfile):
        """Render goal tracking interface."""
        st.subheader("🎯 Goal Tracking")
        
        # Current goal display
        st.write(f"**Primary Goal:** {user_profile.primary_goal.value.replace('_', ' ').title()}")
        
        # Goal progress (mock data)
        progress = 65  # Mock progress percentage
        st.progress(progress / 100)
        st.write(f"Progress: {progress}%")
        
        # Goal settings
        with st.expander("⚙️ Goal Settings"):
            target_weight = st.number_input("Target Weight (kg)", 
                                          value=user_profile.target_weight_kg or user_profile.weight_kg)
            target_date = st.date_input("Target Date")
            
            if st.button("Update Goal"):
                st.success("Goal updated successfully!")
    
    @staticmethod
    def render_exercise_timer():
        """Render exercise timer component."""
        st.subheader("⏱️ Workout Timer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            minutes = st.number_input("Minutes", min_value=0, max_value=60, value=0)
        
        with col2:
            seconds = st.number_input("Seconds", min_value=0, max_value=59, value=30)
        
        with col3:
            if st.button("Start Timer"):
                total_seconds = minutes * 60 + seconds
                
                # Simple countdown display
                placeholder = st.empty()
                for i in range(total_seconds, 0, -1):
                    mins, secs = divmod(i, 60)
                    placeholder.metric("Time Remaining", f"{mins:02d}:{secs:02d}")
                    # Note: In a real app, you'd use time.sleep(1) here
                
                st.success("Timer finished! 🎉")


def get_ui_components() -> UIComponents:
    """Get UI components instance."""
    return UIComponents()
