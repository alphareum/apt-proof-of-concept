"""
Enhanced User Interface Components
Improved Streamlit interface with better UX and modern design

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import json

from models import UserProfile, GoalType, FitnessLevel, ActivityLevel, Gender, EquipmentType
from database import get_database
from recommendation_engine import AdvancedExerciseRecommendationEngine

class ModernUI:
    """Modern UI components and layouts."""
    
    @staticmethod
    def setup_page_config():
        """Setup page configuration."""
        st.set_page_config(
            page_title="AI Fitness Assistant Pro",
            page_icon="🏋️‍♀️"
        )
    
    @staticmethod
    def inject_custom_css():
        """Inject basic CSS."""
        pass
    
    @staticmethod
    def render_header():
        """Render the main application header."""
        st.title("🏋️‍♀️ AI Fitness Assistant Pro")
        st.subheader("Your Intelligent Fitness Companion")
    
    @staticmethod
    def render_metric_card(title: str, value: str, subtitle: str = "", icon: str = "") -> None:
        """Render a metric card with basic styling."""
        st.metric(
            label=f"{icon} {title}" if icon else title,
            value=value,
            help=subtitle if subtitle else None
        )
    
    @staticmethod
    def render_progress_card(title: str, progress: float, target: str = "") -> None:
        """Render a progress card with basic styling."""
        st.subheader(title)
        st.progress(progress / 100 if progress > 1 else progress)
        if target:
            st.write(f"{progress:.1f}% {target}")
    
    @staticmethod
    @staticmethod
    def render_exercise_card(exercise_name: str, details: Dict[str, Any]) -> None:
        """Render an exercise card with details."""
        st.subheader(exercise_name)
        
        cols = st.columns(4)
        if details.get("sets"):
            cols[0].write(f"**Sets:** {details['sets']}")
        if details.get("reps"):
            cols[1].write(f"**Reps:** {details['reps']}")
        if details.get("duration"):
            cols[2].write(f"**Duration:** {details['duration']}")
        if details.get("rest"):
            cols[3].write(f"**Rest:** {details['rest']}")
        
        if details.get("notes"):
            st.write(f"*{details['notes']}*")

class UserProfileManager:
    """Manage user profile creation and updates."""
    
    def __init__(self):
        self.db = get_database()
    
    def render_profile_form(self) -> Optional[UserProfile]:
        """Render comprehensive user profile form."""
        
        st.markdown("### 👤 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age", 
                min_value=18, 
                max_value=100, 
                value=25,
                step=1,
                help="Your current age in years"
            )
            
            weight = st.number_input(
                "Weight (kg)", 
                min_value=30.0, 
                max_value=300.0, 
                value=70.0,
                step=0.5,
                help="Your current weight in kilograms"
            )
            
            activity_level = st.selectbox(
                "Activity Level",
                options=list(ActivityLevel),
                format_func=lambda x: x.value.replace('_', ' ').title(),
                help="Your general activity level throughout the day"
            )
        
        with col2:
            gender = st.selectbox(
                "Gender",
                options=list(Gender),
                format_func=lambda x: x.value.title()
            )
            
            height = st.number_input(
                "Height (cm)", 
                min_value=100.0, 
                max_value=250.0, 
                value=170.0,
                step=0.5,
                help="Your height in centimeters"
            )
            
            fitness_level = st.selectbox(
                "Fitness Level",
                options=list(FitnessLevel),
                format_func=lambda x: x.value.title(),
                help="Your current fitness experience level"
            )
        
        st.markdown("### 🎯 Goals and Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_goal = st.selectbox(
                "Primary Goal",
                options=list(GoalType),
                format_func=lambda x: x.value.replace('_', ' ').title(),
                help="Your main fitness objective"
            )
            
            available_time = st.slider(
                "Available Time per Workout (minutes)",
                min_value=15,
                max_value=120,
                value=30,
                step=5,
                help="How much time you can dedicate to each workout"
            )
        
        with col2:
            workout_days = st.slider(
                "Workout Days per Week",
                min_value=1,
                max_value=7,
                value=3,
                help="How many days per week you want to exercise"
            )
            
            years_training = st.number_input(
                "Years of Training Experience",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="How many years you've been training (0 for beginner)"
            )
        
        st.markdown("### 🏋️‍♀️ Equipment and Constraints")
        
        available_equipment = st.multiselect(
            "Available Equipment",
            options=list(EquipmentType),
            format_func=lambda x: x.value.replace('_', ' ').title(),
            help="Select all equipment you have access to"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            injuries = st.text_area(
                "Injuries or Physical Limitations",
                help="List any injuries or physical limitations (one per line)",
                placeholder="e.g.:\nKnee injury\nLower back pain"
            ).split('\n') if st.text_area(
                "Injuries or Physical Limitations",
                help="List any injuries or physical limitations (one per line)",
                placeholder="e.g.:\nKnee injury\nLower back pain"
            ) else []
            
        with col2:
            medical_conditions = st.text_area(
                "Medical Conditions",
                help="List any relevant medical conditions (one per line)",
                placeholder="e.g.:\nDiabetes\nHigh blood pressure"
            ).split('\n') if st.text_area(
                "Medical Conditions",
                help="List any relevant medical conditions (one per line)",
                placeholder="e.g.:\nDiabetes\nHigh blood pressure"
            ) else []
        
        if st.button("💾 Save Profile", type="primary"):
            try:
                # Clean up empty strings from lists
                injuries = [i.strip() for i in injuries if i.strip()]
                medical_conditions = [c.strip() for c in medical_conditions if c.strip()]
                
                profile = UserProfile(
                    age=age,
                    gender=gender,
                    weight=weight,
                    height=height,
                    activity_level=activity_level,
                    fitness_level=fitness_level,
                    primary_goal=primary_goal,
                    available_time=available_time,
                    workout_days_per_week=workout_days,
                    years_training=years_training,
                    available_equipment=available_equipment,
                    injuries=injuries,
                    medical_conditions=medical_conditions
                )
                
                # Save to database
                success = self.db.save_user_profile(
                    profile.user_id,
                    profile.to_dict()
                )
                
                if success:
                    st.success("✅ Profile saved successfully!")
                    st.session_state.user_profile = profile
                    return profile
                else:
                    st.error("❌ Failed to save profile. Please try again.")
                    
            except ValueError as e:
                st.error(f"❌ Validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
        
        return None
    
    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from database."""
        profile_data = self.db.get_user_profile(user_id)
        if profile_data:
            try:
                return UserProfile(**profile_data['profile'])
            except Exception as e:
                st.error(f"Error loading profile: {e}")
        return None

class DashboardManager:
    """Manage dashboard components and analytics."""
    
    def __init__(self):
        self.db = get_database()
    
    def render_progress_dashboard(self, user_profile: UserProfile):
        """Render comprehensive progress dashboard."""
        
        st.markdown("## 📊 Your Fitness Dashboard")
        
        # Get analytics data
        analytics = self.db.get_analytics_summary(user_profile.user_id)
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ModernUI.render_metric_card(
                "Total Workouts",
                str(analytics.get('total_workouts', 0)),
                "Completed sessions",
                "🏃‍♀️"
            )
        
        with col2:
            ModernUI.render_metric_card(
                "Calories Burned",
                f"{analytics.get('total_calories', 0):,.0f}",
                "Total energy expenditure",
                "🔥"
            )
        
        with col3:
            ModernUI.render_metric_card(
                "Training Hours",
                f"{analytics.get('total_hours', 0):.1f}",
                "Time invested in fitness",
                "⏱️"
            )
        
        with col4:
            ModernUI.render_metric_card(
                "Current Streak",
                f"{analytics.get('recent_streak', 0)} days",
                "Consecutive workout days",
                "🔥"
            )
        
        # Progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_workout_frequency_chart(user_profile.user_id)
        
        with col2:
            self.render_calories_trend_chart(user_profile.user_id)
        
        # Body measurements progress
        self.render_measurements_progress(user_profile.user_id)
    
    def render_workout_frequency_chart(self, user_id: str):
        """Render workout frequency chart."""
        
        workout_history = self.db.get_workout_history(user_id, days=30)
        
        if workout_history:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(workout_history)
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.day_name()
            
            # Count workouts by day of week
            day_counts = df['day_of_week'].value_counts()
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Workouts by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Number of Workouts'},
                color=day_counts.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No workout data available yet. Complete some workouts to see your patterns!")
    
    def render_calories_trend_chart(self, user_id: str):
        """Render calories burned trend chart."""
        
        workout_history = self.db.get_workout_history(user_id, days=30)
        
        if workout_history:
            df = pd.DataFrame(workout_history)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Group by date and sum calories
            daily_calories = df.groupby('date')['calories_burned'].sum().reset_index()
            
            fig = px.line(
                daily_calories,
                x='date',
                y='calories_burned',
                title="Daily Calories Burned",
                labels={'date': 'Date', 'calories_burned': 'Calories Burned'}
            )
            
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No calorie data available yet. Log some workouts to track your progress!")
    
    def render_measurements_progress(self, user_id: str):
        """Render body measurements progress."""
        
        st.markdown("### 📏 Body Measurements Progress")
        
        measurements_df = self.db.get_measurement_history(user_id, days=90)
        
        if not measurements_df.empty:
            # Create subplots for different measurements
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Weight', 'Body Fat %', 'Muscle Mass', 'Waist'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            measurements_df['date'] = pd.to_datetime(measurements_df['date'])
            
            # Weight
            if 'weight' in measurements_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=measurements_df['date'],
                        y=measurements_df['weight'],
                        name='Weight (kg)',
                        line=dict(color='#667eea')
                    ),
                    row=1, col=1
                )
            
            # Body fat percentage
            if 'body_fat_percentage' in measurements_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=measurements_df['date'],
                        y=measurements_df['body_fat_percentage'],
                        name='Body Fat %',
                        line=dict(color='#f093fb')
                    ),
                    row=1, col=2
                )
            
            # Muscle mass
            if 'muscle_mass' in measurements_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=measurements_df['date'],
                        y=measurements_df['muscle_mass'],
                        name='Muscle Mass (kg)',
                        line=dict(color='#28a745')
                    ),
                    row=2, col=1
                )
            
            # Waist measurement
            if 'waist' in measurements_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=measurements_df['date'],
                        y=measurements_df['waist'],
                        name='Waist (cm)',
                        line=dict(color='#ffc107')
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No measurement data available. Add some body measurements to track your progress!")

class RecommendationUI:
    """UI components for exercise recommendations."""
    
    def __init__(self):
        self.engine = AdvancedExerciseRecommendationEngine()
        self.db = get_database()
    
    def render_recommendations(self, user_profile: UserProfile):
        """Render comprehensive exercise recommendations."""
        
        st.markdown("## 💪 Personalized Exercise Recommendations")
        
        # Get user's workout history
        workout_history = self.db.get_workout_history(user_profile.user_id)
        
        with st.spinner("🤖 Generating personalized recommendations..."):
            recommendations = self.engine.generate_personalized_recommendations(
                user_profile, workout_history
            )
        
        if 'error' in recommendations:
            st.error(f"❌ {recommendations['error']}")
            return
        
        # Display confidence score
        confidence = recommendations.get('recommendation_confidence', 0.5)
        st.info(f"🎯 Recommendation Confidence: {confidence*100:.0f}% "
               f"(based on profile completeness and workout history)")
        
        # Recommendations tabs
        rec_tabs = st.tabs(["🏃‍♀️ Cardio", "🏋️‍♀️ Strength", "🧘‍♀️ Flexibility", "📅 Weekly Plan"])
        
        with rec_tabs[0]:
            self.render_cardio_recommendations(recommendations.get('cardio', []))
        
        with rec_tabs[1]:
            self.render_strength_recommendations(recommendations.get('strength', []))
        
        with rec_tabs[2]:
            self.render_flexibility_recommendations(recommendations.get('flexibility', []))
        
        with rec_tabs[3]:
            self.render_weekly_plan(recommendations.get('weekly_plan', {}))
        
        # Additional information
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_nutrition_tips(recommendations.get('nutrition_tips', []))
        
        with col2:
            self.render_safety_guidelines(recommendations.get('safety_guidelines', []))
        
        # Progression plan
        self.render_progression_plan(recommendations.get('progression_plan', {}))
    
    def render_cardio_recommendations(self, cardio_recs: List[Dict]):
        """Render cardio exercise recommendations."""
        
        st.markdown("### 🏃‍♀️ Cardio Exercises")
        
        if not cardio_recs:
            st.warning("No suitable cardio exercises found. Please review your equipment and constraints.")
            return
        
        for i, rec in enumerate(cardio_recs):
            exercise = rec['exercise']
            
            with st.expander(f"🏃‍♀️ {exercise.name} (Score: {rec['score']:.1f}/1.0)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Duration:** {rec['recommended_duration']}")
                    st.markdown(f"**Intensity:** {rec['intensity_level']}")
                    st.markdown(f"**Frequency:** {rec['weekly_frequency']}")
                    st.markdown(f"**Difficulty:** {exercise.difficulty}/5")
                
                with col2:
                    st.markdown("**Target Muscle Groups:**")
                    for muscle in exercise.muscle_groups:
                        st.markdown(f"• {muscle.title()}")
                
                if exercise.instructions:
                    st.markdown("**Instructions:**")
                    for i, instruction in enumerate(exercise.instructions, 1):
                        st.markdown(f"{i}. {instruction}")
                
                if exercise.tips:
                    st.markdown("**Tips:**")
                    for tip in exercise.tips:
                        st.markdown(f"💡 {tip}")
                
                st.markdown(f"**Progression:** {rec['progression_notes']}")
    
    def render_strength_recommendations(self, strength_recs: List[Dict]):
        """Render strength exercise recommendations."""
        
        st.markdown("### 🏋️‍♀️ Strength Exercises")
        
        if not strength_recs:
            st.warning("No suitable strength exercises found. Please review your equipment and constraints.")
            return
        
        for rec in strength_recs:
            exercise = rec['exercise']
            
            with st.expander(f"💪 {exercise.name} (Score: {rec['score']:.1f}/1.0)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Sets:** {rec['recommended_sets']}")
                    st.markdown(f"**Reps:** {rec['recommended_reps']}")
                    st.markdown(f"**Rest:** {rec['rest_time']} seconds")
                    st.markdown(f"**Frequency:** {rec['weekly_frequency']}")
                
                with col2:
                    st.markdown("**Target Muscle Groups:**")
                    for muscle in exercise.muscle_groups:
                        st.markdown(f"• {muscle.title()}")
                
                if exercise.instructions:
                    st.markdown("**Instructions:**")
                    for i, instruction in enumerate(exercise.instructions, 1):
                        st.markdown(f"{i}. {instruction}")
                
                if exercise.tips:
                    st.markdown("**Tips:**")
                    for tip in exercise.tips:
                        st.markdown(f"💡 {tip}")
                
                # Progression plan
                progression = rec.get('progression_plan', {})
                if progression:
                    st.markdown("**Progression Plan:**")
                    for level, description in progression.items():
                        st.markdown(f"• **{level.title()}:** {description}")
    
    def render_flexibility_recommendations(self, flexibility_recs: List[Dict]):
        """Render flexibility exercise recommendations."""
        
        st.markdown("### 🧘‍♀️ Flexibility & Recovery")
        
        if not flexibility_recs:
            st.info("Consider adding flexibility work to your routine for better recovery and mobility.")
            return
        
        for rec in flexibility_recs:
            exercise = rec['exercise']
            
            with st.expander(f"🧘‍♀️ {exercise.name} (Score: {rec['score']:.1f}/1.0)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Duration:** {rec['recommended_duration']}")
                    st.markdown(f"**Frequency:** {rec['frequency']}")
                
                with col2:
                    st.markdown("**Focus Areas:**")
                    for area in rec['focus_areas']:
                        st.markdown(f"• {area.title()}")
                
                st.markdown("**Benefits:**")
                for benefit in rec.get('benefits', []):
                    st.markdown(f"✨ {benefit}")
                
                if exercise.instructions:
                    st.markdown("**Instructions:**")
                    for i, instruction in enumerate(exercise.instructions, 1):
                        st.markdown(f"{i}. {instruction}")
    
    def render_weekly_plan(self, weekly_plan: Dict[str, Dict]):
        """Render weekly workout plan."""
        
        st.markdown("### 📅 Your Weekly Workout Plan")
        
        if not weekly_plan:
            st.warning("No weekly plan generated. Please ensure your profile is complete.")
            return
        
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days_of_week:
            if day in weekly_plan:
                plan = weekly_plan[day]
                
                with st.expander(f"📅 {day} - {plan.get('focus', plan.get('type', 'Workout'))}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Duration:** {plan.get('total_duration', plan.get('duration', 30))} minutes")
                        if 'intensity' in plan:
                            st.markdown(f"**Intensity:** {plan['intensity'].title()}")
                    
                    with col2:
                        if 'estimated_calories' in plan:
                            st.markdown(f"**Est. Calories:** {plan['estimated_calories']}")
                    
                    # Exercises or activities
                    if 'exercises' in plan:
                        st.markdown("**Exercises:**")
                        for exercise in plan['exercises']:
                            if isinstance(exercise, dict):
                                ModernUI.render_exercise_card(exercise['name'], exercise)
                            else:
                                st.markdown(f"• {exercise}")
                    
                    elif 'activities' in plan:
                        st.markdown("**Activities:**")
                        for activity in plan['activities']:
                            st.markdown(f"• {activity}")
                    
                    if 'notes' in plan:
                        st.info(f"💡 {plan['notes']}")
    
    def render_nutrition_tips(self, nutrition_tips: List[str]):
        """Render nutrition recommendations."""
        
        st.markdown("### 🥗 Nutrition Tips")
        
        if nutrition_tips:
            for tip in nutrition_tips:
                st.markdown(f"🍎 {tip}")
        else:
            st.info("Complete your profile for personalized nutrition recommendations.")
    
    def render_safety_guidelines(self, safety_guidelines: List[str]):
        """Render safety guidelines."""
        
        st.markdown("### ⚠️ Safety Guidelines")
        
        if safety_guidelines:
            for guideline in safety_guidelines:
                st.markdown(f"⚠️ {guideline}")
        else:
            st.info("Stay safe and listen to your body during workouts!")
    
    def render_progression_plan(self, progression_plan: Dict[str, str]):
        """Render progression plan."""
        
        st.markdown("### 📈 Your Progression Plan")
        
        if progression_plan:
            for period, description in progression_plan.items():
                st.markdown(f"**{period.replace('_', ' ').title()}:**")
                st.markdown(f"{description}")
                st.markdown("")
        else:
            st.info("Progress gradually and consistently for best results!")

# Initialize UI components
def get_ui_components():
    """Get UI component instances."""
    return {
        'modern_ui': ModernUI(),
        'profile_manager': UserProfileManager(),
        'dashboard_manager': DashboardManager(),
        'recommendation_ui': RecommendationUI()
    }
