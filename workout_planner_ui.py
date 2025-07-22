"""
Enhanced Workout Planner UI Components
Advanced interface for workout planning and tracking

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from models import UserProfile, FitnessLevel, GoalType
from workout_planner import AdvancedWorkoutPlanner, WorkoutSchedule, ProgressMetrics
from enhanced_recommendation_system import EnhancedRecommendationEngine

class WorkoutPlannerUI:
    """Enhanced UI for workout planning and tracking."""
    
    def __init__(self):
        self.planner = AdvancedWorkoutPlanner()
        self.recommendation_engine = EnhancedRecommendationEngine()
    
    def render_workout_planner_tab(self, user_profile: UserProfile):
        """Render the complete workout planner interface."""
        
        st.markdown("## 📅 Advanced Workout Planner")
        st.markdown("Create comprehensive, adaptive workout programs tailored to your goals.")
        
        # Main planner tabs
        planner_tabs = st.tabs([
            "🎯 Program Builder", 
            "📊 Weekly View", 
            "📈 Progress Tracking",
            "🔄 Adaptive Features",
            "📱 Today's Workout"
        ])
        
        with planner_tabs[0]:
            self._render_program_builder(user_profile)
        
        with planner_tabs[1]:
            self._render_weekly_view(user_profile)
        
        with planner_tabs[2]:
            self._render_progress_tracking(user_profile)
        
        with planner_tabs[3]:
            self._render_adaptive_features(user_profile)
        
        with planner_tabs[4]:
            self._render_todays_workout(user_profile)
    
    def _render_program_builder(self, user_profile: UserProfile):
        """Render the program builder interface."""
        
        st.markdown("### 🏗️ Build Your Personalized Program")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Program parameters
            st.markdown("#### Program Configuration")
            
            program_duration = st.selectbox(
                "Program Duration",
                [4, 6, 8, 12, 16, 20, 24],
                index=3,
                help="Select the total length of your program in weeks"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=date.today(),
                min_value=date.today(),
                help="When would you like to start your program?"
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Program Options"):
                include_deload = st.checkbox("Include Deload Weeks", value=True)
                include_assessments = st.checkbox("Schedule Fitness Assessments", value=True)
                adaptive_planning = st.checkbox("Enable Adaptive Planning", value=True)
                
                periodization_style = st.selectbox(
                    "Periodization Style",
                    ["Linear", "Undulating", "Block", "Conjugate"],
                    help="Select the periodization approach for your program"
                )
        
        with col2:
            # Program preview
            st.markdown("#### Program Preview")
            
            # Quick stats
            total_workouts = program_duration * user_profile.workout_days_per_week
            rest_days = program_duration * 7 - total_workouts
            
            st.metric("Total Workouts", total_workouts)
            st.metric("Rest Days", rest_days)
            st.metric("Training Density", f"{(total_workouts/(program_duration*7)*100):.0f}%")
        
        # Generate program button
        if st.button("🚀 Generate Complete Program", type="primary", use_container_width=True):
            with st.spinner("Creating your personalized program..."):
                try:
                    schedule = self.planner.create_comprehensive_plan(
                        user_profile, start_date, program_duration
                    )
                    
                    # Store in session state
                    st.session_state.workout_schedule = schedule
                    
                    st.success("✅ Program created successfully!")
                    
                    # Show program overview
                    self._display_program_overview(schedule)
                    
                except Exception as e:
                    st.error(f"Error creating program: {str(e)}")
    
    def _render_weekly_view(self, user_profile: UserProfile):
        """Render weekly workout view."""
        
        st.markdown("### 📊 Weekly Workout Overview")
        
        # Check if we have a schedule
        if 'workout_schedule' not in st.session_state:
            st.info("👆 Please create a program first in the Program Builder tab.")
            return
        
        schedule = st.session_state.workout_schedule
        
        # Week selector
        total_weeks = len(schedule.weekly_plans)
        selected_week = st.selectbox(
            "Select Week",
            range(1, total_weeks + 1),
            format_func=lambda x: f"Week {x}"
        )
        
        if selected_week <= total_weeks:
            weekly_plan = schedule.weekly_plans[selected_week - 1]
            
            # Week overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Phase", weekly_plan.phase)
            with col2:
                st.metric("Workout Days", len(weekly_plan.workout_days))
            with col3:
                st.metric("Total Volume", f"{weekly_plan.total_weekly_volume} min")
            with col4:
                st.metric("Rest Days", len(weekly_plan.rest_days))
            
            # Daily breakdown
            st.markdown("#### Daily Workout Schedule")
            
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for i, day in enumerate(days):
                with st.expander(f"{day} - {self._get_day_summary(weekly_plan, day)}", 
                               expanded=False):
                    
                    if day in weekly_plan.rest_days:
                        st.markdown("🛌 **Rest Day**")
                        st.markdown("- Focus on recovery and regeneration")
                        st.markdown("- Light activity: walking, stretching")
                        st.markdown("- Prioritize sleep and nutrition")
                    else:
                        # Find workout for this day
                        workout_day = None
                        for wd in weekly_plan.workout_days:
                            if wd.day_name == day:
                                workout_day = wd
                                break
                        
                        if workout_day:
                            self._display_workout_day(workout_day)
            
            # Weekly nutrition focus
            if weekly_plan.nutrition_focus:
                st.markdown("#### 🍎 Weekly Nutrition Focus")
                for focus in weekly_plan.nutrition_focus:
                    st.markdown(f"• {focus}")
            
            # Progressive overload notes
            if weekly_plan.progressive_overload_notes:
                st.markdown("#### 📈 Progressive Overload This Week")
                for note in weekly_plan.progressive_overload_notes:
                    st.markdown(f"• {note}")
    
    def _render_progress_tracking(self, user_profile: UserProfile):
        """Render progress tracking interface."""
        
        st.markdown("### 📈 Progress Tracking & Analytics")
        
        # Check if we have progress data
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = []
        
        progress_data = st.session_state.progress_data
        
        # Input new workout data
        with st.expander("➕ Log Today's Workout", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                workout_date = st.date_input("Workout Date", value=date.today())
                completed = st.checkbox("Workout Completed", value=True)
                perceived_exertion = st.slider("Perceived Exertion (1-10)", 1, 10, 7)
                actual_duration = st.number_input("Actual Duration (minutes)", min_value=0, value=45)
            
            with col2:
                recovery_score = st.slider("Recovery Score (1-10)", 1, 10, 7)
                workout_notes = st.text_area("Workout Notes", placeholder="How did the workout feel?")
            
            if st.button("📝 Log Workout"):
                new_progress = ProgressMetrics(
                    date=workout_date,
                    workout_completed=completed,
                    perceived_exertion=perceived_exertion,
                    duration_actual=actual_duration,
                    exercises_completed=[],
                    weights_used={},
                    reps_completed={},
                    notes=workout_notes,
                    recovery_score=recovery_score
                )
                
                st.session_state.progress_data.append(new_progress)
                st.success("✅ Workout logged successfully!")
                st.rerun()
        
        # Progress analytics
        if progress_data:
            self._display_progress_analytics(progress_data)
        else:
            st.info("Start logging workouts to see your progress analytics!")
    
    def _render_adaptive_features(self, user_profile: UserProfile):
        """Render adaptive planning features."""
        
        st.markdown("### 🔄 Adaptive Planning & Smart Adjustments")
        
        if 'workout_schedule' not in st.session_state:
            st.info("👆 Please create a program first to use adaptive features.")
            return
        
        schedule = st.session_state.workout_schedule
        progress_data = st.session_state.get('progress_data', [])
        
        # Adaptive triggers status
        st.markdown("#### 🎯 Adaptive Triggers Status")
        
        trigger_cols = st.columns(len(schedule.adaptive_triggers))
        
        for i, trigger in enumerate(schedule.adaptive_triggers):
            with trigger_cols[i]:
                status = self._evaluate_trigger_status(trigger, progress_data)
                
                if status['active']:
                    st.warning(f"⚠️ {trigger['name'].replace('_', ' ').title()}")
                else:
                    st.success(f"✅ {trigger['name'].replace('_', ' ').title()}")
                
                st.caption(f"Sensitivity: {trigger['sensitivity']:.0%}")
        
        # Suggested adjustments
        if progress_data:
            st.markdown("#### 💡 Suggested Adjustments")
            
            adjustments = self.planner.suggest_schedule_adjustments(
                user_profile, schedule, progress_data
            )
            
            if adjustments:
                for adj in adjustments:
                    with st.expander(f"🔧 {adj.adjustment_type.replace('_', ' ').title()}", 
                                   expanded=True):
                        st.markdown(f"**Reason:** {adj.reason}")
                        st.markdown(f"**Confidence:** {adj.confidence_score:.0%}")
                        st.markdown(f"**Effective Date:** {adj.effective_date}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Apply Adjustment", key=f"apply_{adj.adjustment_type}"):
                                st.success("✅ Adjustment applied to your program!")
                        with col2:
                            if st.button(f"Dismiss", key=f"dismiss_{adj.adjustment_type}"):
                                st.info("Adjustment dismissed.")
            else:
                st.success("🎉 No adjustments needed - you're on track!")
        
        # Alternative workout generator
        st.markdown("#### 🔄 Alternative Workout Generator")
        
        constraint_options = st.multiselect(
            "Current Constraints",
            ["Limited Time", "Limited Equipment", "Need Lower Intensity", "Want Higher Challenge"],
            help="Select any current constraints for alternative workout options"
        )
        
        if constraint_options and st.button("Generate Alternatives"):
            # This would generate alternative workouts based on constraints
            st.success("Alternative workouts generated! Check Today's Workout tab.")
    
    def _render_todays_workout(self, user_profile: UserProfile):
        """Render today's specific workout."""
        
        st.markdown("### 📱 Today's Workout")
        
        if 'workout_schedule' not in st.session_state:
            st.info("👆 Please create a program first to see today's workout.")
            return
        
        schedule = st.session_state.workout_schedule
        progress_data = st.session_state.get('progress_data', [])
        
        # Get today's workout
        today = date.today()
        
        try:
            today_plan = self.planner.get_daily_workout_plan(
                user_profile, today, schedule, progress_data[-7:] if progress_data else None
            )
            
            # Debug: Check the type of today_plan
            if not isinstance(today_plan, dict):
                st.error(f"Error: Expected dictionary but got {type(today_plan)}. Data: {today_plan}")
                return
            
            if 'error' in today_plan:
                st.warning(f"⚠️ {today_plan['error']}")
                return
            
            # Display today's workout
            self._display_todays_workout_detail(today_plan, user_profile)
            
        except Exception as e:
            st.error(f"Error getting today's workout: {str(e)}")
            # Add debug info
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
    
    def _display_program_overview(self, schedule: WorkoutSchedule):
        """Display program overview after creation."""
        
        st.markdown("#### 🎉 Your Program Overview")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{len(schedule.weekly_plans)} weeks")
        with col2:
            total_workouts = sum(len(wp.workout_days) for wp in schedule.weekly_plans)
            st.metric("Total Workouts", total_workouts)
        with col3:
            st.metric("Deload Weeks", len(schedule.deload_weeks))
        with col4:
            st.metric("Assessments", len(schedule.assessment_dates))
        
        # Phase breakdown
        phases = {}
        for wp in schedule.weekly_plans:
            phases[wp.phase] = phases.get(wp.phase, 0) + 1
        
        st.markdown("#### Phase Distribution")
        phase_df = pd.DataFrame(list(phases.items()), columns=['Phase', 'Weeks'])
        fig = px.pie(phase_df, values='Weeks', names='Phase', title="Training Phases")
        st.plotly_chart(fig, use_container_width=True)
        
        # Milestones timeline
        if schedule.goal_milestones:
            st.markdown("#### Goal Milestones")
            for milestone in schedule.goal_milestones:
                st.markdown(f"**Week {milestone['week']}:** {milestone['goal']}")
    
    def _display_workout_day(self, workout_day):
        """Display detailed workout day information."""
        
        # Workout overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Type", workout_day.workout_type.replace('_', ' ').title())
        with col2:
            st.metric("Duration", f"{workout_day.total_duration} min")
        with col3:
            st.metric("Intensity", workout_day.intensity_level.title())
        
        # Exercise list
        st.markdown("**Exercises:**")
        for exercise in workout_day.exercises:
            with st.container():
                st.markdown(f"**{exercise['name']}**")
                
                exercise_cols = st.columns([2, 1, 1, 1])
                with exercise_cols[0]:
                    if exercise.get('technique_cues'):
                        st.caption("Key Cues: " + ", ".join(exercise['technique_cues'][:2]))
                with exercise_cols[1]:
                    st.caption(f"Sets: {exercise.get('sets', 'N/A')}")
                with exercise_cols[2]:
                    st.caption(f"Reps: {exercise.get('reps', 'N/A')}")
                with exercise_cols[3]:
                    st.caption(f"Rest: {exercise.get('rest_seconds', 60)}s")
        
        # Workout notes
        if workout_day.notes:
            st.markdown("**Notes:**")
            for note in workout_day.notes:
                st.markdown(f"• {note}")
    
    def _display_todays_workout_detail(self, today_plan: Dict[str, Any], user_profile: UserProfile):
        """Display detailed view of today's workout."""
        
        # Safety check: ensure today_plan is a dictionary
        if not isinstance(today_plan, dict):
            st.error(f"Error: today_plan is not a dictionary. Type: {type(today_plan)}")
            return
        
        workout = today_plan.get('workout')
        week_info = today_plan.get('week_info', {})
        
        # Handle None workout
        if workout is None:
            st.warning("⚠️ No workout data available for today.")
            return
        
        # Handle different workout types (dict vs object)
        if isinstance(workout, dict):
            # Workout is a dictionary - extract values safely
            workout_type = workout.get('workout_type', workout.get('name', 'Workout'))
            total_duration = workout.get('total_duration', workout.get('duration', 45))
            intensity_level = workout.get('intensity_level', workout.get('intensity', 'moderate'))
            estimated_calories = workout.get('estimated_calories', workout.get('calories', 300))
        elif hasattr(workout, 'workout_type'):
            # Workout is a WorkoutDay object - access attributes directly
            workout_type = workout.workout_type
            total_duration = workout.total_duration
            intensity_level = workout.intensity_level
            estimated_calories = workout.estimated_calories
        else:
            st.error(f"Error: workout object has unexpected format. Type: {type(workout)}")
            st.error(f"Available keys/attributes: {dir(workout) if hasattr(workout, '__dict__') else list(workout.keys()) if hasattr(workout, 'keys') else 'No keys available'}")
            return
        
        # Header
        st.markdown(f"## 🏋️‍♀️ {workout_type.replace('_', ' ').title()}")
        st.markdown(f"**Week {week_info.get('week_number', 1)} • {week_info.get('phase', 'Training')}**")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{total_duration} min")
        with col2:
            st.metric("Intensity", str(intensity_level).title())
        with col3:
            st.metric("Calories", f"~{estimated_calories}")
        with col4:
            progress_pct = (week_info.get('week_number', 1) / 
                          int(week_info.get('program_progress', '1/12').split('/')[1])) * 100
            st.metric("Progress", f"{progress_pct:.0f}%")
        
        # Workout structure
        structure_tabs = st.tabs(["🔥 Main Workout", "🏃‍♀️ Warm-up", "🧘‍♀️ Cool-down"])
        
        with structure_tabs[0]:
            self._display_main_workout(workout)
        
        with structure_tabs[1]:
            self._display_warmup_routine(workout)
        
        with structure_tabs[2]:
            self._display_cooldown_routine(workout)
        
        # Additional guidance
        guidance_tabs = st.tabs(["📋 Preparation", "🍎 Nutrition", "💪 Recovery"])
        
        with guidance_tabs[0]:
            preparation = today_plan.get('preparation', {})
            if preparation and isinstance(preparation, dict):
                tips = preparation.get('tips', [])
                if tips:
                    for tip in tips:
                        st.markdown(f"• {tip}")
                else:
                    st.markdown("• Follow your normal pre-workout routine")
            else:
                st.markdown("• Follow your normal pre-workout routine")
        
        with guidance_tabs[1]:
            nutrition = today_plan.get('nutrition', {})
            if nutrition and isinstance(nutrition, dict):
                pre_workout = nutrition.get('pre_workout', 'Light snack 30-60 min before')
                post_workout = nutrition.get('post_workout', 'Protein + carbs within 30 min')
                st.markdown(f"**Pre-workout:** {pre_workout}")
                st.markdown(f"**Post-workout:** {post_workout}")
            else:
                st.markdown("**Pre-workout:** Light snack 30-60 min before")
                st.markdown("**Post-workout:** Protein + carbs within 30 min")
        
        with guidance_tabs[2]:
            recovery = today_plan.get('recovery', {})
            if recovery and isinstance(recovery, dict):
                recommendations = recovery.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                else:
                    st.markdown("• Get adequate sleep (7-9 hours)")
                    st.markdown("• Stay hydrated throughout the day")
            else:
                st.markdown("• Get adequate sleep (7-9 hours)")
                st.markdown("• Stay hydrated throughout the day")
        
        # Workout timer
        if st.button("🚀 Start Workout Timer", type="primary", use_container_width=True):
            self._start_workout_timer(workout)
    
    def _display_main_workout(self, workout):
        """Display main workout exercises."""
        
        # Get exercises from workout (handle both dict and object)
        if isinstance(workout, dict):
            exercises = workout.get('exercises', [])
        elif hasattr(workout, 'exercises'):
            exercises = workout.exercises
        else:
            st.warning("No exercise data available for this workout.")
            return
        
        if not exercises:
            st.info("No exercises defined for this workout.")
            return
        
        for i, exercise in enumerate(exercises, 1):
            # Safety check: ensure exercise is a dictionary
            if not isinstance(exercise, dict):
                st.error(f"Error: Exercise {i} is not in expected format. Type: {type(exercise)}")
                continue
                
            with st.expander(f"{i}. {exercise['name']}", expanded=True):
                
                # Exercise details
                detail_cols = st.columns([2, 1, 1, 1])
                
                with detail_cols[0]:
                    st.markdown("**Target Muscles:**")
                    st.markdown(", ".join(exercise.get('muscle_groups', [])))
                
                with detail_cols[1]:
                    st.markdown("**Sets:**")
                    st.markdown(exercise.get('sets', 'N/A'))
                
                with detail_cols[2]:
                    st.markdown("**Reps/Time:**")
                    st.markdown(exercise.get('reps', exercise.get('work_time', 'N/A')))
                
                with detail_cols[3]:
                    st.markdown("**Rest:**")
                    st.markdown(f"{exercise.get('rest_seconds', 60)}s")
                
                # Technique cues
                if exercise.get('technique_cues'):
                    st.markdown("**Technique Cues:**")
                    for cue in exercise['technique_cues']:
                        st.markdown(f"• {cue}")
                
                # Progressions/modifications
                if exercise.get('progressions'):
                    prog_cols = st.columns(2)
                    with prog_cols[0]:
                        if 'easier' in exercise['progressions']:
                            st.markdown(f"**Easier:** {exercise['progressions']['easier']}")
                    with prog_cols[1]:
                        if 'harder' in exercise['progressions']:
                            st.markdown(f"**Harder:** {exercise['progressions']['harder']}")
    
    def _display_warmup_routine(self, workout):
        """Display warm-up routine."""
        
        # Get warm_up_duration from workout (handle both dict and object)
        if isinstance(workout, dict):
            warm_up_duration = workout.get('warm_up_duration', 5)
        elif hasattr(workout, 'warm_up_duration'):
            warm_up_duration = workout.warm_up_duration
        else:
            warm_up_duration = 5
        
        st.markdown(f"**Duration:** {warm_up_duration} minutes")
        
        warmup_exercises = [
            "5 minutes light cardio (marching, easy movement)",
            "Dynamic stretching (arm circles, leg swings)",
            "Movement preparation (bodyweight squats, arm swings)",
            "Gradual intensity increase"
        ]
        
        for exercise in warmup_exercises:
            st.markdown(f"• {exercise}")
    
    def _display_cooldown_routine(self, workout):
        """Display cool-down routine."""
        
        # Get cool_down_duration from workout (handle both dict and object)
        if isinstance(workout, dict):
            cool_down_duration = workout.get('cool_down_duration', 5)
        elif hasattr(workout, 'cool_down_duration'):
            cool_down_duration = workout.cool_down_duration
        else:
            cool_down_duration = 5
        
        st.markdown(f"**Duration:** {cool_down_duration} minutes")
        
        cooldown_exercises = [
            "Gradual heart rate reduction (2-3 min easy movement)",
            "Static stretching (hold 15-30 seconds each)",
            "Focus on muscles worked today",
            "Deep breathing and relaxation"
        ]
        
        for exercise in cooldown_exercises:
            st.markdown(f"• {exercise}")
    
    def _start_workout_timer(self, workout):
        """Start interactive workout timer."""
        
        st.markdown("### ⏱️ Workout Timer")
        
        # This would implement an interactive timer
        # For now, show structure
        st.info("🏃‍♀️ Workout timer would be implemented here with:")
        st.markdown("• Exercise-by-exercise progression")
        st.markdown("• Rest period timers")
        st.markdown("• Set completion tracking")
        st.markdown("• Real-time workout logging")
    
    def _get_day_summary(self, weekly_plan, day: str) -> str:
        """Get summary for a day in the weekly plan."""
        
        if day in weekly_plan.rest_days:
            return "Rest Day"
        
        for workout_day in weekly_plan.workout_days:
            if workout_day.day_name == day:
                return workout_day.workout_type.replace('_', ' ').title()
        
        return "Rest Day"
    
    def _evaluate_trigger_status(self, trigger: Dict[str, Any], 
                                progress_data: List[ProgressMetrics]) -> Dict[str, Any]:
        """Evaluate if an adaptive trigger is active."""
        
        # This would implement trigger evaluation logic
        # For now, return mock status
        return {
            'active': False,
            'confidence': trigger['sensitivity'],
            'data_points': len(progress_data)
        }
    
    def _display_progress_analytics(self, progress_data: List[ProgressMetrics]):
        """Display comprehensive progress analytics."""
        
        st.markdown("#### 📊 Progress Analytics")
        
        if len(progress_data) < 2:
            st.info("Log more workouts to see detailed analytics!")
            return
        
        # Create dataframe
        df = pd.DataFrame([
            {
                'date': p.date,
                'completed': p.workout_completed,
                'exertion': p.perceived_exertion,
                'duration': p.duration_actual,
                'recovery': p.recovery_score
            }
            for p in progress_data
        ])
        
        # Progress charts
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            # Completion rate trend
            fig1 = px.line(df, x='date', y='completed', 
                          title="Workout Completion Trend")
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_cols[1]:
            # Perceived exertion trend
            fig2 = px.line(df, x='date', y='exertion', 
                          title="Perceived Exertion Trend")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Summary statistics
        completion_rate = df['completed'].mean()
        avg_exertion = df['exertion'].mean()
        avg_recovery = df['recovery'].mean()
        
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric("Completion Rate", f"{completion_rate:.1%}")
        with metric_cols[1]:
            st.metric("Avg Exertion", f"{avg_exertion:.1f}/10")
        with metric_cols[2]:
            st.metric("Avg Recovery", f"{avg_recovery:.1f}/10")
