"""
Enhanced UI Components for Comprehensive Fitness Features
Nutrition, social features, progress analytics, and recipe management

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import our new systems
try:
    from nutrition_planner import NutritionPlanner, MacroTargets, NutritionPlan
    from social_features import SocialFitnessSystem, Challenge, Achievement
    from progress_analytics import ProgressAnalytics, BodyMeasurement, WorkoutLog, PerformanceMetric
    from recipe_manager import RecipeManager, Recipe
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    st.warning("⚠️ Enhanced features not fully available. Some functionality may be limited.")

from models import UserProfile

class EnhancedFitnessUI:
    """Enhanced UI components for comprehensive fitness features."""
    
    def __init__(self):
        if ENHANCED_FEATURES_AVAILABLE:
            self.nutrition_planner = NutritionPlanner()
            self.social_system = SocialFitnessSystem()
            self.progress_analytics = ProgressAnalytics()
            self.recipe_manager = RecipeManager()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for enhanced features."""
        
        if 'nutrition_plan' not in st.session_state:
            st.session_state.nutrition_plan = None
        
        if 'user_achievements' not in st.session_state:
            st.session_state.user_achievements = []
        
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = []
        
        if 'favorite_recipes' not in st.session_state:
            st.session_state.favorite_recipes = []
    
    def render_nutrition_tab(self, user_profile: UserProfile):
        """Render comprehensive nutrition planning tab."""
        
        if not ENHANCED_FEATURES_AVAILABLE:
            st.error("Enhanced nutrition features not available. Please install required dependencies.")
            return
        
        st.markdown("## 🍎 Nutrition & Meal Planning")
        st.markdown("Personalized nutrition planning with macro tracking and meal prep guidance.")
        
        # Nutrition tabs
        nutrition_tabs = st.tabs([
            "🎯 Daily Nutrition",
            "📅 Meal Planning", 
            "🍳 Recipes",
            "🛒 Meal Prep",
            "📊 Nutrition Analytics"
        ])
        
        with nutrition_tabs[0]:
            self._render_daily_nutrition(user_profile)
        
        with nutrition_tabs[1]:
            self._render_meal_planning(user_profile)
        
        with nutrition_tabs[2]:
            self._render_recipe_browser(user_profile)
        
        with nutrition_tabs[3]:
            self._render_meal_prep_planner(user_profile)
        
        with nutrition_tabs[4]:
            self._render_nutrition_analytics(user_profile)
    
    def render_social_tab(self, user_profile: UserProfile):
        """Render social and community features tab."""
        
        if not ENHANCED_FEATURES_AVAILABLE:
            st.error("Social features not available. Please install required dependencies.")
            return
        
        st.markdown("## 👥 Community & Social")
        st.markdown("Connect with the fitness community, track achievements, and join challenges.")
        
        # Social tabs
        social_tabs = st.tabs([
            "🏆 Achievements",
            "🎯 Challenges",
            "📈 Leaderboards",
            "📱 Share Workout",
            "🌟 Motivation Hub"
        ])
        
        with social_tabs[0]:
            self._render_achievements_system(user_profile)
        
        with social_tabs[1]:
            self._render_challenges_system(user_profile)
        
        with social_tabs[2]:
            self._render_leaderboards()
        
        with social_tabs[3]:
            self._render_workout_sharing(user_profile)
        
        with social_tabs[4]:
            self._render_motivation_hub(user_profile)
    
    def render_analytics_tab(self, user_profile: UserProfile):
        """Render advanced progress analytics tab."""
        
        if not ENHANCED_FEATURES_AVAILABLE:
            st.error("Analytics features not available. Please install required dependencies.")
            return
        
        st.markdown("## 📊 Advanced Analytics")
        st.markdown("Comprehensive progress tracking with detailed insights and predictions.")
        
        # Analytics tabs
        analytics_tabs = st.tabs([
            "📈 Progress Overview",
            "📏 Body Measurements",
            "💪 Performance Tracking",
            "📸 Progress Photos",
            "🔮 Predictions"
        ])
        
        with analytics_tabs[0]:
            self._render_progress_overview(user_profile)
        
        with analytics_tabs[1]:
            self._render_body_measurements(user_profile)
        
        with analytics_tabs[2]:
            self._render_performance_tracking(user_profile)
        
        with analytics_tabs[3]:
            self._render_progress_photos(user_profile)
        
        with analytics_tabs[4]:
            self._render_goal_predictions(user_profile)
    
    def _render_daily_nutrition(self, user_profile: UserProfile):
        """Render daily nutrition planning interface."""
        
        st.markdown("### 🎯 Daily Macro Targets")
        
        # Calculate macro targets
        macro_targets = self.nutrition_planner.calculate_macro_targets(user_profile)
        
        # Display macro targets
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Calories", f"{macro_targets.calories}")
        
        with col2:
            st.metric("Protein", f"{macro_targets.protein_g}g")
        
        with col3:
            st.metric("Carbs", f"{macro_targets.carbs_g}g")
        
        with col4:
            st.metric("Fat", f"{macro_targets.fat_g}g")
        
        # Additional targets
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Fiber", f"{macro_targets.fiber_g}g")
        with col6:
            st.metric("Water", f"{macro_targets.water_ml/1000:.1f}L")
        
        # Generate today's meal plan
        if st.button("🍽️ Generate Today's Meal Plan", type="primary"):
            with st.spinner("Creating your personalized meal plan..."):
                nutrition_plan = self.nutrition_planner.generate_meal_plan(
                    user_profile, date.today(), macro_targets
                )
                st.session_state.nutrition_plan = nutrition_plan
        
        # Display meal plan if available
        if st.session_state.nutrition_plan:
            self._display_daily_meal_plan(st.session_state.nutrition_plan)
    
    def _render_meal_planning(self, user_profile: UserProfile):
        """Render meal planning interface."""
        
        st.markdown("### 📅 Weekly Meal Planning")
        
        # Planning options
        col1, col2 = st.columns(2)
        
        with col1:
            planning_duration = st.selectbox(
                "Planning Duration",
                [7, 14, 21, 30],
                index=0,
                format_func=lambda x: f"{x} days"
            )
        
        with col2:
            include_meal_prep = st.checkbox("Include Meal Prep Instructions", value=True)
        
        if st.button("📋 Create Weekly Plan", type="primary"):
            with st.spinner("Creating your weekly meal plan..."):
                macro_targets = self.nutrition_planner.calculate_macro_targets(user_profile)
                weekly_plan = self.recipe_manager.create_weekly_meal_plan(user_profile, macro_targets)
                
                st.success("✅ Weekly meal plan created!")
                
                # Display weekly plan
                self._display_weekly_meal_plan(weekly_plan)
    
    def _render_recipe_browser(self, user_profile: UserProfile):
        """Render recipe browsing and search interface."""
        
        st.markdown("### 🍳 Recipe Browser")
        
        # Search filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_filter = st.selectbox(
                "Category",
                ["All", "breakfast", "lunch", "dinner", "snacks"],
                index=0
            )
        
        with col2:
            difficulty_filter = st.selectbox(
                "Difficulty",
                ["All", "easy", "medium", "hard"],
                index=0
            )
        
        with col3:
            max_time = st.slider(
                "Max Cooking Time (min)",
                5, 120, 60
            )
        
        # Dietary tags
        dietary_tags = st.multiselect(
            "Dietary Preferences",
            ["high_protein", "low_calorie", "vegetarian", "vegan", "gluten_free", "healthy", "quick"]
        )
        
        # Search recipes
        search_criteria = {}
        if category_filter != "All":
            search_criteria['category'] = category_filter
        if difficulty_filter != "All":
            search_criteria['difficulty'] = difficulty_filter
        if dietary_tags:
            search_criteria['tags'] = dietary_tags
        
        search_criteria['max_prep_time'] = max_time
        
        recipes = self.recipe_manager.get_recipes_by_criteria(**search_criteria)
        
        # Display recipes
        if recipes:
            st.markdown(f"Found {len(recipes)} recipes:")
            
            for recipe in recipes[:6]:  # Show first 6 recipes
                with st.expander(f"🍽️ {recipe.name} - {recipe.difficulty.title()} ({recipe.prep_time_minutes + recipe.cook_time_minutes} min)"):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {recipe.description}")
                        st.markdown(f"**Cuisine:** {recipe.cuisine_type.title()}")
                        st.markdown(f"**Servings:** {recipe.servings}")
                        
                        if recipe.tags:
                            st.markdown(f"**Tags:** {', '.join(recipe.tags)}")
                    
                    with col2:
                        # Nutritional info
                        st.markdown("**Nutrition (per serving):**")
                        st.metric("Calories", f"{recipe.nutritional_info.get('calories', 0):.0f}")
                        st.metric("Protein", f"{recipe.nutritional_info.get('protein', 0):.1f}g")
                    
                    # Instructions
                    st.markdown("**Instructions:**")
                    for i, instruction in enumerate(recipe.instructions, 1):
                        st.markdown(f"{i}. {instruction}")
                    
                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(f"⭐ Add to Favorites", key=f"fav_{recipe.id}"):
                            if recipe.id not in st.session_state.favorite_recipes:
                                st.session_state.favorite_recipes.append(recipe.id)
                                st.success("Added to favorites!")
                    
                    with btn_col2:
                        if st.button(f"📋 Add to Meal Plan", key=f"plan_{recipe.id}"):
                            st.info("Recipe added to meal plan!")
        else:
            st.info("No recipes found with current filters. Try adjusting your search criteria.")
    
    def _render_meal_prep_planner(self, user_profile: UserProfile):
        """Render meal prep planning interface."""
        
        st.markdown("### 🛒 Meal Prep Planner")
        
        # Meal prep options
        col1, col2 = st.columns(2)
        
        with col1:
            prep_days = st.selectbox(
                "Prep Duration",
                [3, 5, 7],
                index=1,
                format_func=lambda x: f"{x} days"
            )
        
        with col2:
            meal_types = st.multiselect(
                "Meal Types to Prep",
                ["breakfast", "lunch", "dinner", "snacks"],
                default=["lunch", "dinner"]
            )
        
        if st.button("🥘 Generate Meal Prep Plan", type="primary"):
            with st.spinner("Creating your meal prep plan..."):
                # Get suitable recipes for meal prep
                recipes = []
                goal_recipes = self.recipe_manager.get_recipes_for_goals(user_profile)
                
                for meal_type in meal_types:
                    if meal_type in goal_recipes:
                        recipes.extend(goal_recipes[meal_type][:2])  # 2 recipes per meal type
                
                if recipes:
                    meal_prep_plan = self.recipe_manager.create_meal_prep_plan(recipes, prep_days)
                    
                    st.success("✅ Meal prep plan created!")
                    
                    # Display prep plan
                    st.markdown("#### 📋 Prep Schedule")
                    for task in meal_prep_plan.prep_schedule:
                        st.markdown(f"**{task['task']}** ({task['duration_minutes']} min): {task['description']}")
                    
                    # Shopping list
                    st.markdown("#### 🛒 Shopping List")
                    for item, amount in meal_prep_plan.shopping_list.items():
                        st.markdown(f"• {item}: {amount}")
                    
                    # Storage instructions
                    st.markdown("#### 📦 Storage Instructions")
                    for instruction in meal_prep_plan.storage_instructions:
                        st.markdown(instruction)
                else:
                    st.warning("No suitable recipes found for meal prep. Try different meal types.")
    
    def _render_nutrition_analytics(self, user_profile: UserProfile):
        """Render nutrition analytics and tracking."""
        
        st.markdown("### 📊 Nutrition Analytics")
        
        # Mock nutrition tracking data
        st.info("📝 This would show your nutrition tracking analytics including:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Daily Tracking:**
            - Calorie intake vs targets
            - Macro distribution trends
            - Hydration tracking
            - Meal timing patterns
            """)
        
        with col2:
            st.markdown("""
            **Weekly Analysis:**
            - Average macro achievement
            - Meal prep compliance
            - Nutrition goal progress
            - Recipe variety scoring
            """)
        
        # Sample chart
        if st.checkbox("Show Sample Analytics Chart"):
            # Create sample data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            sample_data = pd.DataFrame({
                'Date': dates,
                'Calories': [1800 + i*5 + (i%7)*100 for i in range(30)],
                'Target': [2000] * 30,
                'Protein': [120 + i*2 + (i%5)*20 for i in range(30)]
            })
            
            fig = px.line(sample_data, x='Date', y=['Calories', 'Target'], 
                         title='Daily Calorie Intake vs Target')
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_achievements_system(self, user_profile: UserProfile):
        """Render achievements and badges system."""
        
        st.markdown("### 🏆 Your Achievements")
        
        # Mock user achievements
        user_id = getattr(user_profile, 'user_id', 'user_123')
        achievements = self.social_system.get_user_achievements(user_id)
        
        # Display achievement categories
        achievement_categories = {}
        for achievement in achievements:
            if achievement.category not in achievement_categories:
                achievement_categories[achievement.category] = []
            achievement_categories[achievement.category].append(achievement)
        
        for category, category_achievements in achievement_categories.items():
            st.markdown(f"#### {category.title()} Achievements")
            
            cols = st.columns(min(3, len(category_achievements)))
            
            for i, achievement in enumerate(category_achievements[:3]):
                with cols[i % 3]:
                    if achievement.is_earned:
                        st.success(f"{achievement.icon} **{achievement.name}**")
                        st.caption(f"✅ Earned {achievement.earned_date.strftime('%B %d, %Y') if achievement.earned_date else 'Recently'}")
                    else:
                        st.info(f"🔒 **{achievement.name}**")
                        st.caption(f"Progress: {achievement.progress_current}/{achievement.progress_target}")
                        
                        # Progress bar
                        progress = achievement.progress_current / achievement.progress_target
                        st.progress(min(progress, 1.0))
                    
                    st.caption(achievement.description)
        
        # Achievement stats
        earned_count = sum(1 for a in achievements if a.is_earned)
        total_count = len(achievements)
        
        st.markdown(f"### 📊 Achievement Progress: {earned_count}/{total_count}")
        st.progress(earned_count / total_count)
    
    def _render_challenges_system(self, user_profile: UserProfile):
        """Render challenges and goals system."""
        
        st.markdown("### 🎯 Challenges")
        
        # Active challenges
        user_id = getattr(user_profile, 'user_id', 'user_123')
        active_challenges = self.social_system.get_active_challenges(user_id)
        
        if active_challenges:
            st.markdown("#### Your Active Challenges")
            
            for challenge in active_challenges:
                with st.expander(f"🎯 {challenge.name}", expanded=True):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Progress", f"{challenge.current_value}/{challenge.target_value}")
                    
                    with col2:
                        progress_pct = (challenge.current_value / challenge.target_value) * 100
                        st.metric("Completion", f"{progress_pct:.1f}%")
                    
                    with col3:
                        days_left = (challenge.end_date - date.today()).days
                        st.metric("Days Left", days_left)
                    
                    st.markdown(f"**Description:** {challenge.description}")
                    
                    # Progress bar
                    progress = min(challenge.current_value / challenge.target_value, 1.0)
                    st.progress(progress)
        
        # Suggested challenges
        st.markdown("#### Suggested Challenges")
        suggested = self.social_system.get_suggested_challenges(user_id)
        
        for i, suggestion in enumerate(suggested[:3]):
            with st.container():
                st.markdown(f"**{suggestion['name']}**")
                st.caption(suggestion['description'])
                
                if st.button(f"Accept Challenge", key=f"accept_{i}"):
                    # Create challenge
                    challenge_data = {
                        'name': suggestion['name'],
                        'description': suggestion['description'],
                        'type': suggestion['type'].value,
                        'target': suggestion['target'],
                        'start_date': date.today(),
                        'end_date': date.today() + timedelta(days=30),
                        'category': suggestion['category']
                    }
                    
                    new_challenge = self.social_system.create_personal_challenge(user_id, challenge_data)
                    st.success(f"✅ Challenge '{new_challenge.name}' accepted!")
        
        # Community challenges
        st.markdown("#### Community Challenges")
        community_challenges = self.social_system.get_weekly_community_challenges()
        
        for challenge in community_challenges:
            with st.container():
                st.markdown(f"**{challenge['name']}**")
                st.caption(challenge['description'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Progress", f"{challenge['current']:,}/{challenge['target']:,}")
                
                with col2:
                    st.metric("Participants", challenge['participants'])
                
                with col3:
                    days_left = (challenge['end_date'] - date.today()).days
                    st.metric("Days Left", days_left)
                
                # Progress bar
                progress = challenge['current'] / challenge['target']
                st.progress(min(progress, 1.0))
                
                st.caption(f"🎁 Reward: {challenge['reward']}")
    
    def _render_leaderboards(self):
        """Render community leaderboards."""
        
        st.markdown("### 📈 Community Leaderboards")
        
        # Leaderboard categories
        leaderboard_type = st.selectbox(
            "Leaderboard Category",
            ["total_workouts", "total_minutes", "current_streak", "total_calories"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Get leaderboard data
        leaderboard = self.social_system.get_leaderboard(leaderboard_type)
        
        if leaderboard:
            st.markdown(f"#### Top 10 - {leaderboard_type.replace('_', ' ').title()}")
            
            for i, user_data in enumerate(leaderboard, 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                    
                    with col1:
                        if i == 1:
                            st.markdown("🥇")
                        elif i == 2:
                            st.markdown("🥈")
                        elif i == 3:
                            st.markdown("🥉")
                        else:
                            st.markdown(f"#{i}")
                    
                    with col2:
                        # In a real app, this would show actual usernames
                        st.markdown(f"**User {user_data['user_id'][-3:]}**")
                    
                    with col3:
                        st.metric("Score", user_data['score'])
                    
                    with col4:
                        st.metric("Level", user_data['level'])
        else:
            st.info("No leaderboard data available yet. Start working out to join the rankings!")
    
    def _render_workout_sharing(self, user_profile: UserProfile):
        """Render workout sharing interface."""
        
        st.markdown("### 📱 Share Your Workout")
        
        # Workout sharing form
        with st.form("share_workout"):
            workout_name = st.text_input("Workout Name", "Today's Training Session")
            
            col1, col2 = st.columns(2)
            
            with col1:
                duration = st.number_input("Duration (minutes)", min_value=1, value=45, key="workout_duration")
                calories = st.number_input("Calories Burned", min_value=0, value=300, key="workout_calories")
            
            with col2:
                exercises = st.text_area(
                    "Exercises Completed",
                    "Squats, Push-ups, Planks",
                    help="List the exercises you completed"
                ).split(',')
            
            notes = st.text_area("Notes", "Great workout today! Felt strong and energized.")
            
            if st.form_submit_button("🚀 Share Workout", type="primary"):
                workout_data = {
                    'name': workout_name,
                    'duration': duration,
                    'calories': calories,
                    'exercises': [ex.strip() for ex in exercises],
                    'notes': notes
                }
                
                user_id = getattr(user_profile, 'user_id', 'user_123')
                workout_share = self.social_system.share_workout(user_id, workout_data)
                
                st.success("✅ Workout shared with the community!")
                
                # Update achievements
                newly_earned = self.social_system.check_and_award_achievements(user_id, workout_data)
                
                if newly_earned:
                    st.balloons()
                    st.success(f"🏆 New achievements earned: {', '.join(a.name for a in newly_earned)}")
        
        # Community feed
        st.markdown("### 🌟 Community Feed")
        
        community_feed = self.social_system.get_community_feed(limit=5)
        
        if community_feed:
            for share in community_feed:
                with st.container():
                    st.markdown(f"**User {share.user_id[-3:]}** completed {share.workout_name}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.caption(f"⏱️ {share.duration_minutes} minutes")
                    
                    with col2:
                        st.caption(f"🔥 {share.calories_burned} calories")
                    
                    with col3:
                        st.caption(f"📅 {share.date_completed.strftime('%B %d')}")
                    
                    if share.notes:
                        st.caption(f"💭 {share.notes}")
                    
                    # Engagement actions
                    like_col, comment_col = st.columns(2)
                    with like_col:
                        st.button(f"👍 {share.likes}", key=f"like_{share.id}")
                    with comment_col:
                        st.button("💬 Comment", key=f"comment_{share.id}")
        else:
            st.info("No recent community activity. Be the first to share a workout!")
    
    def _render_motivation_hub(self, user_profile: UserProfile):
        """Render motivation and inspiration hub."""
        
        st.markdown("### 🌟 Motivation Hub")
        
        # Personal motivation message
        user_id = getattr(user_profile, 'user_id', 'user_123')
        motivation_message = self.social_system.get_motivation_message(user_id)
        
        st.success(f"💪 {motivation_message}")
        
        # User stats
        user_stats = self.social_system.get_user_stats(user_id)
        
        st.markdown("#### Your Fitness Journey")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workouts", user_stats.total_workouts)
        
        with col2:
            st.metric("Current Streak", f"{user_stats.current_streak} days")
        
        with col3:
            st.metric("Level", user_stats.level)
        
        with col4:
            st.metric("Total Minutes", user_stats.total_minutes)
        
        # Progress visualization
        if user_stats.total_workouts > 0:
            # Experience bar
            st.markdown("#### Experience Progress")
            xp_for_current_level = (user_stats.level - 1) * 100
            xp_for_next_level = user_stats.level * 100
            current_level_progress = user_stats.experience_points - xp_for_current_level
            level_progress = current_level_progress / 100
            
            st.progress(min(level_progress, 1.0))
            st.caption(f"XP: {user_stats.experience_points} | Next level: {100 - current_level_progress} XP needed")
        
        # Inspirational quotes
        st.markdown("#### Daily Inspiration")
        
        quotes = [
            "The only bad workout is the one that didn't happen.",
            "Your body can stand almost anything. It's your mind you have to convince.",
            "Fitness is not about being better than someone else. It's about being better than you used to be.",
            "The groundwork for all happiness is good health.",
            "Take care of your body. It's the only place you have to live."
        ]
        
        import random
        daily_quote = random.choice(quotes)
        st.info(f"💭 {daily_quote}")
        
        # Weekly challenge suggestion
        st.markdown("#### This Week's Focus")
        
        weekly_challenges = [
            "🎯 Try one new exercise this week",
            "🏃‍♀️ Add 5 minutes to your cardio sessions",
            "💪 Focus on proper form over heavy weights",
            "🧘‍♀️ Include stretching in every workout",
            "📱 Track your workouts consistently"
        ]
        
        weekly_focus = random.choice(weekly_challenges)
        st.markdown(weekly_focus)
    
    def _display_daily_meal_plan(self, nutrition_plan: NutritionPlan):
        """Display daily meal plan."""
        
        st.markdown("#### 🍽️ Today's Meal Plan")
        
        for meal in nutrition_plan.meals:
            with st.expander(f"{meal.meal_type.title()} - {meal.name}", expanded=False):
                
                # Meal overview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Prep Time:** {meal.preparation_time_minutes} minutes")
                    
                    # Calculate meal macros
                    meal_macros = self.nutrition_planner.calculate_meal_macros(meal)
                    st.markdown(f"**Calories:** {meal_macros['calories']:.0f}")
                    st.markdown(f"**Protein:** {meal_macros['protein']:.1f}g")
                
                with col2:
                    st.markdown("**Ingredients:**")
                    for food_item, amount in meal.food_items:
                        st.markdown(f"• {food_item.name}: {amount}g")
                
                # Instructions
                if meal.instructions:
                    st.markdown("**Instructions:**")
                    for i, instruction in enumerate(meal.instructions, 1):
                        st.markdown(f"{i}. {instruction}")
        
        # Hydration reminders
        if nutrition_plan.hydration_reminders:
            st.markdown("#### 💧 Hydration Schedule")
            for reminder in nutrition_plan.hydration_reminders:
                st.markdown(f"• {reminder}")
        
        # Supplements
        if nutrition_plan.supplements:
            st.markdown("#### 💊 Supplement Recommendations")
            for supplement in nutrition_plan.supplements:
                st.markdown(f"• {supplement}")
    
    def _display_weekly_meal_plan(self, weekly_plan: Dict[str, Any]):
        """Display weekly meal plan."""
        
        st.markdown("#### 📅 Weekly Meal Plan")
        
        for day, meals in weekly_plan['days'].items():
            with st.expander(f"{day}", expanded=False):
                
                for meal_type, recipe in meals.items():
                    if recipe:
                        st.markdown(f"**{meal_type.title()}:** {recipe.name}")
                        st.caption(f"Prep: {recipe.prep_time_minutes}min | Cook: {recipe.cook_time_minutes}min")
                    else:
                        st.markdown(f"**{meal_type.title()}:** No recipe assigned")
        
        # Shopping list
        if weekly_plan.get('shopping_list'):
            st.markdown("#### 🛒 Weekly Shopping List")
            for item, amount in list(weekly_plan['shopping_list'].items())[:10]:  # Show first 10 items
                st.markdown(f"• {item}: {amount}")
    
    def _render_progress_overview(self, user_profile: UserProfile):
        """Render comprehensive progress overview."""
        
        st.markdown("### 📈 Progress Overview")
        
        # Generate progress report
        progress_report = self.progress_analytics.generate_progress_report(user_profile, 30)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        performance = progress_report.get('performance', {})
        consistency = progress_report.get('consistency', {})
        
        with col1:
            st.metric("Total Workouts", performance.get('total_workouts', 0))
        
        with col2:
            st.metric("Consistency Rate", f"{consistency.get('consistency_rate', 0):.1f}%")
        
        with col3:
            st.metric("Current Streak", f"{consistency.get('current_streak', 0)} days")
        
        with col4:
            st.metric("Avg Duration", f"{consistency.get('avg_duration', 0):.0f} min")
        
        # Achievements and recommendations
        achievements = progress_report.get('achievements', [])
        if achievements:
            st.markdown("#### 🏆 Recent Achievements")
            for achievement in achievements:
                st.success(achievement)
        
        recommendations = progress_report.get('recommendations', [])
        if recommendations:
            st.markdown("#### 💡 Recommendations")
            for recommendation in recommendations:
                st.info(recommendation)
    
    def _render_body_measurements(self, user_profile: UserProfile):
        """Render body measurements tracking."""
        
        st.markdown("### 📏 Body Measurements")
        
        # Input new measurement
        with st.expander("➕ Log New Measurement", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=user_profile.weight, step=0.1, key="body_measurements_weight")
                body_fat = st.number_input("Body Fat %", min_value=3.0, max_value=50.0, step=0.1, value=None, key="body_measurements_body_fat")
                waist = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, step=0.5, value=None, key="body_measurements_waist")
            
            with col2:
                muscle_mass = st.number_input("Muscle Mass (kg)", min_value=10.0, max_value=100.0, step=0.1, value=None, key="body_measurements_muscle_mass")
                chest = st.number_input("Chest (cm)", min_value=70.0, max_value=200.0, step=0.5, value=None, key="body_measurements_chest")
                bicep = st.number_input("Bicep (cm)", min_value=20.0, max_value=60.0, step=0.5, value=None, key="body_measurements_bicep")
            
            notes = st.text_area("Notes", "")
            
            if st.button("📊 Log Measurement"):
                from progress_analytics import BodyMeasurement
                
                measurement = BodyMeasurement(
                    date=date.today(),
                    weight=weight if weight != user_profile.weight else None,
                    body_fat_percentage=body_fat,
                    muscle_mass_kg=muscle_mass,
                    waist_cm=waist,
                    chest_cm=chest,
                    bicep_cm=bicep,
                    notes=notes
                )
                
                self.progress_analytics.log_body_measurement(measurement)
                st.success("✅ Measurement logged successfully!")
        
        # Display trends
        weight_trend = self.progress_analytics.get_weight_trend(30)
        
        if weight_trend['trend'] != 'insufficient_data':
            st.markdown("#### Weight Trend (30 days)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Trend", weight_trend['trend'].title())
            
            with col2:
                st.metric("Total Change", f"{weight_trend['change']:+.1f} kg")
            
            with col3:
                st.metric("Rate per Week", f"{weight_trend['rate_per_week']:+.2f} kg/week")
        else:
            st.info("📝 Log more measurements to see weight trends and analytics.")
    
    def _render_performance_tracking(self, user_profile: UserProfile):
        """Render performance metrics tracking."""
        
        st.markdown("### 💪 Performance Tracking")
        
        # Add new performance metric
        with st.expander("➕ Log Performance Metric", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                exercise_name = st.selectbox(
                    "Exercise",
                    ["Bench Press", "Squat", "Deadlift", "Overhead Press", "Pull-ups", "Other"]
                )
                
                if exercise_name == "Other":
                    exercise_name = st.text_input("Exercise Name")
                
                metric_type = st.selectbox("Metric Type", ["weight", "reps", "time", "distance"], key="performance_metric_type")
            
            with col2:
                value = st.number_input("Value", min_value=0.0, step=0.5, key="performance_value")
                unit = st.selectbox("Unit", ["kg", "lbs", "reps", "seconds", "minutes", "meters", "km"], key="performance_unit")
            
            notes = st.text_input("Notes", "")
            
            if st.button("📈 Log Performance"):
                from progress_analytics import PerformanceMetric
                
                metric = PerformanceMetric(
                    date=date.today(),
                    exercise_name=exercise_name,
                    metric_type=metric_type,
                    value=value,
                    unit=unit,
                    notes=notes
                )
                
                self.progress_analytics.log_performance_metric(metric)
                st.success("✅ Performance metric logged!")
        
        # Show strength progress
        st.markdown("#### Strength Progress")
        
        key_exercises = ["Bench Press", "Squat", "Deadlift"]
        
        for exercise in key_exercises:
            progress = self.progress_analytics.get_strength_progress(exercise)
            
            if progress['progress'] != 'insufficient_data':
                with st.container():
                    st.markdown(f"**{exercise}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Progress", progress['progress'].title())
                    
                    with col2:
                        if progress['best_performance']:
                            best = progress['best_performance']
                            st.metric("Best", f"{best['value']} {best['unit']}")
                    
                    with col3:
                        if progress['recent_performance']:
                            recent = progress['recent_performance']
                            st.metric("Recent", f"{recent['value']} {recent['unit']}")
            else:
                st.info(f"📝 Log {exercise} performances to see progress.")
    
    def _render_progress_photos(self, user_profile: UserProfile):
        """Render progress photos tracking."""
        
        st.markdown("### 📸 Progress Photos")
        
        # Upload new photo
        st.markdown("#### Upload New Progress Photo")
        
        uploaded_file = st.file_uploader(
            "Choose a progress photo",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a progress photo to track visual changes over time"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                body_part = st.selectbox(
                    "Body Part/Angle",
                    ["Front", "Side", "Back", "Arms", "Legs", "Core", "Full Body"]
                )
                
                notes = st.text_area("Photo Notes", "")
            
            with col2:
                st.image(uploaded_file, caption="Progress Photo Preview", width=200)
            
            if st.button("📷 Save Progress Photo"):
                # In a real app, you would save the photo to storage
                from progress_analytics import ProgressPhoto
                
                photo = ProgressPhoto(
                    date=date.today(),
                    photo_path=f"progress_photos/{uploaded_file.name}",
                    body_part=body_part.lower(),
                    notes=notes
                )
                
                self.progress_analytics.add_progress_photo(photo)
                st.success("✅ Progress photo saved!")
        
        # Display recent photos
        if self.progress_analytics.progress_photos:
            st.markdown("#### Recent Progress Photos")
            
            recent_photos = self.progress_analytics.progress_photos[-6:]  # Last 6 photos
            
            cols = st.columns(min(3, len(recent_photos)))
            
            for i, photo in enumerate(recent_photos):
                with cols[i % 3]:
                    st.markdown(f"**{photo.body_part.title()}**")
                    st.caption(f"{photo.date}")
                    if photo.notes:
                        st.caption(photo.notes)
                    # st.image would display the actual photo in a real app
                    st.info("📷 Photo placeholder")
        else:
            st.info("📷 No progress photos yet. Upload your first photo to start tracking visual progress!")
    
    def _render_goal_predictions(self, user_profile: UserProfile):
        """Render goal predictions and timeline estimates."""
        
        st.markdown("### 🔮 Goal Predictions")
        
        # Goal prediction form
        st.markdown("#### Predict Goal Timeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goal_type = st.selectbox("Goal Type", ["weight"], key="goals_goal_type")  # More types can be added
            
        with col2:
            if goal_type == "weight":
                target_weight = st.number_input(
                    "Target Weight (kg)",
                    min_value=40.0,
                    max_value=200.0,
                    value=user_profile.weight - 5,
                    step=0.5,
                    key="goals_target_weight"
                )
        
        if st.button("🔮 Predict Timeline"):
            if goal_type == "weight":
                prediction = self.progress_analytics.predict_goal_timeline(
                    user_profile, "weight", target_weight
                )
                
                if prediction['prediction'] == 'success':
                    st.success(f"🎯 Estimated timeline: {prediction['estimated_weeks']} weeks")
                    st.info(f"📅 Target date: {prediction['estimated_date']}")
                    st.caption(f"Based on current rate: {prediction['current_rate']:+.2f} kg/week")
                    
                    # Confidence indicator
                    confidence = prediction.get('confidence', 'medium')
                    if confidence == 'high':
                        st.success("🎯 High confidence prediction")
                    elif confidence == 'medium':
                        st.warning("⚠️ Medium confidence - continue tracking for better accuracy")
                    else:
                        st.error("❗ Low confidence - need more data points")
                
                elif prediction['prediction'] == 'insufficient_data':
                    st.warning("📊 Need more measurement data for accurate predictions")
                
                elif prediction['prediction'] == 'goal_passed':
                    st.info(prediction['message'])
                
                else:
                    st.error(prediction.get('message', 'Prediction not available'))
        
        # Goal insights
        st.markdown("#### Goal Insights")
        
        if user_profile.primary_goal:
            goal_insights = {
                'weight_loss': {
                    'tips': [
                        "🥗 Maintain a consistent caloric deficit",
                        "💧 Stay hydrated - drink 2-3L water daily",
                        "🏃‍♀️ Combine cardio with strength training",
                        "😴 Get 7-9 hours of quality sleep"
                    ]
                },
                'muscle_gain': {
                    'tips': [
                        "🥩 Eat 2-2.2g protein per kg body weight",
                        "🏋️‍♀️ Progressive overload in strength training",
                        "😴 Recovery is crucial - don't skip rest days",
                        "📈 Track your lifts to ensure progression"
                    ]
                },
                'endurance': {
                    'tips': [
                        "🏃‍♀️ Gradually increase training volume",
                        "❤️ Monitor heart rate zones",
                        "⏰ Include both steady-state and interval training",
                        "🧘‍♀️ Don't neglect recovery and flexibility"
                    ]
                }
            }
            
            goal_key = user_profile.primary_goal.value
            if goal_key in goal_insights:
                st.markdown(f"**Tips for {goal_key.replace('_', ' ').title()}:**")
                for tip in goal_insights[goal_key]['tips']:
                    st.markdown(f"• {tip}")
        
        # Progress milestones
        st.markdown("#### Milestone Suggestions")
        
        milestones = [
            "🎯 Complete 10 workouts",
            "💪 Increase strength by 10% in key exercises",
            "⏰ Maintain 7-day workout streak",
            "📏 Achieve 1kg of progress toward goal weight",
            "🏆 Complete first fitness challenge"
        ]
        
        for milestone in milestones:
            st.markdown(f"• {milestone}")
