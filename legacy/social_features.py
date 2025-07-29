"""
Social and Community Features for AI Fitness Assistant
Community challenges, social sharing, and motivation system

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChallengeType(Enum):
    STEPS = "steps"
    WORKOUTS = "workouts"
    STREAK = "streak"
    DISTANCE = "distance"
    CALORIES = "calories"
    CUSTOM = "custom"

class ChallengeStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    UPCOMING = "upcoming"

@dataclass
class Achievement:
    """User achievement or badge."""
    id: str
    name: str
    description: str
    icon: str
    category: str
    earned_date: Optional[datetime] = None
    progress_current: int = 0
    progress_target: int = 1
    is_earned: bool = False

@dataclass
class Challenge:
    """Community or personal challenge."""
    id: str
    name: str
    description: str
    challenge_type: ChallengeType
    target_value: int
    current_value: int
    start_date: date
    end_date: date
    status: ChallengeStatus
    participants: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    category: str = "general"

@dataclass
class WorkoutShare:
    """Shared workout result."""
    id: str
    user_id: str
    workout_name: str
    duration_minutes: int
    calories_burned: int
    exercises_completed: List[str]
    achievements_earned: List[str]
    notes: str
    date_completed: datetime
    likes: int = 0
    comments: List[str] = field(default_factory=list)

@dataclass
class UserStats:
    """User fitness statistics."""
    total_workouts: int = 0
    total_minutes: int = 0
    total_calories: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    favorite_exercise: str = ""
    level: int = 1
    experience_points: int = 0

class SocialFitnessSystem:
    """Social and community features system."""
    
    def __init__(self):
        self.achievements_db = self._initialize_achievements()
        self.challenges_db = {}
        self.user_stats_db = {}
        self.workout_shares_db = {}
        self.leaderboards = {}
    
    def get_user_achievements(self, user_id: str) -> List[Achievement]:
        """Get all achievements for a user."""
        
        if user_id not in self.achievements_db:
            self.achievements_db[user_id] = self._initialize_user_achievements()
        
        return self.achievements_db[user_id]
    
    def check_and_award_achievements(self, user_id: str, workout_data: Dict[str, Any]) -> List[Achievement]:
        """Check and award achievements based on workout data."""
        
        user_achievements = self.get_user_achievements(user_id)
        user_stats = self.get_user_stats(user_id)
        newly_earned = []
        
        # Update user stats first
        self._update_user_stats(user_id, workout_data)
        updated_stats = self.get_user_stats(user_id)
        
        # Check each achievement
        for achievement in user_achievements:
            if not achievement.is_earned:
                if self._check_achievement_criteria(achievement, updated_stats, workout_data):
                    achievement.is_earned = True
                    achievement.earned_date = datetime.now()
                    achievement.progress_current = achievement.progress_target
                    newly_earned.append(achievement)
        
        return newly_earned
    
    def create_personal_challenge(self, user_id: str, challenge_data: Dict[str, Any]) -> Challenge:
        """Create a personal challenge."""
        
        challenge = Challenge(
            id=str(uuid.uuid4()),
            name=challenge_data['name'],
            description=challenge_data['description'],
            challenge_type=ChallengeType(challenge_data['type']),
            target_value=challenge_data['target'],
            current_value=0,
            start_date=challenge_data['start_date'],
            end_date=challenge_data['end_date'],
            status=ChallengeStatus.ACTIVE,
            participants=[user_id],
            category=challenge_data.get('category', 'personal')
        )
        
        self.challenges_db[challenge.id] = challenge
        return challenge
    
    def get_active_challenges(self, user_id: str) -> List[Challenge]:
        """Get all active challenges for a user."""
        
        active_challenges = []
        current_date = date.today()
        
        for challenge in self.challenges_db.values():
            if (user_id in challenge.participants and 
                challenge.status == ChallengeStatus.ACTIVE and
                challenge.start_date <= current_date <= challenge.end_date):
                active_challenges.append(challenge)
        
        return active_challenges
    
    def update_challenge_progress(self, user_id: str, challenge_type: ChallengeType, value: int):
        """Update progress for challenges of a specific type."""
        
        active_challenges = self.get_active_challenges(user_id)
        
        for challenge in active_challenges:
            if challenge.challenge_type == challenge_type:
                challenge.current_value += value
                
                # Check if challenge is completed
                if challenge.current_value >= challenge.target_value:
                    challenge.status = ChallengeStatus.COMPLETED
                    self._award_challenge_rewards(user_id, challenge)
    
    def share_workout(self, user_id: str, workout_data: Dict[str, Any]) -> WorkoutShare:
        """Share a workout with the community."""
        
        workout_share = WorkoutShare(
            id=str(uuid.uuid4()),
            user_id=user_id,
            workout_name=workout_data['name'],
            duration_minutes=workout_data['duration'],
            calories_burned=workout_data.get('calories', 0),
            exercises_completed=workout_data.get('exercises', []),
            achievements_earned=workout_data.get('achievements', []),
            notes=workout_data.get('notes', ''),
            date_completed=datetime.now()
        )
        
        self.workout_shares_db[workout_share.id] = workout_share
        return workout_share
    
    def get_community_feed(self, limit: int = 20) -> List[WorkoutShare]:
        """Get recent community workout shares."""
        
        # Sort by date and return most recent
        all_shares = list(self.workout_shares_db.values())
        all_shares.sort(key=lambda x: x.date_completed, reverse=True)
        
        return all_shares[:limit]
    
    def get_leaderboard(self, category: str = "total_workouts", period: str = "all_time") -> List[Dict[str, Any]]:
        """Get leaderboard for specific category and time period."""
        
        leaderboard_data = []
        
        for user_id, stats in self.user_stats_db.items():
            if category == "total_workouts":
                score = stats.total_workouts
            elif category == "total_minutes":
                score = stats.total_minutes
            elif category == "total_calories":
                score = stats.total_calories
            elif category == "current_streak":
                score = stats.current_streak
            elif category == "level":
                score = stats.level
            else:
                score = 0
            
            leaderboard_data.append({
                'user_id': user_id,
                'score': score,
                'level': stats.level,
                'total_workouts': stats.total_workouts
            })
        
        # Sort by score (descending)
        leaderboard_data.sort(key=lambda x: x['score'], reverse=True)
        
        return leaderboard_data[:10]  # Top 10
    
    def get_suggested_challenges(self, user_id: str) -> List[Dict[str, Any]]:
        """Get suggested challenges based on user stats."""
        
        user_stats = self.get_user_stats(user_id)
        suggestions = []
        
        # Workout frequency challenges
        if user_stats.total_workouts < 10:
            suggestions.append({
                'name': "Workout Warrior",
                'description': "Complete 10 workouts this month",
                'type': ChallengeType.WORKOUTS,
                'target': 10,
                'category': "consistency"
            })
        
        # Streak challenges
        if user_stats.longest_streak < 7:
            suggestions.append({
                'name': "7-Day Streak Master",
                'description': "Workout for 7 consecutive days",
                'type': ChallengeType.STREAK,
                'target': 7,
                'category': "consistency"
            })
        
        # Time-based challenges
        if user_stats.total_minutes < 300:
            suggestions.append({
                'name': "5-Hour Finisher",
                'description': "Complete 5 hours of exercise this month",
                'type': ChallengeType.WORKOUTS,
                'target': 300,
                'category': "endurance"
            })
        
        # Advanced challenges for experienced users
        if user_stats.total_workouts > 50:
            suggestions.append({
                'name': "Century Club",
                'description': "Complete 100 total workouts",
                'type': ChallengeType.WORKOUTS,
                'target': 100,
                'category': "milestone"
            })
        
        return suggestions
    
    def get_motivation_message(self, user_id: str) -> str:
        """Get personalized motivation message."""
        
        user_stats = self.get_user_stats(user_id)
        messages = []
        
        # Streak-based messages
        if user_stats.current_streak > 0:
            if user_stats.current_streak >= 7:
                messages.append(f"🔥 Amazing! You're on a {user_stats.current_streak}-day streak!")
            elif user_stats.current_streak >= 3:
                messages.append(f"💪 Great momentum with your {user_stats.current_streak}-day streak!")
            else:
                messages.append(f"🚀 Keep going! {user_stats.current_streak} days strong!")
        
        # Milestone messages
        if user_stats.total_workouts > 0:
            if user_stats.total_workouts % 10 == 0:
                messages.append(f"🎉 Milestone achieved: {user_stats.total_workouts} workouts completed!")
            elif user_stats.total_workouts == 1:
                messages.append("🌟 Welcome to your fitness journey! First workout done!")
        
        # Level-based messages
        if user_stats.level > 1:
            messages.append(f"⭐ You're Level {user_stats.level}! Keep climbing!")
        
        # Fallback motivational messages
        if not messages:
            fallback_messages = [
                "💫 Every workout counts toward your goals!",
                "🎯 You're stronger than your excuses!",
                "🌟 Progress, not perfection!",
                "💪 Your future self will thank you!",
                "🔥 Make today count!"
            ]
            import random
            messages.append(random.choice(fallback_messages))
        
        return messages[0] if messages else "Keep pushing forward! 💪"
    
    def get_user_stats(self, user_id: str) -> UserStats:
        """Get user statistics."""
        
        if user_id not in self.user_stats_db:
            self.user_stats_db[user_id] = UserStats()
        
        return self.user_stats_db[user_id]
    
    def _update_user_stats(self, user_id: str, workout_data: Dict[str, Any]):
        """Update user statistics after a workout."""
        
        stats = self.get_user_stats(user_id)
        
        # Update basic stats
        stats.total_workouts += 1
        stats.total_minutes += workout_data.get('duration', 0)
        stats.total_calories += workout_data.get('calories', 0)
        
        # Update streak
        last_workout_date = workout_data.get('date', datetime.now().date())
        today = date.today()
        
        if last_workout_date == today:
            stats.current_streak += 1
        elif last_workout_date == today - timedelta(days=1):
            stats.current_streak += 1
        else:
            stats.current_streak = 1
        
        # Update longest streak
        if stats.current_streak > stats.longest_streak:
            stats.longest_streak = stats.current_streak
        
        # Update level and XP
        xp_gained = self._calculate_xp_gain(workout_data)
        stats.experience_points += xp_gained
        stats.level = self._calculate_level(stats.experience_points)
        
        # Update favorite exercise
        exercises = workout_data.get('exercises', [])
        if exercises:
            # Simple logic - could be more sophisticated
            stats.favorite_exercise = exercises[0]
    
    def _calculate_xp_gain(self, workout_data: Dict[str, Any]) -> int:
        """Calculate experience points gained from a workout."""
        
        base_xp = 10
        duration_bonus = workout_data.get('duration', 0) // 10 * 5  # 5 XP per 10 minutes
        completion_bonus = 20 if workout_data.get('completed', True) else 0
        
        return base_xp + duration_bonus + completion_bonus
    
    def _calculate_level(self, total_xp: int) -> int:
        """Calculate user level based on total XP."""
        
        # Simple level calculation: 100 XP per level
        return max(1, total_xp // 100 + 1)
    
    def _check_achievement_criteria(self, achievement: Achievement, stats: UserStats, workout_data: Dict[str, Any]) -> bool:
        """Check if achievement criteria are met."""
        
        criteria_map = {
            "first_workout": stats.total_workouts >= 1,
            "workout_warrior": stats.total_workouts >= 10,
            "fitness_enthusiast": stats.total_workouts >= 50,
            "gym_legend": stats.total_workouts >= 100,
            "streak_starter": stats.current_streak >= 3,
            "week_warrior": stats.current_streak >= 7,
            "consistency_king": stats.longest_streak >= 30,
            "time_crusher": stats.total_minutes >= 300,
            "calorie_burner": stats.total_calories >= 1000,
            "level_up": stats.level >= achievement.progress_target,
            "early_bird": workout_data.get('time_of_day') == 'morning',
            "night_owl": workout_data.get('time_of_day') == 'evening'
        }
        
        return criteria_map.get(achievement.id, False)
    
    def _award_challenge_rewards(self, user_id: str, challenge: Challenge):
        """Award rewards for completing a challenge."""
        
        stats = self.get_user_stats(user_id)
        
        # Award XP based on challenge difficulty
        xp_rewards = {
            ChallengeType.WORKOUTS: 50,
            ChallengeType.STREAK: 75,
            ChallengeType.CALORIES: 40,
            ChallengeType.STEPS: 30,
            ChallengeType.DISTANCE: 60,
            ChallengeType.CUSTOM: 100
        }
        
        xp_gain = xp_rewards.get(challenge.challenge_type, 50)
        stats.experience_points += xp_gain
        stats.level = self._calculate_level(stats.experience_points)
    
    def _initialize_achievements(self) -> Dict[str, List[Achievement]]:
        """Initialize default achievements."""
        
        default_achievements = [
            Achievement(
                id="first_workout",
                name="First Steps",
                description="Complete your first workout",
                icon="🌟",
                category="milestone",
                progress_target=1
            ),
            Achievement(
                id="workout_warrior",
                name="Workout Warrior",
                description="Complete 10 workouts",
                icon="⚔️",
                category="consistency",
                progress_target=10
            ),
            Achievement(
                id="fitness_enthusiast",
                name="Fitness Enthusiast",
                description="Complete 50 workouts",
                icon="🏆",
                category="milestone",
                progress_target=50
            ),
            Achievement(
                id="gym_legend",
                name="Gym Legend",
                description="Complete 100 workouts",
                icon="👑",
                category="milestone",
                progress_target=100
            ),
            Achievement(
                id="streak_starter",
                name="Streak Starter",
                description="Maintain a 3-day workout streak",
                icon="🔥",
                category="consistency",
                progress_target=3
            ),
            Achievement(
                id="week_warrior",
                name="Week Warrior",
                description="Maintain a 7-day workout streak",
                icon="💪",
                category="consistency",
                progress_target=7
            ),
            Achievement(
                id="consistency_king",
                name="Consistency King/Queen",
                description="Maintain a 30-day workout streak",
                icon="👸",
                category="consistency",
                progress_target=30
            ),
            Achievement(
                id="time_crusher",
                name="Time Crusher",
                description="Complete 5 hours of exercise",
                icon="⏱️",
                category="endurance",
                progress_target=300
            ),
            Achievement(
                id="calorie_burner",
                name="Calorie Burner",
                description="Burn 1000 calories through exercise",
                icon="🔥",
                category="performance",
                progress_target=1000
            ),
            Achievement(
                id="early_bird",
                name="Early Bird",
                description="Complete a morning workout",
                icon="🌅",
                category="lifestyle",
                progress_target=1
            ),
            Achievement(
                id="night_owl",
                name="Night Owl",
                description="Complete an evening workout",
                icon="🌙",
                category="lifestyle",
                progress_target=1
            )
        ]
        
        return {user_id: default_achievements.copy() for user_id in []}
    
    def _initialize_user_achievements(self) -> List[Achievement]:
        """Initialize achievements for a new user."""
        
        return [
            Achievement(
                id="first_workout",
                name="First Steps",
                description="Complete your first workout",
                icon="🌟",
                category="milestone",
                progress_target=1
            ),
            Achievement(
                id="workout_warrior",
                name="Workout Warrior",
                description="Complete 10 workouts",
                icon="⚔️",
                category="consistency",
                progress_target=10
            ),
            Achievement(
                id="fitness_enthusiast",
                name="Fitness Enthusiast",
                description="Complete 50 workouts",
                icon="🏆",
                category="milestone",
                progress_target=50
            ),
            Achievement(
                id="gym_legend",
                name="Gym Legend",
                description="Complete 100 workouts",
                icon="👑",
                category="milestone",
                progress_target=100
            ),
            Achievement(
                id="streak_starter",
                name="Streak Starter",
                description="Maintain a 3-day workout streak",
                icon="🔥",
                category="consistency",
                progress_target=3
            ),
            Achievement(
                id="week_warrior",
                name="Week Warrior",
                description="Maintain a 7-day workout streak",
                icon="💪",
                category="consistency",
                progress_target=7
            ),
            Achievement(
                id="consistency_king",
                name="Consistency King/Queen",
                description="Maintain a 30-day workout streak",
                icon="👸",
                category="consistency",
                progress_target=30
            ),
            Achievement(
                id="time_crusher",
                name="Time Crusher",
                description="Complete 5 hours of exercise",
                icon="⏱️",
                category="endurance",
                progress_target=300
            ),
            Achievement(
                id="calorie_burner",
                name="Calorie Burner",
                description="Burn 1000 calories through exercise",
                icon="🔥",
                category="performance",
                progress_target=1000
            ),
            Achievement(
                id="early_bird",
                name="Early Bird",
                description="Complete a morning workout",
                icon="🌅",
                category="lifestyle",
                progress_target=1
            ),
            Achievement(
                id="night_owl",
                name="Night Owl",
                description="Complete an evening workout",
                icon="🌙",
                category="lifestyle",
                progress_target=1
            )
        ]

    def get_weekly_community_challenges(self) -> List[Dict[str, Any]]:
        """Get weekly community challenges."""
        
        current_date = date.today()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_end = week_start + timedelta(days=6)
        
        community_challenges = [
            {
                'name': "Community Steps Challenge",
                'description': "Collective goal: 1 million steps this week!",
                'type': ChallengeType.STEPS,
                'target': 1000000,
                'current': 456789,  # Would be calculated from all users
                'participants': 127,
                'reward': "Community achievement badge",
                'end_date': week_end
            },
            {
                'name': "Workout Wednesday",
                'description': "Everyone workout on Wednesday!",
                'type': ChallengeType.WORKOUTS,
                'target': 100,  # 100 people working out
                'current': 67,
                'participants': 89,
                'reward': "Special workout routine unlock",
                'end_date': week_end
            },
            {
                'name': "Weekend Warrior",
                'description': "Complete a workout this weekend",
                'type': ChallengeType.WORKOUTS,
                'target': 1,
                'current': 0,
                'participants': 45,
                'reward': "Weekend Warrior badge",
                'end_date': week_end
            }
        ]
        
        return community_challenges
