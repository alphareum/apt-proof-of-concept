"""
Enhanced Exercise Database with Comprehensive Exercise Library
Advanced exercise data with detailed progressions and variations

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from models import ExerciseCategory, FitnessLevel, GoalType

@dataclass
class EnhancedExercise:
    """Enhanced exercise with comprehensive details."""
    
    name: str
    category: ExerciseCategory
    muscle_groups: List[str]
    equipment_needed: List[str]
    difficulty: int  # 1-5 scale
    calories_per_minute: float
    met_value: float
    
    # Exercise execution
    instructions: List[str]
    technique_cues: List[str]
    breathing_pattern: str
    
    # Programming details
    target_sets: Optional[str] = None
    target_reps: Optional[str] = None
    target_duration: Optional[str] = None
    rest_time_seconds: int = 60
    
    # Safety and modifications
    contraindications: List[str] = None
    modifications: Dict[str, str] = None
    progressions: Dict[str, str] = None
    
    # Additional metadata
    tags: List[str] = None
    benefits: List[str] = None
    
    def __post_init__(self):
        if self.contraindications is None:
            self.contraindications = []
        if self.modifications is None:
            self.modifications = {}
        if self.progressions is None:
            self.progressions = {}
        if self.tags is None:
            self.tags = []
        if self.benefits is None:
            self.benefits = []

class ComprehensiveExerciseDatabase:
    """Comprehensive database of exercises with detailed information."""
    
    def __init__(self):
        self.exercises = self._initialize_exercise_database()
    
    def _initialize_exercise_database(self) -> Dict[str, EnhancedExercise]:
        """Initialize comprehensive exercise database."""
        
        exercises = {}
        
        # CARDIO EXERCISES
        exercises['running'] = EnhancedExercise(
            name="Running",
            category=ExerciseCategory.CARDIO,
            muscle_groups=['legs', 'core', 'cardiovascular'],
            equipment_needed=[],
            difficulty=2,
            calories_per_minute=10.0,
            met_value=8.0,
            instructions=[
                "Start with a 5-minute warm-up walk",
                "Maintain a steady pace you can sustain",
                "Land on midfoot, not heel strike",
                "Keep arms relaxed at 90-degree angle",
                "Breathe rhythmically - in through nose, out through mouth"
            ],
            technique_cues=[
                "Keep head up and eyes forward",
                "Slight forward lean from ankles",
                "Quick, light steps",
                "Pump arms naturally"
            ],
            breathing_pattern="Rhythmic - 2:2 or 3:3 pattern",
            contraindications=['acute knee injury', 'severe ankle sprain', 'active shin splints'],
            modifications={
                'easier': 'Walk-run intervals (1 min run, 2 min walk)',
                'harder': 'Hill running or interval sprints',
                'low_impact': 'Elliptical or stationary bike'
            },
            progressions={
                'beginner': 'Start with 15-20 minutes, increase by 10% weekly',
                'intermediate': 'Add tempo runs and intervals',
                'advanced': 'Include hill training and long runs'
            },
            tags=['outdoor', 'no_equipment', 'fat_burning'],
            benefits=['Cardiovascular health', 'Weight management', 'Mental health', 'Bone density']
        )
        
        exercises['jump_rope'] = EnhancedExercise(
            name="Jump Rope",
            category=ExerciseCategory.CARDIO,
            muscle_groups=['legs', 'shoulders', 'core', 'cardiovascular'],
            equipment_needed=['jump_rope'],
            difficulty=3,
            calories_per_minute=12.0,
            met_value=9.8,
            instructions=[
                "Hold handles with light grip",
                "Keep elbows close to body",
                "Jump on balls of feet",
                "Use wrists to turn rope, not arms",
                "Land softly with slight knee bend"
            ],
            technique_cues=[
                "Small, quick jumps",
                "Stay on balls of feet",
                "Keep rhythm consistent",
                "Relax shoulders"
            ],
            breathing_pattern="Natural rhythm, don't hold breath",
            target_duration="30 seconds to 5 minutes",
            contraindications=['ankle injuries', 'knee problems', 'low ceilings'],
            modifications={
                'easier': 'Imaginary rope or step-tap motion',
                'harder': 'Double unders or criss-cross',
                'low_impact': 'Marching in place with arm circles'
            },
            progressions={
                'beginner': '30 seconds on, 30 seconds rest',
                'intermediate': '2-3 minute rounds',
                'advanced': 'Complex footwork patterns'
            },
            tags=['portable', 'high_intensity', 'coordination'],
            benefits=['Coordination', 'Agility', 'Cardiovascular fitness', 'Calorie burn']
        )
        
        # STRENGTH EXERCISES
        exercises['push_ups'] = EnhancedExercise(
            name="Push-ups",
            category=ExerciseCategory.STRENGTH,
            muscle_groups=['chest', 'shoulders', 'triceps', 'core'],
            equipment_needed=[],
            difficulty=2,
            calories_per_minute=6.0,
            met_value=3.8,
            instructions=[
                "Start in plank position, hands slightly wider than shoulders",
                "Keep body in straight line from head to heels",
                "Lower chest to nearly touch ground",
                "Push back up to starting position",
                "Keep core tight throughout movement"
            ],
            technique_cues=[
                "Engage core to prevent sagging",
                "Keep elbows at 45-degree angle",
                "Control the descent",
                "Full range of motion"
            ],
            breathing_pattern="Inhale down, exhale up",
            target_sets="2-4",
            target_reps="8-20",
            rest_time_seconds=60,
            contraindications=['wrist injury', 'shoulder impingement', 'acute back pain'],
            modifications={
                'easier': 'Knee push-ups or incline push-ups',
                'harder': 'Decline push-ups or diamond push-ups',
                'wrist_friendly': 'Push-ups on fists or with handles'
            },
            progressions={
                'beginner': 'Wall push-ups → Incline → Knee → Full',
                'intermediate': 'Add pauses or single-arm variations',
                'advanced': 'Plyo push-ups or weighted push-ups'
            },
            tags=['no_equipment', 'upper_body', 'compound'],
            benefits=['Upper body strength', 'Core stability', 'Functional movement', 'Convenience']
        )
        
        exercises['squats'] = EnhancedExercise(
            name="Bodyweight Squats",
            category=ExerciseCategory.STRENGTH,
            muscle_groups=['quadriceps', 'glutes', 'hamstrings', 'core'],
            equipment_needed=[],
            difficulty=2,
            calories_per_minute=7.0,
            met_value=5.0,
            instructions=[
                "Stand with feet shoulder-width apart",
                "Keep chest up and core engaged",
                "Lower by pushing hips back and bending knees",
                "Go down until thighs are parallel to ground",
                "Drive through heels to return to standing"
            ],
            technique_cues=[
                "Keep knees aligned over toes",
                "Weight on heels and mid-foot",
                "Chest up, spine neutral",
                "Full depth with control"
            ],
            breathing_pattern="Inhale down, exhale up",
            target_sets="2-4",
            target_reps="10-25",
            rest_time_seconds=60,
            contraindications=['severe knee problems', 'hip impingement', 'acute back injury'],
            modifications={
                'easier': 'Chair-assisted squats or partial range',
                'harder': 'Jump squats or single-leg squats',
                'knee_friendly': 'Shallower range of motion'
            },
            progressions={
                'beginner': 'Chair squats → Box squats → Full squats',
                'intermediate': 'Add pauses or single-leg variations',
                'advanced': 'Pistol squats or weighted squats'
            },
            tags=['no_equipment', 'lower_body', 'functional'],
            benefits=['Lower body strength', 'Functional movement', 'Hip mobility', 'Bone health']
        )
        
        exercises['plank'] = EnhancedExercise(
            name="Plank",
            category=ExerciseCategory.STRENGTH,
            muscle_groups=['core', 'shoulders', 'back'],
            equipment_needed=[],
            difficulty=2,
            calories_per_minute=4.0,
            met_value=3.0,
            instructions=[
                "Start in push-up position on forearms",
                "Keep body in straight line from head to heels",
                "Engage core and glutes",
                "Hold position while breathing normally",
                "Avoid sagging hips or raising butt"
            ],
            technique_cues=[
                "Squeeze glutes and core",
                "Neutral spine alignment",
                "Breathe steadily",
                "Eyes looking down"
            ],
            breathing_pattern="Deep, steady breathing throughout hold",
            target_duration="30 seconds to 2 minutes",
            rest_time_seconds=30,
            contraindications=['lower back injury', 'wrist problems', 'shoulder injury'],
            modifications={
                'easier': 'Knee plank or incline plank',
                'harder': 'Single-arm plank or plank with leg lifts',
                'wrist_friendly': 'Forearm plank'
            },
            progressions={
                'beginner': '15-30 seconds',
                'intermediate': '1-2 minutes with variations',
                'advanced': 'Complex plank movements'
            },
            tags=['no_equipment', 'core', 'isometric'],
            benefits=['Core strength', 'Posture improvement', 'Stability', 'Injury prevention']
        )
        
        exercises['lunges'] = EnhancedExercise(
            name="Lunges",
            category=ExerciseCategory.STRENGTH,
            muscle_groups=['quadriceps', 'glutes', 'hamstrings', 'calves'],
            equipment_needed=[],
            difficulty=3,
            calories_per_minute=6.0,
            met_value=4.0,
            instructions=[
                "Stand with feet hip-width apart",
                "Step forward into lunge position",
                "Lower until both knees at 90 degrees",
                "Keep front knee over ankle",
                "Push off front foot to return to start"
            ],
            technique_cues=[
                "Keep torso upright",
                "Front knee tracks over toe",
                "Back knee points down",
                "Control the movement"
            ],
            breathing_pattern="Inhale down, exhale up",
            target_sets="2-3",
            target_reps="8-15 each leg",
            rest_time_seconds=60,
            contraindications=['knee injuries', 'balance issues', 'ankle problems'],
            modifications={
                'easier': 'Stationary lunges or assisted lunges',
                'harder': 'Walking lunges or jump lunges',
                'balance_help': 'Hold wall or chair for support'
            },
            progressions={
                'beginner': 'Stationary lunges with support',
                'intermediate': 'Walking lunges and side lunges',
                'advanced': 'Plyometric and weighted lunges'
            },
            tags=['no_equipment', 'unilateral', 'functional'],
            benefits=['Leg strength', 'Balance', 'Hip flexibility', 'Core stability']
        )
        
        # FLEXIBILITY EXERCISES
        exercises['yoga_flow'] = EnhancedExercise(
            name="Basic Yoga Flow",
            category=ExerciseCategory.FLEXIBILITY,
            muscle_groups=['full_body', 'flexibility'],
            equipment_needed=['yoga_mat'],
            difficulty=2,
            calories_per_minute=3.0,
            met_value=2.5,
            instructions=[
                "Start in mountain pose",
                "Flow through sun salutation sequence",
                "Hold each pose for 5-8 breaths",
                "Move slowly and mindfully",
                "End in child's pose or savasana"
            ],
            technique_cues=[
                "Focus on breath awareness",
                "Move within your range",
                "Keep movements fluid",
                "Listen to your body"
            ],
            breathing_pattern="Deep, rhythmic breathing throughout",
            target_duration="10-30 minutes",
            contraindications=['acute injuries', 'recent surgery'],
            modifications={
                'easier': 'Chair yoga or gentle stretches',
                'harder': 'Advanced poses and longer holds',
                'limited_mobility': 'Seated variations'
            },
            progressions={
                'beginner': 'Basic poses with shorter holds',
                'intermediate': 'Full sun salutations',
                'advanced': 'Complex sequences and arm balances'
            },
            tags=['flexibility', 'mindfulness', 'stress_relief'],
            benefits=['Flexibility', 'Stress reduction', 'Balance', 'Mind-body connection']
        )
        
        exercises['dynamic_stretching'] = EnhancedExercise(
            name="Dynamic Stretching Routine",
            category=ExerciseCategory.FLEXIBILITY,
            muscle_groups=['full_body', 'mobility'],
            equipment_needed=[],
            difficulty=1,
            calories_per_minute=3.5,
            met_value=3.0,
            instructions=[
                "Start with gentle arm circles",
                "Perform leg swings front to back and side to side",
                "Do walking knee hugs and butt kicks",
                "Include torso twists and hip circles",
                "Gradually increase range of motion"
            ],
            technique_cues=[
                "Controlled movements",
                "Gradually increase range",
                "Don't force the stretch",
                "Stay warm throughout"
            ],
            breathing_pattern="Natural, continuous breathing",
            target_duration="5-15 minutes",
            contraindications=['acute muscle strains', 'joint injuries'],
            modifications={
                'easier': 'Smaller range of motion',
                'harder': 'Faster tempo or larger movements',
                'standing_issues': 'Seated dynamic movements'
            },
            progressions={
                'beginner': 'Basic movements with small range',
                'intermediate': 'Full range dynamic movements',
                'advanced': 'Sport-specific dynamic prep'
            },
            tags=['warm_up', 'mobility', 'pre_workout'],
            benefits=['Injury prevention', 'Movement preparation', 'Circulation', 'Range of motion']
        )
        
        # FUNCTIONAL EXERCISES
        exercises['burpees'] = EnhancedExercise(
            name="Burpees",
            category=ExerciseCategory.FUNCTIONAL,
            muscle_groups=['full_body', 'cardiovascular'],
            equipment_needed=[],
            difficulty=4,
            calories_per_minute=12.0,
            met_value=10.0,
            instructions=[
                "Start standing, drop into squat position",
                "Place hands on ground, jump feet back to plank",
                "Perform push-up (optional)",
                "Jump feet back to squat position",
                "Jump up with arms overhead"
            ],
            technique_cues=[
                "Keep core engaged throughout",
                "Land softly on jumps",
                "Maintain good form over speed",
                "Breathe rhythmically"
            ],
            breathing_pattern="Exhale on exertion phases",
            target_sets="3-5",
            target_reps="5-15",
            rest_time_seconds=90,
            contraindications=['back injuries', 'wrist problems', 'knee issues'],
            modifications={
                'easier': 'Step back to plank instead of jumping',
                'harder': 'Add push-up or increase tempo',
                'low_impact': 'Step-back burpees without jump'
            },
            progressions={
                'beginner': 'Half burpees or step-back version',
                'intermediate': 'Full burpees with good form',
                'advanced': 'Burpee variations and complexes'
            },
            tags=['full_body', 'high_intensity', 'no_equipment'],
            benefits=['Total body conditioning', 'Cardiovascular fitness', 'Functional strength', 'Time efficiency']
        )
        
        exercises['mountain_climbers'] = EnhancedExercise(
            name="Mountain Climbers",
            category=ExerciseCategory.FUNCTIONAL,
            muscle_groups=['core', 'shoulders', 'legs', 'cardiovascular'],
            equipment_needed=[],
            difficulty=3,
            calories_per_minute=8.0,
            met_value=6.5,
            instructions=[
                "Start in plank position",
                "Alternate bringing knees toward chest",
                "Keep hips level and core tight",
                "Maintain plank position with upper body",
                "Move at a controlled pace"
            ],
            technique_cues=[
                "Keep hips down",
                "Land on balls of feet",
                "Stable upper body",
                "Controlled tempo"
            ],
            breathing_pattern="Quick, rhythmic breathing",
            target_duration="30-60 seconds",
            rest_time_seconds=30,
            contraindications=['wrist injuries', 'shoulder problems', 'back pain'],
            modifications={
                'easier': 'Slower pace or hands on elevation',
                'harder': 'Faster pace or cross-body movement',
                'wrist_friendly': 'Hands on handles or dumbbells'
            },
            progressions={
                'beginner': 'Slow, controlled movements',
                'intermediate': 'Standard pace and duration',
                'advanced': 'High intensity intervals'
            },
            tags=['core', 'cardio', 'no_equipment'],
            benefits=['Core strength', 'Cardiovascular fitness', 'Agility', 'Calorie burn']
        )
        
        return exercises
    
    def get_exercises_by_category(self, category: ExerciseCategory) -> Dict[str, EnhancedExercise]:
        """Get exercises filtered by category."""
        return {
            name: exercise for name, exercise in self.exercises.items()
            if exercise.category == category
        }
    
    def get_exercises_by_equipment(self, available_equipment: List[str]) -> Dict[str, EnhancedExercise]:
        """Get exercises that can be performed with available equipment."""
        return {
            name: exercise for name, exercise in self.exercises.items()
            if not exercise.equipment_needed or 
            all(eq in available_equipment for eq in exercise.equipment_needed)
        }
    
    def get_exercises_by_difficulty(self, max_difficulty: int) -> Dict[str, EnhancedExercise]:
        """Get exercises within difficulty range."""
        return {
            name: exercise for name, exercise in self.exercises.items()
            if exercise.difficulty <= max_difficulty
        }
    
    def get_exercises_by_goal(self, goal: GoalType) -> Dict[str, EnhancedExercise]:
        """Get exercises suitable for specific goal."""
        
        goal_exercise_mapping = {
            GoalType.WEIGHT_LOSS: ['cardio', 'functional'],
            GoalType.MUSCLE_GAIN: ['strength'],
            GoalType.ENDURANCE: ['cardio', 'functional'],
            GoalType.STRENGTH: ['strength'],
            GoalType.FLEXIBILITY: ['flexibility'],
            GoalType.GENERAL_FITNESS: ['cardio', 'strength', 'flexibility']
        }
        
        target_categories = goal_exercise_mapping.get(goal, ['cardio', 'strength', 'flexibility'])
        
        return {
            name: exercise for name, exercise in self.exercises.items()
            if exercise.category.value in target_categories
        }
    
    def search_exercises(self, query: str) -> Dict[str, EnhancedExercise]:
        """Search exercises by name, muscle groups, or tags."""
        query_lower = query.lower()
        
        results = {}
        for name, exercise in self.exercises.items():
            # Search in name
            if query_lower in exercise.name.lower():
                results[name] = exercise
                continue
            
            # Search in muscle groups
            if any(query_lower in muscle.lower() for muscle in exercise.muscle_groups):
                results[name] = exercise
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in exercise.tags):
                results[name] = exercise
                continue
        
        return results
    
    def get_exercise_by_name(self, name: str) -> Optional[EnhancedExercise]:
        """Get specific exercise by name."""
        return self.exercises.get(name.lower().replace(' ', '_'))
    
    def get_all_exercises(self) -> Dict[str, EnhancedExercise]:
        """Get all exercises in the database."""
        return self.exercises.copy()
    
    def get_exercise_recommendations(self, user_profile, max_exercises: int = 10) -> List[EnhancedExercise]:
        """Get personalized exercise recommendations."""
        
        # Filter by equipment availability
        available_exercises = self.get_exercises_by_equipment(
            user_profile.available_equipment if hasattr(user_profile, 'available_equipment') else []
        )
        
        # Filter by fitness level (difficulty)
        difficulty_map = {
            FitnessLevel.BEGINNER: 2,
            FitnessLevel.INTERMEDIATE: 3,
            FitnessLevel.ADVANCED: 5
        }
        max_difficulty = difficulty_map.get(user_profile.fitness_level, 2)
        
        suitable_exercises = {
            name: exercise for name, exercise in available_exercises.items()
            if exercise.difficulty <= max_difficulty
        }
        
        # Filter by primary goal
        goal_exercises = self.get_exercises_by_goal(user_profile.primary_goal)
        
        # Combine filters
        recommended = {
            name: exercise for name, exercise in suitable_exercises.items()
            if name in goal_exercises
        }
        
        # If not enough exercises, add from suitable exercises
        if len(recommended) < max_exercises:
            for name, exercise in suitable_exercises.items():
                if name not in recommended:
                    recommended[name] = exercise
                if len(recommended) >= max_exercises:
                    break
        
        return list(recommended.values())[:max_exercises]
