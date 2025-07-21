"""
Nutrition Planning System for AI Fitness Assistant
Comprehensive meal planning and nutrition tracking

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import logging
from dataclasses import dataclass, field
import json
from models import UserProfile, GoalType, ActivityLevel

logger = logging.getLogger(__name__)

@dataclass
class MacroTargets:
    """Daily macronutrient targets."""
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    fiber_g: int
    water_ml: int

@dataclass
class FoodItem:
    """Individual food item with nutritional information."""
    name: str
    calories_per_100g: int
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    fiber_per_100g: float
    category: str  # protein, carbs, vegetables, fats, etc.
    serving_size_g: int = 100

@dataclass
class Meal:
    """Meal with multiple food items."""
    name: str
    meal_type: str  # breakfast, lunch, dinner, snack
    food_items: List[Tuple[FoodItem, int]]  # (food_item, amount_in_grams)
    preparation_time_minutes: int = 15
    instructions: List[str] = field(default_factory=list)

@dataclass
class NutritionPlan:
    """Complete nutrition plan."""
    date: date
    macro_targets: MacroTargets
    meals: List[Meal]
    hydration_reminders: List[str]
    supplements: List[str]
    notes: List[str]

class NutritionPlanner:
    """Advanced nutrition planning system."""
    
    def __init__(self):
        self.food_database = self._initialize_food_database()
        self.meal_templates = self._initialize_meal_templates()
    
    def calculate_macro_targets(self, user_profile: UserProfile, 
                              activity_multiplier: float = 1.0) -> MacroTargets:
        """Calculate daily macro targets based on user profile."""
        
        # Calculate BMR using Mifflin-St Jeor Equation
        if user_profile.gender.value == "male":
            bmr = 10 * user_profile.weight + 6.25 * user_profile.height - 5 * user_profile.age + 5
        else:
            bmr = 10 * user_profile.weight + 6.25 * user_profile.height - 5 * user_profile.age - 161
        
        # Activity level multipliers
        activity_multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTREMELY_ACTIVE: 1.9
        }
        
        # Calculate TDEE
        tdee = bmr * activity_multipliers[user_profile.activity_level] * activity_multiplier
        
        # Adjust calories based on goal
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            calories = int(tdee * 0.8)  # 20% deficit
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            calories = int(tdee * 1.15)  # 15% surplus
        else:
            calories = int(tdee)  # Maintenance
        
        # Calculate macros based on goal
        if user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            protein_g = int(user_profile.weight * 2.2)  # 2.2g per kg
            fat_g = int(calories * 0.25 / 9)  # 25% of calories
            carbs_g = int((calories - (protein_g * 4) - (fat_g * 9)) / 4)
        elif user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            protein_g = int(user_profile.weight * 2.0)  # 2.0g per kg for muscle retention
            fat_g = int(calories * 0.25 / 9)  # 25% of calories
            carbs_g = int((calories - (protein_g * 4) - (fat_g * 9)) / 4)
        else:
            protein_g = int(user_profile.weight * 1.6)  # 1.6g per kg
            fat_g = int(calories * 0.30 / 9)  # 30% of calories
            carbs_g = int((calories - (protein_g * 4) - (fat_g * 9)) / 4)
        
        fiber_g = max(25, int(calories / 1000 * 14))  # 14g per 1000 calories
        water_ml = int(user_profile.weight * 35)  # 35ml per kg body weight
        
        return MacroTargets(
            calories=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            fiber_g=fiber_g,
            water_ml=water_ml
        )
    
    def generate_meal_plan(self, user_profile: UserProfile, 
                          target_date: date,
                          macro_targets: MacroTargets) -> NutritionPlan:
        """Generate a complete meal plan for the day."""
        
        # Distribute calories across meals
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.30,
            'snack': 0.10
        }
        
        meals = []
        for meal_type, portion in meal_distribution.items():
            target_calories = int(macro_targets.calories * portion)
            target_protein = int(macro_targets.protein_g * portion)
            
            meal = self._generate_meal(
                meal_type, target_calories, target_protein, user_profile
            )
            meals.append(meal)
        
        # Generate hydration reminders
        hydration_reminders = self._generate_hydration_reminders(macro_targets.water_ml)
        
        # Suggest supplements based on goals
        supplements = self._suggest_supplements(user_profile)
        
        # Add nutrition notes
        notes = self._generate_nutrition_notes(user_profile, macro_targets)
        
        return NutritionPlan(
            date=target_date,
            macro_targets=macro_targets,
            meals=meals,
            hydration_reminders=hydration_reminders,
            supplements=supplements,
            notes=notes
        )
    
    def _generate_meal(self, meal_type: str, target_calories: int, 
                      target_protein: int, user_profile: UserProfile) -> Meal:
        """Generate a specific meal."""
        
        # Get appropriate meal templates
        templates = self.meal_templates.get(meal_type, [])
        
        # Select template based on user preferences and restrictions
        selected_template = self._select_meal_template(
            templates, user_profile, target_calories
        )
        
        # Adjust portions to meet targets
        adjusted_meal = self._adjust_meal_portions(
            selected_template, target_calories, target_protein
        )
        
        return adjusted_meal
    
    def _select_meal_template(self, templates: List[Meal], 
                            user_profile: UserProfile, 
                            target_calories: int) -> Meal:
        """Select appropriate meal template."""
        
        # Filter based on dietary restrictions (could be extended)
        suitable_templates = templates
        
        # Select template closest to target calories
        if suitable_templates:
            return min(suitable_templates, 
                      key=lambda x: abs(self._calculate_meal_calories(x) - target_calories))
        
        # Fallback to default meal
        return self._create_default_meal(target_calories)
    
    def _adjust_meal_portions(self, meal: Meal, target_calories: int, 
                            target_protein: int) -> Meal:
        """Adjust meal portions to meet targets."""
        
        current_calories = self._calculate_meal_calories(meal)
        scale_factor = target_calories / max(current_calories, 1)
        
        # Scale all portions
        adjusted_food_items = []
        for food_item, amount in meal.food_items:
            new_amount = int(amount * scale_factor)
            adjusted_food_items.append((food_item, new_amount))
        
        return Meal(
            name=meal.name,
            meal_type=meal.meal_type,
            food_items=adjusted_food_items,
            preparation_time_minutes=meal.preparation_time_minutes,
            instructions=meal.instructions
        )
    
    def _calculate_meal_calories(self, meal: Meal) -> int:
        """Calculate total calories in a meal."""
        total_calories = 0
        for food_item, amount in meal.food_items:
            calories = (food_item.calories_per_100g * amount) / 100
            total_calories += calories
        return int(total_calories)
    
    def _generate_hydration_reminders(self, target_water_ml: int) -> List[str]:
        """Generate hydration reminders throughout the day."""
        
        reminders = [
            f"🌅 Morning: Start with 500ml water upon waking",
            f"🥤 Pre-workout: 250ml water 30 mins before exercise",
            f"💧 During workout: 150-200ml every 15-20 mins",
            f"🔄 Post-workout: 150% of fluid lost during exercise",
            f"🍽️ With meals: 250ml with each meal",
            f"🌙 Evening: Final 250ml before bed (2 hours prior)"
        ]
        
        daily_target = f"🎯 Daily target: {target_water_ml}ml ({target_water_ml/1000:.1f} liters)"
        reminders.insert(0, daily_target)
        
        return reminders
    
    def _suggest_supplements(self, user_profile: UserProfile) -> List[str]:
        """Suggest supplements based on user profile and goals."""
        
        supplements = []
        
        # Basic recommendations
        supplements.append("🌟 Multivitamin: Daily comprehensive nutrition support")
        supplements.append("🐟 Omega-3: 1-2g daily for inflammation and heart health")
        
        # Goal-specific supplements
        if user_profile.primary_goal in [GoalType.MUSCLE_GAIN, GoalType.STRENGTH]:
            supplements.extend([
                "💪 Whey Protein: 25-30g post-workout",
                "⚡ Creatine: 3-5g daily for strength and power",
                "🔋 Vitamin D3: 2000-4000 IU daily for bone health"
            ])
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            supplements.extend([
                "🍃 Green Tea Extract: Natural metabolism support",
                "🔥 L-Carnitine: Fat metabolism support",
                "🌿 Fiber Supplement: If not meeting daily fiber goals"
            ])
        
        if user_profile.activity_level in [ActivityLevel.VERY_ACTIVE, ActivityLevel.EXTREMELY_ACTIVE]:
            supplements.extend([
                "⚡ Electrolytes: During/after intense workouts",
                "🧠 Magnesium: 200-400mg for recovery and sleep"
            ])
        
        return supplements
    
    def _generate_nutrition_notes(self, user_profile: UserProfile, 
                                macro_targets: MacroTargets) -> List[str]:
        """Generate helpful nutrition notes."""
        
        notes = []
        
        # Goal-specific notes
        if user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            notes.extend([
                "🎯 Focus on eating protein every 3-4 hours",
                "⏰ Have protein + carbs within 30 mins post-workout",
                "💤 Include casein protein before bed for overnight recovery"
            ])
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            notes.extend([
                "🍽️ Use smaller plates to control portion sizes",
                "🥗 Fill half your plate with vegetables",
                "💧 Drink water before meals to increase satiety"
            ])
        
        # General health notes
        notes.extend([
            f"📊 Daily calorie target: {macro_targets.calories} calories",
            f"🥩 Protein target: {macro_targets.protein_g}g ({macro_targets.protein_g/user_profile.weight:.1f}g/kg)",
            "🌈 Aim for 5-7 servings of fruits and vegetables daily",
            "⏰ Try to eat meals at consistent times each day"
        ])
        
        return notes
    
    def _initialize_food_database(self) -> Dict[str, List[FoodItem]]:
        """Initialize comprehensive food database."""
        
        foods = {
            'proteins': [
                FoodItem("Chicken Breast", 165, 31, 0, 3.6, 0, "lean_protein"),
                FoodItem("Salmon", 208, 20, 0, 13, 0, "fatty_protein"),
                FoodItem("Eggs", 155, 13, 1.1, 11, 0, "complete_protein"),
                FoodItem("Greek Yogurt", 59, 10, 3.6, 0.4, 0, "dairy_protein"),
                FoodItem("Lean Beef", 250, 26, 0, 15, 0, "red_meat"),
                FoodItem("Tofu", 76, 8, 1.9, 4.8, 0.3, "plant_protein"),
                FoodItem("Protein Powder", 400, 80, 5, 5, 2, "supplement")
            ],
            'carbohydrates': [
                FoodItem("Brown Rice", 111, 2.6, 23, 0.9, 1.8, "whole_grain"),
                FoodItem("Oatmeal", 68, 2.4, 12, 1.4, 1.7, "whole_grain"),
                FoodItem("Sweet Potato", 86, 1.6, 20, 0.1, 3, "starchy_vegetable"),
                FoodItem("Quinoa", 120, 4.4, 22, 1.9, 2.8, "pseudo_grain"),
                FoodItem("Banana", 89, 1.1, 23, 0.3, 2.6, "fruit"),
                FoodItem("Whole Wheat Bread", 247, 13, 41, 4.2, 7, "bread")
            ],
            'vegetables': [
                FoodItem("Broccoli", 34, 2.8, 7, 0.4, 2.6, "cruciferous"),
                FoodItem("Spinach", 23, 2.9, 3.6, 0.4, 2.2, "leafy_green"),
                FoodItem("Bell Peppers", 31, 1, 7, 0.3, 2.5, "colorful_vegetable"),
                FoodItem("Carrots", 41, 0.9, 10, 0.2, 2.8, "root_vegetable"),
                FoodItem("Asparagus", 20, 2.2, 3.9, 0.1, 2.1, "green_vegetable")
            ],
            'fats': [
                FoodItem("Avocado", 160, 2, 9, 15, 7, "healthy_fat"),
                FoodItem("Almonds", 579, 21, 22, 50, 12, "nuts"),
                FoodItem("Olive Oil", 884, 0, 0, 100, 0, "cooking_oil"),
                FoodItem("Peanut Butter", 588, 25, 20, 50, 6, "nut_butter"),
                FoodItem("Chia Seeds", 486, 17, 42, 31, 34, "seeds")
            ]
        }
        
        return foods
    
    def _initialize_meal_templates(self) -> Dict[str, List[Meal]]:
        """Initialize meal templates."""
        
        # Get foods from database
        proteins = self.food_database.get('proteins', [])
        carbs = self.food_database.get('carbohydrates', [])
        vegetables = self.food_database.get('vegetables', [])
        fats = self.food_database.get('fats', [])
        
        templates = {
            'breakfast': [
                Meal(
                    name="Protein Oatmeal Bowl",
                    meal_type="breakfast",
                    food_items=[
                        (next(f for f in carbs if f.name == "Oatmeal"), 50),
                        (next(f for f in proteins if f.name == "Protein Powder"), 30),
                        (next(f for f in carbs if f.name == "Banana"), 120),
                        (next(f for f in fats if f.name == "Almonds"), 15)
                    ],
                    preparation_time_minutes=10,
                    instructions=[
                        "Cook oatmeal with water or milk",
                        "Mix in protein powder",
                        "Top with sliced banana and almonds"
                    ]
                )
            ],
            'lunch': [
                Meal(
                    name="Chicken Rice Bowl",
                    meal_type="lunch",
                    food_items=[
                        (next(f for f in proteins if f.name == "Chicken Breast"), 150),
                        (next(f for f in carbs if f.name == "Brown Rice"), 100),
                        (next(f for f in vegetables if f.name == "Broccoli"), 100),
                        (next(f for f in fats if f.name == "Avocado"), 50)
                    ],
                    preparation_time_minutes=25,
                    instructions=[
                        "Grill seasoned chicken breast",
                        "Cook brown rice",
                        "Steam broccoli",
                        "Assemble bowl with sliced avocado"
                    ]
                )
            ],
            'dinner': [
                Meal(
                    name="Salmon Sweet Potato",
                    meal_type="dinner",
                    food_items=[
                        (next(f for f in proteins if f.name == "Salmon"), 150),
                        (next(f for f in carbs if f.name == "Sweet Potato"), 200),
                        (next(f for f in vegetables if f.name == "Asparagus"), 150),
                        (next(f for f in fats if f.name == "Olive Oil"), 10)
                    ],
                    preparation_time_minutes=30,
                    instructions=[
                        "Bake salmon with herbs",
                        "Roast sweet potato",
                        "Sauté asparagus with olive oil",
                        "Serve together"
                    ]
                )
            ],
            'snack': [
                Meal(
                    name="Greek Yogurt Parfait",
                    meal_type="snack",
                    food_items=[
                        (next(f for f in proteins if f.name == "Greek Yogurt"), 150),
                        (next(f for f in fats if f.name == "Almonds"), 20)
                    ],
                    preparation_time_minutes=5,
                    instructions=[
                        "Layer Greek yogurt",
                        "Top with chopped almonds"
                    ]
                )
            ]
        }
        
        return templates
    
    def _create_default_meal(self, target_calories: int) -> Meal:
        """Create a default meal when no templates match."""
        
        # Simple default meal
        proteins = self.food_database.get('proteins', [])
        carbs = self.food_database.get('carbohydrates', [])
        
        return Meal(
            name="Simple Balanced Meal",
            meal_type="meal",
            food_items=[
                (proteins[0], 100),
                (carbs[0], 80)
            ],
            preparation_time_minutes=15,
            instructions=["Cook protein", "Prepare carbohydrate", "Serve together"]
        )

    def calculate_meal_macros(self, meal: Meal) -> Dict[str, float]:
        """Calculate macros for a specific meal."""
        
        totals = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for food_item, amount in meal.food_items:
            ratio = amount / 100
            totals['calories'] += food_item.calories_per_100g * ratio
            totals['protein'] += food_item.protein_per_100g * ratio
            totals['carbs'] += food_item.carbs_per_100g * ratio
            totals['fat'] += food_item.fat_per_100g * ratio
            totals['fiber'] += food_item.fiber_per_100g * ratio
        
        return totals

    def generate_weekly_meal_prep_plan(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate a weekly meal prep plan."""
        
        macro_targets = self.calculate_macro_targets(user_profile)
        
        # Generate 7 days of meals
        weekly_meals = []
        start_date = date.today()
        
        for i in range(7):
            day_date = start_date + timedelta(days=i)
            daily_plan = self.generate_meal_plan(user_profile, day_date, macro_targets)
            weekly_meals.append(daily_plan)
        
        # Generate shopping list
        shopping_list = self._generate_shopping_list(weekly_meals)
        
        # Generate prep instructions
        prep_instructions = self._generate_prep_instructions(weekly_meals)
        
        return {
            'weekly_meals': weekly_meals,
            'shopping_list': shopping_list,
            'prep_instructions': prep_instructions,
            'macro_targets': macro_targets
        }
    
    def _generate_shopping_list(self, weekly_meals: List[NutritionPlan]) -> Dict[str, float]:
        """Generate shopping list for the week."""
        
        shopping_list = {}
        
        for daily_plan in weekly_meals:
            for meal in daily_plan.meals:
                for food_item, amount in meal.food_items:
                    if food_item.name in shopping_list:
                        shopping_list[food_item.name] += amount
                    else:
                        shopping_list[food_item.name] = amount
        
        # Convert to reasonable shopping quantities
        shopping_list_formatted = {}
        for food_name, total_grams in shopping_list.items():
            if total_grams >= 1000:
                shopping_list_formatted[food_name] = f"{total_grams/1000:.1f} kg"
            else:
                shopping_list_formatted[food_name] = f"{total_grams:.0f} g"
        
        return shopping_list_formatted
    
    def _generate_prep_instructions(self, weekly_meals: List[NutritionPlan]) -> List[str]:
        """Generate meal prep instructions for the week."""
        
        instructions = [
            "🛒 Shopping Day (Sunday):",
            "- Purchase all ingredients from shopping list",
            "- Wash and prep vegetables",
            "",
            "📦 Prep Day (Sunday Evening):",
            "- Cook proteins in bulk (chicken, salmon, etc.)",
            "- Prepare grains (rice, quinoa) in large batches",
            "- Chop vegetables for easy assembly",
            "- Portion snacks into containers",
            "",
            "🥘 Daily Assembly:",
            "- Combine pre-cooked proteins and grains",
            "- Add fresh vegetables and healthy fats",
            "- Follow specific meal instructions",
            "",
            "💾 Storage Tips:",
            "- Use glass containers for better freshness",
            "- Store proteins for 3-4 days maximum",
            "- Keep dressings/sauces separate until serving",
            "- Label containers with contents and date"
        ]
        
        return instructions
