"""
Meal Planning and Recipe Management System
Recipe database and meal planning features

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, date
import logging
from dataclasses import dataclass, field
import json
from nutrition_planner import FoodItem, MacroTargets
from models import UserProfile, GoalType

logger = logging.getLogger(__name__)

@dataclass
class Recipe:
    """Recipe with nutritional information and instructions."""
    id: str
    name: str
    description: str
    category: str  # breakfast, lunch, dinner, snack, dessert
    cuisine_type: str  # italian, asian, mexican, etc.
    difficulty: str  # easy, medium, hard
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    ingredients: List[Dict[str, Any]]  # [{'item': FoodItem, 'amount': float, 'unit': str}]
    instructions: List[str]
    nutritional_info: Dict[str, float]
    tags: List[str] = field(default_factory=list)  # vegetarian, vegan, gluten-free, etc.
    rating: float = 0.0
    created_date: datetime = field(default_factory=datetime.now)

@dataclass
class MealPrepPlan:
    """Meal prep planning structure."""
    name: str
    duration_days: int
    target_servings: int
    recipes: List[Recipe]
    shopping_list: Dict[str, str]
    prep_schedule: List[Dict[str, Any]]
    storage_instructions: List[str]
    estimated_cost: float = 0.0

class RecipeManager:
    """Recipe management and meal planning system."""
    
    def __init__(self):
        self.recipes_db = self._initialize_recipe_database()
        self.user_favorites = {}
        self.custom_recipes = {}
    
    def get_recipes_by_criteria(self, 
                               category: str = None,
                               cuisine_type: str = None,
                               difficulty: str = None,
                               max_prep_time: int = None,
                               tags: List[str] = None,
                               max_calories: int = None) -> List[Recipe]:
        """Search recipes by various criteria."""
        
        filtered_recipes = list(self.recipes_db.values())
        
        if category:
            filtered_recipes = [r for r in filtered_recipes if r.category == category]
        
        if cuisine_type:
            filtered_recipes = [r for r in filtered_recipes if r.cuisine_type == cuisine_type]
        
        if difficulty:
            filtered_recipes = [r for r in filtered_recipes if r.difficulty == difficulty]
        
        if max_prep_time:
            total_time = max_prep_time
            filtered_recipes = [r for r in filtered_recipes 
                              if (r.prep_time_minutes + r.cook_time_minutes) <= total_time]
        
        if tags:
            filtered_recipes = [r for r in filtered_recipes 
                              if any(tag in r.tags for tag in tags)]
        
        if max_calories:
            filtered_recipes = [r for r in filtered_recipes 
                              if r.nutritional_info.get('calories', 0) <= max_calories]
        
        return filtered_recipes
    
    def get_recipes_for_goals(self, user_profile: UserProfile) -> Dict[str, List[Recipe]]:
        """Get recipes suited for user's fitness goals."""
        
        goal_recipes = {
            'recommended': [],
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snacks': []
        }
        
        if user_profile.primary_goal == GoalType.WEIGHT_LOSS:
            # Low calorie, high protein recipes
            for category in ['breakfast', 'lunch', 'dinner', 'snacks']:
                recipes = self.get_recipes_by_criteria(
                    category=category,
                    max_calories=400 if category == 'snacks' else 600,
                    tags=['high_protein', 'low_calorie']
                )
                goal_recipes[category] = recipes[:5]  # Top 5
        
        elif user_profile.primary_goal == GoalType.MUSCLE_GAIN:
            # High protein, moderate calorie recipes
            for category in ['breakfast', 'lunch', 'dinner', 'snacks']:
                recipes = self.get_recipes_by_criteria(
                    category=category,
                    tags=['high_protein', 'muscle_building']
                )
                goal_recipes[category] = recipes[:5]
        
        else:
            # Balanced, healthy recipes
            for category in ['breakfast', 'lunch', 'dinner', 'snacks']:
                recipes = self.get_recipes_by_criteria(
                    category=category,
                    tags=['balanced', 'healthy']
                )
                goal_recipes[category] = recipes[:5]
        
        # Overall recommended based on user preferences
        goal_recipes['recommended'] = self._get_personalized_recommendations(user_profile)
        
        return goal_recipes
    
    def create_weekly_meal_plan(self, user_profile: UserProfile, macro_targets: MacroTargets) -> Dict[str, Any]:
        """Create a weekly meal plan with recipes."""
        
        weekly_plan = {
            'days': {},
            'shopping_list': {},
            'prep_instructions': [],
            'nutritional_summary': {}
        }
        
        # Get suitable recipes for user's goals
        suitable_recipes = self.get_recipes_for_goals(user_profile)
        
        # Plan each day
        for day_num in range(7):
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_num]
            
            daily_meals = {
                'breakfast': self._select_meal_recipe(suitable_recipes['breakfast'], macro_targets, 0.25),
                'lunch': self._select_meal_recipe(suitable_recipes['lunch'], macro_targets, 0.35),
                'dinner': self._select_meal_recipe(suitable_recipes['dinner'], macro_targets, 0.30),
                'snack': self._select_meal_recipe(suitable_recipes['snacks'], macro_targets, 0.10)
            }
            
            weekly_plan['days'][day_name] = daily_meals
        
        # Generate shopping list
        weekly_plan['shopping_list'] = self._generate_recipe_shopping_list(weekly_plan['days'])
        
        # Generate prep instructions
        weekly_plan['prep_instructions'] = self._generate_meal_prep_instructions(weekly_plan['days'])
        
        # Calculate nutritional summary
        weekly_plan['nutritional_summary'] = self._calculate_weekly_nutrition(weekly_plan['days'])
        
        return weekly_plan
    
    def create_meal_prep_plan(self, recipes: List[Recipe], target_days: int = 5) -> MealPrepPlan:
        """Create a meal prep plan for batch cooking."""
        
        # Calculate total servings needed
        total_servings = sum(recipe.servings for recipe in recipes) * target_days
        
        # Generate shopping list
        shopping_list = {}
        for recipe in recipes:
            for ingredient in recipe.ingredients:
                item_name = ingredient['item'].name
                amount = ingredient['amount'] * target_days
                unit = ingredient['unit']
                
                if item_name in shopping_list:
                    # Simple addition - in reality would need unit conversion
                    shopping_list[item_name] += f" + {amount} {unit}"
                else:
                    shopping_list[item_name] = f"{amount} {unit}"
        
        # Create prep schedule
        prep_schedule = self._create_prep_schedule(recipes, target_days)
        
        # Storage instructions
        storage_instructions = self._generate_storage_instructions(recipes)
        
        return MealPrepPlan(
            name=f"{target_days}-Day Meal Prep",
            duration_days=target_days,
            target_servings=total_servings,
            recipes=recipes,
            shopping_list=shopping_list,
            prep_schedule=prep_schedule,
            storage_instructions=storage_instructions
        )
    
    def add_custom_recipe(self, user_id: str, recipe_data: Dict[str, Any]) -> Recipe:
        """Add a custom user recipe."""
        
        # Calculate nutritional information
        nutritional_info = self._calculate_recipe_nutrition(recipe_data['ingredients'])
        
        recipe = Recipe(
            id=f"custom_{user_id}_{len(self.custom_recipes)}",
            name=recipe_data['name'],
            description=recipe_data.get('description', ''),
            category=recipe_data['category'],
            cuisine_type=recipe_data.get('cuisine_type', 'custom'),
            difficulty=recipe_data.get('difficulty', 'medium'),
            prep_time_minutes=recipe_data.get('prep_time', 15),
            cook_time_minutes=recipe_data.get('cook_time', 15),
            servings=recipe_data.get('servings', 2),
            ingredients=recipe_data['ingredients'],
            instructions=recipe_data['instructions'],
            nutritional_info=nutritional_info,
            tags=recipe_data.get('tags', [])
        )
        
        if user_id not in self.custom_recipes:
            self.custom_recipes[user_id] = {}
        
        self.custom_recipes[user_id][recipe.id] = recipe
        return recipe
    
    def get_recipe_variations(self, base_recipe: Recipe, dietary_restrictions: List[str] = None) -> List[Recipe]:
        """Generate recipe variations based on dietary restrictions."""
        
        variations = []
        
        if dietary_restrictions:
            if 'vegetarian' in dietary_restrictions:
                vegetarian_recipe = self._create_vegetarian_variation(base_recipe)
                if vegetarian_recipe:
                    variations.append(vegetarian_recipe)
            
            if 'low_carb' in dietary_restrictions:
                low_carb_recipe = self._create_low_carb_variation(base_recipe)
                if low_carb_recipe:
                    variations.append(low_carb_recipe)
            
            if 'high_protein' in dietary_restrictions:
                high_protein_recipe = self._create_high_protein_variation(base_recipe)
                if high_protein_recipe:
                    variations.append(high_protein_recipe)
        
        return variations
    
    def _select_meal_recipe(self, recipes: List[Recipe], macro_targets: MacroTargets, meal_portion: float) -> Recipe:
        """Select the best recipe for a meal based on macro targets."""
        
        target_calories = macro_targets.calories * meal_portion
        
        if not recipes:
            return None
        
        # Find recipe closest to target calories
        best_recipe = min(recipes, 
                         key=lambda r: abs(r.nutritional_info.get('calories', 0) - target_calories))
        
        return best_recipe
    
    def _generate_recipe_shopping_list(self, weekly_meals: Dict[str, Dict[str, Recipe]]) -> Dict[str, str]:
        """Generate shopping list from weekly meal plan."""
        
        shopping_list = {}
        
        for day_meals in weekly_meals.values():
            for meal_recipe in day_meals.values():
                if meal_recipe:
                    for ingredient in meal_recipe.ingredients:
                        item_name = ingredient['item'].name
                        amount = ingredient['amount']
                        unit = ingredient['unit']
                        
                        if item_name in shopping_list:
                            # Simple aggregation - would need proper unit handling
                            current = shopping_list[item_name]
                            shopping_list[item_name] = f"{current} + {amount} {unit}"
                        else:
                            shopping_list[item_name] = f"{amount} {unit}"
        
        return shopping_list
    
    def _generate_meal_prep_instructions(self, weekly_meals: Dict[str, Dict[str, Recipe]]) -> List[str]:
        """Generate meal prep instructions."""
        
        instructions = [
            "📅 Sunday Meal Prep Session:",
            "",
            "🛒 Shopping (30 minutes):",
            "- Get all ingredients from shopping list",
            "- Check spices and condiments",
            "",
            "🔪 Prep Work (45 minutes):",
            "- Wash and chop all vegetables",
            "- Marinate proteins if needed",
            "- Cook grains in bulk (rice, quinoa)",
            "",
            "🍳 Cooking Session (60-90 minutes):",
            "- Cook proteins for the week",
            "- Prepare base sauces and dressings",
            "- Assemble meals that store well",
            "",
            "📦 Storage (15 minutes):",
            "- Use glass containers for better freshness",
            "- Label with contents and date",
            "- Store according to each recipe's requirements",
            "",
            "🔄 Daily Assembly:",
            "- Combine prepped ingredients as needed",
            "- Add fresh elements before eating",
            "- Reheat safely and enjoy!"
        ]
        
        return instructions
    
    def _calculate_weekly_nutrition(self, weekly_meals: Dict[str, Dict[str, Recipe]]) -> Dict[str, float]:
        """Calculate nutritional summary for the week."""
        
        weekly_totals = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for day_meals in weekly_meals.values():
            for meal_recipe in day_meals.values():
                if meal_recipe:
                    for nutrient in weekly_totals.keys():
                        weekly_totals[nutrient] += meal_recipe.nutritional_info.get(nutrient, 0)
        
        # Calculate daily averages
        daily_averages = {f"avg_{k}": v/7 for k, v in weekly_totals.items()}
        
        return {**weekly_totals, **daily_averages}
    
    def _create_prep_schedule(self, recipes: List[Recipe], target_days: int) -> List[Dict[str, Any]]:
        """Create a meal prep schedule."""
        
        schedule = [
            {
                'task': 'Shopping',
                'duration_minutes': 45,
                'description': 'Purchase all ingredients',
                'day': 'Prep Day'
            },
            {
                'task': 'Ingredient Prep',
                'duration_minutes': 30,
                'description': 'Wash, chop, and organize ingredients',
                'day': 'Prep Day'
            }
        ]
        
        # Add cooking tasks for each recipe
        for recipe in recipes:
            schedule.append({
                'task': f'Cook {recipe.name}',
                'duration_minutes': recipe.prep_time_minutes + recipe.cook_time_minutes,
                'description': f'Prepare {recipe.servings * target_days} servings',
                'day': 'Prep Day'
            })
        
        schedule.append({
            'task': 'Portioning & Storage',
            'duration_minutes': 20,
            'description': 'Divide into containers and store properly',
            'day': 'Prep Day'
        })
        
        return schedule
    
    def _generate_storage_instructions(self, recipes: List[Recipe]) -> List[str]:
        """Generate storage instructions for meal prep."""
        
        return [
            "🥶 Refrigeration (3-4 days):",
            "- Cooked proteins and vegetables",
            "- Assembled salads (without dressing)",
            "- Most prepared meals",
            "",
            "🧊 Freezing (up to 3 months):",
            "- Cooked grains and legumes",
            "- Soups and stews",
            "- Marinated raw proteins",
            "",
            "🌡️ Room Temperature:",
            "- Nuts, seeds, and dried fruits",
            "- Unopened condiments",
            "- Fresh fruits (until ripe)",
            "",
            "💡 Pro Tips:",
            "- Keep dressings separate until serving",
            "- Let hot food cool before refrigerating",
            "- Use oldest meals first",
            "- Reheat thoroughly to 165°F (74°C)"
        ]
    
    def _calculate_recipe_nutrition(self, ingredients: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate nutritional information for a recipe."""
        
        totals = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for ingredient in ingredients:
            food_item = ingredient['item']
            amount = ingredient['amount']  # in grams
            
            # Calculate nutrition based on amount
            factor = amount / 100  # food_item nutrition is per 100g
            
            totals['calories'] += food_item.calories_per_100g * factor
            totals['protein'] += food_item.protein_per_100g * factor
            totals['carbs'] += food_item.carbs_per_100g * factor
            totals['fat'] += food_item.fat_per_100g * factor
            totals['fiber'] += food_item.fiber_per_100g * factor
        
        return totals
    
    def _get_personalized_recommendations(self, user_profile: UserProfile) -> List[Recipe]:
        """Get personalized recipe recommendations."""
        
        # Simple recommendation logic - could be much more sophisticated
        all_recipes = list(self.recipes_db.values())
        
        # Filter by dietary preferences if available
        # This would be expanded based on user preferences
        recommended = all_recipes[:10]  # Top 10 for now
        
        return recommended
    
    def _create_vegetarian_variation(self, base_recipe: Recipe) -> Optional[Recipe]:
        """Create vegetarian variation of a recipe."""
        
        # Simple logic to replace meat with plant proteins
        # In reality, this would be much more sophisticated
        if 'vegetarian' in base_recipe.tags:
            return None  # Already vegetarian
        
        # This would implement actual substitutions
        return None
    
    def _create_low_carb_variation(self, base_recipe: Recipe) -> Optional[Recipe]:
        """Create low-carb variation of a recipe."""
        
        # Logic to reduce carbs by substituting ingredients
        return None
    
    def _create_high_protein_variation(self, base_recipe: Recipe) -> Optional[Recipe]:
        """Create high-protein variation of a recipe."""
        
        # Logic to increase protein content
        return None
    
    def _initialize_recipe_database(self) -> Dict[str, Recipe]:
        """Initialize the recipe database with sample recipes."""
        
        recipes = {}
        
        # Create food items for recipes (simplified)
        # In reality, this would reference the food database
        
        # Sample breakfast recipe
        protein_oats = Recipe(
            id="protein_oatmeal_bowl",
            name="Protein Power Oatmeal Bowl",
            description="High-protein breakfast bowl with oats, protein powder, and berries",
            category="breakfast",
            cuisine_type="healthy",
            difficulty="easy",
            prep_time_minutes=5,
            cook_time_minutes=10,
            servings=1,
            ingredients=[
                {'item': FoodItem("Oats", 68, 2.4, 12, 1.4, 10.6, "whole_grain"), 'amount': 50, 'unit': 'g'},
                {'item': FoodItem("Protein Powder", 400, 80, 5, 5, 2, "supplement"), 'amount': 30, 'unit': 'g'},
                {'item': FoodItem("Banana", 89, 1.1, 23, 0.3, 2.6, "fruit"), 'amount': 120, 'unit': 'g'},
                {'item': FoodItem("Berries", 32, 0.7, 7.7, 0.3, 2.4, "fruit"), 'amount': 100, 'unit': 'g'}
            ],
            instructions=[
                "Cook oats with water or milk according to package directions",
                "Stir in protein powder while oats are still warm",
                "Top with sliced banana and fresh berries",
                "Serve immediately"
            ],
            nutritional_info={
                'calories': 420,
                'protein': 35,
                'carbs': 55,
                'fat': 8,
                'fiber': 12
            },
            tags=['high_protein', 'healthy', 'quick', 'muscle_building'],
            rating=4.5
        )
        
        # Sample lunch recipe
        chicken_quinoa_bowl = Recipe(
            id="chicken_quinoa_power_bowl",
            name="Chicken Quinoa Power Bowl",
            description="Balanced lunch bowl with grilled chicken, quinoa, and vegetables",
            category="lunch",
            cuisine_type="healthy",
            difficulty="medium",
            prep_time_minutes=15,
            cook_time_minutes=25,
            servings=1,
            ingredients=[
                {'item': FoodItem("Chicken Breast", 165, 31, 0, 3.6, 0, "lean_protein"), 'amount': 150, 'unit': 'g'},
                {'item': FoodItem("Quinoa", 120, 4.4, 22, 1.9, 2.8, "pseudo_grain"), 'amount': 80, 'unit': 'g'},
                {'item': FoodItem("Broccoli", 34, 2.8, 7, 0.4, 2.6, "cruciferous"), 'amount': 100, 'unit': 'g'},
                {'item': FoodItem("Avocado", 160, 2, 9, 15, 6.7, "healthy_fat"), 'amount': 50, 'unit': 'g'}
            ],
            instructions=[
                "Cook quinoa according to package directions",
                "Season and grill chicken breast until cooked through",
                "Steam broccoli until tender-crisp",
                "Slice avocado",
                "Assemble bowl with quinoa base, top with chicken, broccoli, and avocado",
                "Season with herbs and lemon juice"
            ],
            nutritional_info={
                'calories': 580,
                'protein': 52,
                'carbs': 35,
                'fat': 25,
                'fiber': 15
            },
            tags=['high_protein', 'balanced', 'muscle_building', 'healthy'],
            rating=4.7
        )
        
        # Sample dinner recipe
        salmon_sweet_potato = Recipe(
            id="baked_salmon_sweet_potato",
            name="Baked Salmon with Roasted Sweet Potato",
            description="Omega-3 rich salmon with nutrient-dense sweet potato and asparagus",
            category="dinner",
            cuisine_type="healthy",
            difficulty="medium",
            prep_time_minutes=10,
            cook_time_minutes=30,
            servings=1,
            ingredients=[
                {'item': FoodItem("Salmon", 208, 20, 0, 13, 0, "fatty_protein"), 'amount': 150, 'unit': 'g'},
                {'item': FoodItem("Sweet Potato", 86, 1.6, 20, 0.1, 3, "starchy_vegetable"), 'amount': 200, 'unit': 'g'},
                {'item': FoodItem("Asparagus", 20, 2.2, 3.9, 0.1, 2.1, "green_vegetable"), 'amount': 150, 'unit': 'g'},
                {'item': FoodItem("Olive Oil", 884, 0, 0, 100, 0, "cooking_oil"), 'amount': 10, 'unit': 'ml'}
            ],
            instructions=[
                "Preheat oven to 400°F (200°C)",
                "Cut sweet potato into cubes and toss with olive oil",
                "Roast sweet potato for 20 minutes",
                "Season salmon with herbs and spices",
                "Add salmon and asparagus to the baking sheet",
                "Roast for additional 10-12 minutes until salmon flakes easily",
                "Serve hot"
            ],
            nutritional_info={
                'calories': 520,
                'protein': 35,
                'carbs': 42,
                'fat': 23,
                'fiber': 8
            },
            tags=['omega_3', 'healthy', 'balanced', 'anti_inflammatory'],
            rating=4.8
        )
        
        # Sample snack recipe
        greek_yogurt_parfait = Recipe(
            id="protein_greek_yogurt_parfait",
            name="Protein Greek Yogurt Parfait",
            description="High-protein snack with Greek yogurt, nuts, and berries",
            category="snacks",
            cuisine_type="healthy",
            difficulty="easy",
            prep_time_minutes=5,
            cook_time_minutes=0,
            servings=1,
            ingredients=[
                {'item': FoodItem("Greek Yogurt", 59, 10, 3.6, 0.4, 0, "dairy_protein"), 'amount': 150, 'unit': 'g'},
                {'item': FoodItem("Berries", 32, 0.7, 7.7, 0.3, 2.4, "fruit"), 'amount': 80, 'unit': 'g'},
                {'item': FoodItem("Almonds", 579, 21, 22, 50, 12.5, "nuts"), 'amount': 20, 'unit': 'g'}
            ],
            instructions=[
                "Layer Greek yogurt in a bowl or glass",
                "Add a layer of fresh berries",
                "Top with chopped almonds",
                "Serve immediately"
            ],
            nutritional_info={
                'calories': 230,
                'protein': 20,
                'carbs': 15,
                'fat': 12,
                'fiber': 6
            },
            tags=['high_protein', 'quick', 'healthy', 'probiotic'],
            rating=4.4
        )
        
        recipes[protein_oats.id] = protein_oats
        recipes[chicken_quinoa_bowl.id] = chicken_quinoa_bowl
        recipes[salmon_sweet_potato.id] = salmon_sweet_potato
        recipes[greek_yogurt_parfait.id] = greek_yogurt_parfait
        
        return recipes
