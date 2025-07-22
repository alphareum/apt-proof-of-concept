"""
Comprehensive Recipe Database with Pre-stored Recipes
Addresses feedback about storing recipes in database instead of generating them

Repository: https://github.com/alphareum/apt-proof-of-concept
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class RecipeNutrition:
    """Nutritional information for a recipe."""
    calories_per_serving: int
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float

@dataclass
class StoredRecipe:
    """Pre-stored recipe with complete information."""
    id: str
    name: str
    description: str
    category: str  # 'high_protein', 'low_carb', 'balanced', 'post_workout', 'breakfast', 'dinner', etc.
    cuisine_type: str  # 'mediterranean', 'asian', 'american', 'mexican', etc.
    difficulty: str  # 'easy', 'medium', 'hard'
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    ingredients: List[Dict[str, Any]]  # [{"name": "chicken breast", "amount": "200g", "notes": ""}]
    instructions: List[str]
    nutrition: RecipeNutrition
    tags: List[str]  # ['gluten_free', 'dairy_free', 'vegetarian', 'keto', etc.]
    meal_timing: List[str]  # ['breakfast', 'lunch', 'dinner', 'snack', 'pre_workout', 'post_workout']
    equipment_needed: List[str]
    storage_instructions: str
    recipe_notes: str
    created_date: datetime

class ComprehensiveRecipeDatabase:
    """Database containing pre-stored recipes across various categories."""
    
    def __init__(self, db_path: str = "recipes.db"):
        self.db_path = db_path
        self._initialize_database()
        self._populate_default_recipes()
    
    def _initialize_database(self):
        """Initialize the recipe database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                cuisine_type TEXT,
                difficulty TEXT,
                prep_time_minutes INTEGER,
                cook_time_minutes INTEGER,
                servings INTEGER,
                ingredients TEXT,  -- JSON string
                instructions TEXT, -- JSON string
                nutrition TEXT,    -- JSON string
                tags TEXT,         -- JSON string
                meal_timing TEXT,  -- JSON string
                equipment_needed TEXT, -- JSON string
                storage_instructions TEXT,
                recipe_notes TEXT,
                created_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _populate_default_recipes(self):
        """Populate database with comprehensive recipe collection."""
        
        # Check if recipes already exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM recipes')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            return  # Recipes already populated
        
        default_recipes = self._get_default_recipe_collection()
        
        for recipe in default_recipes:
            self.add_recipe(recipe)
    
    def _get_default_recipe_collection(self) -> List[StoredRecipe]:
        """Get comprehensive collection of default recipes."""
        
        recipes = []
        
        # HIGH PROTEIN RECIPES
        recipes.extend([
            StoredRecipe(
                id="hp_chicken_quinoa_bowl",
                name="Grilled Chicken Quinoa Power Bowl",
                description="Protein-packed bowl with grilled chicken, quinoa, and vegetables",
                category="high_protein",
                cuisine_type="mediterranean",
                difficulty="easy",
                prep_time_minutes=15,
                cook_time_minutes=25,
                servings=2,
                ingredients=[
                    {"name": "chicken breast", "amount": "300g", "notes": "boneless, skinless"},
                    {"name": "quinoa", "amount": "1 cup", "notes": "rinsed"},
                    {"name": "broccoli", "amount": "200g", "notes": "cut into florets"},
                    {"name": "cherry tomatoes", "amount": "150g", "notes": "halved"},
                    {"name": "olive oil", "amount": "2 tbsp", "notes": "extra virgin"},
                    {"name": "lemon", "amount": "1 whole", "notes": "juiced"},
                    {"name": "garlic", "amount": "2 cloves", "notes": "minced"},
                    {"name": "feta cheese", "amount": "50g", "notes": "crumbled"}
                ],
                instructions=[
                    "Season chicken breast with salt, pepper, and 1 tsp olive oil",
                    "Grill chicken for 6-7 minutes each side until cooked through",
                    "Cook quinoa according to package instructions (about 15 minutes)",
                    "Steam broccoli for 5 minutes until tender-crisp",
                    "Whisk together remaining olive oil, lemon juice, and garlic for dressing",
                    "Slice chicken and arrange over quinoa with vegetables",
                    "Drizzle with dressing and top with feta cheese"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=485,
                    protein_g=42.0,
                    carbs_g=38.0,
                    fat_g=18.0,
                    fiber_g=8.0,
                    sugar_g=6.0,
                    sodium_mg=420
                ),
                tags=["gluten_free", "high_protein", "balanced"],
                meal_timing=["lunch", "dinner", "post_workout"],
                equipment_needed=["grill pan", "pot", "steamer"],
                storage_instructions="Refrigerate up to 3 days. Store dressing separately.",
                recipe_notes="Great for meal prep. Can substitute chicken with tofu for vegetarian option.",
                created_date=datetime.now()
            ),
            
            StoredRecipe(
                id="hp_salmon_sweet_potato",
                name="Herb-Crusted Salmon with Sweet Potato",
                description="Omega-3 rich salmon with roasted sweet potato and asparagus",
                category="high_protein",
                cuisine_type="american",
                difficulty="medium",
                prep_time_minutes=10,
                cook_time_minutes=30,
                servings=2,
                ingredients=[
                    {"name": "salmon fillets", "amount": "300g", "notes": "skin-on, pin bones removed"},
                    {"name": "sweet potato", "amount": "400g", "notes": "medium, cubed"},
                    {"name": "asparagus", "amount": "200g", "notes": "trimmed"},
                    {"name": "fresh dill", "amount": "2 tbsp", "notes": "chopped"},
                    {"name": "fresh parsley", "amount": "2 tbsp", "notes": "chopped"},
                    {"name": "olive oil", "amount": "3 tbsp", "notes": "divided"},
                    {"name": "lemon zest", "amount": "1 tsp", "notes": "fresh"},
                    {"name": "garlic powder", "amount": "1 tsp", "notes": ""}
                ],
                instructions=[
                    "Preheat oven to 425°F (220°C)",
                    "Toss sweet potato cubes with 1 tbsp olive oil, salt, and pepper",
                    "Roast sweet potatoes for 15 minutes",
                    "Add asparagus to the same pan and roast 10 more minutes",
                    "Mix herbs, lemon zest, garlic powder, and remaining oil",
                    "Season salmon and top with herb mixture",
                    "Bake salmon for 12-15 minutes until flakes easily",
                    "Serve salmon over roasted vegetables"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=520,
                    protein_g=35.0,
                    carbs_g=32.0,
                    fat_g=26.0,
                    fiber_g=6.0,
                    sugar_g=12.0,
                    sodium_mg=380
                ),
                tags=["gluten_free", "dairy_free", "high_protein", "omega_3"],
                meal_timing=["dinner", "post_workout"],
                equipment_needed=["baking sheet", "oven"],
                storage_instructions="Best consumed fresh. Leftovers keep 2 days refrigerated.",
                recipe_notes="Wild-caught salmon preferred. Don't overcook to maintain moisture.",
                created_date=datetime.now()
            )
        ])
        
        # BALANCED/GENERAL RECIPES
        recipes.extend([
            StoredRecipe(
                id="bal_mediterranean_wrap",
                name="Mediterranean Veggie Wrap",
                description="Fresh vegetables with hummus in a whole wheat wrap",
                category="balanced",
                cuisine_type="mediterranean",
                difficulty="easy",
                prep_time_minutes=10,
                cook_time_minutes=0,
                servings=2,
                ingredients=[
                    {"name": "whole wheat tortillas", "amount": "2 large", "notes": "burrito size"},
                    {"name": "hummus", "amount": "6 tbsp", "notes": "store-bought or homemade"},
                    {"name": "cucumber", "amount": "1 medium", "notes": "sliced"},
                    {"name": "red bell pepper", "amount": "1 medium", "notes": "sliced"},
                    {"name": "red onion", "amount": "1/4 medium", "notes": "thinly sliced"},
                    {"name": "tomato", "amount": "1 large", "notes": "sliced"},
                    {"name": "lettuce", "amount": "4 leaves", "notes": "large leaves"},
                    {"name": "kalamata olives", "amount": "8 olives", "notes": "pitted, sliced"}
                ],
                instructions=[
                    "Warm tortillas in microwave for 30 seconds",
                    "Spread 3 tbsp hummus evenly on each tortilla",
                    "Layer lettuce leaves in center of each tortilla",
                    "Add cucumber, bell pepper, onion, tomato, and olives",
                    "Roll tightly, tucking in sides as you go",
                    "Cut in half diagonally to serve",
                    "Serve immediately or wrap in foil for later"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=320,
                    protein_g=12.0,
                    carbs_g=48.0,
                    fat_g=10.0,
                    fiber_g=8.0,
                    sugar_g=8.0,
                    sodium_mg=580
                ),
                tags=["vegetarian", "balanced", "quick", "portable"],
                meal_timing=["lunch", "snack"],
                equipment_needed=["knife", "cutting board"],
                storage_instructions="Best eaten fresh. Can be made 2 hours ahead if wrapped tightly.",
                recipe_notes="Great for meal prep lunches. Add avocado for extra healthy fats.",
                created_date=datetime.now()
            ),
            
            StoredRecipe(
                id="bal_asian_stir_fry",
                name="Colorful Vegetable Stir-Fry",
                description="Quick and nutritious stir-fry with brown rice",
                category="balanced",
                cuisine_type="asian",
                difficulty="easy",
                prep_time_minutes=15,
                cook_time_minutes=15,
                servings=3,
                ingredients=[
                    {"name": "brown rice", "amount": "1 cup", "notes": "uncooked"},
                    {"name": "mixed vegetables", "amount": "400g", "notes": "frozen or fresh"},
                    {"name": "tofu", "amount": "200g", "notes": "firm, cubed"},
                    {"name": "soy sauce", "amount": "3 tbsp", "notes": "low sodium"},
                    {"name": "sesame oil", "amount": "2 tbsp", "notes": ""},
                    {"name": "garlic", "amount": "3 cloves", "notes": "minced"},
                    {"name": "ginger", "amount": "1 tbsp", "notes": "fresh, grated"},
                    {"name": "green onions", "amount": "3 stalks", "notes": "chopped"}
                ],
                instructions=[
                    "Cook brown rice according to package instructions",
                    "Heat 1 tbsp sesame oil in large wok or pan",
                    "Add tofu and cook until golden on all sides",
                    "Remove tofu and set aside",
                    "Add remaining oil, garlic, and ginger to pan",
                    "Add vegetables and stir-fry for 5-7 minutes",
                    "Return tofu to pan with soy sauce",
                    "Stir-fry 2 more minutes, garnish with green onions",
                    "Serve over brown rice"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=385,
                    protein_g=18.0,
                    carbs_g=52.0,
                    fat_g=12.0,
                    fiber_g=6.0,
                    sugar_g=8.0,
                    sodium_mg=650
                ),
                tags=["vegetarian", "vegan", "gluten_free", "quick"],
                meal_timing=["lunch", "dinner"],
                equipment_needed=["wok", "rice cooker or pot"],
                storage_instructions="Refrigerate up to 4 days. Reheat in microwave or pan.",
                recipe_notes="Very customizable - use any vegetables you have on hand.",
                created_date=datetime.now()
            )
        ])
        
        # LOW CARB RECIPES
        recipes.extend([
            StoredRecipe(
                id="lc_zucchini_lasagna",
                name="Zucchini Lasagna",
                description="Low-carb lasagna using zucchini slices instead of pasta",
                category="low_carb",
                cuisine_type="italian",
                difficulty="medium",
                prep_time_minutes=30,
                cook_time_minutes=45,
                servings=6,
                ingredients=[
                    {"name": "large zucchini", "amount": "3 pieces", "notes": "sliced lengthwise"},
                    {"name": "ground turkey", "amount": "500g", "notes": "lean"},
                    {"name": "marinara sauce", "amount": "2 cups", "notes": "sugar-free"},
                    {"name": "ricotta cheese", "amount": "500g", "notes": "part-skim"},
                    {"name": "mozzarella cheese", "amount": "300g", "notes": "shredded"},
                    {"name": "parmesan cheese", "amount": "100g", "notes": "grated"},
                    {"name": "egg", "amount": "1 large", "notes": "beaten"},
                    {"name": "Italian seasoning", "amount": "2 tsp", "notes": ""}
                ],
                instructions=[
                    "Preheat oven to 375°F (190°C)",
                    "Slice zucchini lengthwise into 1/4 inch strips",
                    "Salt zucchini slices and let drain 30 minutes",
                    "Brown ground turkey in large pan, add marinara sauce",
                    "Mix ricotta, egg, and Italian seasoning",
                    "Pat zucchini dry and layer in baking dish",
                    "Alternate layers: zucchini, ricotta mixture, meat sauce, mozzarella",
                    "Top with parmesan cheese",
                    "Bake covered 30 minutes, then uncovered 15 minutes",
                    "Let rest 10 minutes before serving"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=320,
                    protein_g=28.0,
                    carbs_g=12.0,
                    fat_g=18.0,
                    fiber_g=4.0,
                    sugar_g=8.0,
                    sodium_mg=720
                ),
                tags=["low_carb", "keto_friendly", "high_protein", "gluten_free"],
                meal_timing=["dinner"],
                equipment_needed=["baking dish", "large pan", "mandoline slicer (optional)"],
                storage_instructions="Refrigerate up to 5 days or freeze up to 3 months.",
                recipe_notes="Make sure to drain zucchini well to prevent watery lasagna.",
                created_date=datetime.now()
            )
        ])
        
        # BREAKFAST RECIPES
        recipes.extend([
            StoredRecipe(
                id="bf_protein_pancakes",
                name="Banana Protein Pancakes",
                description="Fluffy protein-packed pancakes perfect for post-workout",
                category="breakfast",
                cuisine_type="american",
                difficulty="easy",
                prep_time_minutes=5,
                cook_time_minutes=10,
                servings=2,
                ingredients=[
                    {"name": "ripe bananas", "amount": "2 medium", "notes": "mashed"},
                    {"name": "eggs", "amount": "3 large", "notes": ""},
                    {"name": "protein powder", "amount": "1 scoop", "notes": "vanilla or unflavored"},
                    {"name": "oats", "amount": "1/4 cup", "notes": "rolled"},
                    {"name": "baking powder", "amount": "1/2 tsp", "notes": ""},
                    {"name": "cinnamon", "amount": "1/2 tsp", "notes": "ground"},
                    {"name": "coconut oil", "amount": "1 tbsp", "notes": "for cooking"},
                    {"name": "berries", "amount": "1 cup", "notes": "for topping"}
                ],
                instructions=[
                    "Mash bananas in a large bowl",
                    "Whisk in eggs until well combined",
                    "Add protein powder, oats, baking powder, and cinnamon",
                    "Mix until just combined, don't overmix",
                    "Heat coconut oil in non-stick pan over medium heat",
                    "Pour 1/4 cup batter per pancake",
                    "Cook 2-3 minutes until bubbles form, then flip",
                    "Cook another 2 minutes until golden",
                    "Serve with fresh berries"
                ],
                nutrition=RecipeNutrition(
                    calories_per_serving=340,
                    protein_g=25.0,
                    carbs_g=32.0,
                    fat_g=12.0,
                    fiber_g=6.0,
                    sugar_g=18.0,
                    sodium_mg=280
                ),
                tags=["high_protein", "gluten_free", "post_workout", "breakfast"],
                meal_timing=["breakfast", "post_workout"],
                equipment_needed=["non-stick pan", "mixing bowl", "whisk"],
                storage_instructions="Best eaten fresh. Batter can be made night before.",
                recipe_notes="Don't overmix batter for fluffiest pancakes. Great with Greek yogurt.",
                created_date=datetime.now()
            )
        ])
        
        return recipes
    
    def add_recipe(self, recipe: StoredRecipe):
        """Add a recipe to the database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            recipe.id,
            recipe.name,
            recipe.description,
            recipe.category,
            recipe.cuisine_type,
            recipe.difficulty,
            recipe.prep_time_minutes,
            recipe.cook_time_minutes,
            recipe.servings,
            json.dumps(recipe.ingredients),
            json.dumps(recipe.instructions),
            json.dumps(asdict(recipe.nutrition)),
            json.dumps(recipe.tags),
            json.dumps(recipe.meal_timing),
            json.dumps(recipe.equipment_needed),
            recipe.storage_instructions,
            recipe.recipe_notes,
            recipe.created_date.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_recipes_by_category(self, category: str) -> List[StoredRecipe]:
        """Get all recipes from a specific category."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM recipes WHERE category = ?', (category,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_recipe(row) for row in rows]
    
    def get_recipes_by_tags(self, tags: List[str]) -> List[StoredRecipe]:
        """Get recipes that match any of the provided tags."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM recipes')
        rows = cursor.fetchall()
        conn.close()
        
        matching_recipes = []
        for row in rows:
            recipe_tags = json.loads(row[12])  # tags column
            if any(tag in recipe_tags for tag in tags):
                matching_recipes.append(self._row_to_recipe(row))
        
        return matching_recipes
    
    def get_recipes_by_meal_timing(self, meal_timing: str) -> List[StoredRecipe]:
        """Get recipes suitable for specific meal timing."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM recipes')
        rows = cursor.fetchall()
        conn.close()
        
        matching_recipes = []
        for row in rows:
            recipe_timings = json.loads(row[13])  # meal_timing column
            if meal_timing in recipe_timings:
                matching_recipes.append(self._row_to_recipe(row))
        
        return matching_recipes
    
    def search_recipes(self, 
                      category: Optional[str] = None,
                      cuisine_type: Optional[str] = None,
                      max_prep_time: Optional[int] = None,
                      max_calories: Optional[int] = None,
                      min_protein: Optional[float] = None,
                      tags: Optional[List[str]] = None) -> List[StoredRecipe]:
        """Search recipes with multiple filters."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM recipes')
        rows = cursor.fetchall()
        conn.close()
        
        filtered_recipes = []
        
        for row in rows:
            recipe = self._row_to_recipe(row)
            
            # Apply filters
            if category and recipe.category != category:
                continue
            if cuisine_type and recipe.cuisine_type != cuisine_type:
                continue
            if max_prep_time and recipe.prep_time_minutes > max_prep_time:
                continue
            if max_calories and recipe.nutrition.calories_per_serving > max_calories:
                continue
            if min_protein and recipe.nutrition.protein_g < min_protein:
                continue
            if tags and not any(tag in recipe.tags for tag in tags):
                continue
            
            filtered_recipes.append(recipe)
        
        return filtered_recipes
    
    def _row_to_recipe(self, row) -> StoredRecipe:
        """Convert database row to StoredRecipe object."""
        
        return StoredRecipe(
            id=row[0],
            name=row[1],
            description=row[2],
            category=row[3],
            cuisine_type=row[4],
            difficulty=row[5],
            prep_time_minutes=row[6],
            cook_time_minutes=row[7],
            servings=row[8],
            ingredients=json.loads(row[9]),
            instructions=json.loads(row[10]),
            nutrition=RecipeNutrition(**json.loads(row[11])),
            tags=json.loads(row[12]),
            meal_timing=json.loads(row[13]),
            equipment_needed=json.loads(row[14]),
            storage_instructions=row[15],
            recipe_notes=row[16],
            created_date=datetime.fromisoformat(row[17])
        )
    
    def get_recipe_recommendations(self, user_profile, meal_type: str = "dinner", 
                                 max_recipes: int = 5) -> List[StoredRecipe]:
        """Get personalized recipe recommendations based on user profile."""
        
        # Determine preferred categories based on user goals
        preferred_categories = []
        preferred_tags = []
        
        if hasattr(user_profile, 'primary_goal'):
            if user_profile.primary_goal.value == 'muscle_gain':
                preferred_categories = ['high_protein', 'balanced']
                preferred_tags = ['high_protein', 'post_workout']
            elif user_profile.primary_goal.value == 'weight_loss':
                preferred_categories = ['low_carb', 'balanced']
                preferred_tags = ['low_carb', 'high_protein']
            else:
                preferred_categories = ['balanced', 'high_protein']
                preferred_tags = ['balanced']
        
        # Get recipes by meal timing
        meal_recipes = self.get_recipes_by_meal_timing(meal_type)
        
        # Filter by preferences
        recommended = []
        for recipe in meal_recipes:
            if recipe.category in preferred_categories or any(tag in recipe.tags for tag in preferred_tags):
                recommended.append(recipe)
        
        # If we don't have enough, add some general recipes
        if len(recommended) < max_recipes:
            all_meal_recipes = [r for r in meal_recipes if r not in recommended]
            recommended.extend(all_meal_recipes[:max_recipes - len(recommended)])
        
        return recommended[:max_recipes]
