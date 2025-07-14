"""
Test script for the enhanced fitness app
Tests the weekly plan generation to ensure no KeyError occurs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced app components
from fitness_app_enhanced import (
    EnhancedUserProfile, 
    SmartRecommendationEngine
)

def test_weekly_plan_generation():
    """Test weekly plan generation for different user profiles."""
    
    print("🧪 Testing Weekly Plan Generation...")
    
    # Create test user profiles
    test_profiles = [
        {
            'name': 'Beginner User',
            'fitness_level': 'beginner',
            'primary_goal': 'general_fitness',
            'workout_days_per_week': 3,
            'available_time': 30
        },
        {
            'name': 'Weight Loss User', 
            'fitness_level': 'intermediate',
            'primary_goal': 'weight_loss',
            'workout_days_per_week': 5,
            'available_time': 45
        },
        {
            'name': 'Muscle Gain User',
            'fitness_level': 'advanced',
            'primary_goal': 'muscle_gain',
            'workout_days_per_week': 6,
            'available_time': 60
        }
    ]
    
    engine = SmartRecommendationEngine()
    
    for profile_data in test_profiles:
        print(f"\n📋 Testing: {profile_data['name']}")
        
        # Create user profile
        user_profile = EnhancedUserProfile(**profile_data)
        
        try:
            # Generate recommendations
            recommendations = engine.generate_recommendations(user_profile)
            
            # Check weekly plan
            weekly_plan = recommendations['weekly_plan']
            
            print(f"  ✅ Weekly plan generated successfully!")
            print(f"  📊 Estimated weekly calories: {recommendations['estimated_weekly_calories']}")
            
            # Verify all required keys exist
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in days:
                plan = weekly_plan[day]
                required_keys = ['type', 'focus', 'duration']
                
                for key in required_keys:
                    if key not in plan:
                        print(f"  ❌ Missing key '{key}' in {day} plan")
                        return False
                
                print(f"    📅 {day}: {plan['type']} - {plan['focus']}")
            
        except Exception as e:
            print(f"  ❌ Error generating plan: {e}")
            return False
    
    print(f"\n🎉 All tests passed! Weekly plan generation is working correctly.")
    return True

def test_recommendation_categories():
    """Test that all recommendation categories work."""
    
    print("\n🔍 Testing Recommendation Categories...")
    
    user_profile = EnhancedUserProfile(
        fitness_level='intermediate',
        primary_goal='general_fitness',
        available_equipment=['dumbbells', 'yoga_mat']
    )
    
    engine = SmartRecommendationEngine()
    
    try:
        recommendations = engine.generate_recommendations(user_profile)
        
        # Check all categories exist
        categories = ['cardio', 'strength', 'flexibility']
        for category in categories:
            if category not in recommendations['recommendations']:
                print(f"  ❌ Missing category: {category}")
                return False
            
            category_recs = recommendations['recommendations'][category]
            print(f"  ✅ {category.title()}: {len(category_recs)} exercises recommended")
        
        print("  🎉 All recommendation categories working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error in recommendations: {e}")
        return False

if __name__ == "__main__":
    print("🏋️‍♀️ Enhanced Fitness App - Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        test1_passed = test_weekly_plan_generation()
        test2_passed = test_recommendation_categories()
        
        if test1_passed and test2_passed:
            print(f"\n🎉 ALL TESTS PASSED! The enhanced fitness app is working correctly.")
            print(f"✅ KeyError for 'focus' has been fixed")
            print(f"✅ Weekly plans generate properly for all user types")
            print(f"✅ Recommendation engine works for all categories")
        else:
            print(f"\n❌ Some tests failed. Please check the output above.")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
