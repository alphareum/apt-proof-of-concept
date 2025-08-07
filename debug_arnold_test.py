#!/usr/bin/env python3
"""
Debug script to test Arnold's body fat calculation directly
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
    
    # Create analyzer
    analyzer = BodyCompositionAnalyzer()
    
    # Arnold-like measurements
    arnold_measurements = {
        'weight': 113,  # kg (250 lbs competition weight)
        'height': 185,  # cm (6'1")
        'analysis_type': 'Basic Body Assessment',
        'focus_areas': ['Overall Body'],
        'notes': 'Testing with Arnold-like muscular physique'
    }
    
    # Arnold-like user profile
    arnold_profile = {
        'age': 25,  # Peak competition age
        'gender': 'male',
        'weight_kg': 113.0,
        'height_cm': 185.0
    }
    
    print("="*60)
    print("TESTING ARNOLD-LIKE PHYSIQUE")
    print("="*60)
    print(f"Weight: {arnold_measurements['weight']} kg")
    print(f"Height: {arnold_measurements['height']} cm")
    print(f"BMI: {arnold_measurements['weight'] / ((arnold_measurements['height']/100)**2):.1f}")
    print()
    
    # Test the enhanced body fat calculation directly
    body_fat = analyzer._calculate_body_fat_enhanced(
        measurements={
            'waist_width': 75,  # Very lean waist for bodybuilder
            'shoulder_width': 60,  # Very wide shoulders
            'neck_circumference': 45,  # Very muscular neck
            'hip_width': 85  # Relatively narrow hips
        },
        age=arnold_profile['age'],
        gender=arnold_profile['gender'],
        weight_kg=arnold_profile['weight_kg'],
        height_cm=arnold_profile['height_cm']
    )
    
    print(f"Enhanced Body Fat Calculation Result: {body_fat:.1f}%")
    print()
    
    # Test athletic detection directly
    is_athletic = analyzer._detect_athletic_build(
        measurements={
            'waist_width': 75,
            'shoulder_width': 60,
            'neck_circumference': 45,
            'hip_width': 85
        },
        bmi=arnold_measurements['weight'] / ((arnold_measurements['height']/100)**2),
        weight_kg=arnold_profile['weight_kg'],
        height_cm=arnold_profile['height_cm']
    )
    
    print(f"Athletic build detected: {is_athletic}")
    print()
    
    if body_fat > 15:
        print("❌ ISSUE: Body fat still too high for Arnold-like physique!")
        print("Expected: 5-10%")
        print(f"Got: {body_fat:.1f}%")
    else:
        print("✅ SUCCESS: Body fat calculation looks reasonable for muscular individual")
        print(f"Result: {body_fat:.1f}% (expected 5-10%)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
