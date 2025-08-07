#!/usr/bin/env python3
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

analyzer = BodyCompositionAnalyzer()

# Let me check what the analyze_image method is actually getting as measurements
dummy_image = 'processed_images/body_analysis_20250714_190834.jpg'

# Test measurements matching the accuracy validation format
test_measurements = {
    'waist_circumference_cm': 80,
    'hips_circumference_cm': 95, 
    'neck_circumference_cm': 35,
    'shoulder_width_cm': 45,
    'height': 175
}

user_profile = {
    'age': 30,
    'gender': 'male',
    'height_cm': 175,
    'weight_kg': 75
}

# Patch the analyzer to print debug info
original_calculate_ml = analyzer._calculate_body_fat_enhanced_ml

def debug_calculate_ml(measurements, age, gender, weight_kg, height_cm):
    print(f"DEBUG - ML method called with measurements: {measurements}")
    print(f"DEBUG - Age: {age}, Gender: {gender}, Weight: {weight_kg}, Height: {height_cm}")
    result = original_calculate_ml(measurements, age, gender, weight_kg, height_cm)
    print(f"DEBUG - ML result: {result}")
    return result

analyzer._calculate_body_fat_enhanced_ml = debug_calculate_ml

try:
    result = analyzer.analyze_image(
        dummy_image,
        physical_measurements=test_measurements,
        user_profile=user_profile
    )
    if result and result.get('success'):
        body_fat = result.get('body_fat_percentage', 'Not found')
        print(f'Final body fat: {body_fat}%')
    else:
        print(f'Analysis failed: {result}')
except Exception as e:
    print(f'Error in image analysis: {e}')
    import traceback
    traceback.print_exc()
