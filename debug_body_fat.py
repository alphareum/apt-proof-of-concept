#!/usr/bin/env python3
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

analyzer = BodyCompositionAnalyzer()
dummy_image = 'processed_images/body_analysis_20250714_190834.jpg'

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

try:
    result = analyzer.analyze_image(
        dummy_image,
        physical_measurements=test_measurements,
        user_profile=user_profile
    )
    if result and result.get('success'):
        body_fat = result.get('body_fat_percentage', 'Not found')
        muscle_mass = result.get('muscle_mass_percentage', 'Not found')
        bmr = result.get('bmr_estimated', 'Not found')
        success = result.get('success')
        print(f'Body fat: {body_fat}%')
        print(f'Muscle mass: {muscle_mass}%')
        print(f'BMR: {bmr}')
        print(f'Success: {success}')
    else:
        print(f'Analysis failed: {result}')
        if result:
            error = result.get('error', 'Unknown error')
            print(f'Error: {error}')
except Exception as e:
    print(f'Error in image analysis: {e}')
    import traceback
    traceback.print_exc()
