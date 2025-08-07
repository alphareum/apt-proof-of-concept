#!/usr/bin/env python3
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

analyzer = BodyCompositionAnalyzer()

# Test measurements with different values
test_measurements = {
    'waist_width': 80,  # 80cm waist
    'hip_width': 95,    # 95cm hips  
    'neck_circumference': 35,  # 35cm neck
    'shoulder_width': 45,  # 45cm shoulders
    'height': 175       # 175cm height
}

print("=== DEBUGGING BODY FAT CALCULATION ===")
print(f"Input measurements: {test_measurements}")

# Test the traditional method directly
try:
    body_fat_traditional = analyzer._calculate_body_fat_enhanced(
        test_measurements, 30, 'male', 75, 175
    )
    print(f'Traditional body fat result: {body_fat_traditional:.1f}%')
except Exception as e:
    print(f'Error in traditional body fat calculation: {e}')

# Test the enhanced ML method
try:
    body_fat_ml = analyzer._calculate_body_fat_enhanced_ml(
        test_measurements, 30, 'male', 75, 175
    )
    print(f'Enhanced ML body fat result: {body_fat_ml:.1f}%')
except Exception as e:
    print(f'Error in ML body fat calculation: {e}')

# Test individual methods
try:
    waist_cm = 80
    neck_cm = 35
    hip_cm = 95
    height_cm = 175
    
    # Navy method
    navy_bf = analyzer._calculate_navy_body_fat(waist_cm, neck_cm, hip_cm, height_cm, 'male')
    print(f'Navy method: {navy_bf:.1f}%')
    
    # YMCA method
    ymca_bf = analyzer._calculate_ymca_body_fat(waist_cm, 75, 'male')
    print(f'YMCA method: {ymca_bf:.1f}%')
    
    # Gallagher method
    bmi = 75 / ((175/100)**2)
    gallagher_bf = analyzer._calculate_gallagher_body_fat(bmi, 30, 'male')
    print(f'Gallagher method: {gallagher_bf:.1f}%')
    
    print(f'BMI: {bmi:.1f}')
    print(f'Waist-to-height ratio: {waist_cm/height_cm:.3f}')
    
except Exception as e:
    print(f'Error testing individual methods: {e}')

# Test with different waist measurements
print("\n=== TESTING DIFFERENT WAIST MEASUREMENTS ===")
for waist in [70, 75, 80, 85, 90, 95]:
    test_measurements['waist_width'] = waist
    try:
        body_fat = analyzer._calculate_body_fat_enhanced(
            test_measurements, 30, 'male', 75, 175
        )
        wth_ratio = waist / 175
        print(f'Waist {waist}cm (WtH: {wth_ratio:.3f}): {body_fat:.1f}% body fat')
    except Exception as e:
        print(f'Error with waist {waist}cm: {e}')
