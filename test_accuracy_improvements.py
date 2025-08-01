#!/usr/bin/env python3
"""
Test script to validate improved body composition calculation accuracy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

def test_body_fat_accuracy():
    """Test improved body fat calculation methods."""
    analyzer = BodyCompositionAnalyzer()
    
    # Test case 1: Athletic male
    test_cases = [
        {
            "name": "Athletic Male, 25 years",
            "measurements": {
                'waist_width': 80,  # 80cm waist
                'neck_circumference': 38,  # 38cm neck
                'hip_width': 95,  # 95cm hips
                'height_cm': 180
            },
            "age": 25,
            "gender": "male",
            "weight_kg": 75,
            "expected_range": (8, 15)  # Expected body fat range for athletic male
        },
        {
            "name": "Average Female, 35 years",
            "measurements": {
                'waist_width': 75,  # 75cm waist
                'neck_circumference': 32,  # 32cm neck
                'hip_width': 95,  # 95cm hips
                'height_cm': 165
            },
            "age": 35,
            "gender": "female",
            "weight_kg": 60,
            "expected_range": (18, 28)  # Expected body fat range for average female
        },
        {
            "name": "Elderly Male, 70 years",
            "measurements": {
                'waist_width': 95,  # 95cm waist
                'neck_circumference': 40,  # 40cm neck
                'hip_width': 100,  # 100cm hips
                'height_cm': 175
            },
            "age": 70,
            "gender": "male",
            "weight_kg": 80,
            "expected_range": (15, 25)  # Expected body fat range for elderly male
        }
    ]
    
    print("=== Body Fat Calculation Test ===")
    for test_case in test_cases:
        try:
            # Test enhanced body fat calculation
            body_fat = analyzer._calculate_body_fat_enhanced(
                test_case["measurements"],
                test_case["age"],
                test_case["gender"],
                test_case["weight_kg"],
                test_case["measurements"]["height_cm"]
            )
            
            expected_min, expected_max = test_case["expected_range"]
            is_within_range = expected_min <= body_fat <= expected_max
            
            print(f"\n{test_case['name']}:")
            print(f"  Calculated Body Fat: {body_fat:.1f}%")
            print(f"  Expected Range: {expected_min}-{expected_max}%")
            print(f"  Result: {'✓ PASS' if is_within_range else '✗ FAIL'}")
            
            # Test individual methods for comparison
            navy_bf = analyzer._calculate_navy_body_fat(
                test_case["measurements"]["waist_width"],
                test_case["measurements"]["neck_circumference"],
                test_case["measurements"]["hip_width"],
                test_case["measurements"]["height_cm"],
                test_case["gender"]
            )
            
            gallagher_bf = analyzer._calculate_gallagher_body_fat(
                test_case["weight_kg"] / ((test_case["measurements"]["height_cm"] / 100) ** 2),
                test_case["age"],
                test_case["gender"]
            )
            
            print(f"  Navy Method: {navy_bf:.1f}%")
            print(f"  Gallagher Method: {gallagher_bf:.1f}%")
            
        except Exception as e:
            print(f"\n{test_case['name']}: ERROR - {e}")

def test_muscle_mass_accuracy():
    """Test improved muscle mass calculation methods."""
    analyzer = BodyCompositionAnalyzer()
    
    test_cases = [
        {
            "name": "Athletic Male, 25 years",
            "measurements": {
                'arm_circumference': 32,  # 32cm arm
                'thigh_circumference': 55,  # 55cm thigh
                'height_cm': 180
            },
            "age": 25,
            "gender": "male",
            "weight_kg": 75,
            "body_fat": 10.0,
            "expected_range": (40, 55)  # Expected muscle mass range
        },
        {
            "name": "Average Female, 35 years",
            "measurements": {
                'arm_circumference': 26,  # 26cm arm
                'thigh_circumference': 50,  # 50cm thigh
                'height_cm': 165
            },
            "age": 35,
            "gender": "female",
            "weight_kg": 60,
            "body_fat": 22.0,
            "expected_range": (25, 40)  # Expected muscle mass range
        }
    ]
    
    print("\n\n=== Muscle Mass Calculation Test ===")
    for test_case in test_cases:
        try:
            # Test enhanced muscle mass calculation
            muscle_mass = analyzer._calculate_muscle_mass_enhanced(
                test_case["measurements"],
                test_case["age"],
                test_case["gender"],
                test_case["weight_kg"],
                test_case["measurements"]["height_cm"],
                test_case["body_fat"]
            )
            
            expected_min, expected_max = test_case["expected_range"]
            is_within_range = expected_min <= muscle_mass <= expected_max
            
            print(f"\n{test_case['name']}:")
            print(f"  Calculated Muscle Mass: {muscle_mass:.1f}%")
            print(f"  Expected Range: {expected_min}-{expected_max}%")
            print(f"  Result: {'✓ PASS' if is_within_range else '✗ FAIL'}")
            
            # Test individual methods for comparison
            lee_mm = analyzer._calculate_lee_muscle_mass(
                test_case["measurements"],
                test_case["measurements"]["height_cm"],
                test_case["gender"]
            )
            
            janssen_mm = analyzer._calculate_janssen_muscle_mass(
                test_case["weight_kg"],
                test_case["measurements"]["height_cm"],
                test_case["age"],
                test_case["gender"],
                test_case["body_fat"]
            )
            
            print(f"  Lee Method: {lee_mm:.1f}%")
            print(f"  Janssen Method: {janssen_mm:.1f}%")
            
        except Exception as e:
            print(f"\n{test_case['name']}: ERROR - {e}")

def test_bmr_accuracy():
    """Test improved BMR calculation methods."""
    analyzer = BodyCompositionAnalyzer()
    
    test_cases = [
        {
            "name": "Athletic Male, 25 years",
            "weight_kg": 75,
            "height_cm": 180,
            "age": 25,
            "gender": "male",
            "muscle_mass": 45.0,
            "expected_range": (1800, 2200)  # Expected BMR range
        },
        {
            "name": "Average Female, 35 years",
            "weight_kg": 60,
            "height_cm": 165,
            "age": 35,
            "gender": "female",
            "muscle_mass": 30.0,
            "expected_range": (1300, 1600)  # Expected BMR range
        }
    ]
    
    print("\n\n=== BMR Calculation Test ===")
    for test_case in test_cases:
        try:
            # Test enhanced BMR calculation
            bmr = analyzer._estimate_bmr_enhanced(
                test_case["weight_kg"],
                test_case["height_cm"],
                test_case["age"],
                test_case["gender"],
                test_case["muscle_mass"]
            )
            
            expected_min, expected_max = test_case["expected_range"]
            is_within_range = expected_min <= bmr <= expected_max
            
            print(f"\n{test_case['name']}:")
            print(f"  Calculated BMR: {bmr} kcal/day")
            print(f"  Expected Range: {expected_min}-{expected_max} kcal/day")
            print(f"  Result: {'✓ PASS' if is_within_range else '✗ FAIL'}")
            
        except Exception as e:
            print(f"\n{test_case['name']}: ERROR - {e}")

if __name__ == "__main__":
    print("Testing Enhanced Body Composition Calculation Accuracy")
    print("=" * 60)
    
    test_body_fat_accuracy()
    test_muscle_mass_accuracy() 
    test_bmr_accuracy()
    
    print("\n" + "=" * 60)
    print("Test completed! Check results above for accuracy validation.")
