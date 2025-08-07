#!/usr/bin/env python3
"""
Test script to validate body fat calculation for regular (non-athletic) individuals.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

def test_regular_person():
    """Test body fat calculation with average person measurements."""
    
    # Create analyzer
    analyzer = BodyCompositionAnalyzer()
    
    # Average adult male measurements
    regular_measurements = {
        'shoulder_width': 42,    # Average shoulders
        'waist_width': 85,       # Average waist
        'hip_width': 95,         # Average hips  
        'neck_circumference': 38, # Average neck
        'arm_circumference': 32,  # Average arms
        'thigh_circumference': 55, # Average legs
        'chest_circumference': 100, # Average chest
        'body_height': 175,      # Average height
        'height_cm': 175
    }
    
    # Average person stats
    age = 35
    gender = 'male'
    weight_kg = 80  # Average weight
    height_cm = 175
    
    print("=== Testing Regular Person Body Fat Calculation ===")
    print(f"Age: {age}")
    print(f"Weight: {weight_kg} kg")
    print(f"Height: {height_cm} cm")
    print(f"BMI: {weight_kg / ((height_cm / 100) ** 2):.1f}")
    print()
    
    print("Measurements:")
    for key, value in regular_measurements.items():
        print(f"  {key}: {value} cm")
    print()
    
    # Test the enhanced body fat calculation
    body_fat = analyzer._calculate_body_fat_enhanced(
        regular_measurements, age, gender, weight_kg, height_cm
    )
    
    print(f"Calculated Body Fat: {body_fat:.1f}%")
    
    # Test athletic detection
    bmi = weight_kg / ((height_cm / 100) ** 2)
    is_athletic = analyzer._detect_athletic_build(regular_measurements, bmi, weight_kg, height_cm)
    print(f"Athletic Build Detected: {is_athletic}")
    
    # Expected vs calculated
    print(f"\n=== Results Assessment ===")
    expected_range = "15-20%"  # Healthy range for average adult male
    print(f"Expected body fat range for average adult male: {expected_range}")
    print(f"Calculated body fat: {body_fat:.1f}%")
    
    if 12 <= body_fat <= 22:
        print("✅ EXCELLENT: Result is within realistic range for average adult")
    elif 8 <= body_fat <= 25:
        print("⚠️  ACCEPTABLE: Result is reasonable")
    else:
        print("❌ POOR: Result is unrealistic")

if __name__ == "__main__":
    test_regular_person()
