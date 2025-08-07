#!/usr/bin/env python3
"""
Test script to validate body fat calculation improvements for athletic individuals like Arnold.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer

def test_arnold_measurements():
    """Test body fat calculation with Arnold-like measurements."""
    
    # Create analyzer
    analyzer = BodyCompositionAnalyzer()
    
    # Arnold's approximate measurements during competition (based on historical data)
    arnold_measurements = {
        'shoulder_width': 55,    # Very wide shoulders
        'waist_width': 30,       # Very narrow waist (competition ready)
        'hip_width': 42,         # Narrow hips  
        'neck_circumference': 45, # Very muscular neck
        'arm_circumference': 50,  # Massive arms
        'thigh_circumference': 68, # Massive legs
        'chest_circumference': 130, # Massive chest
        'body_height': 183,      # Arnold's height
        'height_cm': 183
    }
    
    # Arnold's stats during competition
    age = 25  # Peak competition age
    gender = 'male'
    weight_kg = 105  # Competition weight
    height_cm = 183
    
    print("=== Testing Arnold Schwarzenegger Body Fat Calculation ===")
    print(f"Age: {age}")
    print(f"Weight: {weight_kg} kg")
    print(f"Height: {height_cm} cm")
    print(f"BMI: {weight_kg / ((height_cm / 100) ** 2):.1f}")
    print()
    
    print("Measurements:")
    for key, value in arnold_measurements.items():
        print(f"  {key}: {value} cm")
    print()
    
    # Test the enhanced body fat calculation
    body_fat = analyzer._calculate_body_fat_enhanced(
        arnold_measurements, age, gender, weight_kg, height_cm
    )
    
    print(f"Calculated Body Fat: {body_fat:.1f}%")
    
    # Test athletic detection
    bmi = weight_kg / ((height_cm / 100) ** 2)
    is_athletic = analyzer._detect_athletic_build(arnold_measurements, bmi, weight_kg, height_cm)
    print(f"Athletic Build Detected: {is_athletic}")
    
    # Test individual methods
    print("\n=== Individual Method Results ===")
    
    # Convert measurements
    waist_cm = arnold_measurements['waist_width']
    neck_cm = arnold_measurements['neck_circumference'] 
    hip_cm = arnold_measurements['hip_width']
    
    # Test each method
    navy = analyzer._calculate_navy_body_fat(waist_cm, neck_cm, hip_cm, height_cm, gender)
    print(f"Navy Method: {navy:.1f}%")
    
    jp = analyzer._calculate_jackson_pollock_adapted(arnold_measurements, age, gender, weight_kg, height_cm)
    print(f"Jackson-Pollock Adapted: {jp:.1f}%")
    
    ymca = analyzer._calculate_ymca_body_fat(waist_cm, weight_kg, gender)
    print(f"YMCA Method: {ymca:.1f}%")
    
    bailey = analyzer._calculate_bailey_body_fat(bmi, waist_cm, height_cm, age, gender)
    print(f"Bailey Method: {bailey:.1f}%")
    
    gallagher = analyzer._calculate_gallagher_body_fat(bmi, age, gender)
    print(f"Gallagher Method: {gallagher:.1f}%")
    
    # Test with athletic corrections
    if is_athletic:
        print("\n=== With Athletic Corrections ===")
        navy_corrected = analyzer._apply_athletic_correction(navy, "navy")
        jp_corrected = analyzer._apply_athletic_correction(jp, "jackson_pollock")
        ymca_corrected = analyzer._apply_athletic_correction(ymca, "ymca")
        bailey_corrected = analyzer._apply_athletic_correction(bailey, "bailey")
        gallagher_corrected = analyzer._apply_athletic_correction(gallagher, "gallagher")
        
        print(f"Navy Method (corrected): {navy_corrected:.1f}%")
        print(f"Jackson-Pollock (corrected): {jp_corrected:.1f}%")
        print(f"YMCA Method (corrected): {ymca_corrected:.1f}%")
        print(f"Bailey Method (corrected): {bailey_corrected:.1f}%")
        print(f"Gallagher Method (corrected): {gallagher_corrected:.1f}%")
    
    # Test method weights
    weights = analyzer._get_method_weights(navy, jp, ymca, bailey, gallagher, age, bmi, is_athletic)
    print(f"\nMethod Weights: {[f'{w:.2f}' for w in weights]}")
    print("(Navy, Jackson-Pollock, YMCA, Bailey, Gallagher)")
    
    # Expected vs calculated
    print(f"\n=== Results Assessment ===")
    expected_range = "5-8%"  # Arnold's competition body fat was extremely low
    print(f"Expected body fat range for competition Arnold: {expected_range}")
    print(f"Calculated body fat: {body_fat:.1f}%")
    
    if 5 <= body_fat <= 10:
        print("✅ EXCELLENT: Result is within realistic range for elite bodybuilder")
    elif 10 < body_fat <= 15:
        print("⚠️  ACCEPTABLE: Result is reasonable for athletic individual")
    elif 15 < body_fat <= 25:
        print("❌ POOR: Result is too high for elite athlete")
    else:
        print("❌ VERY POOR: Result is unrealistic")

if __name__ == "__main__":
    test_arnold_measurements()
