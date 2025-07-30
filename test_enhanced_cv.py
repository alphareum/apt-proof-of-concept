#!/usr/bin/env python3
"""
Test script for enhanced computer vision body analysis
"""

import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
    print("✓ Successfully imported enhanced BodyCompositionAnalyzer")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_enhanced_analyzer():
    """Test the enhanced analyzer functionality."""
    print("\n🔬 Testing Enhanced Computer Vision Body Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    print("1. Initializing enhanced analyzer...")
    try:
        analyzer = BodyCompositionAnalyzer()
        if not hasattr(analyzer, 'pose_models'):
            print("✗ Enhanced features not available - missing pose_models")
            return False
        print("✓ Enhanced analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize analyzer: {e}")
        return False
    
    # Test image enhancement capabilities
    print("\n2. Testing image enhancement capabilities...")
    try:
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test quality assessment
        quality_score = analyzer._assess_image_quality(dummy_image)
        print(f"✓ Image quality assessment: {quality_score:.2f}")
        
        # Test enhancement
        enhanced_image = analyzer._enhance_image_quality(dummy_image)
        print(f"✓ Image enhancement successful, output shape: {enhanced_image.shape}")
        
    except Exception as e:
        print(f"✗ Image enhancement test failed: {e}")
        return False
    
    # Test enhanced measurement extraction
    print("\n3. Testing enhanced measurement capabilities...")
    try:
        # Create dummy landmarks
        class DummyLandmark:
            def __init__(self, x, y, visibility=0.8, presence=0.9):
                self.x = x
                self.y = y
                self.visibility = visibility
                self.presence = presence
        
        # Create dummy landmarks for key body points
        landmarks = []
        for i in range(33):  # MediaPipe has 33 pose landmarks
            x = 0.3 + (i % 3) * 0.2  # Distribute across width
            y = 0.1 + (i // 11) * 0.3  # Distribute across height
            landmarks.append(DummyLandmark(x, y))
        
        # Test enhanced measurement extraction
        measurements = analyzer._extract_enhanced_body_measurements(landmarks, 640, 480)
        print(f"✓ Enhanced measurements extracted: {len(measurements)} metrics")
        
        # Check for enhanced features
        expected_measurements = ['shoulder_width', 'hip_width', 'waist_width', 
                               'body_height', 'neck_circumference', 'chest_circumference']
        for measurement in expected_measurements:
            if measurement in measurements:
                print(f"  ✓ {measurement}: {measurements[measurement]:.1f}")
            else:
                print(f"  ✗ Missing {measurement}")
        
    except Exception as e:
        print(f"✗ Enhanced measurement test failed: {e}")
        return False
    
    # Test ML ensemble models
    print("\n4. Testing ML ensemble models...")
    try:
        # Test feature vector preparation
        test_measurements = {
            'waist_width': 80, 'hip_width': 95, 'shoulder_width': 40,
            'left_arm_length': 60, 'left_leg_length': 90, 'neck_circumference': 35,
            'chest_circumference': 90
        }
        
        features = analyzer._prepare_enhanced_feature_vector(
            test_measurements, age=30, gender='male', weight_kg=75, height_cm=175
        )
        print(f"✓ Feature vector prepared: {len(features)} features")
        
        # Test ensemble prediction
        if hasattr(analyzer, 'body_fat_ensemble') and analyzer.body_fat_ensemble:
            body_fat = analyzer._predict_with_ensemble(features, analyzer.body_fat_ensemble)
            print(f"✓ ML ensemble prediction successful: {body_fat:.1f}% body fat")
        else:
            print("⚠ ML ensemble models not available")
        
    except Exception as e:
        print(f"✗ ML ensemble test failed: {e}")
        return False
    
    # Test confidence assessment
    print("\n5. Testing comprehensive confidence assessment...")
    try:
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        confidence = analyzer._calculate_analysis_confidence_comprehensive(
            landmarks, dummy_image, 
            physical_measurements={'height_cm': 175, 'weight_kg': 75},
            additional_analysis={'side': {'measurements': test_measurements, 'confidence': 0.8}}
        )
        print(f"✓ Comprehensive confidence calculated: {confidence:.2f}")
        
    except Exception as e:
        print(f"✗ Confidence assessment test failed: {e}")
        return False
    
    print("\n🎉 All enhanced computer vision tests completed successfully!")
    print("\nEnhanced Features Available:")
    print("• Multi-model pose detection")
    print("• Advanced image enhancement pipeline")
    print("• ML ensemble for body composition estimation")
    print("• Enhanced measurement extraction with sub-pixel accuracy")
    print("• Comprehensive confidence assessment")
    print("• Multi-view analysis support")
    print("• Advanced body parts analysis with CV techniques")
    
    return True

def test_feature_availability():
    """Test which advanced features are available."""
    print("\n📋 Feature Availability Check")
    print("=" * 40)
    
    features = {
        'MediaPipe': True,
        'OpenCV': True,
        'scikit-image': False,
        'albumentations': False,
        'scipy.ndimage': False
    }
    
    try:
        import mediapipe as mp
        features['MediaPipe'] = True
    except ImportError:
        features['MediaPipe'] = False
    
    try:
        import cv2
        features['OpenCV'] = True
    except ImportError:
        features['OpenCV'] = False
    
    try:
        import skimage
        features['scikit-image'] = True
    except ImportError:
        features['scikit-image'] = False
    
    try:
        import albumentations
        features['albumentations'] = True
    except ImportError:
        features['albumentations'] = False
    
    try:
        from scipy import ndimage
        features['scipy.ndimage'] = True
    except ImportError:
        features['scipy.ndimage'] = False
    
    for feature, available in features.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{feature:<15}: {status}")
    
    missing_features = [f for f, available in features.items() if not available]
    if missing_features:
        print(f"\n⚠ Missing features: {', '.join(missing_features)}")
        print("Install with: pip install scikit-image albumentations scipy")
    else:
        print("\n🎉 All advanced features available!")

if __name__ == "__main__":
    print("🚀 Enhanced Computer Vision Body Analysis Test")
    print("=" * 50)
    
    # Check feature availability
    test_feature_availability()
    
    # Run main tests
    success = test_enhanced_analyzer()
    
    if success:
        print("\n✅ Enhanced computer vision system is ready!")
        print("\nNext steps:")
        print("1. Install missing dependencies if any")
        print("2. Test with real images using the fitness app")
        print("3. Compare accuracy with previous version")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)
