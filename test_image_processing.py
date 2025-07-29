#!/usr/bin/env python3
"""
Simple test to verify image processing pipeline works.
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def test_image_processing():
    """Test the image processing pipeline similar to what the app does."""
    print("Testing image processing pipeline...")
    
    # Create a sample body photo-like image
    height, width = 800, 600
    test_image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw a more realistic human figure
    center_x = width // 2
    
    # Head
    cv2.ellipse(test_image, (center_x, 120), (40, 55), 0, 0, 360, (220, 180, 140), -1)  # Skin tone
    
    # Neck
    cv2.rectangle(test_image, (center_x-15, 175), (center_x+15, 200), (220, 180, 140), -1)
    
    # Torso (shirt)
    cv2.rectangle(test_image, (center_x-80, 200), (center_x+80, 500), (100, 100, 200), -1)  # Blue shirt
    
    # Arms
    cv2.rectangle(test_image, (center_x-120, 220), (center_x-80, 420), (220, 180, 140), -1)  # Left arm
    cv2.rectangle(test_image, (center_x+80, 220), (center_x+120, 420), (220, 180, 140), -1)   # Right arm
    
    # Legs (pants)
    cv2.rectangle(test_image, (center_x-70, 500), (center_x-20, 750), (50, 50, 50), -1)   # Left leg
    cv2.rectangle(test_image, (center_x+20, 500), (center_x+70, 750), (50, 50, 50), -1)   # Right leg
    
    # Add some noise/texture to make it more realistic
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    print("✅ Test image created (realistic human figure)")
    
    # Test MediaPipe processing
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Test different image formats and preprocessing
    formats_to_test = [
        ("Original BGR", test_image),
        ("RGB", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)),
        ("Enhanced contrast", cv2.convertScaleAbs(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), alpha=1.2, beta=10)),
    ]
    
    for format_name, img in formats_to_test:
        print(f"\nTesting {format_name}...")
        results = pose.process(img)
        
        if results.pose_landmarks:
            print(f"   ✅ Pose detected! ({len(results.pose_landmarks.landmark)} landmarks)")
            
            # Test measurement calculation
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            nose = landmarks[0]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Calculate measurements
            shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * width
            avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
            body_height_px = abs(nose.y - avg_ankle_y) * height
            
            print(f"   • Shoulder width: {shoulder_width_px:.1f} px")
            print(f"   • Body height: {body_height_px:.1f} px")
            print(f"   • Aspect ratio: {shoulder_width_px/body_height_px:.3f}")
            
            # Test confidence
            visibilities = [landmarks[i].visibility for i in [0, 11, 12, 23, 24, 27, 28]]
            avg_visibility = np.mean(visibilities)
            print(f"   • Average visibility: {avg_visibility:.3f}")
            
            break
        else:
            print(f"   ❌ No pose detected")
    
    pose.close()
    print("\n✅ Image processing test completed!")

if __name__ == "__main__":
    test_image_processing()
