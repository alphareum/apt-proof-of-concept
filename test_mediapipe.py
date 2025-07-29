#!/usr/bin/env python3
"""
Simple test script to verify MediaPipe pose detection is working.
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def test_mediapipe():
    """Test MediaPipe pose detection with a simple image."""
    print("Testing MediaPipe pose detection...")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Create pose detector
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ MediaPipe Pose initialized successfully")
    
    # Create a simple test image (stick figure)
    height, width = 400, 300
    test_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a simple stick figure
    # Head
    cv2.circle(test_image, (150, 50), 20, (0, 0, 0), 2)
    # Body
    cv2.line(test_image, (150, 70), (150, 250), (0, 0, 0), 3)
    # Arms
    cv2.line(test_image, (150, 120), (100, 160), (0, 0, 0), 3)  # Left arm
    cv2.line(test_image, (150, 120), (200, 160), (0, 0, 0), 3)  # Right arm
    # Legs
    cv2.line(test_image, (150, 250), (120, 350), (0, 0, 0), 3)  # Left leg
    cv2.line(test_image, (150, 250), (180, 350), (0, 0, 0), 3)  # Right leg
    
    print("✅ Test image created")
    
    # Process with MediaPipe
    results = pose.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        print("✅ Pose detected successfully!")
        print(f"   Number of landmarks: {len(results.pose_landmarks.landmark)}")
        
        # Check key landmarks
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        print(f"   Nose position: ({nose.x:.3f}, {nose.y:.3f})")
        print(f"   Left shoulder: ({left_shoulder.x:.3f}, {left_shoulder.y:.3f})")
        print(f"   Right shoulder: ({right_shoulder.x:.3f}, {right_shoulder.y:.3f})")
        
    else:
        print("❌ No pose detected in test image")
    
    # Test with a larger, more realistic dummy image
    print("\nTesting with larger dummy image...")
    
    # Create a more realistic test figure
    height, width = 600, 400
    test_image2 = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a more detailed figure
    center_x = width // 2
    
    # Head (larger)
    cv2.circle(test_image2, (center_x, 80), 35, (0, 0, 0), 3)
    
    # Neck
    cv2.line(test_image2, (center_x, 115), (center_x, 140), (0, 0, 0), 5)
    
    # Shoulders
    cv2.line(test_image2, (center_x - 60, 160), (center_x + 60, 160), (0, 0, 0), 5)
    
    # Torso
    cv2.line(test_image2, (center_x, 140), (center_x, 350), (0, 0, 0), 5)
    
    # Arms
    cv2.line(test_image2, (center_x - 60, 160), (center_x - 80, 280), (0, 0, 0), 4)  # Left arm
    cv2.line(test_image2, (center_x + 60, 160), (center_x + 80, 280), (0, 0, 0), 4)  # Right arm
    
    # Hips
    cv2.line(test_image2, (center_x - 40, 350), (center_x + 40, 350), (0, 0, 0), 5)
    
    # Legs
    cv2.line(test_image2, (center_x - 40, 350), (center_x - 50, 520), (0, 0, 0), 4)  # Left leg
    cv2.line(test_image2, (center_x + 40, 350), (center_x + 50, 520), (0, 0, 0), 4)  # Right leg
    
    # Process with MediaPipe
    results2 = pose.process(cv2.cvtColor(test_image2, cv2.COLOR_BGR2RGB))
    
    if results2.pose_landmarks:
        print("✅ Pose detected in larger image!")
        
        # Calculate some basic measurements
        landmarks = results2.pose_landmarks.landmark
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Shoulder width in pixels
        shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * width
        
        # Body height in pixels  
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        body_height_px = abs(nose.y - avg_ankle_y) * height
        
        print(f"   Shoulder width: {shoulder_width_px:.1f} pixels")
        print(f"   Body height: {body_height_px:.1f} pixels")
        
    else:
        print("❌ No pose detected in larger image")
    
    pose.close()
    print("\n✅ MediaPipe test completed!")

if __name__ == "__main__":
    test_mediapipe()
