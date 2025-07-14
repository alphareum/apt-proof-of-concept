"""
Body Composition Analyzer using Computer Vision
Analyzes body composition from images using deep learning models
"""

import cv2
import numpy as np
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import asdict

# Scientific libraries
import mediapipe as mp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Database integration
from database import get_database, BodyCompositionAnalysis, BodyPartMeasurement

logger = logging.getLogger(__name__)

class BodyCompositionAnalyzer:
    """Analyze body composition from images using computer vision."""
    
    def __init__(self):
        """Initialize the body composition analyzer."""
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # Initialize pose estimation
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )
        
        # Initialize segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # General model
        )
        
        # Body composition estimation models (simplified)
        self.body_fat_model = None
        self.muscle_mass_model = None
        self._init_estimation_models()
        
        # Body part landmarks mapping
        self.body_parts = {
            'chest': [11, 12, 23, 24],  # shoulders and hips
            'waist': [23, 24],  # hip landmarks
            'arms': [11, 13, 15, 12, 14, 16],  # shoulder, elbow, wrist
            'thighs': [23, 25, 27, 24, 26, 28],  # hip, knee, ankle
            'neck': [0, 11, 12]  # nose and shoulders
        }
        
        self.db = get_database()
    
    def _init_estimation_models(self):
        """Initialize machine learning models for body composition estimation."""
        # Simplified models - in production, these would be trained on large datasets
        self.body_fat_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.muscle_mass_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Train with synthetic data (in production, use real anthropometric data)
        self._train_synthetic_models()
    
    def _train_synthetic_models(self):
        """Train models with synthetic anthropometric data."""
        # Generate synthetic training data based on anthropometric research
        np.random.seed(42)
        n_samples = 1000
        
        # Features: body measurements and ratios
        waist_to_height = np.random.normal(0.5, 0.1, n_samples)
        waist_to_hip = np.random.normal(0.85, 0.15, n_samples)
        shoulder_to_waist = np.random.normal(1.3, 0.2, n_samples)
        arm_to_height = np.random.normal(0.44, 0.05, n_samples)
        leg_to_height = np.random.normal(0.5, 0.05, n_samples)
        body_symmetry = np.random.normal(0.95, 0.05, n_samples)
        
        features = np.column_stack([
            waist_to_height, waist_to_hip, shoulder_to_waist,
            arm_to_height, leg_to_height, body_symmetry
        ])
        
        # Target variables based on anthropometric correlations
        body_fat = (waist_to_height * 30 + waist_to_hip * 10 + 
                   np.random.normal(0, 3, n_samples)).clip(5, 35)
        muscle_mass = (50 - body_fat * 0.8 + shoulder_to_waist * 5 + 
                      np.random.normal(0, 2, n_samples)).clip(25, 55)
        
        # Train models
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        self.body_fat_model.fit(features_scaled, body_fat)
        self.muscle_mass_model.fit(features_scaled, muscle_mass)
    
    def analyze_image(self, image_path: str, user_id: str, 
                     additional_images: Dict[str, str] = None) -> Dict[str, Any]:
        """Analyze body composition from image(s)."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract pose landmarks
            pose_results = self.pose.process(image_rgb)
            
            if not pose_results.pose_landmarks:
                return {"error": "No pose detected in image", "confidence": 0.0}
            
            # Extract body segmentation
            segmentation_results = self.segmentation.process(image_rgb)
            
            # Analyze body composition
            analysis_results = self._analyze_body_composition(
                pose_results, segmentation_results, image_rgb
            )
            
            # Generate processed image with annotations
            processed_image_path = self._create_processed_image(
                image, pose_results, analysis_results, image_path
            )
            
            # Create analysis record
            analysis_id = hashlib.md5(
                f"{user_id}_{datetime.now()}_{image_path}".encode()
            ).hexdigest()
            
            composition_analysis = BodyCompositionAnalysis(
                analysis_id=analysis_id,
                user_id=user_id,
                image_path=image_path,
                analysis_date=datetime.now(),
                body_fat_percentage=analysis_results["body_fat_percentage"],
                muscle_mass_percentage=analysis_results["muscle_mass_percentage"],
                visceral_fat_level=analysis_results["visceral_fat_level"],
                bmr_estimated=analysis_results["bmr_estimated"],
                body_shape_classification=analysis_results["body_shape"],
                confidence_score=analysis_results["confidence"],
                front_image_path=additional_images.get("front") if additional_images else None,
                side_image_path=additional_images.get("side") if additional_images else None,
                processed_image_path=processed_image_path,
                body_measurements=analysis_results["measurements"],
                composition_breakdown=analysis_results["breakdown"]
            )
            
            # Save to database
            success = self.db.save_body_composition_analysis(composition_analysis)
            
            if success:
                # Save body part measurements
                self._save_body_part_measurements(analysis_id, analysis_results["body_parts"])
                
                return {
                    "analysis_id": analysis_id,
                    "success": True,
                    **analysis_results,
                    "processed_image_path": processed_image_path
                }
            else:
                return {"error": "Failed to save analysis", "success": False}
                
        except Exception as e:
            logger.error(f"Error analyzing body composition: {e}")
            return {"error": str(e), "success": False}
    
    def _analyze_body_composition(self, pose_results, segmentation_results, 
                                image: np.ndarray) -> Dict[str, Any]:
        """Analyze body composition from pose and segmentation data."""
        landmarks = pose_results.pose_landmarks.landmark
        height, width = image.shape[:2]
        
        # Extract key body measurements
        measurements = self._extract_body_measurements(landmarks, width, height)
        
        # Calculate body ratios
        ratios = self._calculate_body_ratios(measurements)
        
        # Estimate body composition using ML models
        features = np.array([
            ratios["waist_to_height"],
            ratios["waist_to_hip"],
            ratios["shoulder_to_waist"],
            ratios["arm_to_height"],
            ratios["leg_to_height"],
            ratios["body_symmetry"]
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        
        body_fat_percentage = float(self.body_fat_model.predict(features_scaled)[0])
        muscle_mass_percentage = float(self.muscle_mass_model.predict(features_scaled)[0])
        
        # Calculate additional metrics
        visceral_fat_level = self._estimate_visceral_fat(ratios, body_fat_percentage)
        bmr = self._estimate_bmr(measurements, body_fat_percentage, muscle_mass_percentage)
        body_shape = self._classify_body_shape(ratios)
        
        # Analyze body segmentation for fat distribution
        body_parts_analysis = self._analyze_body_parts(landmarks, segmentation_results.segmentation_mask)
        
        # Calculate confidence based on pose visibility and image quality
        confidence = self._calculate_analysis_confidence(landmarks, image)
        
        return {
            "body_fat_percentage": round(body_fat_percentage, 1),
            "muscle_mass_percentage": round(muscle_mass_percentage, 1),
            "visceral_fat_level": visceral_fat_level,
            "bmr_estimated": bmr,
            "body_shape": body_shape,
            "confidence": confidence,
            "measurements": measurements,
            "ratios": ratios,
            "body_parts": body_parts_analysis,
            "breakdown": {
                "fat_mass_kg": round((body_fat_percentage / 100) * measurements.get("estimated_weight", 70), 1),
                "muscle_mass_kg": round((muscle_mass_percentage / 100) * measurements.get("estimated_weight", 70), 1),
                "bone_mass_kg": round(measurements.get("estimated_weight", 70) * 0.15, 1),
                "water_percentage": round(100 - body_fat_percentage - muscle_mass_percentage, 1)
            }
        }
    
    def _extract_body_measurements(self, landmarks, width: int, height: int) -> Dict[str, float]:
        """Extract body measurements from pose landmarks."""
        measurements = {}
        
        try:
            # Convert normalized coordinates to pixels
            def get_point(landmark_idx):
                return (landmarks[landmark_idx].x * width, landmarks[landmark_idx].y * height)
            
            # Shoulder width
            left_shoulder = get_point(11)
            right_shoulder = get_point(12)
            measurements["shoulder_width"] = np.linalg.norm(
                np.array(left_shoulder) - np.array(right_shoulder)
            )
            
            # Hip width
            left_hip = get_point(23)
            right_hip = get_point(24)
            measurements["hip_width"] = np.linalg.norm(
                np.array(left_hip) - np.array(right_hip)
            )
            
            # Waist estimation (midpoint between lowest rib and hip)
            measurements["waist_width"] = (measurements["shoulder_width"] + measurements["hip_width"]) / 2
            
            # Body height (head to ankle)
            nose = get_point(0)
            left_ankle = get_point(27)
            right_ankle = get_point(28)
            avg_ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
            measurements["body_height"] = abs(nose[1] - avg_ankle[1])
            
            # Arm length (shoulder to wrist)
            left_wrist = get_point(15)
            measurements["left_arm_length"] = np.linalg.norm(
                np.array(left_shoulder) - np.array(left_wrist)
            )
            
            # Leg length (hip to ankle)
            measurements["left_leg_length"] = np.linalg.norm(
                np.array(left_hip) - np.array(left_ankle)
            )
            
            # Estimate weight based on body dimensions (simplified)
            body_volume_ratio = (measurements["shoulder_width"] * measurements["hip_width"] * 
                               measurements["body_height"]) / (width * height * height)
            measurements["estimated_weight"] = max(40, min(120, body_volume_ratio * 70000))
            
        except Exception as e:
            logger.error(f"Error extracting measurements: {e}")
            # Default measurements if extraction fails
            measurements = {
                "shoulder_width": width * 0.2,
                "hip_width": width * 0.18,
                "waist_width": width * 0.16,
                "body_height": height * 0.8,
                "left_arm_length": height * 0.35,
                "left_leg_length": height * 0.45,
                "estimated_weight": 70
            }
        
        return measurements
    
    def _calculate_body_ratios(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Calculate important body ratios for composition analysis."""
        ratios = {}
        
        try:
            ratios["waist_to_height"] = measurements["waist_width"] / measurements["body_height"]
            ratios["waist_to_hip"] = measurements["waist_width"] / measurements["hip_width"]
            ratios["shoulder_to_waist"] = measurements["shoulder_width"] / measurements["waist_width"]
            ratios["arm_to_height"] = measurements["left_arm_length"] / measurements["body_height"]
            ratios["leg_to_height"] = measurements["left_leg_length"] / measurements["body_height"]
            ratios["body_symmetry"] = 0.95  # Simplified - would need bilateral analysis
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            # Default ratios
            ratios = {
                "waist_to_height": 0.5,
                "waist_to_hip": 0.85,
                "shoulder_to_waist": 1.3,
                "arm_to_height": 0.44,
                "leg_to_height": 0.5,
                "body_symmetry": 0.95
            }
        
        return ratios
    
    def _estimate_visceral_fat(self, ratios: Dict[str, float], body_fat: float) -> int:
        """Estimate visceral fat level (1-20 scale)."""
        waist_to_height = ratios.get("waist_to_height", 0.5)
        
        # Simplified estimation based on waist-to-height ratio and body fat
        visceral_estimate = (waist_to_height - 0.4) * 30 + body_fat * 0.3
        return max(1, min(20, int(visceral_estimate)))
    
    def _estimate_bmr(self, measurements: Dict[str, float], body_fat: float, 
                     muscle_mass: float) -> int:
        """Estimate Basal Metabolic Rate."""
        weight = measurements.get("estimated_weight", 70)
        height_cm = measurements.get("body_height", 170)  # Convert to cm if needed
        
        # Katch-McArdle Formula (using lean body mass)
        lean_mass = weight * (muscle_mass / 100)
        bmr = 370 + (21.6 * lean_mass)
        
        return int(bmr)
    
    def _classify_body_shape(self, ratios: Dict[str, float]) -> str:
        """Classify body shape based on ratios."""
        shoulder_to_waist = ratios.get("shoulder_to_waist", 1.3)
        waist_to_hip = ratios.get("waist_to_hip", 0.85)
        
        if shoulder_to_waist > 1.4 and waist_to_hip < 0.8:
            return "Athletic/V-shape"
        elif shoulder_to_waist > 1.3 and waist_to_hip < 0.85:
            return "Inverted Triangle"
        elif 0.85 <= waist_to_hip <= 0.95:
            return "Rectangle"
        elif waist_to_hip > 0.95:
            return "Apple"
        else:
            return "Pear"
    
    def _analyze_body_parts(self, landmarks, segmentation_mask) -> Dict[str, Dict[str, float]]:
        """Analyze individual body parts."""
        body_parts_data = {}
        
        for part_name, landmark_indices in self.body_parts.items():
            try:
                # Calculate area and muscle definition for each body part
                part_data = {
                    "area_percentage": self._calculate_part_area(landmark_indices, segmentation_mask),
                    "muscle_definition": self._calculate_muscle_definition(landmark_indices, landmarks),
                    "fat_distribution": self._calculate_fat_distribution(landmark_indices, landmarks),
                    "symmetry_score": self._calculate_symmetry(landmark_indices, landmarks)
                }
                body_parts_data[part_name] = part_data
                
            except Exception as e:
                logger.error(f"Error analyzing {part_name}: {e}")
                body_parts_data[part_name] = {
                    "area_percentage": 5.0,
                    "muscle_definition": 0.5,
                    "fat_distribution": 0.5,
                    "symmetry_score": 0.9
                }
        
        return body_parts_data
    
    def _calculate_part_area(self, landmark_indices: List[int], 
                           segmentation_mask) -> float:
        """Calculate relative area of body part."""
        # Simplified area calculation
        return np.random.uniform(3, 8)  # Placeholder
    
    def _calculate_muscle_definition(self, landmark_indices: List[int], 
                                   landmarks) -> float:
        """Calculate muscle definition score (0-1)."""
        # Simplified muscle definition based on landmark visibility
        visibility_scores = [landmarks[i].visibility for i in landmark_indices]
        return np.mean(visibility_scores)
    
    def _calculate_fat_distribution(self, landmark_indices: List[int], 
                                  landmarks) -> float:
        """Calculate fat distribution score (0-1)."""
        # Simplified fat distribution assessment
        return np.random.uniform(0.3, 0.8)  # Placeholder
    
    def _calculate_symmetry(self, landmark_indices: List[int], landmarks) -> float:
        """Calculate body part symmetry score (0-1)."""
        # Simplified symmetry calculation
        return np.random.uniform(0.85, 0.98)  # Placeholder
    
    def _calculate_analysis_confidence(self, landmarks, image: np.ndarray) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence on landmark visibility and image quality
        visibility_scores = [landmark.visibility for landmark in landmarks]
        avg_visibility = np.mean(visibility_scores)
        
        # Image quality assessment (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)  # Normalize
        
        # Combine factors
        confidence = (avg_visibility * 0.6 + sharpness_score * 0.4)
        return round(confidence, 2)
    
    def _create_processed_image(self, image: np.ndarray, pose_results, 
                              analysis_results: Dict[str, Any], 
                              original_path: str) -> str:
        """Create annotated image with analysis results."""
        try:
            # Create output directory
            output_dir = Path("processed_images")
            output_dir.mkdir(exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"body_analysis_{timestamp}.jpg"
            output_path = output_dir / filename
            
            # Draw pose landmarks
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, 
                pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
            
            # Add analysis text
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            
            analysis_text = [
                f"Body Fat: {analysis_results['body_fat_percentage']}%",
                f"Muscle Mass: {analysis_results['muscle_mass_percentage']}%",
                f"Body Shape: {analysis_results['body_shape']}",
                f"BMR: {analysis_results['bmr_estimated']} cal/day",
                f"Confidence: {analysis_results['confidence']:.2f}"
            ]
            
            for i, text in enumerate(analysis_text):
                cv2.putText(annotated_image, text, (10, y_offset + i * 25), 
                          font, 0.6, (0, 255, 0), 2)
            
            # Save processed image
            cv2.imwrite(str(output_path), annotated_image)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating processed image: {e}")
            return ""
    
    def _save_body_part_measurements(self, analysis_id: str, 
                                   body_parts_data: Dict[str, Dict[str, float]]):
        """Save individual body part measurements."""
        for part_name, part_data in body_parts_data.items():
            measurement_id = hashlib.md5(
                f"{analysis_id}_{part_name}".encode()
            ).hexdigest()
            
            measurement = BodyPartMeasurement(
                measurement_id=measurement_id,
                analysis_id=analysis_id,
                body_part=part_name,
                circumference_cm=0.0,  # Would need additional calculation
                area_percentage=part_data["area_percentage"],
                muscle_definition_score=part_data["muscle_definition"],
                fat_distribution_score=part_data["fat_distribution"],
                symmetry_score=part_data["symmetry_score"]
            )
            
            self.db.save_body_part_measurement(measurement)
    
    def compare_analyses(self, user_id: str, analysis_id1: str, 
                        analysis_id2: str) -> Dict[str, Any]:
        """Compare two body composition analyses."""
        try:
            analyses = self.db.get_body_composition_history(user_id, 365)
            
            analysis1 = next((a for a in analyses if a["analysis_id"] == analysis_id1), None)
            analysis2 = next((a for a in analyses if a["analysis_id"] == analysis_id2), None)
            
            if not analysis1 or not analysis2:
                return {"error": "One or both analyses not found"}
            
            comparison = {
                "analysis1_date": analysis1["analysis_date"],
                "analysis2_date": analysis2["analysis_date"],
                "body_fat_change": analysis2["body_fat_percentage"] - analysis1["body_fat_percentage"],
                "muscle_mass_change": analysis2["muscle_mass_percentage"] - analysis1["muscle_mass_percentage"],
                "visceral_fat_change": analysis2["visceral_fat_level"] - analysis1["visceral_fat_level"],
                "bmr_change": analysis2["bmr_estimated"] - analysis1["bmr_estimated"],
                "body_shape_change": {
                    "from": analysis1["body_shape_classification"],
                    "to": analysis2["body_shape_classification"]
                },
                "time_difference_days": (
                    datetime.fromisoformat(analysis2["analysis_date"]) - 
                    datetime.fromisoformat(analysis1["analysis_date"])
                ).days
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing analyses: {e}")
            return {"error": str(e)}

# Singleton instance
_analyzer_instance = None

def get_body_analyzer() -> BodyCompositionAnalyzer:
    """Get singleton body composition analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = BodyCompositionAnalyzer()
    return _analyzer_instance
