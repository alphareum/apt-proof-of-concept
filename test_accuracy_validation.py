#!/usr/bin/env python3
"""
Accuracy validation test for body composition analyzer using ground truth data.
This script validates the analyzer against real measurements from the tests/ directory.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
from apt_fitness.core.models import UserProfile, Gender

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AccuracyValidator:
    """Validates body composition analyzer accuracy against ground truth data."""
    
    def __init__(self):
        self.analyzer = BodyCompositionAnalyzer()
        self.test_data_dir = Path(__file__).parent / "tests"
        self.results = {}
        
    def load_test_case(self, test_case_dir: Path) -> Dict[str, Any]:
        """Load a test case with images and measurements."""
        test_case = {
            'name': test_case_dir.name,
            'measurements': {},
            'images': {}
        }
        
        # Load measurements
        measurements_files = list(test_case_dir.glob("measurements*.json"))
        if measurements_files:
            with open(measurements_files[0], 'r') as f:
                test_case['measurements'] = json.load(f)
        
        # Load images
        for img_file in test_case_dir.glob("*.jpg"):
            img_type = img_file.stem.replace('_' + test_case_dir.name.replace(' ', '_'), '')
            test_case['images'][img_type] = str(img_file)
        
        return test_case
    
    def calculate_accuracy_metrics(self, predicted: Dict[str, float], 
                                 ground_truth: Dict[str, str]) -> Dict[str, float]:
        """Calculate accuracy metrics between predicted and ground truth values."""
        metrics = {
            'mae': [],  # Mean Absolute Error
            'mape': [],  # Mean Absolute Percentage Error
            'rmse': [],  # Root Mean Square Error
            'exact_matches': 0,
            'within_5_percent': 0,
            'within_10_percent': 0,
            'total_comparisons': 0
        }
        
        # Measurement mapping from analyzer output to ground truth
        measurement_mappings = {
            'waist_width': 'waist_circumference_cm',
            'waist_width_cm': 'waist_circumference_cm',
            'hip_width': 'hips_circumference_cm',
            'hip_width_cm': 'hips_circumference_cm',
            'shoulder_width': 'shoulder_width_cm',
            'shoulder_width_cm': 'shoulder_width_cm',
            'chest_circumference': 'chest_circumference_cm',
            'neck_circumference': 'neck_circumference_cm',
            'thigh_circumference': 'thigh_circumference_cm',
            'arm_circumference': 'arm_length_cm',
            'body_height': 'height',
            'height_cm': 'height',
            'estimated_weight': 'weight'
        }
        
        for pred_key, gt_key in measurement_mappings.items():
            if pred_key in predicted and gt_key in ground_truth:
                try:
                    pred_val = float(predicted[pred_key])
                    gt_val = float(str(ground_truth[gt_key]).replace('_tbr', ''))
                    
                    if gt_val > 0:  # Avoid division by zero
                        # Calculate metrics
                        abs_error = abs(pred_val - gt_val)
                        percentage_error = (abs_error / gt_val) * 100
                        
                        metrics['mae'].append(abs_error)
                        metrics['mape'].append(percentage_error)
                        metrics['rmse'].append((pred_val - gt_val) ** 2)
                        metrics['total_comparisons'] += 1
                        
                        # Count accuracy levels
                        if abs_error < 0.5:
                            metrics['exact_matches'] += 1
                        if percentage_error <= 5:
                            metrics['within_5_percent'] += 1
                        if percentage_error <= 10:
                            metrics['within_10_percent'] += 1
                        
                        logger.info(f"{pred_key}: Predicted={pred_val:.1f}, Ground Truth={gt_val:.1f}, Error={abs_error:.1f}cm ({percentage_error:.1f}%)")
                
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not compare {pred_key} and {gt_key}: {e}")
        
        # Calculate final metrics
        if metrics['mae']:
            metrics['mae'] = np.mean(metrics['mae'])
            metrics['mape'] = np.mean(metrics['mape'])
            metrics['rmse'] = np.sqrt(np.mean(metrics['rmse']))
        else:
            metrics['mae'] = metrics['mape'] = metrics['rmse'] = 0
        
        if metrics['total_comparisons'] > 0:
            metrics['accuracy_5_percent'] = (metrics['within_5_percent'] / metrics['total_comparisons']) * 100
            metrics['accuracy_10_percent'] = (metrics['within_10_percent'] / metrics['total_comparisons']) * 100
            metrics['exact_accuracy'] = (metrics['exact_matches'] / metrics['total_comparisons']) * 100
        else:
            metrics['accuracy_5_percent'] = metrics['accuracy_10_percent'] = metrics['exact_accuracy'] = 0
        
        return metrics
    
    def create_user_profile(self, measurements: Dict[str, str]) -> UserProfile:
        """Create a user profile from test measurements."""
        try:
            age = int(measurements.get('age', 30))
            gender_str = measurements.get('gender', 'male').lower()
            gender = Gender.MALE if gender_str == 'male' else Gender.FEMALE
            height_cm = float(measurements.get('height', 170))
            weight_kg = float(measurements.get('weight', 70))
            
            return UserProfile(
                name=f"Test User",
                age=age,
                gender=gender,
                height_cm=height_cm,
                weight_kg=weight_kg
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Error creating user profile: {e}")
            return UserProfile(name="Default", age=30, gender=Gender.MALE)
    
    def analyze_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single test case and return results."""
        logger.info(f"Analyzing test case: {test_case['name']}")
        
        results = {
            'name': test_case['name'],
            'ground_truth': test_case['measurements'],
            'predictions': {},
            'metrics': {},
            'images_analyzed': list(test_case['images'].keys())
        }
        
        # Create user profile
        user_profile = self.create_user_profile(test_case['measurements'])
        
        # Analyze primary image (front view preferred)
        primary_image_key = None
        for key in ['front_img', 'selfie_img', 'side_img']:
            if key in test_case['images']:
                primary_image_key = key
                break
        
        if not primary_image_key:
            logger.error(f"No suitable image found for {test_case['name']}")
            return results
        
        try:
            # Load and preprocess image
            image_path = test_case['images'][primary_image_key]
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return results
            
            # Prepare physical measurements from ground truth
            physical_measurements = {}
            for key, value in test_case['measurements'].items():
                if key.endswith('_cm') or key in ['height', 'weight']:
                    try:
                        physical_measurements[key] = float(str(value).replace('_tbr', ''))
                    except ValueError:
                        continue
            
            # Prepare additional images if available
            additional_images = {}
            for img_key, img_path in test_case['images'].items():
                if img_key != primary_image_key:
                    additional_images[img_key] = img_path
            
            # Run analysis using the image path directly
            analysis_result = self.analyzer.analyze_image(
                image_data=image_path,  # Pass the file path instead of image data
                user_profile=user_profile.__dict__,
                physical_measurements=physical_measurements,
                additional_images=additional_images
            )
            
            if analysis_result and analysis_result.get('success', False):
                results['predictions'] = analysis_result.get('measurements', {})
                results['body_fat'] = analysis_result.get('body_fat_percentage', None)
                results['muscle_mass'] = analysis_result.get('muscle_mass_percentage', None)
                results['bmr'] = analysis_result.get('bmr_estimated', None)
                
                # Calculate accuracy metrics
                results['metrics'] = self.calculate_accuracy_metrics(
                    results['predictions'], 
                    test_case['measurements']
                )
            else:
                logger.error(f"Analysis failed for {test_case['name']}")
                results['error'] = "Analysis failed"
        
        except Exception as e:
            logger.error(f"Error analyzing {test_case['name']}: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run validation on all test cases."""
        logger.info("Starting accuracy validation...")
        
        validation_results = {
            'test_cases': [],
            'overall_metrics': {},
            'summary': {}
        }
        
        # Find all test case directories
        test_dirs = [d for d in self.test_data_dir.iterdir() 
                    if d.is_dir() and any(d.glob("measurements*.json"))]
        
        if not test_dirs:
            logger.error("No test cases found in tests/ directory")
            return validation_results
        
        logger.info(f"Found {len(test_dirs)} test cases")
        
        all_metrics = []
        
        for test_dir in test_dirs:
            try:
                # Load and analyze test case
                test_case = self.load_test_case(test_dir)
                result = self.analyze_test_case(test_case)
                validation_results['test_cases'].append(result)
                
                if 'metrics' in result and result['metrics'].get('total_comparisons', 0) > 0:
                    all_metrics.append(result['metrics'])
                
            except Exception as e:
                logger.error(f"Error processing test case {test_dir.name}: {e}")
        
        # Calculate overall metrics
        if all_metrics:
            overall = {
                'mae': np.mean([m['mae'] for m in all_metrics]),
                'mape': np.mean([m['mape'] for m in all_metrics]),
                'rmse': np.mean([m['rmse'] for m in all_metrics]),
                'accuracy_5_percent': np.mean([m['accuracy_5_percent'] for m in all_metrics]),
                'accuracy_10_percent': np.mean([m['accuracy_10_percent'] for m in all_metrics]),
                'exact_accuracy': np.mean([m['exact_accuracy'] for m in all_metrics]),
                'total_test_cases': len(all_metrics)
            }
            validation_results['overall_metrics'] = overall
            
            # Create summary
            validation_results['summary'] = {
                'total_test_cases': len(test_dirs),
                'successful_analyses': len(all_metrics),
                'average_accuracy_within_5_percent': f"{overall['accuracy_5_percent']:.1f}%",
                'average_accuracy_within_10_percent': f"{overall['accuracy_10_percent']:.1f}%",
                'mean_absolute_error': f"{overall['mae']:.2f}cm",
                'mean_percentage_error': f"{overall['mape']:.1f}%"
            }
        
        return validation_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "accuracy_validation_results.json"):
        """Save validation results to file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of validation results."""
        print("\n" + "="*60)
        print("BODY COMPOSITION ANALYZER ACCURACY VALIDATION")
        print("="*60)
        
        if 'summary' in results and results['summary']:
            summary = results['summary']
            print(f"Total Test Cases: {summary['total_test_cases']}")
            print(f"Successful Analyses: {summary['successful_analyses']}")
            print(f"Average Accuracy (±5%): {summary['average_accuracy_within_5_percent']}")
            print(f"Average Accuracy (±10%): {summary['average_accuracy_within_10_percent']}")
            print(f"Mean Absolute Error: {summary['mean_absolute_error']}")
            print(f"Mean Percentage Error: {summary['mean_percentage_error']}")
        
        print("\nDetailed Results by Test Case:")
        print("-"*60)
        
        for test_case in results.get('test_cases', []):
            print(f"\nTest Case: {test_case['name']}")
            if 'metrics' in test_case and test_case['metrics'].get('total_comparisons', 0) > 0:
                metrics = test_case['metrics']
                print(f"  Accuracy (±5%): {metrics['accuracy_5_percent']:.1f}%")
                print(f"  Accuracy (±10%): {metrics['accuracy_10_percent']:.1f}%")
                print(f"  MAE: {metrics['mae']:.2f}cm")
                print(f"  MAPE: {metrics['mape']:.1f}%")
                print(f"  Comparisons: {metrics['total_comparisons']}")
            elif 'error' in test_case:
                print(f"  Error: {test_case['error']}")
            else:
                print("  No valid measurements to compare")
        
        print("\n" + "="*60)


def main():
    """Main function to run accuracy validation."""
    validator = AccuracyValidator()
    
    # Run validation
    results = validator.run_validation()
    
    # Print summary
    validator.print_summary(results)
    
    # Save results
    validator.save_results(results)
    
    return results


if __name__ == "__main__":
    main()
