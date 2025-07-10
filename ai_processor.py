import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from config import get_config

config = get_config()

class PoseAnalyzer:
    """Real-time pose analysis using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_poses_from_video(self, video_path: str) -> Dict:
        """Extract pose landmarks from video file"""
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        poses_data = []
        frame_number = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            timestamp = frame_number / fps
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = []
                visibility_scores = []
                
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                    visibility_scores.append(landmark.visibility)
                
                poses_data.append({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'landmarks': landmarks,
                    'visibility_scores': visibility_scores
                })
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        
        processing_time = time.time() - start_time
        
        return {
            'poses': poses_data,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': duration,
                'processing_time_seconds': processing_time
            }
        }

class FormAnalyzer:
    """Analyze exercise form from pose data"""
    
    def __init__(self):
        # Key pose landmarks for exercise analysis
        self.POSE_LANDMARKS = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle between three points"""
        a = np.array([point1['x'], point1['y']])
        b = np.array([point2['x'], point2['y']])
        c = np.array([point3['x'], point3['y']])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def analyze_lat_pulldown(self, poses_data: List[Dict]) -> Dict:
        """Analyze lat pull-down exercise form"""
        if not poses_data:
            return {"error": "No pose data available"}
        
        analysis = {
            'exercise_type': 'lat_pulldown',
            'rep_count': 0,
            'form_issues': [],
            'metrics': {},
            'rep_phases': []
        }
        
        # Extract key angles throughout the exercise
        elbow_angles = []
        shoulder_positions = []
        back_arch_scores = []
        
        for pose in poses_data:
            landmarks = pose['landmarks']
            
            if len(landmarks) >= 33:  # Ensure we have all landmarks
                # Calculate elbow angle (shoulder-elbow-wrist)
                try:
                    left_elbow_angle = self.calculate_angle(
                        landmarks[self.POSE_LANDMARKS['left_shoulder']],
                        landmarks[self.POSE_LANDMARKS['left_elbow']],
                        landmarks[self.POSE_LANDMARKS['left_wrist']]
                    )
                    elbow_angles.append(left_elbow_angle)
                    
                    # Shoulder position (check for shrugging)
                    left_shoulder_y = landmarks[self.POSE_LANDMARKS['left_shoulder']]['y']
                    right_shoulder_y = landmarks[self.POSE_LANDMARKS['right_shoulder']]['y']
                    shoulder_positions.append((left_shoulder_y + right_shoulder_y) / 2)
                    
                    # Back arch assessment (hip-shoulder angle)
                    hip_y = (landmarks[self.POSE_LANDMARKS['left_hip']]['y'] + 
                            landmarks[self.POSE_LANDMARKS['right_hip']]['y']) / 2
                    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                    back_arch_scores.append(abs(hip_y - shoulder_y))
                    
                except (KeyError, IndexError):
                    continue
        
        # Analyze patterns
        if elbow_angles:
            # Rep counting based on elbow angle peaks/valleys
            analysis['rep_count'] = self.count_reps_from_angles(elbow_angles)
            
            # Form analysis
            avg_elbow_angle = np.mean(elbow_angles)
            min_elbow_angle = np.min(elbow_angles)
            max_elbow_angle = np.max(elbow_angles)
            
            analysis['metrics'] = {
                'avg_elbow_angle': float(avg_elbow_angle),
                'min_elbow_angle': float(min_elbow_angle),
                'max_elbow_angle': float(max_elbow_angle),
                'range_of_motion': float(max_elbow_angle - min_elbow_angle),
                'avg_back_arch_score': float(np.mean(back_arch_scores)) if back_arch_scores else 0
            }
            
            # Form issue detection
            if analysis['metrics']['range_of_motion'] < 60:
                analysis['form_issues'].append("Incomplete range of motion - try to fully extend arms")
            
            if analysis['metrics']['avg_back_arch_score'] > 0.15:
                analysis['form_issues'].append("Excessive back arching detected - engage core muscles")
            
            if len(set(np.round(shoulder_positions, 2))) > len(shoulder_positions) * 0.3:
                analysis['form_issues'].append("Shoulder instability - focus on keeping shoulders down")
        
        return analysis
    
    def analyze_pullup(self, poses_data: List[Dict]) -> Dict:
        """Analyze pull-up exercise form"""
        analysis = {
            'exercise_type': 'pullup',
            'rep_count': 0,
            'form_issues': [],
            'metrics': {},
        }
        
        wrist_positions = []
        chin_positions = []
        
        for pose in poses_data:
            landmarks = pose['landmarks']
            
            if len(landmarks) >= 33:
                try:
                    # Track wrist height (key indicator for pull-ups)
                    left_wrist_y = landmarks[self.POSE_LANDMARKS['left_wrist']]['y']
                    right_wrist_y = landmarks[self.POSE_LANDMARKS['right_wrist']]['y']
                    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
                    wrist_positions.append(avg_wrist_y)
                    
                    # Track chin position
                    nose_y = landmarks[self.POSE_LANDMARKS['nose']]['y']
                    chin_positions.append(nose_y)
                    
                except (KeyError, IndexError):
                    continue
        
        if wrist_positions and chin_positions:
            analysis['rep_count'] = self.count_reps_from_positions(wrist_positions)
            
            # Check if chin goes above wrists (full rep)
            min_chin_y = np.min(chin_positions)
            avg_wrist_y = np.mean(wrist_positions)
            
            analysis['metrics'] = {
                'chin_clearance': float(avg_wrist_y - min_chin_y),
                'avg_wrist_height': float(avg_wrist_y)
            }
            
            if analysis['metrics']['chin_clearance'] < 0.05:
                analysis['form_issues'].append("Chin not clearing bar - aim to pull higher")
        
        return analysis
    
    def count_reps_from_angles(self, angles: List[float]) -> int:
        """Count repetitions based on angle peaks and valleys"""
        if len(angles) < 10:
            return 0
        
        # Smooth the data
        smoothed = np.convolve(angles, np.ones(5)/5, mode='valid')
        
        # Find peaks (extended position) and valleys (contracted position)
        peaks = []
        valleys = []
        
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if not peaks or smoothed[i] - smoothed[peaks[-1]] > 10:  # Minimum angle difference
                    peaks.append(i)
            elif smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                if not valleys or smoothed[valleys[-1]] - smoothed[i] > 10:
                    valleys.append(i)
        
        # Rep count is the minimum of peaks and valleys
        return min(len(peaks), len(valleys))
    
    def count_reps_from_positions(self, positions: List[float]) -> int:
        """Count reps based on vertical position changes"""
        if len(positions) < 10:
            return 0
        
        smoothed = np.convolve(positions, np.ones(3)/3, mode='valid')
        
        # Find significant position changes
        peaks = []
        valleys = []
        threshold = np.std(smoothed) * 0.5
        
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:  # Valley (top position)
                if not valleys or abs(smoothed[i] - smoothed[valleys[-1]]) > threshold:
                    valleys.append(i)
            elif smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:  # Peak (bottom position)
                if not peaks or abs(smoothed[i] - smoothed[peaks[-1]]) > threshold:
                    peaks.append(i)
        
        return min(len(peaks), len(valleys))

class LLMFeedbackGenerator:
    """Generate exercise feedback using LLM APIs (Cloud or Local)"""
    
    def __init__(self, api_key: str = "", provider: str = "kolosal", kolosal_url: str = "http://localhost:8080"):
        self.provider = provider
        self.kolosal_url = kolosal_url
        
        if provider == "openai":
            import openai
            openai.api_key = api_key
        elif provider == "anthropic":
            self.anthropic_client = Anthropic(api_key=api_key)
        elif provider == "kolosal":
            # No special setup needed for local Kolosal.AI
            print(f"Using local Kolosal.AI server at: {kolosal_url}")
    
    def generate_feedback(self, exercise_name: str, analysis_data: Dict, user_name: str = "User") -> Dict:
        """Generate personalized feedback using LLM"""
        
        # Create detailed prompt
        prompt = self._build_feedback_prompt(exercise_name, analysis_data, user_name)
        
        try:
            if self.provider == "openai":
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert personal trainer and biomechanics specialist. Provide concise, actionable feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                feedback_text = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                feedback_text = response.content[0].text
                
            elif self.provider == "kolosal":
                # Call local Kolosal.AI server
                feedback_text = self._call_kolosal_api(prompt)
            
            # Parse feedback into structured format
            return self._parse_feedback(feedback_text, analysis_data)
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            return self._fallback_feedback(analysis_data)
    
    def _call_kolosal_api(self, prompt: str) -> str:
        """Call local Kolosal.AI server"""
        import requests
        
        print(f"🤖 Calling Kolosal.AI with prompt: {prompt[:50]}...")
        
        try:
            # Use the confirmed working endpoint and model name
            response = requests.post(
                f"{self.kolosal_url}/v1/chat/completions",
                json={
                    "model": "Gemma 3 4B:4-bit",  # Exact model name from Kolosal.AI
                    "messages": [
                        {"role": "system", "content": "You are an expert personal trainer. Provide concise, actionable feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"🔍 Kolosal.AI response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 Response keys: {list(result.keys())}")
                
                # Handle different response formats
                if "choices" in result:
                    choices = result["choices"]
                    print(f"🔍 Choices type: {type(choices)}, content: {choices}")
                    
                    # Handle choices as dict (Kolosal.AI specific format)
                    if isinstance(choices, dict) and choices:
                        # Try to extract content from dict format
                        for key, value in choices.items():
                            if isinstance(value, dict) and "message" in value:
                                return value["message"].get("content", "")
                            elif isinstance(value, str):
                                return value
                    
                    # Handle choices as list (standard OpenAI format)
                    elif isinstance(choices, list) and len(choices) > 0:
                        choice = choices[0]
                        if "message" in choice:
                            return choice["message"].get("content", "")
                        elif "text" in choice:
                            return choice["text"]
                
                # Fallback: look for other response fields
                if "response" in result:
                    return result["response"]
                elif "text" in result:
                    return result["text"]
                elif "content" in result:
                    return result["content"]
                
                # If we can't find content, return the whole response as debug
                print(f"⚠️ Unexpected response format: {result}")
                return f"Response received but format unclear: {str(result)[:200]}"
            
            else:
                print(f"❌ HTTP Error: {response.status_code}, {response.text}")
                return f"HTTP Error {response.status_code}: {response.text}"
        
        except Exception as e:
            print(f"❌ Kolosal.AI API Error: {e}")
            raise Exception(f"Could not connect to Kolosal.AI server at {self.kolosal_url}. Error: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to LLM provider"""
        try:
            if self.provider == "kolosal":
                # Test with a simple prompt
                test_response = self._call_kolosal_api("Hello, respond with 'Working' if you can see this.")
                print(f"🔍 Test response: {test_response}")
                return len(test_response) > 0 and "error" not in test_response.lower()
            else:
                # Test other providers
                test_feedback = self.generate_feedback("test", {"rep_count": 1, "form_issues": []}, "test")
                return test_feedback.get('llm_generated', False)
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def _build_feedback_prompt(self, exercise_name: str, analysis_data: Dict, user_name: str) -> str:
        """Build detailed prompt for LLM"""
        
        form_issues = analysis_data.get('form_issues', [])
        metrics = analysis_data.get('metrics', {})
        rep_count = analysis_data.get('rep_count', 0)
        
        prompt = f"""
Analyze this {exercise_name} workout performance for {user_name}:

PERFORMANCE DATA:
- Rep Count: {rep_count}
- Form Issues Detected: {', '.join(form_issues) if form_issues else 'None detected'}
- Key Metrics: {json.dumps(metrics, indent=2)}

ANALYSIS REQUIREMENTS:
1. Provide a concise summary (2-3 sentences)
2. Rate form quality (0.0-1.0 scale)
3. Assess injury risk (low/medium/high)
4. Give 2-3 specific, actionable recommendations

RESPONSE FORMAT:
Summary: [Brief analysis of performance]
Form Score: [0.0-1.0]
Injury Risk: [low/medium/high]
Recommendations: [Numbered list of specific improvements]

Be direct, encouraging, and focus on the most important improvements.
"""
        return prompt
    
    def _parse_feedback(self, feedback_text: str, analysis_data: Dict) -> Dict:
        """Parse LLM response into structured feedback"""
        
        lines = feedback_text.strip().split('\n')
        
        feedback = {
            'summary': '',
            'form_score': 0.75,  # Default
            'injury_risk_level': 'low',  # Default
            'recommendations': '',
            'llm_generated': True
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Summary:'):
                current_section = 'summary'
                feedback['summary'] = line.replace('Summary:', '').strip()
            elif line.startswith('Form Score:'):
                try:
                    score_text = line.replace('Form Score:', '').strip()
                    # Extract number from text (could be "0.8" or "0.8/1.0" or "80%")
                    import re
                    numbers = re.findall(r'0?\.\d+|\d+', score_text)
                    if numbers:
                        score = float(numbers[0])
                        if score > 1:  # If it's a percentage
                            score = score / 100
                        feedback['form_score'] = max(0.0, min(1.0, score))
                except:
                    pass
            elif line.startswith('Injury Risk:'):
                risk = line.replace('Injury Risk:', '').strip().lower()
                if risk in ['low', 'medium', 'high']:
                    feedback['injury_risk_level'] = risk
            elif line.startswith('Recommendations:'):
                current_section = 'recommendations'
                feedback['recommendations'] = line.replace('Recommendations:', '').strip()
            elif current_section == 'recommendations' and line:
                feedback['recommendations'] += '\n' + line
            elif current_section == 'summary' and line and not any(line.startswith(prefix) for prefix in ['Form Score:', 'Injury Risk:', 'Recommendations:']):
                feedback['summary'] += ' ' + line
        
        return feedback
    
    def _fallback_feedback(self, analysis_data: Dict) -> Dict:
        """Fallback feedback if LLM fails"""
        form_issues = analysis_data.get('form_issues', [])
        rep_count = analysis_data.get('rep_count', 0)
        
        if form_issues:
            summary = f"Completed {rep_count} reps. Key areas for improvement: {', '.join(form_issues[:2])}"
            form_score = 0.6
            injury_risk = 'medium' if len(form_issues) > 2 else 'low'
        else:
            summary = f"Good performance! Completed {rep_count} reps with solid form."
            form_score = 0.8
            injury_risk = 'low'
        
        return {
            'summary': summary,
            'form_score': form_score,
            'injury_risk_level': injury_risk,
            'recommendations': 'Continue with current technique and focus on consistency.',
            'llm_generated': False
        }

# Configuration should be imported from config.py
# from config import AIConfig