"""
Utility modules for AI Fitness Assistant
Common utilities, helpers, and decorators
"""

import functools
import time
import logging
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
import io
import base64
import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Performance Monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = self.metrics.get(operation, []) + [duration]
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation in self.metrics:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return 0.0
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for operation, times in self.metrics.items():
            summary[operation] = {
                'count': len(times),
                'total': sum(times),
                'average': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        return summary

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Decorators
def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        operation = f"{func.__module__}.{func.__name__}"
        performance_monitor.start_timer(operation)
        try:
            result = func(*args, **kwargs)
            duration = performance_monitor.end_timer(operation)
            logger.debug(f"{operation} took {duration:.3f} seconds")
            return result
        except Exception as e:
            performance_monitor.end_timer(operation)
            raise e
    return wrapper

def cache_result(ttl: int = 3600, max_size: int = 128):
    """Decorator to cache function results with TTL."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = _create_cache_key(func.__name__, args, kwargs)
            current_time = time.time()
            
            # Check if result is in cache and not expired
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    del cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Implement simple LRU by removing oldest entries if cache is full
            if len(cache) >= max_size:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[key] = (result, current_time)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'max_size': max_size,
            'ttl': ttl
        }
        
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function execution on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time:.2f}s")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

def async_to_sync(func: Callable) -> Callable:
    """Decorator to run async functions synchronously."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper

# Image Processing Utilities
class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    @timer
    def validate_image(image_data: bytes, max_size_mb: int = 10) -> Tuple[bool, str]:
        """Validate image data."""
        try:
            # Check size
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"Image size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)"
            
            # Try to decode image
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            if width > 4000 or height > 4000:
                return False, "Image too large (maximum 4000x4000 pixels)"
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                return False, f"Unsupported format: {image.format}"
            
            return True, "Valid image"
        
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    @staticmethod
    @timer
    def resize_image(image_data: bytes, max_width: int = 1024, max_height: int = 1024, quality: int = 85) -> bytes:
        """Resize image while maintaining aspect ratio."""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Calculate new dimensions
            width, height = image.size
            if width <= max_width and height <= max_height:
                return image_data  # No resizing needed
            
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = io.BytesIO()
            format_to_save = 'JPEG' if image.format in ['JPEG', 'JPG'] else 'PNG'
            
            if format_to_save == 'JPEG':
                resized_image = resized_image.convert('RGB')
                resized_image.save(output, format=format_to_save, quality=quality, optimize=True)
            else:
                resized_image.save(output, format=format_to_save, optimize=True)
            
            return output.getvalue()
        
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_data
    
    @staticmethod
    @timer
    def enhance_image(image_data: bytes) -> bytes:
        """Enhance image for better analysis."""
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return image_data
            
            # Apply enhancement
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buffer.tobytes()
        
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_data
    
    @staticmethod
    def image_to_base64(image_data: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_string: str) -> bytes:
        """Convert base64 string to image bytes."""
        return base64.b64decode(base64_string)

# Data Validation Utilities
class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], 
                             max_val: Union[int, float], field_name: str) -> Tuple[bool, str]:
        """Validate that a numeric value is within a specified range."""
        try:
            value = float(value)
            if min_val <= value <= max_val:
                return True, ""
            return False, f"{field_name} must be between {min_val} and {max_val}"
        except (ValueError, TypeError):
            return False, f"{field_name} must be a valid number"
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, ""
        return False, "Invalid email format"
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    @staticmethod
    def validate_user_profile(profile_data: Dict) -> Tuple[bool, List[str]]:
        """Validate user profile data comprehensively."""
        errors = []
        
        # Required fields
        required_fields = ['age', 'gender', 'weight', 'height', 'activity_level']
        for field in required_fields:
            if field not in profile_data or profile_data[field] is None:
                errors.append(f"{field} is required")
        
        # Age validation
        if 'age' in profile_data:
            valid, msg = DataValidator.validate_numeric_range(profile_data['age'], 18, 100, 'Age')
            if not valid:
                errors.append(msg)
        
        # Weight validation
        if 'weight' in profile_data:
            valid, msg = DataValidator.validate_numeric_range(profile_data['weight'], 30.0, 300.0, 'Weight')
            if not valid:
                errors.append(msg)
        
        # Height validation
        if 'height' in profile_data:
            valid, msg = DataValidator.validate_numeric_range(profile_data['height'], 100.0, 250.0, 'Height')
            if not valid:
                errors.append(msg)
        
        # Gender validation
        if 'gender' in profile_data:
            if profile_data['gender'] not in ['male', 'female']:
                errors.append("Gender must be 'male' or 'female'")
        
        # Activity level validation
        if 'activity_level' in profile_data:
            valid_levels = ['sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extremely_active']
            if profile_data['activity_level'] not in valid_levels:
                errors.append(f"Activity level must be one of: {', '.join(valid_levels)}")
        
        return len(errors) == 0, errors

# Math and Calculation Utilities
class MathUtils:
    """Utility class for mathematical operations."""
    
    @staticmethod
    def calculate_angle(point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points."""
        try:
            # Convert to numpy arrays
            p1 = np.array(point1[:2])  # Use only x, y coordinates
            p2 = np.array(point2[:2])
            p3 = np.array(point3[:2])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        try:
            p1 = np.array(point1[:2])
            p2 = np.array(point2[:2])
            return np.linalg.norm(p1 - p2)
        except (ValueError, IndexError):
            return 0.0
    
    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to range [0, 1]."""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    @staticmethod
    def calculate_bmi_category(bmi: float) -> str:
        """Categorize BMI value."""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obesity"
    
    @staticmethod
    def interpolate_values(values: List[float], target_length: int) -> List[float]:
        """Interpolate values to target length."""
        if len(values) == target_length:
            return values
        
        if len(values) < 2:
            return values * target_length if values else [0.0] * target_length
        
        # Linear interpolation
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, values).tolist()

# String and Text Utilities
class TextUtils:
    """Utility class for text processing."""
    
    @staticmethod
    def format_exercise_name(name: str) -> str:
        """Format exercise name for display."""
        return name.replace('_', ' ').title()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def pluralize(word: str, count: int) -> str:
        """Simple pluralization."""
        if count == 1:
            return word
        
        # Simple rules
        if word.endswith('y') and word[-2] not in 'aeiou':
            return word[:-1] + 'ies'
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word + 'es'
        else:
            return word + 's'
    
    @staticmethod
    def format_duration(seconds: int) -> str:
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds} second{'' if seconds == 1 else 's'}"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'' if minutes == 1 else 's'}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes == 0:
                return f"{hours} hour{'' if hours == 1 else 's'}"
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def format_calories(calories: float) -> str:
        """Format calories for display."""
        if calories >= 1000:
            return f"{calories/1000:.1f}k cal"
        return f"{calories:.0f} cal"

# Date and Time Utilities
class DateTimeUtils:
    """Utility class for date and time operations."""
    
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    @staticmethod
    def format_date(date: datetime, format_string: str = "%Y-%m-%d") -> str:
        """Format date to string."""
        return date.strftime(format_string)
    
    @staticmethod
    def parse_date(date_string: str, format_string: str = "%Y-%m-%d") -> Optional[datetime]:
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_string, format_string)
        except ValueError:
            return None
    
    @staticmethod
    def get_week_start(date: datetime) -> datetime:
        """Get the start of the week (Monday) for a given date."""
        days_since_monday = date.weekday()
        return date - timedelta(days=days_since_monday)
    
    @staticmethod
    def get_days_between(start_date: datetime, end_date: datetime) -> int:
        """Get number of days between two dates."""
        return (end_date - start_date).days

# File and Storage Utilities
class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def generate_filename(prefix: str = "file", extension: str = "txt") -> str:
        """Generate unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename."""
        return filename.split('.')[-1].lower() if '.' in filename else ""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in bytes to human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.1f} {units[unit_index]}"

# Helper Functions
def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a cache key from function name and arguments."""
    # Convert args and kwargs to a hashable format
    hashable_data = {
        'function': func_name,
        'args': str(args),
        'kwargs': str(sorted(kwargs.items()))
    }
    
    # Create hash
    data_string = json.dumps(hashable_data, sort_keys=True)
    return hashlib.md5(data_string.encode()).hexdigest()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))

def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default."""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

# Async utilities for future enhancements
async def async_image_processing(image_data: bytes, operations: List[str]) -> bytes:
    """Asynchronously process image with multiple operations."""
    result = image_data
    
    for operation in operations:
        if operation == 'resize':
            result = await asyncio.to_thread(ImageProcessor.resize_image, result)
        elif operation == 'enhance':
            result = await asyncio.to_thread(ImageProcessor.enhance_image, result)
    
    return result

async def batch_process_images(image_list: List[bytes], max_workers: int = 4) -> List[bytes]:
    """Process multiple images in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            asyncio.to_thread(ImageProcessor.enhance_image, img) 
            for img in image_list
        ]
        return await asyncio.gather(*tasks)

# Streamlit-specific utilities
class StreamlitUtils:
    """Utilities for Streamlit applications."""
    
    @staticmethod
    def show_metric_card(title: str, value: str, delta: Optional[str] = None, 
                        help_text: Optional[str] = None):
        """Display a styled metric card."""
        delta_html = f"<div style='color: #28a745; font-size: 0.8rem;'>{delta}</div>" if delta else ""
        help_html = f"<div style='color: #6c757d; font-size: 0.7rem; margin-top: 0.5rem;'>{help_text}</div>" if help_text else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: bold; color: #495057;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #212529; margin: 0.5rem 0;">{value}</div>
            {delta_html}
            {help_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_progress_ring(progress: float, title: str = "", size: int = 100):
        """Display a circular progress indicator."""
        progress = clamp(progress, 0.0, 1.0)
        circumference = 2 * 3.14159 * 45  # radius = 45
        stroke_dasharray = circumference
        stroke_dashoffset = circumference * (1 - progress)
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <svg width="{size}" height="{size}" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#e9ecef" stroke-width="10"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="#667eea" stroke-width="10"
                        stroke-dasharray="{stroke_dasharray}" stroke-dashoffset="{stroke_dashoffset}"
                        stroke-linecap="round" transform="rotate(-90 50 50)"/>
                <text x="50" y="50" text-anchor="middle" dy="0.3em" font-size="16" font-weight="bold" fill="#212529">
                    {progress:.0%}
                </text>
            </svg>
            {f"<div style='margin-top: 0.5rem; font-weight: bold;'>{title}</div>" if title else ""}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_loading_spinner(text: str = "Loading..."):
        """Display a loading spinner with text."""
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div class="loading-spinner"></div>
            <div style="margin-top: 1rem; color: #6c757d;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing utility functions...")
    
    # Test performance monitoring
    @timer
    def sample_function():
        time.sleep(0.1)
        return "result"
    
    result = sample_function()
    print(f"Function result: {result}")
    print(f"Performance metrics: {performance_monitor.get_metrics_summary()}")
    
    # Test caching
    @cache_result(ttl=60)
    def cached_function(x):
        time.sleep(0.1)  # Simulate expensive operation
        return x * 2
    
    start = time.time()
    result1 = cached_function(5)
    time1 = time.time() - start
    
    start = time.time()
    result2 = cached_function(5)  # Should be cached
    time2 = time.time() - start
    
    print(f"First call: {result1} ({time1:.3f}s)")
    print(f"Second call: {result2} ({time2:.3f}s)")
    print(f"Cache info: {cached_function.cache_info()}")
    
    print("Utility module tests completed!")