"""
Utility functions for APT Fitness Assistant
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique ID based on timestamp and prefix."""
    timestamp = datetime.now().isoformat()
    unique_string = f"{prefix}_{timestamp}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("apt_fitness")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_image_file(file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
    """Validate image file for processing."""
    path = Path(file_path)
    
    if not path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return {"valid": False, "error": f"File too large: {size_mb:.1f}MB > {max_size_mb}MB"}
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if path.suffix.lower() not in valid_extensions:
        return {"valid": False, "error": "Invalid file format"}
    
    return {"valid": True, "size_mb": size_mb}


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight and height."""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def get_bmi_category(bmi: float) -> str:
    """Get BMI category string."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable string."""
    if minutes < 60:
        return f"{minutes}m"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {remaining_minutes}m"


def calculate_calories_burned(activity: str, duration_minutes: int, weight_kg: float) -> float:
    """Calculate calories burned for an activity."""
    # MET values for different activities
    met_values = {
        "walking": 3.5,
        "jogging": 7.0,
        "running": 8.0,
        "cycling": 6.0,
        "swimming": 8.0,
        "weightlifting": 3.0,
        "yoga": 2.5,
        "dancing": 4.5
    }
    
    met = met_values.get(activity.lower(), 4.0)  # Default to moderate activity
    calories_per_minute = (met * weight_kg * 3.5) / 200
    return calories_per_minute * duration_minutes


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_value, min(value, max_value))


def create_safe_filename(filename: str) -> str:
    """Create a filesystem-safe filename."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Limit length
    if len(safe_filename) > 255:
        name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
        max_name_length = 255 - len(ext) - 1
        safe_filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return safe_filename


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_week_dates(start_date: datetime) -> List[datetime]:
    """Get list of dates for a week starting from given date."""
    return [start_date + timedelta(days=i) for i in range(7)]


def format_exercise_name(name: str) -> str:
    """Format exercise name for display."""
    return name.replace('_', ' ').title()


def calculate_body_fat_navy(waist_cm: float, neck_cm: float, height_cm: float, 
                           is_male: bool = True) -> float:
    """Calculate body fat percentage using US Navy method."""
    import math
    
    if is_male:
        # Male formula
        body_fat = 495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 
                         0.15456 * math.log10(height_cm)) - 450
    else:
        # Female formula (requires hip measurement - using waist as approximation)
        hip_cm = waist_cm * 1.1  # Rough approximation
        body_fat = 495 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 
                         0.22100 * math.log10(height_cm)) - 450
    
    return clamp(body_fat, 3.0, 50.0)  # Realistic range


def estimate_one_rep_max(weight: float, reps: int) -> float:
    """Estimate one-rep max using Epley formula."""
    if reps == 1:
        return weight
    return weight * (1 + reps / 30.0)


def get_relative_date_string(date: datetime) -> str:
    """Get human-readable relative date string."""
    now = datetime.now()
    diff = now - date
    
    if diff.days == 0:
        return "Today"
    elif diff.days == 1:
        return "Yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    elif diff.days < 30:
        weeks = diff.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif diff.days < 365:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
