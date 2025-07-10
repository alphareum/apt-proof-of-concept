# API Configuration for APT AI Integration
# Copy this file and add your actual API keys

import os

class AIConfig:
    """Configuration for AI services"""
    
    # ============================================================================
    # API KEYS - REPLACE WITH YOUR ACTUAL KEYS (Optional for local LLM)
    # ============================================================================
    
    # OpenAI API Key (for GPT-4 feedback generation) - Optional if using local LLM
    # Get from: https://platform.openai.com/api-keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    
    # Anthropic API Key (alternative to OpenAI) - Optional if using local LLM
    # Get from: https://console.anthropic.com/
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")
    
    # ============================================================================
    # LOCAL LLM SETTINGS (Kolosal.AI)
    # ============================================================================
    
    # Kolosal.AI Local Server Configuration
    KOLOSAL_API_URL = os.getenv("KOLOSAL_API_URL", "http://localhost:8080")  # Default Kolosal.AI port
    KOLOSAL_API_KEY = os.getenv("KOLOSAL_API_KEY", "")  # Usually not needed for local
    
    # Local LLM Model Settings
    LOCAL_MODEL_NAME = "default"  # Exact name from Kolosal.AI
    LOCAL_MAX_TOKENS = 300
    LOCAL_TEMPERATURE = 0.3
    
    # ============================================================================
    # AI SERVICE SETTINGS
    # ============================================================================
    
    # Choose LLM provider: "openai", "anthropic", or "kolosal"
    LLM_PROVIDER = "kolosal"  # Using local Kolosal.AI instead of cloud APIs
    
    # LLM Models
    OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    
    # ============================================================================
    # VIDEO PROCESSING SETTINGS
    # ============================================================================
    
    # Maximum video file size (in MB)
    MAX_VIDEO_SIZE_MB = 100
    
    # Supported video formats
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Pose detection confidence thresholds
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # ============================================================================
    # DIRECTORY SETTINGS
    # ============================================================================
    
    # Upload directories
    UPLOAD_FOLDER = 'uploads'
    POSE_DATA_FOLDER = 'pose_data'
    
    # ============================================================================
    # EXERCISE-SPECIFIC SETTINGS
    # ============================================================================
    
    # Supported exercises for AI analysis
    SUPPORTED_EXERCISES = {
        'lat_pulldown': {
            'aliases': ['lat pull-down', 'lat pulldown', 'pulldown'],
            'analyzer': 'analyze_lat_pulldown'
        },
        'pullup': {
            'aliases': ['pull-up', 'pull up', 'pullup'],
            'analyzer': 'analyze_pullup'
        },
        'row': {
            'aliases': ['seated row', 'cable row', 'rowing'],
            'analyzer': 'analyze_row'
        }
    }
    
    # Form analysis thresholds
    FORM_THRESHOLDS = {
        'min_range_of_motion': 60,  # degrees
        'max_back_arch_score': 0.15,
        'min_rep_consistency': 0.8
    }

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

def setup_instructions():
    """Print setup instructions for API integration"""
    
    print("🤖 APT AI Integration Setup Instructions")
    print("=" * 50)
    print()
    
    print("OPTION A: LOCAL LLM (Recommended - FREE)")
    print("1. Start Kolosal.AI:")
    print("   - Load your model")
    print("   - Start the server (usually on port 8080)")
    print("   - Test: http://localhost:8080")
    print()
    
    print("2. Set LLM_PROVIDER = 'kolosal' in config.py")
    print()
    
    print("OPTION B: CLOUD APIs (Paid)")
    print("1. GET API KEYS:")
    print("   OpenAI: https://platform.openai.com/api-keys")
    print("   Anthropic: https://console.anthropic.com/")
    print()
    
    print("2. SET ENVIRONMENT VARIABLES (Recommended):")
    print("   export OPENAI_API_KEY='your-actual-key-here'")
    print("   export ANTHROPIC_API_KEY='your-actual-key-here'")
    print()
    
    print("3. INSTALL AI DEPENDENCIES:")
    print("   pip install opencv-python mediapipe requests")
    print()
    
    print("4. TEST AI STATUS:")
    print("   python app_with_ai.py")
    print("   Visit: http://localhost:5000/ai-status")
    print()
    
    print("🎯 Ready to process real exercise videos with AI!")

if __name__ == "__main__":
    setup_instructions()