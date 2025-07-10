#!/usr/bin/env python3
"""
APT (AI Personal Trainer) Setup Script
Helps set up the project for development or production
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create Python virtual environment"""
    if os.path.exists("venv"):
        print("📁 Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install Python dependencies"""
    # Determine the correct python and pip paths based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
        pip_cmd = f"{python_cmd} -m pip"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
        pip_cmd = f"{python_cmd} -m pip"
    
    commands = [
        f"{pip_cmd} install --upgrade pip",
        f"{pip_cmd} install -r requirements.txt"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    return True

def setup_environment_file():
    """Set up environment configuration"""
    if os.path.exists(".env"):
        print("📝 .env file already exists")
        return True
    
    if os.path.exists(".env.template"):
        try:
            shutil.copy(".env.template", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your configuration")
            return True
        except Exception as e:
            print(f"❌ Failed to copy .env template: {e}")
            return False
    else:
        print("⚠️  .env.template not found, creating basic .env file")
        basic_env = """# Basic APT Configuration
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=dev-key-change-in-production
LLM_PROVIDER=kolosal
KOLOSAL_API_URL=http://localhost:8080
KOLOSAL_MODEL=Gemma 3 4B:4-bit
"""
        try:
            with open(".env", "w") as f:
                f.write(basic_env)
            print("✅ Created basic .env file")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "pose_data", "logs"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    required_packages = [
        "flask",
        "flask_sqlalchemy", 
        "flask_cors",
        "cv2",
        "mediapipe",
        "numpy",
        "werkzeug"
    ]
    
    # Use the virtual environment Python
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    for package in required_packages:
        test_cmd = f"{python_cmd} -c \"import {package}; print('✅ {package} imported successfully')\""
        if not run_command(test_cmd, f"Testing {package} import"):
            print(f"❌ Failed to import {package}")
            return False
    
    print("✅ All required packages imported successfully")
    return True

def initialize_database():
    """Initialize the database"""
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    # Simple database initialization
    init_script = """
from models import db
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///apt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    print('Database initialized')
"""
    
    try:
        with open('temp_init.py', 'w') as f:
            f.write(init_script)
        
        result = run_command(f"{python_cmd} temp_init.py", "Initializing database")
        os.remove('temp_init.py')
        return result
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        if os.path.exists('temp_init.py'):
            os.remove('temp_init.py')
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*50)
    print("🎉 APT Setup Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n2. Edit .env file with your configuration")
    print("   - Set your API keys if using cloud LLMs")
    print("   - Configure Kolosal.AI URL if using local LLM")
    
    print("\n3. Start your Kolosal.AI server:")
    print("   - Load your Gemma 3 4B model")
    print("   - Start server on port 8080")
    
    print("\n4. Start the API server:")
    print("   python app_with_ai.py")
    
    print("\n5. (Optional) Start the Streamlit frontend:")
    print("   streamlit run frontend.py")
    
    print("\n6. Test the system:")
    print("   curl http://localhost:5000/")
    print("   curl http://localhost:5000/ai-status")
    
    print("\n📚 For more help, check the README.md file")

def main():
    """Main setup function"""
    print("🤖 APT (AI Personal Trainer) Setup")
    print("="*40)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        print("Please make sure you're in the correct directory")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment file", setup_environment_file),
        ("Creating directories", create_directories),
        ("Testing imports", test_imports),
        ("Initializing database", initialize_database)
    ]
    
    for description, step_function in steps:
        print(f"\n📋 {description}...")
        if not step_function():
            print(f"❌ Setup failed at: {description}")
            
            # Provide helpful error message for common issues
            if description == "Installing dependencies":
                print("\n💡 Try manual installation:")
                if os.name == 'nt':
                    print("   venv\\Scripts\\activate")
                    print("   venv\\Scripts\\python -m pip install --upgrade pip")
                    print("   venv\\Scripts\\python -m pip install -r requirements.txt")
                else:
                    print("   source venv/bin/activate")
                    print("   pip install --upgrade pip")
                    print("   pip install -r requirements.txt")
            
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()