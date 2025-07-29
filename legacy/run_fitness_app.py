"""
AI Fitness Assistant Demo Runner

This script helps you set up and run the AI Fitness Assistant application.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("🔧 Installing required packages...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("fitness_venv"):
        print("📋 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "fitness_venv"])
            print("✅ Virtual environment created!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    
    # Determine the correct python executable
    if os.name == 'nt':  # Windows
        python_exe = os.path.join("fitness_venv", "Scripts", "python.exe")
        pip_exe = os.path.join("fitness_venv", "Scripts", "pip.exe")
    else:  # Unix-like
        python_exe = os.path.join("fitness_venv", "bin", "python")
        pip_exe = os.path.join("fitness_venv", "bin", "pip")
    
    try:
        # Upgrade pip
        print("🔧 Upgrading pip...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("📦 Installing requirements...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "-r", "requirements_fitness.txt"])
        
        print("✅ Requirements installed successfully!")
        return True, python_exe
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False, None

def run_streamlit_app(python_exe=None):
    """Run the Streamlit application."""
    print("🚀 Starting AI Fitness Assistant...")
    
    if python_exe is None:
        if os.path.exists("fitness_venv"):
            if os.name == 'nt':  # Windows
                python_exe = os.path.join("fitness_venv", "Scripts", "python.exe")
            else:  # Unix-like
                python_exe = os.path.join("fitness_venv", "bin", "python")
        else:
            python_exe = sys.executable
    
    try:
        subprocess.run([
            python_exe, "-m", "streamlit", "run", "fitness_app.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main demo runner."""
    print("🏋️‍♀️ AI Fitness Assistant Demo")
    print("=" * 40)
    
    # Check if requirements file exists
    if not os.path.exists("requirements_fitness.txt"):
        print("❌ requirements_fitness.txt not found!")
        return
    
    # Check if main app file exists
    if not os.path.exists("fitness_app.py"):
        print("❌ fitness_app.py not found!")
        return
    
    # Ask user if they want to install requirements
    install = input("📦 Install/update requirements? (y/n): ").lower().strip()
    
    python_exe = None
    if install in ['y', 'yes']:
        success, python_exe = install_requirements()
        if not success:
            return
    
    print("\n🎯 Starting the application...")
    print("📱 The app will open in your browser at: http://localhost:8502")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 40)
    
    run_streamlit_app(python_exe)

if __name__ == "__main__":
    main()
