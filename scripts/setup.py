#!/usr/bin/env python3
"""
Setup script for APT Fitness Assistant
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is supported."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def create_virtual_environment():
    """Create virtual environment."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def get_python_executable():
    """Get the Python executable path for the virtual environment."""
    if os.name == 'nt':  # Windows
        return Path(".venv") / "Scripts" / "python.exe"
    else:  # Unix-like
        return Path(".venv") / "bin" / "python"


def install_dependencies():
    """Install project dependencies."""
    python_exe = get_python_executable()
    
    if not python_exe.exists():
        print("❌ Virtual environment not found")
        return False
    
    try:
        print("🔧 Upgrading pip...")
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        
        print("📦 Installing dependencies...")
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "-e", "."])
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_tests():
    """Run test suite."""
    python_exe = get_python_executable()
    
    try:
        print("🧪 Running tests...")
        result = subprocess.run([str(python_exe), "-m", "pytest", "tests/"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠️ pytest not found, running basic tests...")
        try:
            subprocess.check_call([str(python_exe), "-m", "unittest", "discover", "-s", "tests"])
            print("✅ Basic tests passed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Tests failed")
            return False


def main():
    """Main setup function."""
    print("🏋️‍♀️ APT Fitness Assistant - Setup")
    print("=" * 40)
    
    if not check_python_version():
        return 1
    
    if not create_virtual_environment():
        return 1
    
    if not install_dependencies():
        return 1
    
    print("\n🧪 Running tests...")
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To run the application:")
    print("   python main.py")
    print("   or")
    print("   streamlit run main.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
