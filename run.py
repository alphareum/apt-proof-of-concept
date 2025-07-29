#!/usr/bin/env python3
"""
APT Fitness Assistant - Cross-platform Setup and Runner
Supports both UV and traditional pip/venv workflows
"""

import subprocess
import sys
import os
import platform
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def check_uv_available():
    """Check if UV is available."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ UV {result.stdout.strip()} detected")
            return True
    except FileNotFoundError:
        pass
    return False


def install_uv():
    """Install UV package manager."""
    print("📦 Installing UV package manager...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        print("✅ UV installed successfully")
        
        # Try to refresh PATH or use full path
        import site
        # Add user site-packages to path for UV
        user_site = site.getusersitepackages()
        if platform.system() == "Windows":
            scripts_path = os.path.join(user_site, "Scripts")
        else:
            scripts_path = os.path.join(user_site, "bin")
        
        if os.path.exists(scripts_path) and scripts_path not in os.environ["PATH"]:
            os.environ["PATH"] = f"{scripts_path}{os.pathsep}{os.environ['PATH']}"
        
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install UV")
        return False


def create_venv_uv(venv_path: Path, force: bool = False):
    """Create virtual environment using UV."""
    if force and venv_path.exists():
        print("🧹 Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    if not venv_path.exists():
        print("📋 Creating virtual environment with UV...")
        try:
            subprocess.check_call(["uv", "venv", str(venv_path)], shell=True)
            print("✅ Virtual environment created with UV")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ UV venv failed, falling back to python -m venv...")
            return create_venv_standard(venv_path)
    else:
        print("📁 Virtual environment already exists")
        return True


def create_venv_standard(venv_path: Path):
    """Create virtual environment using standard venv."""
    print("📋 Creating virtual environment with venv...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False


def get_python_executable(venv_path: Path):
    """Get the Python executable path in the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies_uv(venv_path: Path, dev: bool = False):
    """Install dependencies using UV."""
    python_exe = get_python_executable(venv_path)
    
    print("🚀 Installing dependencies with UV...")
    try:
        if dev:
            cmd = ["uv", "pip", "install", "-e", ".[dev]", "--python", str(python_exe)]
        else:
            cmd = ["uv", "pip", "install", "-r", "requirements.txt", "--python", str(python_exe)]
        
        subprocess.check_call(cmd, shell=True)
        print("✅ Dependencies installed with UV")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ UV installation failed, falling back to pip...")
        return install_dependencies_pip(python_exe, dev)


def install_dependencies_pip(python_exe: Path, dev: bool = False):
    """Install dependencies using pip."""
    print("📦 Installing dependencies with pip...")
    try:
        # Upgrade pip
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install dependencies
        if dev:
            subprocess.check_call([str(python_exe), "-m", "pip", "install", "-e", ".[dev]"])
        else:
            subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed with pip")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def run_app(venv_path: Path, port: int = 8501, host: str = "localhost"):
    """Run the Streamlit application."""
    python_exe = get_python_executable(venv_path)
    
    if not Path("main.py").exists():
        print("❌ main.py not found!")
        return False
    
    print(f"🚀 Starting APT Fitness Assistant on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        env = os.environ.copy()
        if platform.system() == "Windows":
            # Add Scripts to PATH for Windows
            scripts_path = venv_path / "Scripts"
            env["PATH"] = f"{scripts_path}{os.pathsep}{env['PATH']}"
        else:
            # Add bin to PATH for Unix-like systems
            bin_path = venv_path / "bin"
            env["PATH"] = f"{bin_path}{os.pathsep}{env['PATH']}"
        
        subprocess.check_call([
            str(python_exe), "-m", "streamlit", "run", "main.py",
            "--server.port", str(port),
            "--server.address", host
        ], env=env)
        
    except subprocess.CalledProcessError:
        print("❌ Failed to start the application")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="APT Fitness Assistant Setup and Runner")
    parser.add_argument("--setup", action="store_true", help="Setup the environment")
    parser.add_argument("--run", action="store_true", help="Run the application")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--force", action="store_true", help="Force recreate virtual environment")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on")
    parser.add_argument("--host", default="localhost", help="Host to run the app on")
    parser.add_argument("--no-uv", action="store_true", help="Don't use UV, use pip instead")
    
    args = parser.parse_args()
    
    print("🏋️‍♀️ APT Fitness Assistant")
    print("=" * 40)
    
    if not check_python_version():
        return 1
    
    venv_path = Path(".venv")
    use_uv = not args.no_uv
    
    # Check UV availability if requested
    if use_uv:
        if not check_uv_available():
            if install_uv():
                use_uv = True
            else:
                print("⚠️ Falling back to standard pip/venv workflow")
                use_uv = False
    
    # Setup if requested or if running and venv doesn't exist
    if args.setup or (args.run and not venv_path.exists()):
        print("\n📋 Setting up environment...")
        
        # Create virtual environment
        if use_uv:
            if not create_venv_uv(venv_path, args.force):
                return 1
        else:
            if not create_venv_standard(venv_path):
                return 1
        
        # Install dependencies
        if use_uv:
            if not install_dependencies_uv(venv_path, args.dev):
                return 1
        else:
            python_exe = get_python_executable(venv_path)
            if not install_dependencies_pip(python_exe, args.dev):
                return 1
        
        print("\n🎉 Setup complete!")
        
        if not args.run:
            print("\n🚀 To run the application:")
            print(f"   python {__file__} --run")
            print(f"   or: {get_python_executable(venv_path)} -m streamlit run main.py")
    
    # Run the application
    if args.run:
        if not venv_path.exists():
            print("❌ Virtual environment not found! Run with --setup first.")
            return 1
        
        print("\n🚀 Starting application...")
        if not run_app(venv_path, args.port, args.host):
            return 1
    
    # If no specific action, show help
    if not args.setup and not args.run:
        print("\n🔧 Usage:")
        print(f"   Setup:  python {__file__} --setup")
        print(f"   Run:    python {__file__} --run")
        print(f"   Both:   python {__file__} --setup --run")
        print(f"   Help:   python {__file__} --help")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
