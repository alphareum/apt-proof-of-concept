#!/usr/bin/env python3
"""
APT Fitness Assistant - Troubleshooting Script
Helps identify and fix common issues including AxiosError status 400
"""

import sys
import os
import platform
import subprocess
import json
from pathlib import Path


def check_streamlit_config():
    """Check Streamlit configuration for common issues."""
    print("🔍 Checking Streamlit configuration...")
    
    config_path = Path(".streamlit/config.toml")
    if config_path.exists():
        print("✅ Streamlit config found")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            print("📄 Config content:")
            print(content)
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print("⚠️ No Streamlit config found")
    print()


def check_port_usage():
    """Check if port 8501 is available."""
    print("🔍 Checking port usage...")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-Command", "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                print("⚠️ Port 8501 is in use:")
                print(result.stdout)
            else:
                print("✅ Port 8501 is available")
        else:
            result = subprocess.run(["lsof", "-i", ":8501"], capture_output=True, text=True)
            if result.stdout.strip():
                print("⚠️ Port 8501 is in use:")
                print(result.stdout)
            else:
                print("✅ Port 8501 is available")
    except Exception as e:
        print(f"❌ Error checking port: {e}")
    print()


def check_browser_cache():
    """Provide instructions for clearing browser cache."""
    print("🔍 Browser cache troubleshooting:")
    print("If you're experiencing AxiosError 400, try:")
    print("1. Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (Mac)")
    print("2. Clear browser cache and cookies for localhost")
    print("3. Try incognito/private browsing mode")
    print("4. Try a different browser")
    print()


def check_file_permissions():
    """Check file permissions."""
    print("🔍 Checking file permissions...")
    
    files_to_check = ["main.py", "requirements.txt", ".streamlit/config.toml"]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    f.read(1)  # Try to read
                print(f"✅ {file_path} - readable")
            except Exception as e:
                print(f"❌ {file_path} - {e}")
        else:
            print(f"⚠️ {file_path} - not found")
    print()


def check_python_environment():
    """Check Python environment."""
    print("🔍 Checking Python environment...")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment")
    
    # Check key packages
    required_packages = ["streamlit", "pandas", "numpy", "pillow"]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
    print()


def fix_common_issues():
    """Attempt to fix common issues."""
    print("🔧 Attempting to fix common issues...")
    
    # Clear Streamlit cache
    try:
        cache_dir = Path.home() / ".streamlit"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir / "cache", ignore_errors=True)
            print("✅ Cleared Streamlit cache")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")
    
    # Recreate config with safe settings
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    safe_config = """[server]
port = 8501
headless = true
maxUploadSize = 200
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = false
runOnSave = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
base = "light"

[logger]
level = "info"
"""
    
    try:
        with open(config_dir / "config.toml", 'w') as f:
            f.write(safe_config)
        print("✅ Updated Streamlit config with safe settings")
    except Exception as e:
        print(f"❌ Could not update config: {e}")
    
    print()


def generate_debug_info():
    """Generate debug information."""
    print("🔍 Generating debug information...")
    
    debug_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "environment_variables": {
            key: value for key, value in os.environ.items() 
            if any(term in key.lower() for term in ['python', 'path', 'streamlit'])
        }
    }
    
    try:
        with open("debug_info.json", 'w') as f:
            json.dump(debug_info, f, indent=2)
        print("✅ Debug info saved to debug_info.json")
    except Exception as e:
        print(f"❌ Could not save debug info: {e}")
    
    print()


def main():
    """Main troubleshooting function."""
    print("🏋️‍♀️ APT Fitness Assistant - Troubleshooter")
    print("=" * 50)
    print()
    
    check_python_environment()
    check_streamlit_config()
    check_port_usage()
    check_file_permissions()
    check_browser_cache()
    
    print("🔧 Would you like to attempt automatic fixes? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            fix_common_issues()
            generate_debug_info()
            
            print("🎉 Troubleshooting complete!")
            print()
            print("📋 Next steps:")
            print("1. Restart the application: python run.py --setup --run")
            print("2. If issues persist, try: python run.py --setup --force --run")
            print("3. Use incognito/private browsing mode")
            print("4. Check debug_info.json for detailed system information")
        else:
            print("Troubleshooting cancelled.")
    except KeyboardInterrupt:
        print("\nTroubleshooting cancelled.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
