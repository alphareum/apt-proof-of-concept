@echo off
REM APT Fitness Assistant Setup - UV Package Manager
REM Windows batch script for quick setup

echo 🏋️‍♀️ APT Fitness Assistant Setup (UV)
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if UV is installed, install if not
echo.
echo 🔍 Checking UV installation...

uv --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing UV package manager...
    python -m pip install uv
    if errorlevel 1 (
        echo ❌ Failed to install UV. Please install manually: pip install uv
        pause
        exit /b 1
    )
    echo ✅ UV installed successfully
) else (
    echo ✅ UV already installed
)

REM Create virtual environment
echo.
echo 📋 Setting up virtual environment...

if not exist ".venv" (
    echo Creating .venv with UV...
    uv venv .venv
    if errorlevel 1 (
        echo ⚠️ UV venv failed, falling back to python -m venv...
        python -m venv .venv
    )
    echo ✅ Virtual environment created
) else (
    echo 📁 Virtual environment already exists
)

REM Install dependencies
echo.
echo 📦 Installing dependencies...

echo 🚀 Using UV for fast dependency installation...
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

if errorlevel 1 (
    echo ⚠️ UV installation failed, trying with pip...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo 🎉 Setup Complete!
echo ========================================
echo.
echo 🚀 To run the application:
echo 1. Activate the environment: .venv\Scripts\activate.bat
echo 2. Run the fitness app: streamlit run app.py
echo.
echo 🔧 Or use the quick start: run-app.bat
echo.
echo 📚 For more options, see README.md

pause
