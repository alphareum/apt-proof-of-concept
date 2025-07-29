@echo off
echo 🏋️‍♀️ AI Fitness Assistant Setup
echo ========================================

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected

:: Create virtual environment for fitness app
echo.
echo 📋 Creating virtual environment for fitness app...
if not exist "fitness_venv" (
    python -m venv fitness_venv
    echo ✅ Virtual environment created
) else (
    echo 📁 Virtual environment already exists
)

:: Activate virtual environment and install dependencies
echo.
echo 📋 Installing fitness app dependencies...
call fitness_venv\Scripts\activate.bat

echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

echo 🔧 Installing requirements...
python -m pip install -r requirements_fitness.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements
    echo.
    echo 💡 Manual installation steps:
    echo    1. fitness_venv\Scripts\activate
    echo    2. pip install --upgrade pip
    echo    3. pip install -r requirements_fitness.txt
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 To run the fitness app:
echo    1. fitness_venv\Scripts\activate
echo    2. streamlit run fitness_app.py
echo.
echo Or simply run: run_fitness_app.py
echo.
pause
