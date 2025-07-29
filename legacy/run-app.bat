@echo off
REM APT Fitness Assistant - Quick Run Script
REM Activates .venv and runs the application

echo 🏋️‍♀️ Starting APT Fitness Assistant...

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup-uv.bat first to create the environment.
    pause
    exit /b 1
)

REM Check if app.py exists
if not exist "app.py" (
    echo ❌ app.py not found!
    echo Please ensure you're in the correct directory.
    pause
    exit /b 1
)

echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo 🚀 Starting Streamlit server...
echo 📱 App will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run streamlit
streamlit run app.py --server.port 8501

echo.
echo 👋 Application stopped.
pause
