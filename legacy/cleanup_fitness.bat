@echo off
echo 🧹 AI Fitness Assistant - Quick Cleanup
echo =====================================
echo.
echo This will remove all APT (original project) files and keep only
echo the fitness app files you need.
echo.
echo Files to keep:
echo   ✅ fitness_app.py
echo   ✅ requirements_fitness.txt  
echo   ✅ run_fitness_app.py
echo   ✅ setup files and documentation
echo   ✅ fitness_venv (if exists)
echo.
echo Files to remove:
echo   ❌ All original APT project files
echo   ❌ Original requirements.txt and dependencies
echo   ❌ Docker files and configurations
echo   ❌ Database and upload directories
echo.

set /p confirm="Do you want to proceed? (y/n): "
if /i "%confirm%" neq "y" (
    echo ❌ Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo 🧹 Starting cleanup...

:: Remove APT application files
if exist "ai_processor.py" del "ai_processor.py" && echo    📄 Removed ai_processor.py
if exist "app.py" del "app.py" && echo    📄 Removed app.py
if exist "config.py" del "config.py" && echo    📄 Removed config.py
if exist "demo.py" del "demo.py" && echo    📄 Removed demo.py
if exist "frontend_improved.py" del "frontend_improved.py" && echo    📄 Removed frontend_improved.py
if exist "main.py" del "main.py" && echo    📄 Removed main.py
if exist "models.py" del "models.py" && echo    📄 Removed models.py

:: Remove APT documentation
if exist "CHANGELOG.md" del "CHANGELOG.md" && echo    📄 Removed CHANGELOG.md
if exist "DEPLOYMENT.md" del "DEPLOYMENT.md" && echo    📄 Removed DEPLOYMENT.md
if exist "README.md" del "README.md" && echo    📄 Removed README.md

:: Remove APT configuration files  
if exist "requirements.txt" del "requirements.txt" && echo    📄 Removed requirements.txt
if exist "pyproject.toml" del "pyproject.toml" && echo    📄 Removed pyproject.toml
if exist "setup.py" del "setup.py" && echo    📄 Removed setup.py
if exist "uv.lock" del "uv.lock" && echo    📄 Removed uv.lock
if exist ".python-version" del ".python-version" && echo    📄 Removed .python-version

:: Remove Docker files
if exist "docker-compose.yml" del "docker-compose.yml" && echo    📄 Removed docker-compose.yml
if exist "Dockerfile" del "Dockerfile" && echo    📄 Removed Dockerfile
if exist "Dockerfile.frontend" del "Dockerfile.frontend" && echo    📄 Removed Dockerfile.frontend
if exist ".dockerignore" del ".dockerignore" && echo    📄 Removed .dockerignore

:: Remove APT directories
if exist "instance" rmdir /s /q "instance" && echo    📁 Removed instance/
if exist "logs" rmdir /s /q "logs" && echo    📁 Removed logs/
if exist "nginx" rmdir /s /q "nginx" && echo    📁 Removed nginx/
if exist "pose_data" rmdir /s /q "pose_data" && echo    📁 Removed pose_data/
if exist "uploads" rmdir /s /q "uploads" && echo    📁 Removed uploads/
if exist "__pycache__" rmdir /s /q "__pycache__" && echo    📁 Removed __pycache__/
if exist "venv" rmdir /s /q "venv" && echo    📁 Removed venv/
if exist ".venv" rmdir /s /q ".venv" && echo    📁 Removed .venv/

echo.
echo ✅ Cleanup completed!
echo.
echo 🏋️‍♀️ Your fitness app is now ready!
echo.
echo 🚀 To run the app:
echo    Option 1: python run_fitness_app.py
echo    Option 2: setup_fitness_app.bat
echo.
pause
