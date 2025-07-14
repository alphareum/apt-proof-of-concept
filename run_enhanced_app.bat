@echo off
echo ====================================
echo   AI Fitness Assistant Pro v3.0
echo ====================================

REM Activate virtual environment
call fitness_venv\Scripts\activate.bat

REM Install requirements if needed
echo Installing/updating requirements...
pip install -r requirements_enhanced.txt

REM Run the enhanced fitness app
echo Starting AI Fitness Assistant Pro...
streamlit run fitness_app_enhanced.py --server.port 8502

pause
