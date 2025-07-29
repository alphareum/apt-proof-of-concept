@echo off
echo 🏋️‍♀️ Starting APT Fitness Assistant...
echo ========================================

cd /d "d:\Works\Researchs\apt-proof-of-concept"

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️ No virtual environment found, using global Python
)

REM Start Streamlit app
echo 🚀 Starting Streamlit app...
python -m streamlit run app.py --server.port 8501 --server.address localhost

pause
