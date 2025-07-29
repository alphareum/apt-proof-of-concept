@echo off
echo =======================================================
echo Body Composition Analysis Setup
echo =======================================================
echo.

echo Installing required dependencies...
echo.

echo Installing computer vision packages...
pip install opencv-python==4.9.0.80
pip install mediapipe==0.10.9

echo Installing machine learning packages...
pip install scikit-learn==1.3.2
pip install tensorflow>=2.12.0
pip install scipy>=1.10.0

echo Installing image processing packages...
pip install Pillow==10.1.0
pip install imagehash>=4.3.0

echo Installing data science packages...
pip install numpy==1.26.2
pip install pandas==2.1.4

echo Installing web framework packages...
pip install streamlit==1.29.0
pip install flask>=2.3.0
pip install flask-cors>=4.0.0

echo.
echo =======================================================
echo Testing installation...
echo =======================================================

python -c "import cv2; print('✓ OpenCV installed')"
python -c "import mediapipe; print('✓ MediaPipe installed')"
python -c "import sklearn; print('✓ Scikit-learn installed')"
python -c "import tensorflow; print('✓ TensorFlow installed')"
python -c "import streamlit; print('✓ Streamlit installed')"

echo.
echo =======================================================
echo Setup complete!
echo =======================================================
echo.
echo To test the body composition analysis:
echo.
echo 1. Run the demo:
echo    streamlit run body_composition_demo.py
echo.
echo 2. Run the main fitness app:
echo    streamlit run fitness_app_enhanced.py
echo.
echo 3. Start the REST API:
echo    python body_composition_api.py
echo.
echo =======================================================

pause
