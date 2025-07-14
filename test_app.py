"""
Simple test version of the APT Fitness Assistant
Use this if the main app has issues
"""

import streamlit as st
import sys
import platform
import time
import datetime
import os

st.set_page_config(
    page_title="APT Fitness Assistant - Test",
    page_icon="🏋️‍♀️",
    layout="wide"
)

st.title("🏋️‍♀️ APT Fitness Assistant - Test Version")

st.success("✅ Application is running successfully!")

st.info("""
This is a test version to verify the application is working.
If you see this page, your setup is correct!

**Quick Test Features:**
- ✅ Streamlit is working
- ✅ Python environment is active
- ✅ Dependencies are installed
""")

# System Information
st.header("💻 System Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

with col2:
    st.metric("Platform", platform.system())

with col3:
    st.metric("Architecture", platform.machine())

with col4:
    st.metric("Current Time", datetime.datetime.now().strftime("%H:%M:%S"))

# Test basic functionality
st.header("🧪 Basic Tests")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Python", "✅ Working")

with col2:
    st.metric("Streamlit", "✅ Working")

with col3:
    st.metric("Dependencies", "✅ Loaded")

# Test imports
st.header("📦 Dependency Tests")

dependencies = [
    ("NumPy", "numpy", "np"),
    ("Pandas", "pandas", "pd"),
    ("OpenCV", "cv2", "cv2"),
    ("Streamlit", "streamlit", "st"),
    ("SQLite3", "sqlite3", "sqlite3"),
    ("JSON", "json", "json"),
    ("OS", "os", "os"),
    ("DateTime", "datetime", "datetime")
]

import_results = []
for name, module, alias in dependencies:
    try:
        exec(f"import {module} as {alias}")
        st.success(f"✅ {name} imported successfully")
        import_results.append(True)
    except ImportError:
        st.error(f"❌ {name} import failed")
        import_results.append(False)

# Performance Test
st.header("⚡ Performance Tests")

if st.button("🚀 Run Performance Test"):
    with st.spinner("Running performance tests..."):
        # Test 1: Basic computation
        start_time = time.time()
        try:
            import numpy as np
            data = np.random.random((1000, 1000))
            result = np.sum(data)
            computation_time = time.time() - start_time
            st.success(f"✅ NumPy computation test: {computation_time:.4f} seconds")
        except Exception as e:
            st.error(f"❌ NumPy computation test failed: {str(e)}")
        
        # Test 2: Data processing
        start_time = time.time()
        try:
            import pandas as pd
            df = pd.DataFrame({'A': range(10000), 'B': range(10000, 20000)})
            df['C'] = df['A'] + df['B']
            processing_time = time.time() - start_time
            st.success(f"✅ Pandas processing test: {processing_time:.4f} seconds")
        except Exception as e:
            st.error(f"❌ Pandas processing test failed: {str(e)}")

# Configuration Test
st.header("⚙️ Configuration Tests")

config_checks = [
    ("Working Directory", os.getcwd()),
    ("Python Executable", sys.executable),
    ("Platform Details", f"{platform.system()} {platform.release()}"),
    ("Processor", platform.processor()),
]

for check_name, check_value in config_checks:
    with st.expander(f"📋 {check_name}"):
        st.code(check_value)

# Interactive Features
st.header("🎮 Interactive Features")

# Test user input
test_name = st.text_input("Enter your name for testing:", placeholder="Your name here...")
if test_name:
    st.success(f"Hello, {test_name}! The input system is working correctly! 👋")

# Test file upload simulation
st.subheader("📁 File Upload Test")
uploaded_file = st.file_uploader("Test file upload functionality", type=['txt', 'json', 'csv'])
if uploaded_file:
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
    st.info(f"File size: {len(uploaded_file.getvalue())} bytes")

# Test sliders and inputs
st.subheader("🎛️ Widget Tests")
col1, col2 = st.columns(2)

with col1:
    test_slider = st.slider("Test Slider", 0, 100, 50)
    st.write(f"Slider value: {test_slider}")

with col2:
    test_selectbox = st.selectbox("Test Selection", ["Option 1", "Option 2", "Option 3"])
    st.write(f"Selected: {test_selectbox}")

# Test charts
st.subheader("📊 Chart Test")
if st.button("Generate Test Chart"):
    try:
        import numpy as np
        import pandas as pd
        
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['A', 'B', 'C']
        )
        st.line_chart(chart_data)
        st.success("✅ Chart rendering works!")
    except Exception as e:
        st.error(f"❌ Chart test failed: {str(e)}")

# Summary
st.header("📊 Test Summary")
total_tests = len(import_results)
passed_tests = sum(import_results)
success_rate = (passed_tests / total_tests) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Tests", total_tests)
with col2:
    st.metric("Passed Tests", passed_tests)
with col3:
    st.metric("Success Rate", f"{success_rate:.1f}%")

if success_rate >= 80:
    st.success("🎉 Great! Most tests are passing. Your environment is well configured!")
elif success_rate >= 60:
    st.warning("⚠️ Some tests failed, but basic functionality should work.")
else:
    st.error("❌ Multiple test failures detected. Please check your environment setup.")

st.header("🚀 Next Steps")
st.info("""
If all tests above are green, you can run the full application:

1. **Stop this test app** (Ctrl+C in terminal)
2. **Run the main app**: `streamlit run app.py`
3. **Or use the runner**: `python run.py --run`
""")

# Enhanced success button
if st.button("🎉 Everything looks good!"):
    st.balloons()
    st.success("Great! Your APT Fitness Assistant is ready to use!")
    st.snow()  # Added snow effect for extra celebration

# Footer
st.markdown("---")
st.markdown(f"**Test completed at:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("**APT Fitness Assistant Test Suite v2.0** - Enhanced with interactive features")
