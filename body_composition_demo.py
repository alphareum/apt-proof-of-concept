"""
Body Composition Analysis Demo
Demonstrates the body composition analysis functionality
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Body Composition Analysis Demo",
        page_icon="🏋️",
        layout="wide"
    )
    
    st.title("🏋️ Body Composition Analysis Demo")
    st.markdown("---")
    
    st.markdown("""
    ## What is Body Composition Analysis?
    
    This feature uses computer vision and AI to analyze your body composition from photos:
    
    ### 🎯 Key Features:
    - **Body Fat Percentage**: Estimated using pose landmarks and body ratios
    - **Muscle Mass**: Calculated based on body shape and proportions  
    - **BMR Estimation**: Basal Metabolic Rate for calorie planning
    - **Body Shape Classification**: Athletic, pear, apple, etc.
    - **Progress Tracking**: Compare changes over time
    - **Detailed Measurements**: Shoulder width, waist-to-hip ratio, etc.
    
    ### 🔬 How it works:
    1. **Pose Detection**: MediaPipe extracts 33 body landmarks
    2. **Measurement Extraction**: Calculate body ratios and proportions
    3. **ML Estimation**: Random Forest models predict composition
    4. **Analysis**: Generate detailed breakdown and recommendations
    5. **Progress Tracking**: Store results for comparison over time
    
    ### 📊 What you get:
    - Body fat percentage (with health assessment)
    - Muscle mass percentage
    - Visceral fat level (1-20 scale)
    - BMR estimation for diet planning
    - Body shape classification
    - Detailed body measurements
    - Progress visualization
    - Comparison tools
    """)
    
    # Try to load the actual component
    try:
        from body_composition_ui import render_body_composition_analysis
        
        st.success("✅ Body composition analysis is available!")
        
        if st.button("🚀 Launch Body Composition Analysis"):
            st.markdown("---")
            render_body_composition_analysis()
            
    except ImportError as e:
        st.warning("⚠️ Body composition analysis not fully available.")
        st.error(f"Import error: {e}")
        
        st.markdown("### 📦 Required Dependencies:")
        st.code("""
        # Install required packages
        pip install opencv-python
        pip install mediapipe
        pip install scikit-learn
        pip install tensorflow
        pip install streamlit
        pip install pandas
        pip install numpy
        pip install Pillow
        """)
        
        # Show mock interface
        st.markdown("---")
        st.markdown("### 📸 Mock Interface Preview")
        
        mock_demo()

def mock_demo():
    """Show a mock demo of the interface."""
    
    # Mock upload
    uploaded_file = st.file_uploader(
        "Upload body image for analysis", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a full-body image for analysis"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        if st.button("🔍 Analyze Body Composition (Mock)"):
            # Mock analysis
            with st.spinner("Analyzing... (This is a mock demo)"):
                import time
                time.sleep(2)
            
            # Mock results
            st.success("✅ Analysis Complete (Mock Results)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Body Fat %", "18.5%", help="Estimated body fat percentage")
            
            with col2:
                st.metric("Muscle Mass %", "42.3%", help="Estimated muscle mass percentage")
            
            with col3:
                st.metric("BMR", "1,847 cal/day", help="Estimated Basal Metabolic Rate")
            
            with col4:
                st.metric("Confidence", "0.85", help="Analysis confidence score")
            
            # Mock details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Body Assessment")
                st.write("**Body Shape:** Athletic")
                st.write("**Visceral Fat Level:** 8/20")
                st.write("**Health Status:** ✅ Healthy")
            
            with col2:
                st.markdown("### 📈 Composition Breakdown")
                st.write("**Fat Mass:** 13.7 kg")
                st.write("**Muscle Mass:** 31.4 kg")
                st.write("**Bone Mass:** 11.1 kg")
                st.write("**Water %:** 39.2%")
            
            st.info("💡 This is a mock demonstration. Install the required dependencies to use the real analysis.")

if __name__ == "__main__":
    main()
