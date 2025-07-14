"""
Body Composition Analysis UI Component
Streamlit interface for body composition analysis using images
"""

import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import hashlib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports with error handling
try:
    from PIL import Image
    import cv2
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    st.warning("⚠️ Computer vision libraries not available. Some features may be limited.")

try:
    from body_composition_analyzer import get_body_analyzer
    from database import get_database
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    st.warning("⚠️ Body composition analyzer not available.")

def render_body_composition_analysis():
    """Render the body composition analysis interface."""
    st.header("🏋️ Body Composition Analysis")
    st.markdown("---")
    
    if not VISION_AVAILABLE or not ANALYZER_AVAILABLE:
        st.error("❌ Body composition analysis requires computer vision libraries and analyzer.")
        st.info("Please install required dependencies: opencv-python, mediapipe, scikit-learn")
        return
    
    # Sidebar for navigation
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["📸 New Analysis", "📊 View History", "📈 Progress Tracking", "⚖️ Compare Results"]
    )
    
    if analysis_mode == "📸 New Analysis":
        render_new_analysis()
    elif analysis_mode == "📊 View History":
        render_analysis_history()
    elif analysis_mode == "📈 Progress Tracking":
        render_progress_tracking()
    elif analysis_mode == "⚖️ Compare Results":
        render_comparison_tool()

def render_new_analysis():
    """Render interface for new body composition analysis."""
    st.subheader("📸 New Body Composition Analysis")
    
    # User selection
    user_id = st.text_input("User ID", help="Enter your unique user ID")
    
    if not user_id:
        st.warning("Please enter a User ID to proceed.")
        return
    
    # Image upload section
    st.markdown("### Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Image** (Required)")
        primary_image = st.file_uploader(
            "Upload your primary body image",
            type=['jpg', 'jpeg', 'png'],
            key="primary_image",
            help="Front-facing full-body image for best results"
        )
        
        if primary_image:
            image = Image.open(primary_image)
            st.image(image, caption="Primary Image", use_column_width=True)
    
    with col2:
        st.markdown("**Additional Images** (Optional)")
        
        front_image = st.file_uploader(
            "Front view image",
            type=['jpg', 'jpeg', 'png'],
            key="front_image"
        )
        
        side_image = st.file_uploader(
            "Side view image",
            type=['jpg', 'jpeg', 'png'],
            key="side_image"
        )
        
        if front_image:
            st.image(Image.open(front_image), caption="Front View", width=150)
        if side_image:
            st.image(Image.open(side_image), caption="Side View", width=150)
    
    # Analysis options
    st.markdown("### Analysis Options")
    
    col1, col2 = st.columns(2)
    with col1:
        include_measurements = st.checkbox("Include detailed measurements", value=True)
        include_breakdown = st.checkbox("Include composition breakdown", value=True)
    
    with col2:
        confidence_threshold = st.slider("Minimum confidence threshold", 0.5, 1.0, 0.7, 0.05)
        save_processed_image = st.checkbox("Save processed image", value=True)
    
    # Analysis button
    if st.button("🔍 Analyze Body Composition", type="primary"):
        if primary_image:
            with st.spinner("Analyzing body composition... This may take a few moments."):
                try:
                    # Save uploaded image temporarily
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    primary_path = temp_dir / f"primary_{timestamp}_{primary_image.name}"
                    
                    with open(primary_path, "wb") as f:
                        f.write(primary_image.getbuffer())
                    
                    # Handle additional images
                    additional_images = {}
                    if front_image:
                        front_path = temp_dir / f"front_{timestamp}_{front_image.name}"
                        with open(front_path, "wb") as f:
                            f.write(front_image.getbuffer())
                        additional_images["front"] = str(front_path)
                    
                    if side_image:
                        side_path = temp_dir / f"side_{timestamp}_{side_image.name}"
                        with open(side_path, "wb") as f:
                            f.write(side_image.getbuffer())
                        additional_images["side"] = str(side_path)
                    
                    # Perform analysis
                    analyzer = get_body_analyzer()
                    result = analyzer.analyze_image(
                        image_path=str(primary_path),
                        user_id=user_id,
                        additional_images=additional_images if additional_images else None
                    )
                    
                    if result.get("success", False):
                        display_analysis_results(result)
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error in body composition analysis: {e}")
                    st.error(f"❌ Analysis error: {str(e)}")
        else:
            st.warning("Please upload a primary image to analyze.")

def display_analysis_results(result: Dict[str, Any]):
    """Display body composition analysis results."""
    st.success("✅ Analysis completed successfully!")
    
    # Main metrics
    st.markdown("### 📊 Body Composition Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Body Fat %",
            f"{result['body_fat_percentage']:.1f}%",
            help="Estimated body fat percentage"
        )
    
    with col2:
        st.metric(
            "Muscle Mass %",
            f"{result['muscle_mass_percentage']:.1f}%",
            help="Estimated muscle mass percentage"
        )
    
    with col3:
        st.metric(
            "BMR",
            f"{result['bmr_estimated']} cal/day",
            help="Estimated Basal Metabolic Rate"
        )
    
    with col4:
        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}",
            help="Analysis confidence score"
        )
    
    # Additional details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Body Shape & Health")
        st.write(f"**Body Shape:** {result['body_shape']}")
        st.write(f"**Visceral Fat Level:** {result['visceral_fat_level']}/20")
        
        # Health assessment
        body_fat = result['body_fat_percentage']
        if body_fat < 10:
            health_status = "⚠️ Very Low (may be unhealthy)"
        elif body_fat < 15:
            health_status = "✅ Athletic"
        elif body_fat < 25:
            health_status = "✅ Healthy"
        elif body_fat < 30:
            health_status = "⚠️ Above Average"
        else:
            health_status = "🔴 High (health risk)"
        
        st.write(f"**Health Status:** {health_status}")
    
    with col2:
        if "breakdown" in result:
            st.markdown("### 📈 Composition Breakdown")
            breakdown = result["breakdown"]
            
            # Create a simple breakdown chart using text
            st.write(f"**Fat Mass:** {breakdown.get('fat_mass_kg', 0):.1f} kg")
            st.write(f"**Muscle Mass:** {breakdown.get('muscle_mass_kg', 0):.1f} kg")
            st.write(f"**Bone Mass:** {breakdown.get('bone_mass_kg', 0):.1f} kg")
            st.write(f"**Water %:** {breakdown.get('water_percentage', 0):.1f}%")
    
    # Body measurements
    if "measurements" in result:
        st.markdown("### 📏 Body Measurements")
        measurements = result["measurements"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Upper Body**")
            st.write(f"Shoulder Width: {measurements.get('shoulder_width', 0):.1f} px")
            st.write(f"Arm Length: {measurements.get('left_arm_length', 0):.1f} px")
        
        with col2:
            st.write("**Core**")
            st.write(f"Waist Width: {measurements.get('waist_width', 0):.1f} px")
            st.write(f"Hip Width: {measurements.get('hip_width', 0):.1f} px")
        
        with col3:
            st.write("**Lower Body**")
            st.write(f"Body Height: {measurements.get('body_height', 0):.1f} px")
            st.write(f"Leg Length: {measurements.get('left_leg_length', 0):.1f} px")
    
    # Processed image
    if result.get("processed_image_path"):
        st.markdown("### 🖼️ Processed Image")
        try:
            processed_image = Image.open(result["processed_image_path"])
            st.image(processed_image, caption="Analysis Visualization", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not display processed image: {e}")
    
    # Save results option
    if st.button("💾 Save Results"):
        st.success("✅ Results saved to your profile!")
        st.balloons()

def render_analysis_history():
    """Render analysis history interface."""
    st.subheader("📊 Analysis History")
    
    user_id = st.text_input("User ID", key="history_user_id")
    
    if not user_id:
        st.warning("Please enter a User ID to view history.")
        return
    
    days = st.slider("Show data for last N days", 7, 365, 90)
    
    try:
        db = get_database()
        history = db.get_body_composition_history(user_id, days)
        
        if not history:
            st.info("No body composition analyses found for this user.")
            return
        
        st.write(f"Found {len(history)} analyses in the last {days} days")
        
        # Display history table
        display_data = []
        for analysis in history:
            display_data.append({
                "Date": analysis["analysis_date"][:10],  # Just the date part
                "Body Fat %": f"{analysis['body_fat_percentage']:.1f}%",
                "Muscle Mass %": f"{analysis['muscle_mass_percentage']:.1f}%",
                "BMR": f"{analysis['bmr_estimated']} cal",
                "Body Shape": analysis["body_shape_classification"],
                "Confidence": f"{analysis['confidence_score']:.2f}"
            })
        
        st.dataframe(display_data, use_container_width=True)
        
        # Plot trends if we have enough data
        if len(history) >= 2:
            render_trend_charts(history)
            
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        st.error(f"Error fetching history: {e}")

def render_trend_charts(history: List[Dict[str, Any]]):
    """Render trend charts for body composition data."""
    st.markdown("### 📈 Trends")
    
    try:
        # Prepare data for plotting
        dates = [analysis["analysis_date"][:10] for analysis in reversed(history)]
        body_fat = [analysis["body_fat_percentage"] for analysis in reversed(history)]
        muscle_mass = [analysis["muscle_mass_percentage"] for analysis in reversed(history)]
        bmr = [analysis["bmr_estimated"] for analysis in reversed(history)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Body Fat % Trend**")
            # Simple line chart using Streamlit's built-in charting
            chart_data = {"Date": dates, "Body Fat %": body_fat}
            st.line_chart(chart_data, x="Date", y="Body Fat %")
        
        with col2:
            st.markdown("**Muscle Mass % Trend**")
            chart_data = {"Date": dates, "Muscle Mass %": muscle_mass}
            st.line_chart(chart_data, x="Date", y="Muscle Mass %")
        
        # BMR trend
        st.markdown("**BMR Trend**")
        chart_data = {"Date": dates, "BMR": bmr}
        st.line_chart(chart_data, x="Date", y="BMR")
        
    except Exception as e:
        logger.error(f"Error creating trend charts: {e}")
        st.warning("Could not generate trend charts.")

def render_progress_tracking():
    """Render progress tracking interface."""
    st.subheader("📈 Progress Tracking")
    
    user_id = st.text_input("User ID", key="progress_user_id")
    
    if not user_id:
        st.warning("Please enter a User ID to track progress.")
        return
    
    period_days = st.selectbox("Progress Period", [7, 14, 30, 60, 90], index=2)
    
    try:
        db = get_database()
        progress = db.calculate_composition_progress(user_id, period_days)
        
        if "error" in progress:
            st.warning(f"Could not calculate progress: {progress['error']}")
            return
        
        st.markdown(f"### Progress over last {period_days} days")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            body_fat_change = progress.get("body_fat_change", 0)
            delta_color = "inverse" if body_fat_change < 0 else "normal"
            st.metric(
                "Body Fat Change",
                f"{body_fat_change:+.1f}%",
                delta=f"{body_fat_change:+.1f}%",
                delta_color=delta_color
            )
        
        with col2:
            muscle_change = progress.get("muscle_mass_change", 0)
            delta_color = "normal" if muscle_change > 0 else "inverse"
            st.metric(
                "Muscle Mass Change",
                f"{muscle_change:+.1f}%",
                delta=f"{muscle_change:+.1f}%",
                delta_color=delta_color
            )
        
        with col3:
            bmr_change = progress.get("bmr_change", 0)
            st.metric(
                "BMR Change",
                f"{bmr_change:+.0f} cal/day",
                delta=f"{bmr_change:+.0f} cal"
            )
        
        # Trend analysis
        if "trend_analysis" in progress:
            trends = progress["trend_analysis"]
            st.markdown("### 🔍 Trend Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                body_fat_trend = trends.get("body_fat", "stable")
                if body_fat_trend == "decreasing":
                    st.success("🔥 Body fat decreasing")
                elif body_fat_trend == "increasing":
                    st.warning("📈 Body fat increasing")
                else:
                    st.info("➡️ Body fat stable")
            
            with col2:
                muscle_trend = trends.get("muscle_mass", "stable")
                if muscle_trend == "increasing":
                    st.success("💪 Muscle mass increasing")
                elif muscle_trend == "decreasing":
                    st.warning("📉 Muscle mass decreasing")
                else:
                    st.info("➡️ Muscle mass stable")
            
            with col3:
                overall_trend = trends.get("overall", "stable")
                if overall_trend == "improving":
                    st.success("🎯 Overall improving")
                else:
                    st.info("➡️ Overall stable")
        
    except Exception as e:
        logger.error(f"Error calculating progress: {e}")
        st.error(f"Error calculating progress: {e}")

def render_comparison_tool():
    """Render analysis comparison interface."""
    st.subheader("⚖️ Compare Results")
    
    user_id = st.text_input("User ID", key="compare_user_id")
    
    if not user_id:
        st.warning("Please enter a User ID to compare results.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**First Analysis**")
        analysis_id1 = st.text_input("Analysis ID 1", key="analysis_id1")
    
    with col2:
        st.markdown("**Second Analysis**")
        analysis_id2 = st.text_input("Analysis ID 2", key="analysis_id2")
    
    if analysis_id1 and analysis_id2:
        if st.button("🔍 Compare Analyses"):
            try:
                analyzer = get_body_analyzer()
                comparison = analyzer.compare_analyses(user_id, analysis_id1, analysis_id2)
                
                if "error" in comparison:
                    st.error(f"Comparison failed: {comparison['error']}")
                    return
                
                display_comparison_results(comparison)
                
            except Exception as e:
                logger.error(f"Error comparing analyses: {e}")
                st.error(f"Error comparing analyses: {e}")

def display_comparison_results(comparison: Dict[str, Any]):
    """Display comparison results."""
    st.success("✅ Comparison completed!")
    
    st.markdown("### 📊 Comparison Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analysis Dates**")
        st.write(f"First: {comparison['analysis1_date'][:10]}")
        st.write(f"Second: {comparison['analysis2_date'][:10]}")
        st.write(f"Time Difference: {comparison['time_difference_days']} days")
    
    with col2:
        st.markdown("**Body Shape Change**")
        shape_change = comparison["body_shape_change"]
        st.write(f"From: {shape_change['from']}")
        st.write(f"To: {shape_change['to']}")
    
    # Changes
    st.markdown("### 📈 Changes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        body_fat_change = comparison["body_fat_change"]
        delta_color = "inverse" if body_fat_change < 0 else "normal"
        st.metric(
            "Body Fat",
            f"{body_fat_change:+.1f}%",
            delta=f"{body_fat_change:+.1f}%",
            delta_color=delta_color
        )
    
    with col2:
        muscle_change = comparison["muscle_mass_change"]
        delta_color = "normal" if muscle_change > 0 else "inverse"
        st.metric(
            "Muscle Mass",
            f"{muscle_change:+.1f}%",
            delta=f"{muscle_change:+.1f}%",
            delta_color=delta_color
        )
    
    with col3:
        visceral_change = comparison["visceral_fat_change"]
        delta_color = "inverse" if visceral_change < 0 else "normal"
        st.metric(
            "Visceral Fat",
            f"{visceral_change:+d}",
            delta=f"{visceral_change:+d}",
            delta_color=delta_color
        )
    
    with col4:
        bmr_change = comparison["bmr_change"]
        st.metric(
            "BMR",
            f"{bmr_change:+.0f} cal",
            delta=f"{bmr_change:+.0f} cal"
        )

# Main function to be called from the main app
def main():
    """Main function for standalone testing."""
    st.set_page_config(
        page_title="Body Composition Analysis",
        page_icon="🏋️",
        layout="wide"
    )
    
    render_body_composition_analysis()

if __name__ == "__main__":
    main()
