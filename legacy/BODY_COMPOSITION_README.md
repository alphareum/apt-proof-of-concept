# Body Composition Analysis Documentation

## Overview

The Body Composition Analysis feature provides advanced computer vision-based analysis of body composition from images. This feature integrates cutting-edge AI technologies to deliver accurate estimations of body fat, muscle mass, and other health metrics.

## Features

### 🎯 Core Analysis Capabilities
- **Body Fat Percentage**: AI-estimated using pose landmarks and anthropometric ratios
- **Muscle Mass Percentage**: Calculated based on body shape and proportions
- **Visceral Fat Level**: Estimated on a 1-20 scale for health assessment
- **BMR Estimation**: Basal Metabolic Rate calculation for diet planning
- **Body Shape Classification**: Athletic, pear, apple, rectangle, inverted triangle
- **Confidence Scoring**: Analysis reliability assessment

### 📊 Detailed Measurements
- Shoulder width and arm length
- Waist width and hip circumference
- Body height and leg length
- Body ratios (waist-to-hip, shoulder-to-waist, etc.)
- Symmetry assessment

### 📈 Progress Tracking
- Historical analysis comparison
- Trend analysis over time
- Progress visualization
- Goal tracking integration

## Technical Implementation

### Computer Vision Pipeline

1. **Image Upload & Preprocessing**
   - Support for JPG, PNG, JPEG formats
   - Image quality assessment
   - Pose detection readiness check

2. **Pose Landmark Extraction**
   - MediaPipe Pose model with 33 landmarks
   - High-precision body keypoint detection
   - Visibility scoring for quality assessment

3. **Body Segmentation**
   - MediaPipe Selfie Segmentation
   - Body area calculation
   - Fat distribution analysis

4. **Measurement Extraction**
   - Geometric calculations from landmarks
   - Anthropometric ratio computation
   - Body proportion analysis

5. **ML-Based Estimation**
   - Random Forest models for composition prediction
   - Trained on anthropometric research data
   - Feature engineering from body ratios

### Machine Learning Models

#### Body Fat Estimation Model
- **Algorithm**: Random Forest Regressor
- **Features**: 
  - Waist-to-height ratio
  - Waist-to-hip ratio
  - Shoulder-to-waist ratio
  - Arm-to-height ratio
  - Leg-to-height ratio
  - Body symmetry score

#### Muscle Mass Estimation Model
- **Algorithm**: Random Forest Regressor
- **Training**: Synthetic anthropometric data
- **Validation**: Cross-validated for accuracy

### Database Schema

```sql
-- Body composition analysis table
CREATE TABLE body_composition_analysis (
    analysis_id TEXT PRIMARY KEY,
    user_id TEXT,
    image_path TEXT,
    analysis_date TIMESTAMP,
    body_fat_percentage REAL,
    muscle_mass_percentage REAL,
    visceral_fat_level INTEGER,
    bmr_estimated INTEGER,
    body_shape_classification TEXT,
    confidence_score REAL,
    analysis_method TEXT,
    front_image_path TEXT,
    side_image_path TEXT,
    processed_image_path TEXT,
    body_measurements_json TEXT,
    composition_breakdown_json TEXT
);

-- Body part measurements table
CREATE TABLE body_part_measurements (
    measurement_id TEXT PRIMARY KEY,
    analysis_id TEXT,
    body_part TEXT,
    circumference_cm REAL,
    area_percentage REAL,
    muscle_definition_score REAL,
    fat_distribution_score REAL,
    symmetry_score REAL
);

-- Progress tracking table
CREATE TABLE composition_progress (
    progress_id TEXT PRIMARY KEY,
    user_id TEXT,
    start_analysis_id TEXT,
    end_analysis_id TEXT,
    progress_type TEXT,
    change_percentage REAL,
    time_period_days INTEGER,
    trend_direction TEXT
);
```

## API Endpoints

### Flask REST API

```python
# Body composition analysis
POST /analyze-body-composition
- Upload image(s) for analysis
- Returns detailed composition results

# User history
GET /user/{user_id}/body-composition-history?days=90
- Retrieve analysis history

# Latest analysis
GET /user/{user_id}/latest-body-composition
- Get most recent analysis

# Progress calculation
GET /user/{user_id}/composition-progress?period_days=30
- Calculate progress over time

# Analysis comparison
POST /compare-analyses
- Compare two analyses

# Processed images
GET /processed-image/{filename}
- Retrieve annotated analysis images
```

## Usage Guide

### For Users

1. **Taking Photos**
   - Use good lighting (natural light preferred)
   - Wear fitted clothing
   - Stand straight facing camera
   - Full body should be visible
   - Use consistent conditions for progress tracking

2. **Uploading Images**
   - Primary image (required): Front-facing full body
   - Additional images (optional): Front view, side view
   - Supported formats: JPG, PNG, JPEG
   - Maximum file size: 16MB

3. **Understanding Results**
   - **Body Fat %**: 
     - <10%: Very Low (may be unhealthy)
     - 10-15%: Athletic
     - 15-25%: Healthy
     - 25-30%: Above Average
     - >30%: High (health risk)
   
   - **Muscle Mass %**: Higher is generally better
   - **Visceral Fat Level**: Lower is better (1-20 scale)
   - **BMR**: Calories needed at rest

### For Developers

1. **Installation**
   ```bash
   pip install -r requirements_fitness.txt
   ```

2. **Basic Usage**
   ```python
   from body_composition_analyzer import get_body_analyzer
   
   analyzer = get_body_analyzer()
   result = analyzer.analyze_image(
       image_path="path/to/image.jpg",
       user_id="user123"
   )
   ```

3. **Integration with Streamlit**
   ```python
   from body_composition_ui import render_body_composition_analysis
   
   # In your Streamlit app
   render_body_composition_analysis()
   ```

## Dependencies

### Required Packages
- **opencv-python**: Image processing
- **mediapipe**: Pose detection and segmentation
- **scikit-learn**: Machine learning models
- **tensorflow**: Deep learning backend
- **streamlit**: Web interface
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **Pillow**: Image handling

### Optional Packages
- **flask**: REST API server
- **flask-cors**: CORS support
- **werkzeug**: HTTP utilities

## Configuration

### Model Parameters
```python
# Pose detection settings
POSE_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 2  # 0, 1, or 2

# Analysis thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
MIN_VISIBILITY_SCORE = 0.5

# File settings
MAX_FILE_SIZE_MB = 16
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']
```

## Limitations and Considerations

### Current Limitations
1. **Accuracy**: Estimations based on visual analysis, not medical-grade
2. **Lighting Dependency**: Results affected by photo quality
3. **Clothing**: Fitted clothing provides better results
4. **Demographics**: Models trained on general population data
5. **Individual Variation**: Results may vary based on body type

### Future Improvements
1. **Enhanced Models**: Training on larger, more diverse datasets
2. **Multiple View Analysis**: Combining front, side, and back views
3. **Calibration**: Using reference objects for scale
4. **Medical Integration**: Validation with DEXA scan data
5. **Real-time Analysis**: Video-based analysis

## Privacy and Security

### Data Handling
- Images processed locally when possible
- Only pose landmarks sent to cloud (if applicable)
- Raw images can be automatically deleted after processing
- User consent required for data storage

### Compliance
- GDPR compliant data handling
- User data export/deletion capabilities
- Anonymized analytics only

## Performance Metrics

### Target Performance
- **Analysis Speed**: <5 seconds per image
- **Accuracy**: ±3% for body fat estimation
- **Reliability**: 95% successful pose detection
- **Uptime**: 99.9% API availability

### Monitoring
- Analysis success rate tracking
- Confidence score distribution
- User feedback integration
- Performance benchmarking

## Support and Troubleshooting

### Common Issues
1. **"No pose detected"**: Check lighting and full body visibility
2. **Low confidence scores**: Improve image quality
3. **Import errors**: Install all required dependencies
4. **Memory issues**: Reduce image size or upgrade hardware

### Getting Help
- Check logs for detailed error messages
- Verify all dependencies are installed
- Ensure adequate system resources
- Contact support with error details

## License and Attribution

This body composition analysis system uses:
- MediaPipe (Apache 2.0)
- OpenCV (Apache 2.0)  
- Scikit-learn (BSD 3-Clause)
- TensorFlow (Apache 2.0)

Commercial use requires appropriate licensing for all components.
