# 🏋️‍♀️ AI Fitness Assistant

A comprehensive Streamlit-based fitness application that provides AI-powered body composition analysis, personalized exercise recommendations, and workout form correction.

## ✨ Features

### 📊 Tab 1: Body Fat Analysis
- **Image-based body composition analysis** using computer vision
- **Anthropometric calculations** (Navy Method, BMI-based estimations)
- **Activity level adjustments** for more accurate results
- **Visual feedback** with body detection and analysis

### 💪 Tab 2: Exercise Recommendations
- **Personalized workout plans** based on user goals and preferences
- **Equipment-aware recommendations** (works with or without gym equipment)
- **Injury-conscious planning** that avoids problematic exercises
- **Caloric burn estimates** and BMR calculations
- **Weekly workout schedules** with balanced training

### 🎯 Tab 3: Workout Form Correction
- **Real-time pose detection** using Google MediaPipe
- **Exercise-specific form analysis** (squats, push-ups, deadlifts, etc.)
- **Visual pose annotation** with skeleton overlay
- **Detailed feedback** on form improvements
- **Safety warnings** for injury prevention

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or camera (optional, for live analysis)

### Installation & Setup

1. **Clone or download the project files:**
   ```bash
   # Make sure you have these files:
   # - fitness_app.py
   # - requirements_fitness.txt
   # - run_fitness_app.py
   ```

2. **Run the demo script:**
   ```bash
   python run_fitness_app.py
   ```
   
   The script will:
   - Install required packages automatically
   - Start the Streamlit application
   - Open your browser to `http://localhost:8502`

3. **Manual installation (alternative):**
   ```bash
   pip install -r requirements_fitness.txt
   streamlit run fitness_app.py --server.port 8502
   ```

## 🎮 How to Use

### Getting Started
1. **Fill out your profile** in the sidebar (age, weight, height, activity level)
2. **Navigate between tabs** to use different features
3. **Upload images** for analysis or follow the guided forms

### Tab 1: Body Fat Analysis
1. Upload a clear, full-body photo
2. Optionally enter body measurements (waist, neck, hip)
3. Get instant body composition analysis
4. View recommendations based on your results

### Tab 2: Exercise Recommendations
1. Set your fitness goals (weight loss, muscle gain, etc.)
2. Specify available equipment and any injuries
3. Choose workout duration and frequency
4. Get a personalized weekly workout plan

### Tab 3: Form Correction
1. Select your exercise type (squat, push-up, etc.)
2. Upload a photo of yourself performing the exercise
3. Get detailed form analysis with visual feedback
4. Follow specific tips for improvement

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Google's pose detection library
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Data visualization
- **Pillow**: Image handling

### Body Fat Analysis Methods
1. **Navy Method**: Uses body measurements for accurate calculation
2. **BMI-based Estimation**: Considers age, activity level, and demographics
3. **Computer Vision**: Analyzes body shape and composition from images

### Pose Detection
- Uses Google MediaPipe for 33-point body landmark detection
- Analyzes joint angles and body alignment
- Provides exercise-specific form feedback
- Real-time visual annotations

## 📱 Screenshots & Examples

### Body Fat Analysis
- Upload body photos for composition analysis
- Get detailed metrics including body fat percentage
- Receive health category classification
- View measurement-based calculations

### Exercise Recommendations
- Personalized plans based on goals and equipment
- Weekly schedules with balanced workouts
- Calorie burn estimates and BMR calculations
- Equipment-aware exercise selection

### Form Correction
- Real-time pose detection and analysis
- Visual skeleton overlay on your photos
- Exercise-specific feedback and tips
- Safety warnings for injury prevention

## 🤝 Support & Troubleshooting

### Common Issues

**MediaPipe Installation:**
```bash
pip install mediapipe
```

**Camera/Image Upload Issues:**
- Ensure good lighting for pose detection
- Use contrasting clothing against background
- Show full body in frame for best results

**Performance Issues:**
- Use smaller image sizes for faster processing
- Close other applications to free up memory
- Ensure stable internet connection for initial setup

### Tips for Best Results

**Body Fat Analysis:**
- Use well-lit, full-body photos
- Wear form-fitting clothing
- Stand against plain background
- Include body measurements when possible

**Exercise Recommendations:**
- Be honest about fitness level and goals
- Update profile information regularly
- Consider any injuries or limitations
- Start with shorter workout durations

**Form Correction:**
- Ensure entire body is visible in frame
- Use good lighting conditions
- Wear contrasting colors
- Follow exercise-specific setup tips

## 📊 Data & Privacy

- **No data storage**: All analysis is done locally
- **No personal information shared**: Your photos and data stay on your device
- **Offline capable**: Most features work without internet connection
- **Privacy-first design**: No user tracking or data collection

## 🔄 Updates & Roadmap

### Current Version: 1.0.0
- ✅ Body fat analysis from images
- ✅ Personalized exercise recommendations
- ✅ Pose-based form correction
- ✅ Multi-method body composition analysis

### Planned Features
- 🔄 Video analysis for dynamic form correction
- 🔄 Live camera feed integration
- 🔄 Progress tracking and history
- 🔄 Nutrition recommendations
- 🔄 Workout timer and guidance
- 🔄 Social features and challenges

## 📄 License

This project is open source and available under the MIT License.

## 🙋‍♀️ Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Disclaimer**: This application provides general fitness guidance and should not replace professional medical or fitness advice. Always consult with healthcare professionals before starting new exercise programs.
