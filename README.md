# 🔬 APT Proof of Concept - AI Fitness Assistant

A comprehensive research project demonstrating Advanced Pose Tracking (APT) technologies through a practical fitness application.

## 🎯 Project Overview

This proof-of-concept showcases the integration of computer vision, pose detection, and AI technologies to create a real-world fitness application. The project demonstrates various aspects of modern AI development including:

- **Pose Detection & Analysis** using Google MediaPipe
- **Computer Vision** for body composition analysis
- **Machine Learning** for personalized recommendations
- **Web Application Development** with Streamlit
- **Modern Python Packaging** with UV and traditional tools

## 🚀 Quick Start

### Choose Your Installation Method:

#### 🖱️ One-Click Setup (Windows)
- **Batch File**: Double-click `setup_fitness_app.bat`
- **PowerShell**: Right-click `setup_fitness_app.ps1` → "Run with PowerShell"

#### ⚡ UV Package Manager (Recommended)
```bash
# Install UV (ultra-fast Python package manager)
pip install uv

# Setup and run
uv venv fitness_env
source fitness_env/bin/activate  # Windows: fitness_env\Scripts\activate
uv pip install -r requirements_fitness.txt
uvx streamlit run fitness_app.py --server.port 8502
```

#### 🐍 Traditional Python
```bash
python run_fitness_app.py
```

## 📋 What's Included

### 🏋️‍♀️ Fitness Application
- **Body Fat Analysis**: AI-powered body composition from photos
- **Exercise Recommendations**: Personalized workout plans
- **Form Correction**: Real-time pose analysis and feedback

### 🛠️ Setup Tools
- `setup_fitness_app.bat` - Windows batch setup
- `setup_fitness_app.ps1` - PowerShell setup with enhanced features  
- `run_fitness_app.py` - Cross-platform Python installer
- `cleanup_fitness.py/.bat` - Environment cleanup utilities

### 📚 Documentation
- `FITNESS_README.md` - Complete application documentation
- `QUICK_START.md` - Rapid deployment guide
- This `README.md` - Project overview

## 🔧 Technical Stack

- **Frontend**: Streamlit web application
- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Package Management**: UV, pip, virtual environments

## 📖 Documentation

- **[Detailed Setup Guide](FITNESS_README.md)** - Complete installation and usage instructions
- **[Quick Start Guide](QUICK_START.md)** - Get running in minutes
- **[Requirements](requirements_fitness.txt)** - Python dependencies

## 🎮 Usage

1. **Start the application** using any of the setup methods above
2. **Open your browser** to `http://localhost:8502`
3. **Explore the three main features**:
   - Body fat analysis from photos
   - Personalized exercise recommendations
   - Real-time workout form correction

## 🔬 Research Applications

This proof-of-concept demonstrates:

### Computer Vision Research
- Real-time pose detection and tracking
- Body composition analysis from images
- Multi-modal data fusion (images + measurements)

### AI/ML Applications
- Personalized recommendation systems
- Real-time feedback algorithms
- Health and fitness analytics

### Software Engineering
- Modern Python packaging and distribution
- Cross-platform deployment strategies
- User-friendly setup automation

## 🤝 Contributing

This is a research project showcasing various technologies. Contributions and feedback are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙋‍♀️ Support

- Check the **[Quick Start Guide](QUICK_START.md)** for common issues
- Review the **[Detailed Documentation](FITNESS_README.md)** for comprehensive help
- Use the provided cleanup scripts if you encounter environment issues

---

**Disclaimer**: This application provides general fitness guidance and should not replace professional medical or fitness advice. Always consult with healthcare professionals before starting new exercise programs.

**Research Purpose**: This is a proof-of-concept demonstrating AI and computer vision technologies for educational and research purposes.
