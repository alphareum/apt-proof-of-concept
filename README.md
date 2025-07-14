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

#### ⚡ UV Package Manager (Recommended)
```bash
# One-command setup and run
python run.py --setup --run

# Or step by step:
python run.py --setup     # Setup environment
python run.py --run       # Run application
```

#### 🖱️ One-Click Setup (Windows)
- **UV Setup**: Double-click `setup-uv.bat` or run `setup-uv.ps1`
- **Quick Run**: Double-click `run-app.bat` or run `run-app.ps1`

#### 🔧 Manual UV Setup
```bash
# Install UV if not available
pip install uv

# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install -r requirements.txt --python .venv/Scripts/python

# Run application
.venv/Scripts/activate  # Windows
# or source .venv/bin/activate  # Unix-like
streamlit run app.py
```

#### 🐍 Traditional Python (Fallback)
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

## � Project Structure

```
├── app.py                      # Main Streamlit application
├── pyproject.toml             # Modern Python project configuration
├── requirements.txt           # Dependencies for UV/pip
├── .venv/                     # Virtual environment (standard location)
├── run.py                     # Cross-platform setup and runner
├── setup-uv.ps1/.bat         # UV setup scripts for Windows
├── run-app.ps1/.bat          # Quick run scripts for Windows
├── models.py                  # Data models and database
├── database.py                # Database operations
├── recommendation_engine.py   # AI recommendation system
├── body_composition_*.py      # Body analysis modules
├── ui_components.py           # Streamlit UI components
└── utils.py                   # Utility functions
```

## �📋 What's Included

### 🏋️‍♀️ Fitness Application
- **Body Fat Analysis**: AI-powered body composition from photos
- **Exercise Recommendations**: Personalized workout plans
- **Form Correction**: Real-time pose analysis and feedback

### 🛠️ Setup Tools (New UV-based)
- `run.py` - Cross-platform setup and runner (Python)
- `setup-uv.ps1/.bat` - UV-based setup scripts for Windows
- `run-app.ps1/.bat` - Quick run scripts for Windows
- `pyproject.toml` - Modern Python packaging configuration

### 🛠️ Legacy Setup Tools
- `setup_fitness_app.bat` - Original Windows batch setup
- `setup_fitness_app.ps1` - Original PowerShell setup
- `run_fitness_app.py` - Original cross-platform installer
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
