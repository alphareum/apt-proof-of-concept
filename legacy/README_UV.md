# 🔬 APT Proof of Concept - AI Fitness Assistant (UV-Optimized)

A comprehensive research project demonstrating Advanced Pose Tracking (APT) technologies through a practical fitness application, now restructured with modern Python tooling.

## 🎯 Project Overview

This proof-of-concept showcases the integration of computer vision, pose detection, and AI technologies using **modern Python packaging** with UV and standard virtual environments.

**Key Technologies:**
- **Pose Detection & Analysis** using Google MediaPipe
- **Computer Vision** for body composition analysis
- **Machine Learning** for personalized recommendations
- **Web Application Development** with Streamlit
- **Modern Python Packaging** with UV and pyproject.toml

## 🚀 Quick Start

### ⚡ Method 1: UV Package Manager (Recommended - Fastest)
```bash
# Setup and run in one command
python run.py --setup --run

# Or step by step:
python run.py --setup     # Setup environment
python run.py --run       # Run application
```

### 🖱️ Method 2: One-Click Setup (Windows)
- **UV Setup**: Double-click `setup-uv.bat` or run `setup-uv.ps1`
- **Quick Run**: Double-click `run-app.bat` or run `run-app.ps1`

### 🔧 Method 3: Manual UV Setup
```bash
# Install UV if not available
pip install uv

# Create virtual environment
uv venv .venv

# Install dependencies (Windows)
uv pip install -r requirements.txt --python .venv/Scripts/python
# Install dependencies (Linux/macOS)
uv pip install -r requirements.txt --python .venv/bin/python

# Activate and run
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/macOS
streamlit run app.py
```

### 🐍 Method 4: Traditional Python (Fallback)
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Modern Project Structure

```
├── app.py                      # Main Streamlit application
├── pyproject.toml             # Modern Python project configuration
├── requirements.txt           # Dependencies for UV/pip
├── .venv/                     # Virtual environment (standard location)
│
├── run.py                     # Cross-platform setup and runner
├── setup-uv.ps1/.bat         # UV setup scripts for Windows
├── run-app.ps1/.bat          # Quick run scripts for Windows
│
├── models.py                  # Data models and database
├── database.py                # Database operations
├── recommendation_engine.py   # AI recommendation system
├── body_composition_*.py      # Body analysis modules
├── ui_components.py           # Streamlit UI components
├── utils.py                   # Utility functions
│
├── .vscode/tasks.json         # VS Code tasks for development
├── .env.template              # Environment configuration template
└── README.md                  # This file
```

## 📋 Features

### 🏋️‍♀️ Fitness Application
- **Body Fat Analysis**: AI-powered body composition from photos
- **Exercise Recommendations**: Personalized workout plans
- **Form Correction**: Real-time pose analysis and feedback
- **Progress Tracking**: Monitor fitness improvements over time

### 🛠️ Development Tools (New UV-based)
- `run.py` - Cross-platform setup and runner (Python)
- `setup-uv.ps1/.bat` - UV-based setup scripts for Windows
- `run-app.ps1/.bat` - Quick run scripts for Windows
- `pyproject.toml` - Modern Python packaging configuration

### 🛠️ Legacy Setup Tools (Still Available)
- `setup_fitness_app.bat` - Original Windows batch setup
- `setup_fitness_app.ps1` - Original PowerShell setup
- `run_fitness_app.py` - Original cross-platform installer

## 🔧 Technical Stack

### Core Technologies
- **Frontend**: Streamlit web application
- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: PIL, Albumentations, ImageHash
- **Visualization**: Matplotlib, Seaborn, Plotly

### Development Tools
- **Package Manager**: UV (ultra-fast Python package manager)
- **Project Config**: pyproject.toml (modern Python standard)
- **Virtual Environment**: Standard `.venv` directory
- **Task Runner**: VS Code tasks for common operations

## 🎮 Usage Options

### VS Code Integration
Use the Command Palette (`Ctrl+Shift+P`) → "Run Task":
- **Setup and run APT fitness app (UV)** - Complete setup and launch
- **Setup environment only (UV)** - Just create environment
- **Run APT fitness app** - Launch with existing environment
- **Clean and reinstall environment** - Fresh start

### Command Line Options
```bash
python run.py --help                    # Show all options
python run.py --setup                   # Setup only
python run.py --run                     # Run only
python run.py --setup --run             # Setup and run
python run.py --setup --dev             # Setup with dev dependencies
python run.py --setup --force           # Force recreate environment
python run.py --run --port 8502         # Run on custom port
python run.py --run --host 0.0.0.0      # Run on all interfaces
```

## 🌟 What's New in the UV-Optimized Version

### ✅ Improvements
- **⚡ Faster Setup**: UV can install dependencies 10-100x faster than pip
- **📁 Standard Structure**: Uses `.venv` instead of custom `fitness_venv`
- **🔧 Modern Config**: `pyproject.toml` for project metadata and dependencies
- **🎯 Cross-Platform**: Single `run.py` script works on Windows, Linux, macOS
- **🔄 Fallback Support**: Automatically falls back to pip if UV unavailable
- **📝 Better Tasks**: Improved VS Code tasks for development workflow

### 🔄 Migration from Legacy Setup
Your old `fitness_venv` has been replaced with the standard `.venv`. To migrate:

1. **Clean old environment**: The old `fitness_venv` directory was removed
2. **Run new setup**: Use any of the quick start methods above
3. **Update bookmarks**: Change any scripts pointing to `fitness_venv` to `.venv`

## 🔍 Troubleshooting

### UV Not Found
If you see "uv is not recognized", the script will automatically:
1. Try to install UV via pip
2. Fall back to standard pip installation
3. Still create a `.venv` virtual environment

### Dependencies Issues
```bash
# Force reinstall everything
python run.py --setup --force

# Or manually clean and reinstall
Remove-Item -Recurse -Force .venv  # Windows
rm -rf .venv                       # Linux/macOS
python run.py --setup
```

### Permission Issues (Windows)
If PowerShell scripts won't run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📊 Performance Comparison

| Setup Method | Speed | Compatibility | Ease of Use |
|-------------|-------|---------------|-------------|
| UV + run.py | ⚡⚡⚡ | 🟢 Excellent | 🟢 One Command |
| Setup-UV scripts | ⚡⚡ | 🟡 Windows Only | 🟢 Double-click |
| Traditional pip | ⚡ | 🟢 Universal | 🟡 Manual steps |

## 🤝 Contributing

The project now uses modern Python standards:
- Dependencies: Add to `pyproject.toml`
- Development: Use `python run.py --setup --dev`
- Testing: Run via VS Code tasks or `python run.py --run`

## 📈 Roadmap

- [ ] Add `uv.lock` file for dependency locking
- [ ] Implement pre-commit hooks
- [ ] Add automated testing with pytest
- [ ] Docker containerization
- [ ] CI/CD pipeline integration

---

🎯 **Ready to start?** Run `python run.py --setup --run` and you'll have the fitness app running in minutes!
