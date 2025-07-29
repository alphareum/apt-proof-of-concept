# 🚀 Quick Start Guide - AI Fitness Assistant

## The Problem You Encountered
You got a dependency conflict between different versions of Werkzeug (2.3.7 and 3.0.1) in the main requirements.txt file. This happens when different packages require different versions of the same dependency.

## ✅ Solution: Use the Fitness App Setup

I've created a separate, clean environment specifically for the fitness app to avoid conflicts with your main APT project.

### 🎯 Option 1: Automated Setup (Recommended)

**Windows Batch File:**
```bash
setup_fitness_app.bat
```

**PowerShell (Enhanced features):**
```powershell
.\setup_fitness_app.ps1
```

**Python Script (Cross-platform):**
```bash
python run_fitness_app.py
```

### 🎯 Option 2: UV Package Manager (Ultra-Fast)

If you have [UV](https://github.com/astral-sh/uv) installed:

```bash
# Install UV if you don't have it
pip install uv

# Quick setup and run
uv venv fitness_env
source fitness_env/bin/activate  # Windows: fitness_env\Scripts\activate
uv pip install -r requirements_fitness.txt
uvx streamlit run fitness_app.py --server.port 8502
```

**Why UV?**
- ⚡ 10-100x faster than pip
- 🔒 Better dependency resolution
- 🎯 More reliable installations

### 🎯 Option 3: Manual Setup

1. **Create isolated environment:**
   ```bash
   python -m venv fitness_venv
   ```

2. **Activate environment:**
   ```bash
   # Windows
   fitness_venv\Scripts\activate
   
   # Linux/Mac
   source fitness_venv/bin/activate
   ```

3. **Install fitness app dependencies:**
   ```bash
   # Standard pip
   pip install --upgrade pip
   pip install -r requirements_fitness.txt
   
   # OR with UV (faster)
   uv pip install -r requirements_fitness.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run fitness_app.py --server.port 8502
   ```

## 📁 What This Setup Does

1. **Creates a separate virtual environment** (`fitness_venv`) that won't conflict with your main APT project
2. **Uses clean dependencies** in `requirements_fitness.txt` (no version conflicts)
3. **Installs only what's needed** for the fitness app:
   - Streamlit for the web interface
   - OpenCV for computer vision
   - MediaPipe for pose detection
   - NumPy, Pandas for data processing
   - Matplotlib, Seaborn for visualizations

## 🔧 Files Created for You

- `fitness_app.py` - Main application with 3 tabs
- `requirements_fitness.txt` - Clean dependencies 
- `setup_fitness_app.bat` - Windows setup script
- `setup_fitness_app.ps1` - PowerShell setup script
- `run_fitness_app.py` - Python runner script
- `FITNESS_README.md` - Full documentation

## 🏋️‍♀️ What the App Does

### Tab 1: Body Fat Analysis
- Upload photos for body composition analysis
- Uses Navy Method and BMI calculations
- Computer vision analysis of body shape

### Tab 2: Exercise Recommendations  
- Personalized workout plans based on goals
- Equipment-aware recommendations
- Weekly schedules with calorie estimates

### Tab 3: Form Correction
- Real-time pose detection using MediaPipe
- Exercise-specific form analysis (squats, push-ups, etc.)
- Visual feedback with pose annotations

## 🚨 If You Still Get Errors

1. **Make sure you're in the right directory** where the fitness app files are located
2. **Try the PowerShell script** if the batch file doesn't work
3. **Use UV for faster, more reliable installs**:
   ```bash
   pip install uv
   uv pip install -r requirements_fitness.txt
   ```
4. **Check Python version** - needs Python 3.8 or higher
5. **Clear package cache** if needed:
   ```bash
   # Standard pip
   pip cache purge
   
   # With UV
   uv cache clean
   ```
6. **Try the cleanup scripts** to start fresh:
   ```bash
   python cleanup_fitness.py
   # OR
   cleanup_fitness.bat
   ```

## 🔄 To Keep Both Projects

- **Main APT project**: Use the existing `venv` and `requirements.txt`
- **Fitness app**: Use the new `fitness_venv` and `requirements_fitness.txt`

This way, you can work on both projects without dependency conflicts!

---

**Ready to try?** Run one of the setup scripts above and you should be good to go! 🎉
