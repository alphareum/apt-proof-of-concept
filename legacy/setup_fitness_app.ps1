# AI Fitness Assistant Setup - PowerShell Version
Write-Host "🏋️‍♀️ AI Fitness Assistant Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion detected" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "📋 Setting up virtual environment..." -ForegroundColor Yellow

if (!(Test-Path "fitness_venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv fitness_venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "📁 Virtual environment already exists" -ForegroundColor Blue
}

# Activate virtual environment and install dependencies
Write-Host ""
Write-Host "📋 Installing dependencies..." -ForegroundColor Yellow

# Activate virtual environment
& "fitness_venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "🔧 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "🔧 Installing fitness app requirements..." -ForegroundColor Yellow
python -m pip install -r requirements_fitness.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 To run the fitness app:" -ForegroundColor Cyan
    Write-Host "   Method 1: Run 'python run_fitness_app.py'" -ForegroundColor White
    Write-Host "   Method 2: Manual steps:" -ForegroundColor White
    Write-Host "      1. fitness_venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "      2. streamlit run fitness_app.py" -ForegroundColor Gray
    Write-Host ""
    
    $runNow = Read-Host "Do you want to start the app now? (y/n)"
    if ($runNow -eq "y" -or $runNow -eq "yes") {
        Write-Host "🚀 Starting AI Fitness Assistant..." -ForegroundColor Green
        streamlit run fitness_app.py --server.port 8502
    }
} else {
    Write-Host ""
    Write-Host "❌ Failed to install requirements" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Manual installation steps:" -ForegroundColor Yellow
    Write-Host "   1. fitness_venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "   2. pip install --upgrade pip" -ForegroundColor Gray
    Write-Host "   3. pip install -r requirements_fitness.txt" -ForegroundColor Gray
}

Read-Host "Press Enter to exit"
