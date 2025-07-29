#!/usr/bin/env pwsh
# APT Fitness Assistant Setup - UV Package Manager (Recommended)
# PowerShell script for Windows with UV integration

param(
    [switch]$ForceReinstall,
    [switch]$DevMode
)

Write-Host "🏋️‍♀️ APT Fitness Assistant Setup (UV)" -ForegroundColor Cyan
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

# Check if UV is installed, install if not
Write-Host ""
Write-Host "🔍 Checking UV installation..." -ForegroundColor Yellow

try {
    $uvVersion = uv --version 2>&1
    Write-Host "✅ UV $uvVersion detected" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing UV package manager..." -ForegroundColor Yellow
    try {
        python -m pip install uv
        Write-Host "✅ UV installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install UV. Falling back to pip." -ForegroundColor Red
        $useUV = $false
    }
}

# Clean up old virtual environment if force reinstall
if ($ForceReinstall -and (Test-Path ".venv")) {
    Write-Host "🧹 Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

# Create virtual environment with UV
Write-Host ""
Write-Host "📋 Setting up virtual environment..." -ForegroundColor Yellow

if (!(Test-Path ".venv")) {
    Write-Host "Creating .venv with UV..." -ForegroundColor Yellow
    try {
        uv venv .venv
        Write-Host "✅ Virtual environment created with UV" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ UV venv failed, falling back to python -m venv..." -ForegroundColor Yellow
        python -m venv .venv
        Write-Host "✅ Virtual environment created with venv" -ForegroundColor Green
    }
} else {
    Write-Host "📁 Virtual environment already exists" -ForegroundColor Blue
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow

try {
    # Try UV first
    Write-Host "🚀 Using UV for fast dependency installation..." -ForegroundColor Cyan
    
    if ($DevMode) {
        uv pip install -e ".[dev]" --python .venv/Scripts/python.exe
    } else {
        uv pip install -r requirements.txt --python .venv/Scripts/python.exe
    }
    
    Write-Host "✅ Dependencies installed with UV" -ForegroundColor Green
    $installSuccess = $true
} catch {
    Write-Host "⚠️ UV installation failed, trying with pip..." -ForegroundColor Yellow
    
    # Activate virtual environment
    & ".venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    if ($DevMode) {
        python -m pip install -e ".[dev]"
    } else {
        python -m pip install -r requirements.txt
    }
    
    Write-Host "✅ Dependencies installed with pip" -ForegroundColor Green
    $installSuccess = $true
}

if ($installSuccess) {
    Write-Host ""
    Write-Host "🎉 Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 To run the application:" -ForegroundColor Cyan
    Write-Host "1. Activate the environment:" -ForegroundColor White
    Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Run the fitness app:" -ForegroundColor White
    Write-Host "   streamlit run app.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Or use the quick start:" -ForegroundColor Cyan
    Write-Host "   .\run-app.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📚 For more options, see README.md" -ForegroundColor Blue
} else {
    Write-Host ""
    Write-Host "❌ Setup failed!" -ForegroundColor Red
    Write-Host "Please check the error messages above and try again." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
