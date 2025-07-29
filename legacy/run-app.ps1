#!/usr/bin/env pwsh
# APT Fitness Assistant - Quick Run Script
# Activates .venv and runs the application

param(
    [int]$Port = 8501,
    [string]$Host = "localhost",
    [switch]$OpenBrowser = $true
)

Write-Host "🏋️‍♀️ Starting APT Fitness Assistant..." -ForegroundColor Cyan

# Check if virtual environment exists
if (!(Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup-uv.ps1 first to create the environment." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if app.py exists
if (!(Test-Path "app.py")) {
    Write-Host "❌ app.py not found!" -ForegroundColor Red
    Write-Host "Please ensure you're in the correct directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
try {
    & ".venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Build streamlit command
$streamlitArgs = @(
    "run", "app.py",
    "--server.port", $Port,
    "--server.address", $Host
)

if (-not $OpenBrowser) {
    $streamlitArgs += "--server.headless", "true"
}

Write-Host ""
Write-Host "🚀 Starting Streamlit server..." -ForegroundColor Green
Write-Host "📱 App will be available at: http://${Host}:${Port}" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run streamlit
try {
    & streamlit @streamlitArgs
} catch {
    Write-Host ""
    Write-Host "❌ Failed to start the application" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "👋 Application stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
