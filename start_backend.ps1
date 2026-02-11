# Quick start script for the AI ROI backend API

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "AI ROI Prediction API - Starting Backend" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Check if model exists
if (-Not (Test-Path "backend\models\roi_model.pkl")) {
    Write-Host ""
    Write-Host "WARNING: Model file not found!" -ForegroundColor Red
    Write-Host "Training model now..." -ForegroundColor Yellow
    python backend\train_roi_model.py
    Write-Host ""
}

# Start the API server
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Set-Location backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
