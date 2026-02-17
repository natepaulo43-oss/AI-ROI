# AI ROI Web Tool - Start Script
# Runs backend and frontend as background processes

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "AI ROI Prediction Tool - Starting Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "Then install dependencies: .venv\Scripts\pip install -r backend\requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Check if models exist
if (-Not (Test-Path "backend\models\roi_model.pkl") -or -Not (Test-Path "backend\models\roi_classifier_best.pkl")) {
    Write-Host "WARNING: Model files not found!" -ForegroundColor Red
    Write-Host "Training models now..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
    python backend\train_roi_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Model training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# Check for port conflicts
$backendPort = netstat -ano | Select-String ":8000" | Select-String "LISTENING"
if ($backendPort) {
    Write-Host "WARNING: Port 8000 is already in use!" -ForegroundColor Yellow
    Write-Host "Please stop the existing service or run .\stop.ps1 first" -ForegroundColor Yellow
    exit 1
}

$frontendPort = netstat -ano | Select-String ":3000" | Select-String "LISTENING"
if ($frontendPort) {
    Write-Host "WARNING: Port 3000 is already in use!" -ForegroundColor Yellow
    Write-Host "Please stop the existing service or run .\stop.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Get absolute paths
$rootPath = $PWD.Path
$venvPython = Join-Path $rootPath ".venv\Scripts\python.exe"
$backendPath = Join-Path $rootPath "backend"
$frontendPath = Join-Path $rootPath "frontend"

# Start Backend
Write-Host "Starting Backend API..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    param($pythonPath, $backendDir)
    Set-Location $backendDir
    & $pythonPath -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
} -ArgumentList $venvPython, $backendPath

Write-Host "  Backend job started (ID: $($backendJob.Id))" -ForegroundColor Gray

# Wait for backend to start
Write-Host "  Waiting for backend to initialize..." -ForegroundColor Gray
$backendReady = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "  Backend is ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Still starting up
    }
}

if (-not $backendReady) {
    Write-Host "  WARNING: Backend may still be starting (check logs with: Receive-Job $($backendJob.Id))" -ForegroundColor Yellow
}

# Check if node_modules exists
if (-Not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location frontend
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: npm install failed!" -ForegroundColor Red
        Stop-Job $backendJob
        Remove-Job $backendJob
        Set-Location ..
        exit 1
    }
    Set-Location ..
}

# Start Frontend
Write-Host "Starting Frontend..." -ForegroundColor Green
$frontendJob = Start-Job -ScriptBlock {
    param($frontendDir)
    Set-Location $frontendDir
    npm run dev
} -ArgumentList $frontendPath

Write-Host "  Frontend job started (ID: $($frontendJob.Id))" -ForegroundColor Gray
Write-Host "  Waiting for frontend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "All Services Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs:     http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Frontend:     http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job IDs (for monitoring):" -ForegroundColor White
Write-Host "  Backend:  $($backendJob.Id)" -ForegroundColor Gray
Write-Host "  Frontend: $($frontendJob.Id)" -ForegroundColor Gray
Write-Host ""
Write-Host "View logs:" -ForegroundColor White
Write-Host "  Receive-Job $($backendJob.Id) -Keep" -ForegroundColor Gray
Write-Host "  Receive-Job $($frontendJob.Id) -Keep" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop services, run: .\stop.ps1" -ForegroundColor Magenta
Write-Host ""
