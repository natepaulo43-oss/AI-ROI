# AI ROI Web Tool - Stop Script
# Stops both backend and frontend services

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "AI ROI Prediction Tool - Stopping Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$stoppedCount = 0

# Stop background jobs
$jobs = Get-Job -ErrorAction SilentlyContinue
if ($jobs) {
    Write-Host "Stopping background jobs..." -ForegroundColor Yellow
    foreach ($job in $jobs) {
        Write-Host "  Stopping job $($job.Id) ($($job.Name))..." -ForegroundColor Gray
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -Force -ErrorAction SilentlyContinue
        $stoppedCount++
    }
    Write-Host "  Background jobs stopped ($($jobs.Count) job(s))" -ForegroundColor Green
} else {
    Write-Host "No background jobs running" -ForegroundColor Gray
}

# Stop processes on port 8000 (Backend)
Write-Host "Checking Backend (port 8000)..." -ForegroundColor Yellow
$backendConnections = netstat -ano | Select-String ":8000" | Select-String "LISTENING"
if ($backendConnections) {
    $backendConnections | ForEach-Object {
        $line = $_.ToString().Trim()
        $parts = $line -split '\s+'
        $pid = $parts[-1]
        if ($pid -match '^\d+$') {
            try {
                $process = Get-Process -Id $pid -ErrorAction Stop
                Write-Host "  Stopping process $pid ($($process.ProcessName))..." -ForegroundColor Gray
                Stop-Process -Id $pid -Force -ErrorAction Stop
                $stoppedCount++
                Write-Host "  Backend stopped (PID: $pid)" -ForegroundColor Green
            } catch {
                Write-Host "  Could not stop process $pid" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "  Backend not running on port 8000" -ForegroundColor Gray
}

# Stop processes on port 3000 (Frontend)
Write-Host "Checking Frontend (port 3000)..." -ForegroundColor Yellow
$frontendConnections = netstat -ano | Select-String ":3000" | Select-String "LISTENING"
if ($frontendConnections) {
    $frontendConnections | ForEach-Object {
        $line = $_.ToString().Trim()
        $parts = $line -split '\s+'
        $pid = $parts[-1]
        if ($pid -match '^\d+$') {
            try {
                $process = Get-Process -Id $pid -ErrorAction Stop
                Write-Host "  Stopping process $pid ($($process.ProcessName))..." -ForegroundColor Gray
                Stop-Process -Id $pid -Force -ErrorAction Stop
                $stoppedCount++
                Write-Host "  Frontend stopped (PID: $pid)" -ForegroundColor Green
            } catch {
                Write-Host "  Could not stop process $pid" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "  Frontend not running on port 3000" -ForegroundColor Gray
}

# Additional cleanup: Stop any uvicorn processes in this project directory
$rootPath = $PWD.Path
$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*$rootPath*"
}
if ($pythonProcesses) {
    Write-Host "Cleaning up Python processes..." -ForegroundColor Yellow
    foreach ($proc in $pythonProcesses) {
        try {
            Write-Host "  Stopping Python process $($proc.Id)..." -ForegroundColor Gray
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            $stoppedCount++
        } catch {
            # Already stopped or access denied
        }
    }
}

# Additional cleanup: Stop any node processes in this project directory
$nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*$rootPath*"
}
if ($nodeProcesses) {
    Write-Host "Cleaning up Node processes..." -ForegroundColor Yellow
    foreach ($proc in $nodeProcesses) {
        try {
            Write-Host "  Stopping Node process $($proc.Id)..." -ForegroundColor Gray
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            $stoppedCount++
        } catch {
            # Already stopped or access denied
        }
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
if ($stoppedCount -gt 0) {
    Write-Host "Services Stopped ($stoppedCount process(es))" -ForegroundColor Green
} else {
    Write-Host "No Services Were Running" -ForegroundColor Gray
}
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
