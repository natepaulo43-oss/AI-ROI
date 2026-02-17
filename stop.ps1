Write-Host "Stopping services..." -ForegroundColor Yellow

Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "node"} | ForEach-Object {
    try {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped $($_.ProcessName) (PID: $($_.Id))"
    } catch {}
}

Write-Host "Done" -ForegroundColor Green
