# Full AI ROI Data Pipeline
# This script runs the complete workflow: data integration -> model training -> API restart

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "AI ROI FULL PIPELINE" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

# Step 1: Integrate datasets
Write-Host "`n[1/3] Integrating datasets..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\data"
python integrate_datasets.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Data integration failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Train model
Write-Host "`n[2/3] Training model..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\backend"
python train_roi_model_improved.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Model training failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Restart backend (optional - user can do manually)
Write-Host "`n[3/3] Pipeline complete!" -ForegroundColor Green
Write-Host "`n✅ New model trained and saved to backend/models/" -ForegroundColor Green
Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Restart the backend API to load the new model" -ForegroundColor White
Write-Host "   2. Test predictions with the updated model" -ForegroundColor White
Write-Host "`n   To restart backend, run:" -ForegroundColor Cyan
Write-Host "   cd backend && uvicorn app.main:app --reload" -ForegroundColor White

Set-Location $PSScriptRoot\..
