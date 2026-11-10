# PowerShell Startup Script for eKYC Platform
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Starting Agentic Multimodal eKYC Verification Platform (HSR-AGAI-029)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

# 1. Check if .venv exists
$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating local virtual environment in .venv..." -ForegroundColor Yellow
    $sysPy = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    if (-not (Test-Path $sysPy)) {
        $sysPy = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
        if (-not (Test-Path $sysPy)) { $sysPy = "python" }
    }
    & $sysPy -m venv .venv --system-site-packages
}

# 2. Activate venv
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

$pythonExec = ".\.venv\Scripts\python.exe"
Write-Host "Using Virtual Environment: $pythonExec" -ForegroundColor Green

Write-Host "`n[1/3] Checking synthetic benchmark dataset..." -ForegroundColor Yellow
if (-not (Test-Path "backend\data\processed\bundles_manifest.json")) {
    Write-Host "Building 200 identity bundles from Kaggle datasets..."
    & $pythonExec backend\data\build_from_kaggle.py
} else {
    Write-Host "Dataset found." -ForegroundColor Green
}

Write-Host "`n[2/3] Running automated test suite in .venv..." -ForegroundColor Yellow
& $pythonExec tests\test_full_suite.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Test suite failed! Please review test failures." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[3/3] Launching FastAPI Backend and HITL Reviewer Console on http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "Access Reviewer Console Dashboard at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API Documentation at: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Press CTRL+C to terminate the server.`n" -ForegroundColor DarkGray

& $pythonExec -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
