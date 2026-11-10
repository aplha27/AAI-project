@echo off
setlocal enabledelayedexpansion

echo =========================================================================
echo Starting Agentic Multimodal eKYC Verification Platform (HSR-AGAI-029)
echo =========================================================================

:: 1. Check if .venv already exists in current workspace
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
    goto :PYTHON_READY
)

:: 2. If not, locate Python 3.12 and create .venv
set "SYS_PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "SYS_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) else if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" (
    set "SYS_PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
) else (
    set "SYS_PYTHON=python"
)

echo Creating local virtual environment in .venv...
"!SYS_PYTHON!" -m venv .venv --system-site-packages
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=!SYS_PYTHON!"
)

:PYTHON_READY
echo Using Virtual Environment: "!PYTHON_EXEC!"

:: Activate venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo.
echo [1/3] Checking synthetic benchmark dataset...
if not exist "backend\data\processed\bundles_manifest.json" (
    echo Building 200 identity bundles from Kaggle datasets...
    "!PYTHON_EXEC!" backend\data\build_from_kaggle.py
) else (
    echo Dataset found.
)

echo.
echo [2/3] Running automated test suite in .venv...
"!PYTHON_EXEC!" tests\test_full_suite.py
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Test suite failed! Please review test failures.
    pause
    exit /b !errorlevel!
)

echo.
echo [3/3] Launching FastAPI Backend and HITL Reviewer Console on http://127.0.0.1:8000
echo Access Reviewer Console Dashboard at: http://127.0.0.1:8000
echo API Documentation at: http://127.0.0.1:8000/docs
echo Press CTRL+C to terminate the server.
echo.

"!PYTHON_EXEC!" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
